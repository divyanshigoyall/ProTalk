import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import pymongo

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import io

load_dotenv()

st.set_page_config(
    page_title="E-commerce Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
/* Give enough space so the pinned chat_input doesn't overlap content */
.main .block-container { padding-bottom: 120px; }

/* Chat messages container spacing */
div[data-testid="chatMessage"] { margin-bottom: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

@st.cache_resource
def init_database():
    DB_CONNECT = os.getenv("CONNECTION_DB")
    GEMINI_API = os.getenv("GEMINI_API")
    os.environ["GOOGLE_API_KEY"] = GEMINI_API

    client = pymongo.MongoClient(DB_CONNECT)
    db = client.E_Commerce
    collection = db.Products
    chat_history_collection = db.chat_history
    return client, db, collection, chat_history_collection

client, db, collection, chat_history_collection = init_database()

@st.cache_resource
def init_ai_components():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, max_tokens=790)

    vectorStore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index",
        text_key="embedding_text",
        embedding_key="embedding",
    )

    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return llm, retriever

llm, retriever = init_ai_components()

def text_to_speech(text: str):
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def save_message_to_db(session_id, message, sender="user", response=None):
    chat_doc = {
        "session_id": session_id,
        "message": message,
        "sender": sender,
        "timestamp": datetime.utcnow(),
        "response": response,
    }
    chat_history_collection.insert_one(chat_doc)

def get_chat_history(session_id, limit=50):
    history = list(
        chat_history_collection.find({"session_id": session_id})
        .sort("timestamp", 1)
        .limit(limit)
    )
    return [
        {
            "message": doc["message"],
            "sender": doc["sender"],
            "timestamp": doc["timestamp"],
            "response": doc.get("response"),
        }
        for doc in history
    ]

def get_all_sessions():
    pipeline = [
        {"$sort": {"timestamp": -1}},
        {
            "$group": {
                "_id": "$session_id",
                "latest_message": {"$first": "$message"},
                "latest_timestamp": {"$first": "$timestamp"},
                "message_count": {"$sum": 1},
            }
        },
        {"$sort": {"latest_timestamp": -1}},
        {"$limit": 20},
    ]
    return list(chat_history_collection.aggregate(pipeline))

def process_message(user_text: str):
    if not user_text or not user_text.strip():
        return

    # UI history
    st.session_state.messages.append({"role": "user", "content": user_text})

    # DB save
    save_message_to_db(st.session_state.session_id, user_text, "user")

    try:
        # Build memory from stored chat
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        history = get_chat_history(st.session_state.session_id, limit=10)
        for item in history[:-1]:  # exclude current just-saved user msg
            if item["sender"] == "user" and item.get("response"):
                memory.chat_memory.add_user_message(item["message"])
                memory.chat_memory.add_ai_message(item["response"])

        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
        )

        result = rag_chain.invoke({"question": user_text})
        response = result.get("answer", str(result)) if isinstance(result, dict) else str(result)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # update DB user doc with response
        chat_history_collection.update_one(
            {"session_id": st.session_state.session_id, "message": user_text, "sender": "user"},
            {"$set": {"response": response}},
            upsert=False,
        )

    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"I encountered an error: {str(e)}"})

# Sidebar
with st.sidebar:
    st.title("🛒 Product Chat")

    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.subheader("Previous Chats")
    sessions = get_all_sessions()

    if sessions:
        for s in sessions[:10]:
            session_id = s["_id"]
            latest_message = s.get("latest_message", "")
            display_message = latest_message[:40] + "..." if len(latest_message) > 40 else latest_message

            if st.button(display_message or "(empty)", key=session_id, use_container_width=True):
                st.session_state.session_id = session_id
                history = get_chat_history(session_id)

                st.session_state.messages = []
                for item in history:
                    if item["sender"] == "user":
                        st.session_state.messages.append({"role": "user", "content": item["message"]})
                    if item.get("response"):
                        st.session_state.messages.append({"role": "assistant", "content": item["response"]})
                st.rerun()
    else:
        st.info("No previous chats found")

# Main UI
st.title("🛒 E-commerce Assistant")
st.markdown("Ask me about products, get recommendations, or explore product details!")

# Put messages in a scrollable container (recommended approach)
chat_container = st.container(height=520)  # adjust height as you like

with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if st.button("🔊", key=f"tts_{i}", help="Listen to this response"):
                    audio_data = text_to_speech(message["content"])
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")

# Voice-to-text (renders as a separate widget)
voice_text = speech_to_text(
    language="en",
    start_prompt="🎤",
    stop_prompt="⏹️",
    just_once=True,
    use_container_width=True,
    key="voice_input_integrated",
)

# Chat input MUST be top-level to stay pinned to bottom
prompt = st.chat_input("Type your message or click the microphone...")

if voice_text:
    process_message(voice_text)
    st.rerun()

if prompt:
    process_message(prompt)
    st.rerun()
