import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import pymongo
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from bson import ObjectId
from bson.errors import InvalidId
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import io
import base64
# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="E-commerce Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Database configuration
@st.cache_resource
def init_database():
    DB_CONNECT = os.getenv("CONNECTION_DB")
    GEMINI_API = os.getenv("GEMINI_API")
    os.environ["GOOGLE_API_KEY"] = GEMINI_API
    client = pymongo.MongoClient(DB_CONNECT)
    db = client.E_Commerce
    collection = db.Products
    chat_history_collection = db.chat_history
    cart_col = db.Carts
    orders_col = db.Orders
    return client, db, collection, chat_history_collection, cart_col, orders_col

client, db, collection, chat_history_collection, cart_col, orders_col = init_database()

# Initialize AI components
@st.cache_resource
def init_ai_components():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=790
    )
    vectorStore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index",
        text_key="embedding_text",
        embedding_key="embedding"
    )
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return llm, retriever

llm, retriever = init_ai_components()

# Tool functions
def add_to_cart_tool(input_str: str) -> str:
    """
    Uses RAG retrieval to find the product by keywords, then adds it to the cart.
    Input: "<product keywords>[, <quantity>]".
    """
    parts    = [p.strip() for p in input_str.split(",")]
    query    = parts[0]
    quantity = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1

    # 1) Vector‐search for relevant products
    results = retriever.get_relevant_documents(query)
    if not results:
        return f"❌ I couldn’t find any products matching “{query}.”"
    doc  = results[0]
    meta = doc.metadata

    # 2) Extract the name (or any unique metadata key)
    name  = meta.get("title") or meta.get("name") or meta.get("product_name")
    price = float(meta.get("price", 0))

    if not name:
        return "❌ Retrieved product has no name in metadata."

    # 3) Look up the real product in MongoDB by name
    prod = collection.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})
    if prod:
        prod_id = prod["_id"]
    else:
        # fallback: store the string raw_id if you really must
        prod_id = meta.get("_id") or meta.get("id")

    # 4) Push into cart
    cart_col.update_one(
        {"_id": "session_cart"},
        {"$push": {
            "items": {
                "product_id": prod_id,
                "name":       name,
                "price":      price,
                "quantity":   quantity,
                "added_at":   datetime.utcnow()
            }
        }},
        upsert=True
    )

    return f"✅ Added {quantity}× “{name}” (₹{price} each) to your cart."


def remove_from_cart_tool(input_str: str) -> str:
    parts = [p.strip() for p in input_str.split(",")]
    target = parts[0]
    qty    = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None

    # 1) Determine the product_id and name via RAG or direct ID
    try:
        # try direct ObjectId first
        prod_id = ObjectId(target)
        prod = {"_id": prod_id}
    except (InvalidId, TypeError):
        # fallback: RAG-based lookup by keywords
        results = retriever.get_relevant_documents(target)
        if not results:
            return f"❌ No product found matching “{target}.”"
        doc = results[0]
        meta = doc.metadata
        raw_id = meta.get("_id") or meta.get("id")
        try:
            prod_id = ObjectId(raw_id)
        except (InvalidId, TypeError):
            prod_id = raw_id
        prod = {"_id": prod_id, "name": meta.get("name", target)}

    # 2) Fetch the current cart
    cart = cart_col.find_one({"_id": "session_cart"})
    if not cart or not cart.get("items"):
        return "❌ Your cart is already empty."

    # 3) Locate the line item
    for item in cart["items"]:
        if item["product_id"] == prod_id:
            current_qty = item["quantity"]
            name = item.get("name", "that product")
            break
    else:
        return f"❌ “{target}” isn’t in your cart."

    # 4) Compute removal behavior
    if qty is None or qty >= current_qty:
        # remove entire line
        cart_col.update_one(
            {"_id": "session_cart"},
            {"$pull": {"items": {"product_id": prod_id}}}
        )
        return f"🗑️ Removed all of “{name}” from your cart."
    else:
        # decrement quantity
        cart_col.update_one(
            {"_id": "session_cart", "items.product_id": prod_id},
            {"$inc": {"items.$.quantity": -qty}}
        )
        return f"🗑️ Removed {qty}× “{name}” (remaining {current_qty - qty}) from your cart."

def show_cart() -> str:
    cart = cart_col.find_one({"_id": "session_cart"})
    if not cart or not cart.get("items"):
        return "🛒 Your cart is empty."
    lines = ["🛒 Cart contents:"]
    total = 0
    for idx, item in enumerate(cart["items"], 1):
        subtotal = item["price"] * item["quantity"]
        total += subtotal
        lines.append(f"{idx}. {item['name']} — {item['quantity']}×₹{item['price']} = ₹{subtotal}")
    lines.append(f"**Total: ₹{total}**")
    return "\n".join(lines)

def place_order() -> str:
    cart = cart_col.find_one({"_id": "session_cart"})
    if not cart or not cart.get("items"):
        return "❌ Cannot place order: cart is empty."
    total = sum(i["price"] * i["quantity"] for i in cart["items"])
    orders_col.insert_one({
        "items": cart["items"],
        "total": total,
        "placed_at": datetime.utcnow(),
        "status": "placed"
    })
    cart_col.delete_one({"_id": "session_cart"})
    return f"🎉 Order placed! Total was ₹{total}."

def rag_qa(query: str) -> str:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )
    result = chain.invoke({"question": query})
    return result["answer"] if "answer" in result else str(result)

# Initialize agent
@st.cache_resource
def init_agent():
    tools = [
        Tool(name="add_to_cart", func=add_to_cart_tool, 
            description="Add product to cart. Input: 'product_name, quantity'"),
        Tool(name="show_cart", func=lambda _: show_cart(),
            description="Display cart contents"),
        Tool(name="place_order", func=lambda _: place_order(),
            description="Place order for cart items"),
        Tool(name="product_qa", func=rag_qa,
            description="Answer product questions"),
        Tool(name="remove_from_cart", func=remove_from_cart_tool,
            description="Remove from cart. Input: 'product_name/id, quantity'")
    ]
    
    agent = initialize_agent(
        tools, llm, 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=False
    )
    return agent

agent = init_agent()

# Helper functions
def save_message_to_db(session_id, message, sender="user"):
    chat_doc = {
        "session_id": session_id,
        "message": message,
        "sender": sender,
        "timestamp": datetime.utcnow()
    }
    chat_history_collection.insert_one(chat_doc)

def get_chat_history(session_id, limit=50):
    history = list(chat_history_collection.find(
        {"session_id": session_id}
    ).sort("timestamp", 1).limit(limit))
    return [{
        "message": doc["message"],
        "sender": doc["sender"],
        "timestamp": doc["timestamp"]
    } for doc in history]

def get_all_sessions():
    pipeline = [
        {"$sort": {"timestamp": -1}},
        {"$group": {
            "_id": "$session_id",
            "latest_message": {"$first": "$message"},
            "latest_timestamp": {"$first": "$timestamp"},
            "message_count": {"$sum": 1}
        }},
        {"$sort": {"latest_timestamp": -1}},
        {"$limit": 20}
    ]
    return list(chat_history_collection.aggregate(pipeline))

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# Process chat message
def process_message(prompt):
    if not prompt.strip():
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message_to_db(st.session_state.session_id, prompt, "user")
    
    try:
        response = agent.run(prompt)
    except Exception as e:
        response = f"I encountered an error: {str(e)}"
    
    st.session_state.messages.append({"role": "assistant", "content": response.removesuffix("```")})
    save_message_to_db(st.session_state.session_id, response.removesuffix("```"), "assistant")

with st.sidebar:
    st.title("🛒 Product Chat")
    
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.subheader("Previous Chats")
    sessions = get_all_sessions()
    
    if sessions:
        for session in sessions[:10]:  # Show last 10 sessions
            session_id = session["_id"]
            latest_message = session["latest_message"]
            
            # Truncate message for display
            display_message = latest_message[:40] + "..." if len(latest_message) > 40 else latest_message
            
            if st.button(f"{display_message}", key=session_id, use_container_width=True):
                # Load selected session
                st.session_state.session_id = session_id
                history = get_chat_history(session_id)
                
                # Rebuild messages for display
                st.session_state.messages = []  
                for item in history:
                    if item["sender"] == "user":
                        st.session_state.messages.append({"role": "user", "content": item["message"]})
                    if item.get("response"):
                        st.session_state.messages.append({"role": "assistant", "content": item["response"]})
                st.rerun()
    else:
        st.info("No previous chats found")

# Main chat interface
# Create two columns
with open("logo.png", "rb") as img_file:
    b64 = base64.b64encode(img_file.read()).decode()
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 12px;">
      <img src="data:image/png;base64,{b64}" width="50" />
      <h1 style="margin: 0; line-height: 1;">Pro-Talk</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "Ask me about products, get recommendations, or explore product details!"

)

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add text-to-speech autoplay for assistant messages
        if message["role"] == "assistant":
            if st.button("🔊", key=f"tts_{i}"):
                audio_data = text_to_speech(message["content"])
                if audio_data:
                    # Read and Base64‑encode the MP3 bytes
                    audio_bytes = audio_data.read()
                    b64 = base64.b64encode(audio_bytes).decode()
                    # Inject hidden autoplaying audio element
                    st.markdown(
                        f"""
                        <audio autoplay style="display:none" src="data:audio/mp3;base64,{b64}"></audio>
                        """,
                        unsafe_allow_html=True
                    )


# Enhanced voice-integrated chat input using st._bottom container
with st._bottom:
    st.markdown('<div class="fixed-chat-container">', unsafe_allow_html=True)
    
    # Create columns for voice and text input
    col1, col2 = st.columns([1, 6])
    
    with col1:
        # Voice input with auto-submit
        voice_text = speech_to_text(
            language="en",
            start_prompt="🎤",
            stop_prompt="⏹️",
            just_once=True,
            use_container_width=True,
            key="voice_input"
        )
    
    with col2:
        # Regular chat input
        prompt = st.chat_input("Type your message or click the microphone...")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle voice input with auto-submit
    if voice_text:
        process_message(voice_text)
        st.rerun()
    
    # Handle text input
    if prompt:
        process_message(prompt)
        st.rerun()