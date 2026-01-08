import os
from dotenv import load_dotenv
from datetime import datetime
import pymongo
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()
DB_CONNECT = os.getenv("CONNECTION_DB")
GEMINI_API = os.getenv("GEMINI_API")

os.environ["GOOGLE_API_KEY"] = GEMINI_API

# Connect to MongoDB
client = pymongo.MongoClient(DB_CONNECT)
db = client.E_Commerce
collection = db.Products

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, max_tokens=790)

# Vector search setup
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index",
    text_key="embedding_text",
    embedding_key="embedding"
)
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Custom prompt
# system_prompt = (
#     "You are a helpful assistant that provides product recommendations and information. "
#     "Use the following pieces of retrieved context to answer the question. "
#     "If you don't know the answer, say you don't know. "
#     "Use three sentences maximum.\n\n{context}"
# )
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}")
# ])

# Memory to store conversation context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Conversational RAG Chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)

# # Chat loop
# print("Bot is ready! Type 'exit' to end the chat.\n")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Bot: Goodbye!!")
#         break

#     result = conversational_chain.invoke({"question": user_input})
#     # Safely access answer
#     if isinstance(result, dict) and "answer" in result:
#         print("Bot:", result["answer"])
#     elif hasattr(result, "answer"):
#         print("Bot:", result.answer)
#     else:
#         print("Bot:", result)


from bson import ObjectId
cart_col   = db.Carts
orders_col = db.Orders

from bson import ObjectId
from bson.errors import InvalidId

def add_to_cart_tool(input_str: str) -> str:

    parts   = [p.strip() for p in input_str.split(",")]
    query   = parts[0]
    quantity = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1

    # 1) Vector search for relevant products
    results = retriever.get_relevant_documents(query)
    if not results:
        return f"❌ I couldn’t find any products matching “{query}.”"
    doc  = results[0]
    meta = doc.metadata

    # 2) Extract ID, name, price
    raw_id = meta.get("_id") or meta.get("id")
    try:
        # try converting to BSON ObjectId; if it fails, fall back to the raw string
        prod_id = ObjectId(raw_id)
    except (InvalidId, TypeError):
        prod_id = raw_id

    name  = meta.get("title", "Unknown Product")
    price = float(meta.get("price", 0))

    # 3) Push into cart (using string or ObjectId as-is)
    cart_col.update_one(
        {"_id": "session_cart"},
        {"$push": {
            "items": {
                "product_id": prod_id,
                "name":        name,
                "price":       price,
                "quantity":    quantity,
                "added_at":    datetime.utcnow()
            }
        }},
        upsert=True
    )

    return f"✅ Added {quantity}× “{name}” (₹{price} each) to your cart."

def remove_from_cart_tool(input_str: str) -> str:
    """
    Removes a product (or some quantity) from the session cart.
    Input format: "<product keywords or ID>[, <quantity>]"
      • If you supply a name/keywords string, we run RAG retrieval to pick the right product.
      • If you supply a raw ID, we use it directly.
      • If quantity is omitted, we remove the entire item from the cart.
    """
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
    result = conversational_chain.invoke({"question": query})
    # extract the answer
    if isinstance(result, dict) and "answer" in result:
        return result["answer"]
    elif hasattr(result, "answer"):
        return result.answer
    else:
        return str(result)


tools = [
    Tool(
        name="add_to_cart",
        func=add_to_cart_tool,
        description=(
            "Use this to add a product to the cart by its name. "
            "Input should be: '<product name>[, <quantity>]'.  "
            "If quantity is omitted, it defaults to 1."
        )
    ),
    Tool(
        name="show_cart",
        func=lambda _: show_cart(),
        description="Use this to display the current cart contents."
    ),
    Tool(
        name="place_order",
        func=lambda _: place_order(),
        description="Use this to place an order for everything currently in the cart."
    ),
    Tool(
        name="product_qa",
        func=rag_qa,
        description=(
            "Answer any product-related question using our product catalog. "
            "Input should be a natural language question."
        )
    ),
    Tool(
    name="remove_from_cart",
    func=remove_from_cart_tool,
    description=(
        "Remove a product from the cart by name or ID. "
        "Input: '<product keywords or ID>[, <quantity>]'. "
        "Quantity omitted → remove the entire item; otherwise remove that many."
        )
    ),
]




agent = initialize_agent(
    tools, llm, memory=memory, 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
    verbose=False
)


print("Chatbot with AI-driven cart & checkout. Type ‘exit’ to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    response = agent.run(user_input)
    print("Bot:", response)