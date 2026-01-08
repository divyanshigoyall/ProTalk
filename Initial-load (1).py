import os
from dotenv import load_dotenv
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pymongo
load_dotenv()
DB_CONNECT = os.getenv("CONNECTION_DB")
GEMINI_API = os.getenv("GEMINI_API")

os.environ["GOOGLE_API_KEY"] = GEMINI_API

client = pymongo.MongoClient(DB_CONNECT)
db = client.E_Commerce
collection = db.Products


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


from datetime import datetime

def serialize_product_for_embedding(doc):
    title = doc.get("title", "No Title")
    short_desc = doc.get("shortDescription", "")
    detailed_desc = doc.get("detailedDescription", "")
    category = ", ".join(doc.get("category", []))
    features = ", ".join(doc.get("features", []))

    specs = doc.get("specifications", {})
    weight = specs.get("weight", "N/A")
    color = specs.get("color", "N/A")
    warranty = specs.get("warranty", "N/A")

    pros_list = doc.get("pros", [])
    pros = "\n  - " + "\n  - ".join(pros_list) if pros_list else "None"

    cons_list = doc.get("cons", [])
    cons = "\n  - " + "\n  - ".join(cons_list) if cons_list else "None"

    usage = ", ".join(doc.get("usageScenarios", []))

    price = doc.get("price", "N/A")
    brand = doc.get("brand", "Unknown Brand")
    rating = doc.get("rating", "No Rating")
    review_summary = doc.get("reviewSummary", "No review summary available.")

    keywords = ", ".join(doc.get("keywords", []))

    # Sort reviews by most recent and get one per rating level
    reviews = doc.get("reviews", [])
    sorted_reviews = sorted(
        reviews,
        key=lambda r: datetime.fromisoformat(r.get("createdAt", "1970-01-01T00:00:00")),
        reverse=True
    )

    recent_review_per_rating = {}
    for r in sorted_reviews:
        r_rating = r.get("rating")
        if r_rating in [1, 2, 3, 4, 5] and r_rating not in recent_review_per_rating:
            recent_review_per_rating[r_rating] = r

    review_text_sections = []
    for rating_level in range(5, 0, -1):
        if rating_level in recent_review_per_rating:
            rev = recent_review_per_rating[rating_level]
            reviewer = rev.get("user", {}).get("name", "Anonymous")
            title_r = rev.get("title", "")
            body = rev.get("body", "")
            review_text_sections.append(
                f"⭐ {rating_level}-Star Review by {reviewer}:\nTitle: {title_r}\nBody: {body}"
            )

    review_text = "\n\n".join(review_text_sections) if review_text_sections else "No user reviews available."

    created_at = doc.get("createdAt", "")
    updated_at = doc.get("updatedAt", "")

    serialized = f"""Product: {title}
Short Description: {short_desc}
Detailed Description: {detailed_desc}
Category: {category}
Features: {features}

Specifications:
  - Weight: {weight}
  - Color: {color}
  - Warranty: {warranty}

Pros:{pros}

Cons:{cons}

Usage Scenarios: {usage}
Price: ${price}
Brand: {brand}
Rating: {rating} stars
Review Summary: {review_summary}

Top Recent Reviews by Rating:
{review_text}

Keywords: {keywords}
Created At: {created_at}
Last Updated: {updated_at}
"""
    return serialized

print("Databases:", client.list_database_names())
print("Collections:", db.list_collection_names())
for doc in collection.find({}):
    if "embedding" in doc:
        print("Already embedded")
        continue 
    text = serialize_product_for_embedding(doc)
    embedding = embeddings.embed_query(text)
    # Save both the text and the embedding
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "embedding_text": text,
            "embedding": embedding
        }}
    )

print("Success!")