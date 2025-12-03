import os
from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
from urllib.parse import quote_plus
from pymongo import MongoClient
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq


load_dotenv()
MONGO_PASSWORD = os.getenv("MONGODB_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Debug: Check if API key is loaded
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

print(f"API Key loaded: {GROQ_API_KEY[:10]}...") # Print first 10 chars for verification

encoded_password = quote_plus(MONGO_PASSWORD)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")


mongo_client = MongoClient(
    f"mongodb+srv://sujan:{encoded_password}@cluster0.6eacggy.mongodb.net/?retryWrites=true&w=majority"
)
db = mongo_client["HospitalDB"]
users_collection = db["users"]


chroma_client = chromadb.PersistentClient(path="./chromadb")
collection_chroma = chroma_client.get_or_create_collection(
    name="employee_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

groq_client = Groq(api_key=GROQ_API_KEY)


def get_rag_response(question: str, sources: list) -> str:
    """Generate response using Groq API with retrieved sources"""
    # Combine ChromaDB retrieved sources into a single string
    source_texts = "\n".join([f"- {s['content']}" for s in sources])
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided sources.

Sources:
{source_texts}

Question: {question}

Provide a factual and concise answer based only on the sources above. If the sources don't contain relevant information, say so."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided sources."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users_collection.find_one({"username": username, "password": password})
        if user:
            session["username"] = username
            return redirect(url_for("chat"))
        else:
            error = "Invalid username or password"
    
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]
    messages = []

    if request.method == "POST":
        query = request.form["query"]

        # Query ChromaDB
        results = collection_chroma.query(
            query_texts=[query],
            n_results=3
        )
        sources = []
        for i, doc in enumerate(results["documents"][0]):
            sources.append({
                "content": doc,
                "full_content": doc,
                "similarity_score": results["distances"][0][i]
            })

        # Get RAG response using Groq
        response_text = get_rag_response(question=query, sources=sources)

        messages.append({"is_user": True, "text": query})
        messages.append({"is_user": False, "text": response_text, "sources": sources})

    return render_template("chat.html", username=username, messages=messages)


@app.route("/quick_query", methods=["GET"])
def quick_query():
    if "username" not in session:
        return redirect(url_for("login"))

    q = request.args.get("q")
    if not q:
        return redirect(url_for("chat"))

    # Query ChromaDB
    results = collection_chroma.query(
        query_texts=[q],
        n_results=3
    )
    sources = []
    for i, doc in enumerate(results["documents"][0]):
        sources.append({
            "content": doc,
            "full_content": doc,
            "similarity_score": results["distances"][0][i]
        })

    # Get RAG response using Groq
    response_text = get_rag_response(question=q, sources=sources)

    messages = [
        {"is_user": True, "text": q},
        {"is_user": False, "text": response_text, "sources": sources}
    ]

    return render_template("chat.html", username=session.get("username"), messages=messages)


if __name__ == "__main__":
    app.run(debug=True)

