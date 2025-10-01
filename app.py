from flask import Flask, render_template, request, redirect, url_for, session
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from pymongo import MongoClient
from urllib.parse import quote_plus
import secrets
import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))

try:
    raw_password = os.getenv('MONGODB_PASSWORD')
    encoded_password = quote_plus(raw_password)
    
    mongo_client = MongoClient(
        f"mongodb+srv://sujan:{encoded_password}@cluster0.6eacggy.mongodb.net/?retryWrites=true&w=majority"
    )
    db = mongo_client["HospitalDB"]
    users_collection = db["users"]
    
    # Test connection
    mongo_client.admin.command('ping')
    print("MongoDB connected successfully")
    
    # Check if users collection exists and has users
    user_count = users_collection.count_documents({})
    print(f"Found {user_count} users in the database")
    
    if user_count == 0:
        print("WARNING: No users found in database. Please add users to MongoDB first.")
    
except Exception as e:
    print(f"MongoDB setup error: {e}")
    users_collection = None

try:
    chroma_client = chromadb.PersistentClient(path="./chromadb")
    collection_chroma = chroma_client.get_or_create_collection(
        name="employee_collection",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    print("ChromaDB initialized successfully")
except Exception as e:
    print(f"ChromaDB setup error: {e}")
    collection_chroma = None


try:
    groq_api_key = os.getenv('GROQ_API_KEY')
    client_groq = Groq(api_key=groq_api_key)
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Groq setup error: {e}")
    client_groq = None


try:
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    # Initialize ChatGroq LLM
    langchain_llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024
    )
    
    # Initialize embeddings for LangChain
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create prompt template for RAG
    prompt_template = """You are a helpful AI assistant for employee information queries.

Use the following context to answer the user's question. If the context doesn't contain enough information, please say so.

Context: {context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    print("LangChain components initialized successfully")
    
except Exception as e:
    print(f"LangChain setup error: {e}")
    langchain_llm = None
    embeddings = None
    PROMPT = None

def authenticate_user(username, password):
    """Authenticate user against MongoDB"""
    if not users_collection:
        return False
    
    try:
        user = users_collection.find_one({"username": username, "password": password})
        return user is not None
    except Exception as e:
        print(f"Authentication error: {e}")
        return False

# Store chat messages in session
def get_chat_messages():
    if 'chat_messages' not in session:
        session['chat_messages'] = []
    return session['chat_messages']

def add_message(message, is_user=True):
    messages = get_chat_messages()
    messages.append({'text': message, 'is_user': is_user})
    session['chat_messages'] = messages

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('chatbot'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    
    if authenticate_user(username, password):
        session['username'] = username
        session['chat_messages'] = []  # Clear previous chat
        return redirect(url_for('chatbot'))
    else:
        return render_template('login.html', error="Invalid username or password")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        user_query = request.form.get('query', '').strip()
        if user_query:
            add_message(user_query, is_user=True)
            bot_response = rag_query_langchain(user_query)
            add_message(bot_response, is_user=False)
    
    messages = get_chat_messages()
    return render_template('chatbot.html', username=session['username'], messages=messages)

@app.route('/quick_query')
def quick_query():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    query = request.args.get('q', '').strip()
    if query:
        add_message(query, is_user=True)
        bot_response = rag_query_langchain(query)
        add_message(bot_response, is_user=False)
    
    return redirect(url_for('chatbot'))

def rag_query_langchain(user_query, top_k=3):
    """Perform RAG query using ChromaDB and LangChain"""
    if not collection_chroma:
        return "ChromaDB is not available. Please check the database setup."
    
    if not langchain_llm:
        return "LangChain LLM is not available. Please check your API key."
    
    try:
        print(f"Processing query with LangChain: {user_query}")
        
        results = collection_chroma.query(
            query_texts=[user_query],
            n_results=top_k
        )
        
        if not results["documents"] or not results["documents"][0]:
            return "No relevant information found in the database for your query."
        
        retrieved_docs = results["documents"][0]
        context_text = "\n".join(retrieved_docs)
        
        print(f"Found {len(retrieved_docs)} relevant documents")
        
        formatted_prompt = PROMPT.format(context=context_text, question=user_query)
        response = langchain_llm.invoke(formatted_prompt)
        
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        print("Generated response successfully with LangChain")
        return response_text
    
    except Exception as e:
        error_msg = f"Error processing query with LangChain: {str(e)}"
        print(error_msg)
        return rag_query_fallback(user_query, top_k)

def rag_query_fallback(user_query, top_k=3):
    """Fallback RAG query using direct Groq API"""
    if not client_groq:
        return "Both LangChain and Groq client are not available."
    
    try:
        print(f"Using fallback Groq API for: {user_query}")
        
        results = collection_chroma.query(
            query_texts=[user_query],
            n_results=top_k
        )
        
        if not results["documents"] or not results["documents"][0]:
            return "No relevant information found in the database."
        
        retrieved_docs = results["documents"][0]
        context_text = "\n".join(retrieved_docs)
        
        prompt = f"""You are a helpful AI assistant for employee information queries. 
        
Please answer the user's question using ONLY the information provided in the context below. 
If the context doesn't contain enough information to answer the question, please say so.

CONTEXT:
{context_text}

USER QUESTION: {user_query}

ANSWER:"""
        
        completion = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        
        response = completion.choices[0].message.content
        print("Generated response successfully with fallback")
        return response
    
    except Exception as e:
        error_msg = f"Error in fallback query: {str(e)}"
        print(error_msg)
        return error_msg

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
