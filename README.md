RAG Employee Knowledgebase Chatbot

This project is a Retrieval-Augmented Generation (RAG) application built with LangChain, ChromaDB, and Groq API. It enables natural language querying over an employee dataset (~3000 records) and returns accurate, context-aware responses.

Features

Employee Knowledgebase – Upload & query ~3000 employee records.

LangChain Integration – Handles retrieval + LLM pipeline.

ChromaDB Vector Store – Efficient storage and retrieval of embeddings.

Groq API (LLM) – Fast and reliable text generation.

Flask Backend – Simple API server to interact with the RAG pipeline.

Session Memory – Keeps conversation flow natural.

Tech Stack

Backend: Flask (Python)

Vector DB: ChromaDB

LLM: Groq API

LangChain: Orchestrates RAG pipeline

Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (or any other)

Dataset: Employee dataset (3000 entries)
