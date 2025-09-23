# HProject with MongoDB, CSV Import, and VectorDB (Chroma)

This project demonstrates how to:
1. Connect to a **MongoDB Atlas cluster**.
2. Import employee data from a CSV file into a collection.
3. Create a **Vector Database using Chroma** for information retrieval (RAG).
4. Add a **Users collection** with login credentials.

---

## 🚀 Features
- Uploads `employee_data.csv` into MongoDB (`employees` collection).
- Preprocesses data (handles NaN/NaT values).
- Creates a **vector store** from employee data using `chromadb` + embeddings.
- Adds authentication (`users` collection with username/password).

---

## 📂 Project Structure
