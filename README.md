# BasisQuest

**BasisQuest** is an intelligent document-retrieval chatbot designed to explore and query large collections of SAP Basis or enterprise automation documents.

It demonstrates a practical **Retrieval-Augmented Generation (RAG)** architecture combining OpenAI embeddings, FAISS vector search, and a Streamlit interface.

---

## 1. Overview

BasisQuest enables users to:

- Load multiple document formats (`.pdf`, `.docx`, `.txt`, `.html`)
- Split text into overlapping semantic chunks
- Generate embeddings using the OpenAI API
- Perform similarity search using **FAISS**
- Re-rank retrieved results with cosine normalization
- Handle follow-up questions with conversational memory
- Display real-time progress during document loading and embedding

This project is intended for demonstration and learning, showing how RAG systems can turn enterprise documentation into an interactive knowledge assistant.

---

## 2. Architecture

```
Documents → Text Splitter → Embeddings (OpenAI) → FAISS Index
↑ ↓
Chat Query ← Context Retrieval ← Reranking (Cosine)
↓
OpenAI Chat Completion
↓
Streamlit Front-End
```

---

## 3. Project Structure

```
BasisQuest/
│
├─ BasisQuest.py # Core logic: document loading, embeddings, RAG functions
├─ app.py # Streamlit interface with progress display
├─ .env # API key file (not committed to Git)
├─ requirements.txt # Python dependencies
├─ .gitignore # Files excluded from version control
└─ README.md # Project overview
```

---

## 4. Installation

### Step 1: Clone the repository
```
git clone https://github.com/lokhanglee/BasisQuest.git
cd BasisQuest
```

### Step 2: Set up a virtual environment
```
python -m venv env
env\Scripts\activate     # On Windows
```

### Step 3: Install dependencies
```
pip install -r requirements.txt
```

### Step 4: Set your OpenAI API key
Create a file named .env in the project folder:
```
OPENAI_API_KEY=your_api_key_here
```

## 5. Run the Streamlit app
```
streamlit run app.py
```

Then open your browser and visit:
```
https://basisquest.streamlit.app[https://basisquest.streamlit.app]
```

You can ask questions such as:
* What is SAP HANA Cloud?
* Explain the role of the Basis layer in SAP architecture.
* Compare SAP transport management and version control.

---

## 6. Key Features
|Feature|Description|
|---|---|
|Multi-format ingestion|Reads `.pdf`, `.docx`, `.txt`, `.html` automatically|
|Chunking with overlap|Splits text into overlapping segments for better context|
|Vector search (FAISS)|Enables efficient high-dimensional similarity search|
|Cosine normalization|Improves ranking accuracy of relevant results|
|Conversation memory|Supports short-term follow-up dialogue|
|Progress tracking|Displays document loading and embedding progress in Streamlit|

---

## 7. Technology Stack

* Python 3.11
* OpenAI API
* FAISS
* Streamlit
* tqdm
* dotenv
* BeautifulSoup4
* PyPDF2
* python-docx
* NumPy

---

## 8. License

This project is distributed for educational and demonstration purposes only.\
All documents and API credentials must comply with their respective licenses and terms of use

---

## 10. Author

**Louis Lee**\
LinkedIn: [https://www.linkedin.com/in/lokhanglee/]\
GitHub: [https://github.com/lokhanglee]