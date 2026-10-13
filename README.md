# ⚖️ Smart Legal Assistance

### AI-Powered Legal Information and Assistance System

## 📌 Project Overview

Smart Legal Assistance is an AI-powered application designed to help users understand legal information through simple, user-friendly explanations. The system uses Artificial Intelligence (AI), Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG) to retrieve relevant information from legal documents and generate easy-to-understand responses.

The project aims to make legal information more accessible to students, citizens, and people who need help understanding Indian criminal law.

## 🎯 Objectives

- Provide easy-to-understand explanations of legal concepts.
- Retrieve relevant information from legal documents.
- Answer user questions using AI-powered conversational assistance.
- Reduce the difficulty of understanding complex legal terminology.
- Organize and manage legal documents efficiently.
- Provide a user-friendly interface for interacting with legal information.

## ✨ Key Features

- **AI Legal Assistant:** Ask questions about legal topics in natural language.
- **RAG-Based Question Answering:** Retrieve relevant passages from legal documents to support responses.
- **Legal Document Processing:** Extract, preprocess, and organize legal document content.
- **Conversational Interface:** Interact with the system through a chat interface.
- **User Authentication:** Support user registration and login.
- **Chat History:** Maintain conversation records for reference.
- **Feedback System:** Allow users to provide feedback on responses.
- **Database Management:** Store user information, documents, document chunks, chat history, and feedback.

## 🛠️ Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend / API | Flask |
| Database | MySQL |
| AI Approach | Retrieval-Augmented Generation (RAG) |
| Language Processing | NLP |
| Vector Search | FAISS (if enabled in the implementation) |
| LLM Integration | Ollama |
| Development Tools | VS Code, Git, GitHub |

## 🏗️ System Architecture

1. **User Interface:** Users enter legal questions through the application.
2. **Query Processing:** The system processes and prepares the question for retrieval.
3. **Knowledge Base:** Legal documents are ingested, cleaned, and divided into smaller text chunks.
4. **Vector Search:** Relevant document chunks are retrieved using semantic similarity.
5. **AI Response Generation:** The retrieved context is provided to the language model to generate a clear response.
6. **Response Display:** The application presents the answer to the user.
7. **Database:** User details, document metadata, chat history, and feedback are managed using MySQL.

## 🔄 Working Workflow

```text
User Login
    ↓
Enter Legal Question
    ↓
Process User Query
    ↓
Search Legal Knowledge Base
    ↓
Retrieve Relevant Documents
    ↓
Generate Answer Using LLM
    ↓
Display Simple Legal Explanation
    ↓
Save Chat History and Feedback
```

## 📂 Project Structure

```text
Smart_Legal_Assistance/
│
├── streamlit_app.py
├── enhanced_rag_engine.py
├── vector_store.py
├── document_ingestion.py
├── retriever.py
├── login.py
├── requirements.txt
├── README.md
│
└── Additional modules, datasets, and configuration files
```

*The structure above is representative. Update it to match the actual files in your repository.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12
- MySQL Server
- Git
- Ollama, if using a locally hosted language model

### Step 1: Clone the Repository

```bash
git clone https://github.com/ASIMFATHIMA/Smart-Legal-Assistance.git
cd Smart-Legal-Assistance
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python executable directly instead.

### Step 3: Install Dependencies

If your project contains a `requirements.txt` file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure the Database

1. Start MySQL Server.
2. Create the database:

```sql
CREATE DATABASE smart_legal_assistance;
```

3. Configure your database connection settings using environment variables or your project's configuration file.
4. Ensure that the required tables are created before running the application.

**Security:** Do not commit database passwords, API keys, or other secrets to GitHub.

### Step 5: Configure Ollama

If the project uses Ollama, install it and download the language model configured in your application.

Example:

```bash
ollama pull qwen3:8b
```

Use this model only if your application is configured to support it and your computer has sufficient memory. Otherwise, select a smaller compatible model.

### Step 6: Run the Application

For the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

If your project requires a separate Flask backend, start it using the entry-point file and command configured in your application.

## 💡 Example Use Cases

- Understanding legal terminology in simple language.
- Finding relevant information in uploaded legal documents.
- Learning about Indian criminal law provisions.
- Exploring legal concepts for educational and research purposes.

## 🔐 Privacy and Security

- Keep database credentials and secret keys outside the source code.
- Restrict access to user records and chat history.
- Validate uploaded documents and user inputs.
- Follow appropriate security practices when storing personal information.

## ⚠️ Disclaimer

Smart Legal Assistance is an educational and informational project. It does not replace a qualified lawyer or provide a substitute for professional legal advice. AI-generated answers may be incomplete or inaccurate. Users should verify important legal information against authoritative legal sources and consult a qualified legal professional for advice about specific cases.

## 🚀 Future Enhancements

- Support for additional Indian laws and legal documents.
- Multilingual legal assistance, including Tamil and English.
- Voice-based question answering.
- Improved retrieval accuracy and source citations.
- Document summarization and legal-topic classification.
- Enhanced evaluation of answer relevance and factual accuracy.
- Improved accessibility and mobile-friendly design.

## 👩‍💻 Developer

**ASIM FATHIMA P**

B.Tech – Artificial Intelligence and Data Science

Nadar Saraswathi College of Engineering and Technology, Theni, Tamil Nadu, India.

## 📄 License

This project is developed for educational and project demonstration purposes. Add a suitable open-source license if you intend to permit reuse and redistribution.

---

**Smart Legal Assistance — Making Legal Information Easier to Understand with AI.**
