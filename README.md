# 🚀 Lab08-RAG

## 📌 Introduction
This repository contains the implementation of a Retrieval-Augmented Generator (RAG) using LangChain, OpenAI for embeddings and language generation, and an InMemoryVectorStore for document retrieval. The goal of this lab is to demonstrate how to build a RAG system that retrieves relevant information from a vector store and generates context-aware responses using a large language model (LLM).

## 📂 Project Structure
```
flypzed-lab08-rag/
├── README.md                # Documentation for the project
└── Lab08_RAG.ipynb         # Jupyter Notebook with the RAG implementation
```

## 🔧 Requirements
Before running the notebook, ensure you have the following dependencies installed:
```bash
pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph
pip install -qU "langchain[openai]"
pip install -qU langchain-openai
pip install -qU langchain-core
```

## ⚙️ Setup
### 🔑 1. OpenAI API Key
This project uses OpenAI for embeddings and language generation. To configure your API key:
```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
```

### 🔑 2. Initialize OpenAI Components
- **Embeddings:** Use OpenAI's `text-embedding-3-large` model for generating embeddings.
- **LLM:** Use OpenAI's `gpt-4` model for generating responses.

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4")
```

## 🌟 Key Features
- **Document Loading:** Load and preprocess documents from a web source (e.g., a blog post).
- **Text Splitting:** Split documents into smaller chunks for efficient retrieval.
- **Vector Store:** Use an in-memory vector store to index and retrieve document embeddings.
- **Retrieval-Augmented Generation:** Combine retrieval and generation to produce context-aware responses.
- **Customizable Prompting:** Use a custom prompt template for question-answering tasks.

## ▶️ Usage
### Load and Preprocess Documents
Use `WebBaseLoader` to load a blog post and split it into chunks.
```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
```

### Index Documents
Add document embeddings to the vector store.
```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)
```

### Define the RAG Pipeline
Retrieve relevant documents and generate answers using a custom prompt.
```python
from langchain import hub
from langchain_core.prompts import PromptTemplate

prompt = hub.pull("rlm/rag-prompt")
```

### Run the RAG System
Invoke the RAG pipeline with a question.
```python
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
```

## 📌 Example Output
Here’s an example of the RAG system in action:
```
Question: What is Task Decomposition?
Answer: Task Decomposition is the process of breaking down a complex task into smaller, more manageable steps. It can be achieved through techniques such as Chain of Thought (CoT) prompting, where models are guided to think step by step, or through task-specific instructions. This method enhances performance on difficult tasks and clarifies the model's reasoning process.
```

## 🔗 Quick Access
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FlypZed/Lab08-RAG/blob/main/Lab08_RAG.ipynb)

## 📌 Repository Link
[GitHub Repository](https://github.com/FlypZed/Lab08-RAG)

## 📜 License
This project is licensed under the MIT License.
