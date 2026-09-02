COLAB LINK : https://colab.research.google.com/drive/1daa66jU8IoXZdWwCNSQSc5OwVvnD_Jkk?usp=sharing
Hybrid AI Copilot (RAG + Web Search)
A full-stack conversational AI application that combines local document retrieval (RAG) with live internet search capabilities. Built with Python, Flask, LangChain, and open-source Hugging Face models.

Features
Three AI Modes:
Offline RAG: Answers questions based purely on the documents you upload.
Internet Search Only: Bypasses local data to fetch live answers from the web.
Hybrid Mode: Searches your local knowledge base first, and falls back to the internet if it doesn't know the answer.
Local Vector Database: Uses ChromaDB to securely store and search your uploaded documents completely offline.
Multi-Format Support: Upload and chat with .txt, .pdf, .docx, and image files (.png, .jpg).
Open-Source LLMs: Powered by TinyLlama-1.1B-Chat and sentence-transformers for completely free, local text generation and embeddings.
Note on Hosting
This application requires a Python backend and local machine learning models to function. It cannot be run directly via GitHub Pages. To use this AI, please follow the local installation steps below to run it on your own machine.

Prerequisites
Before you begin, ensure you have the following installed on your local machine:

Python 3.9+
Tesseract-OCR: Required for extracting text from images.
Windows: Download and install the Tesseract executable.
Mac/Linux: Install via Homebrew (brew install tesseract) or apt (sudo apt install tesseract-ocr).
Installation & Setup
