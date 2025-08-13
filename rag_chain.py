import os
import pdfplumber
import docx
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from dotenv import load_dotenv

# ---------------------
# Load environment variables
# ---------------------
load_dotenv()  # reads from .env if present
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "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")

MODEL_NAME = os.getenv("FLAN_MODEL", "google/flan-t5-large")
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------
# Text splitter & vector DB (persistent storage)
# ---------------------
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorDB = Chroma(collection_name="rag_docs", embedding_function=embedding_model, persist_directory="chroma_store")
retriever = vectorDB.as_retriever()

# ---------------------
# LLM setup
# ---------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=generator)

# ---------------------
# Prompt / RAG chain
# ---------------------
prompt_template = PromptTemplate.from_template("""
Use the context below to answer the question.
If you don’t know the answer, say "I don't know."

Context:
{context}

Question: {question}
Answer:
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# ---------------------
# Web search tool (Tavily)
# ---------------------
search_tool = TavilySearchResults()

# ---------------------
# File text extraction
# ---------------------
def extract_text_from_file(path):
    _, ext = os.path.splitext(path.lower())
    ext = ext.strip(".")
    if ext in ("txt", "md", "py", "java", "cpp", "c", "js", "ts", "html", "css", "json"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == "pdf":
        texts = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text() or ""
                texts.append(page_text)
        return "\n".join(texts)
    elif ext == "docx":
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext in ("png", "jpg", "jpeg"):
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def add_document_from_path(path):
    text = extract_text_from_file(path)
    if not text or len(text.strip()) == 0:
        raise ValueError("No text extracted from file.")
    chunks = text_splitter.create_documents([text])
    vectorDB.add_documents(chunks)
    vectorDB.persist()  # save to disk
    global retriever
    retriever = vectorDB.as_retriever()

# ---------------------
# Hybrid RAG function
# ---------------------
def hybrid_rag(query):
    q = query.strip()
    if q.lower() in ["hi", "hello", "hii", "hey"]:
        return f"{q} 😊 How can I assist you today?"

    banned = ["bomb", "explosive", "sex", "illicit"]
    if any(b in q.lower() for b in banned):
        return "I'm not able to help with that."

    # Local search
    answer = rag_chain.invoke(q)

    # Fallback: Web + auto enrichment
    if "I don't know" in answer or len(answer.strip()) < 15:
        web_docs = search_tool.run(q)
        if not web_docs:
            return "🤖 Sorry, I couldn't find an answer."

        combined = "\n".join([d.get("content", "") for d in web_docs])
        new_chunks = text_splitter.create_documents([combined])
        vectorDB.add_documents(new_chunks)
        vectorDB.persist()
        global retriever
        retriever = vectorDB.as_retriever()
        answer = rag_chain.invoke(q)

    return answer

if __name__ == "__main__":
    print("📚 Hybrid RAG System Ready")
    while True:
        user_input = input("🧠 Your Question (or 'upload <path>', 'exit'): ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower().startswith("upload"):
            path = user_input.split(" ", 1)[1]
            try:
                add_document_from_path(path)
                print(f"✅ File '{path}' added to knowledge base.")
            except Exception as e:
                print("❌ Upload error:", e)
            continue
        print("\n✅ Answer:", hybrid_rag(user_input))
