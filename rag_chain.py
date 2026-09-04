import os
import re
import sys
import webbrowser
import warnings
from datetime import date

import pdfplumber
import docx
import pytesseract
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from dotenv import load_dotenv

# Suppress library deprecation and generation warnings
warnings.filterwarnings("ignore")

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# --------------------------------------------------
# Hardware & Configuration
# --------------------------------------------------
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n" + "=" * 50)
print(f" HARDWARE DETECTED: {device.upper()}")
print("=" * 50 + "\n")

# --------------------------------------------------
# Text Splitter & Vector Store
# --------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=300, separators=["\n\n", "\n", " ", ""]
)

emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": device})
vectorDB = Chroma(collection_name="rag_docs", embedding_function=emb, persist_directory="chroma_store")
retriever = vectorDB.as_retriever(search_kwargs={"k": 10})

# --------------------------------------------------
# LLM Pipeline (Optimized to Prevent Warnings)
# --------------------------------------------------
print(" Loading AI Brain (this takes a moment)...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

if hasattr(model.generation_config, "max_length"):
    model.generation_config.max_length = None
if hasattr(model.config, "max_length"):
    model.config.max_length = None

model.generation_config.max_new_tokens = 768

# Removed temperature=0.0 when do_sample=False to stop the transformer flag warnings
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    max_new_tokens=768,
    max_length=None,
    return_full_text=False,
    do_sample=False,
    repetition_penalty=1.15,
)
llm = HuggingFacePipeline(pipeline=gen)

# --------------------------------------------------
# Keyword Definitions
# --------------------------------------------------
IMAGE_KEYWORDS = [
    "image", "images", "photo", "photos", "picture", "pictures",
    "pic", "pics", "wallpaper", "show me an image", "show me photos"
]
VIDEO_KEYWORDS = [
    "video", "videos", "clip", "clips", "watch", "youtube",
    "documentary", "footage", "tutorial video"
]
SUMMARY_KEYWORDS = [
    "summarize", "summary", "brief overview", "key points", "tldr",
    "recap", "synopsis", "outline the main points"
]
CODE_KEYWORDS = [
    "code", "program", "function", "algorithm", "leetcode", "leet code",
    "class ", "implement", "script", "compile", "syntax", "debug", "bug",
    "python", "java", "javascript", "typescript", "c++", "c#", "sql",
    "html", "css", "react", "node", "regex", "api", "loop", "recursion",
    "array", "linked list", "sort", "data structure", "compiler",
    "write a program", "write code", "fix this code", "stack trace"
]
TRANSLATE_KEYWORDS = [
    "translate", "transulate", "translation", "meaning in",
    "in telugu", "in hindi", "to telugu", "to hindi", "to english", "in english"
]
TIME_SENSITIVE_KEYWORDS = [
    "today", "current", "currently", "now", "latest", "right now",
    "this week", "this month", "net worth", "stock price", "share price",
    "exchange rate", "weather", "score", "live", "up to date", "as of"
]

def extract_search_keywords(query: str) -> str:
    filler_patterns = [
        r"\b(please|can you|could you|tell me|what is|who is|give me|provide)\b",
        r"\b(show me|search for|look up|find out|i want to know about)\b",
        r"\b(images? of|photos? of|pictures? of|videos? of|clips? of)\b",
        r"\b(links about|link for|links for|provide link)\b",
        r"\b(create an|create a)\b"
    ]
    cleaned = query.strip()
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    keywords = " ".join(cleaned.split())
    return keywords if keywords else query.strip()

def classify_query(query: str) -> str:
    q = query.lower()
    if any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in IMAGE_KEYWORDS): return "image"
    if any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in VIDEO_KEYWORDS): return "video"
    if any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in SUMMARY_KEYWORDS): return "summary"
    if any(kw in q for kw in TRANSLATE_KEYWORDS): return "translate"
    if any(kw in q for kw in CODE_KEYWORDS): return "code"
    return "general"

def is_time_sensitive(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in TIME_SENSITIVE_KEYWORDS)

def format_history_for_prompt(history: list) -> str:
    if not history:
        return "No prior conversation context."
    formatted = []
    for q, a in history[-3:]:
        short_a = a if len(a) < 180 else a[:180] + "..."
        formatted.append(f"User: {q}\nAI: {short_a}")
    return "\n\n".join(formatted)

# Few-shot examples ensure 1.5B models resolve pronouns accurately
contextualize_prompt = PromptTemplate.from_template(
    "Given the conversation history and a follow-up question, rewrite the question into a standalone query by replacing pronouns (he, she, it, his, her, their) with the specific person, company, or subject discussed.\n\n"
    "Example 1:\n"
    "History:\n"
    "User: who is the ceo of google?\n"
    "AI: The CEO of Google is Sundar Pichai.\n"
    "Follow-up: where did he complete his bachelors degree?\n"
    "Standalone Query: Where did Sundar Pichai complete his bachelors degree?\n\n"
    "Example 2:\n"
    "History:\n"
    "User: what is python?\n"
    "AI: Python is a high-level programming language created by Guido van Rossum.\n"
    "Follow-up: when was it created?\n"
    "Standalone Query: When was Python created?\n\n"
    "Now perform the task:\n"
    "History:\n{history}\n\n"
    "Follow-up: {question}\n"
    "Standalone Query:"
)

def build_search_query(query: str, chat_history: list) -> str:
    if not chat_history:
        return query.strip()

    pronouns = {"he", "she", "it", "they", "his", "her", "their", "this", "that", "him", "them"}
    query_words = set(re.findall(r'\b\w+\b', query.lower()))

    # Check if this is a pronoun follow-up or short phrase
    if not (query_words.intersection(pronouns) or len(query_words) < 4):
        return query.strip()

    formatted_hist = format_history_for_prompt(chat_history)
    print(" [*] Resolving follow-up question context...")

    try:
        standalone_query = llm.invoke(
            contextualize_prompt.format(history=formatted_hist, question=query)
        ).strip().strip('"').strip("'")

        if "\n" in standalone_query:
            standalone_query = standalone_query.split("\n")[0].strip()

        # Check if the rewrite succeeded
        if standalone_query and len(standalone_query) < 120 and not any(p in standalone_query.lower().split() for p in ["their", "someone", "unknown"]):
            print(f" [Memory] Rewrote to: '{standalone_query}'")
            return standalone_query
    except Exception as e:
        print(f" [!] Contextualize error: {e}")

    # Fallback to appending previous keywords if rewriting fails
    last_user_q, last_ai_ans = chat_history[-1]
    subject = extract_search_keywords(last_ai_ans[:80]) or extract_search_keywords(last_user_q)
    fallback = f"{subject} {query}".strip()
    print(f" [Memory Fallback] Rewrote to: '{fallback}'")
    return fallback

# --------------------------------------------------
# Handlers: Image, Video, Translation
# --------------------------------------------------
def handle_image_request(query: str) -> str:
    subject = extract_search_keywords(query)
    print(f" [*] Opening Google Images for: '{subject}'...")
    encoded = subject.replace(" ", "+")
    search_url = f"https://www.google.com/search?tbm=isch&q={encoded}"
    try:
        webbrowser.open(search_url)
    except Exception:
        pass
    return f"🖼️ Opened Google Images for '{subject}'.\nDirect Link: <a href='{search_url}' target='_blank'>{search_url}</a>"

def handle_video_request(query: str) -> str:
    subject = extract_search_keywords(query)
    print(f" [*] Opening YouTube for: '{subject}'...")
    encoded = subject.replace(" ", "+")
    search_url = f"https://www.youtube.com/results?search_query={encoded}"
    try:
        webbrowser.open(search_url)
    except Exception:
        pass
    return f"📺 Opened YouTube videos for '{subject}'.\nDirect Link: <a href='{search_url}' target='_blank'>{search_url}</a>"

def handle_translation(query: str) -> str:
    if GoogleTranslator is None:
        return "Translation engine unavailable. Please install deep-translator."

    query_lower = query.lower()
    lang_map = {
        "telugu": "te", "hindi": "hi", "spanish": "es", "french": "fr",
        "german": "de", "japanese": "ja", "chinese": "zh-CN", "english": "en"
    }
    target_lang = "te"
    for lang_name, lang_code in lang_map.items():
        if lang_name in query_lower:
            target_lang = lang_code
            break

    quoted_text = re.findall(r'"([^"]*)"|\'([^\']*)\'', query)
    if quoted_text:
        extracted = next((item for sublist in quoted_text for item in sublist if item), None)
        text_to_translate = extracted.strip() if extracted else ""
    else:
        remove_words = [
            "translate", "transulate", "translation", "meaning",
            "in", "to", "on", "into", "word", "phrase", "say", "write"
        ] + list(lang_map.keys())
        words = query.split()
        cleaned_words = [w for w in words if w.lower() not in remove_words]
        text_to_translate = " ".join(cleaned_words).strip()

    if not text_to_translate:
        return "Please specify the text to translate inside quotes (e.g., translate \"hello\" to telugu)."

    try:
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text_to_translate)
        return f"Translated '{text_to_translate}' to {target_lang.upper()}:\n\n**{translated_text}**"
    except Exception as e:
        return f"Translation error: {e}"

# --------------------------------------------------
# Prompts
# --------------------------------------------------
offline_prompt = PromptTemplate.from_template(
    "You are a strict data extraction assistant.\n"
    "Conversation History:\n{history}\n\n"
    "Answer the question using ONLY the provided context. If the context contains relevant information "
    "(such as projects, skills, or experience), extract and list them clearly.\n"
    "If the context does not contain the answer, say \"Information is missing.\"\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

general_prompt = PromptTemplate.from_template(
    "Today's real date is {today}. Treat this as the current date.\n"
    "Conversation History:\n{history}\n\n"
    "You are a factual, concise AI assistant answering using the provided web notes.\n"
    "CRITICAL RULES:\n"
    "1. Base your answer directly on the facts in the context.\n"
    "2. Answer the question directly without filler or disclaimers.\n"
    "3. Do not invent fake links.\n\n"
    "Web notes:\n{context}\n\nQuestion: {question}\nAnswer:"
)

summary_prompt = PromptTemplate.from_template(
    "Today's date is {today}.\n"
    "Conversation History:\n{history}\n\n"
    "Task: Provide a short executive summary and key bullet points based ONLY on the context below.\n"
    "Context:\n{context}\n\nSubject to Summarize: {question}\nSummary:"
)

code_prompt = PromptTemplate.from_template(
    "Today's date is {today}.\n"
    "Conversation History:\n{history}\n\n"
    "Answer with exactly ONE clean code block. Do not add disclaimers or dialog.\n"
    "Reference notes:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# --------------------------------------------------
# Search & Processing
# --------------------------------------------------    
def web_search(query: str, max_results: int = 5) -> str:
    today_str = date.today().strftime("%B %d, %Y")
    clean_kw = extract_search_keywords(query)
    search_query = f"{clean_kw} {today_str}" if is_time_sensitive(query) else clean_kw

    if TAVILY_AVAILABLE and TAVILY_API_KEY:
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            kwargs = dict(max_results=max_results, search_depth="advanced", include_answer=True)
            if is_time_sensitive(query):
                kwargs["topic"] = "news"
                kwargs["days"] = 7
            res = client.search(search_query, **kwargs)

            chunks = [f"(Search run on: {today_str})"]
            if res.get("answer"):
                chunks.append(f"Quick answer: {res['answer']}")
            for r in res.get("results", []):
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")
                chunks.append(f"Source: {title} (URL: {url})\n{content}")
            return "\n\n".join(chunks) if len(chunks) > 1 else "No web results found."
        except Exception as e:
            print(f"Tavily search error: {e}")

    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        ddg = DuckDuckGoSearchResults(num_results=max_results, output_format="list")
        results = ddg.invoke(search_query)
        if not results:
            return "No web results found."
        header = f"(Search run on: {today_str})"
        return header + "\n\n" + "\n\n".join(
            f"{r.get('title', '')}: {r.get('snippet', '')} (URL: {r.get('link', '')})" for r in results
        )
    except Exception as e:
        return f"Web search failed: {e}"

def append_verified_urls(answer: str, web_text: str) -> str:
    urls = re.findall(r'\(URL:\s*(https?://[^\s\)]+)\)', web_text)
    seen = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]

    if unique_urls:
        answer += "\n\n🌐 **Verified Web Links:**\n"
        for u in unique_urls:
            answer += f"- <a href='{u}' target='_blank'>{u}</a>\n"
    return answer

def extract_text_from_file(path):
    _, ext = os.path.splitext(path.lower())
    ext = ext.strip(".")
    if ext in ("txt", "md", "py", "java", "cpp", "c", "js", "ts", "html", "css", "json"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == "pdf":
        texts = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                texts.append(p.extract_text() or "")
        return "\n".join(texts)
    if ext == "docx":
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if ext in ("png", "jpg", "jpeg"):
        return pytesseract.image_to_string(Image.open(path))
    raise ValueError(f"Unsupported file type: {ext}")

# =====================================================
#  FILE TEXT EXTRACTION
# =====================================================
def add_document_from_path(path):
    text = extract_text_from_file(path)
    if not text or not text.strip():
        raise ValueError("No text extracted from file.")
    
    # Notice: vectorDB.delete_collection() has been removed!
    
    chunks = splitter.create_documents([text])
    vectorDB.add_documents(chunks)

def answer_with_llm(query: str, context: str, chat_history: list, is_offline: bool = False) -> str:
    intent = classify_query(query)
    if intent == "summary":
        template = summary_prompt
    elif intent == "code":
        template = code_prompt
    elif is_offline:
        template = offline_prompt
    else:
        template = general_prompt

    today_str = date.today().strftime("%B %d, %Y")
    formatted_hist = format_history_for_prompt(chat_history)

    raw_answer = llm.invoke(template.format(
        context=context,
        question=query,
        today=today_str,
        history=formatted_hist
    )).strip()

    stop_phrases = ["\nUser:", "User:", "\nQuestion:", "\nAI:"]
    for phrase in stop_phrases:
        if phrase in raw_answer:
            raw_answer = raw_answer.split(phrase)[0].strip()

    return raw_answer

# --------------------------------------------------
# Core Modes (Using the Context-Resolved Query)
# --------------------------------------------------
def online_rag(resolved_query: str, chat_history: list) -> str:
    print(" Searching the web (Tavily)..." if (TAVILY_AVAILABLE and TAVILY_API_KEY) else " Searching the web (DuckDuckGo fallback)...")
    web_text = web_search(resolved_query)
    answer = answer_with_llm(resolved_query, web_text, chat_history, is_offline=False)
    return append_verified_urls(answer, web_text)

def offline_rag(resolved_query: str, chat_history: list) -> str:
    print(f" [*] Reading local documents for: '{resolved_query}'...")
    docs = retriever.invoke(resolved_query)
    context = "\n\n".join(d.page_content for d in docs)
    return answer_with_llm(resolved_query, context, chat_history, is_offline=True)

def hybrid_rag(resolved_query: str, chat_history: list) -> str:
    answer = offline_rag(resolved_query, chat_history)
    if "information is missing" in answer.lower() or len(answer) < 20:
        print(" [!] Missing in database. Consulting live web...")
        web_text = web_search(resolved_query)
        ai_answer = answer_with_llm(resolved_query, web_text, chat_history, is_offline=False)
        return append_verified_urls(ai_answer, web_text)
    return answer

# --------------------------------------------------
# Universal Flask Interface
# --------------------------------------------------
global_chat_history = []

def process_query_with_mode(query: str, mode: str = "hybrid") -> str:
    global global_chat_history

    intent = classify_query(query)

    try:
        if intent == "image":
            return handle_image_request(query)
        if intent == "video":
            return handle_video_request(query)
        if intent == "translate":
            return handle_translation(query)

        # 1. Resolve context into a standalone query (e.g. 'Where did Sundar Pichai complete his bachelors degree?')
        resolved_query = build_search_query(query, global_chat_history)

        # 2. Pass the resolved query directly to the active retrieval mode
        if mode in ("internet", "1"):
            final_answer = online_rag(resolved_query, global_chat_history)
        elif mode in ("offline", "2"):
            final_answer = offline_rag(resolved_query, global_chat_history)
        else:
            final_answer = hybrid_rag(resolved_query, global_chat_history)

        # 3. Track the original question and its answer in the conversation history
        global_chat_history.append((query, final_answer))
        if len(global_chat_history) > 5:
            global_chat_history.pop(0)

        return final_answer
    except Exception as e:
        return f"Error: {e}"
