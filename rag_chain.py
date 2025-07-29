import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Set environment keys
os.environ["TAVILY_API_KEY"] = "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"

# Load local knowledge
with open("C:\\Users\\sures\\OneDrive\\Desktop\\AI PROJECT\\2024_state_of_the_union.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents([raw_text])

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorDB = Chroma(collection_name="rag_docs", embedding_function=embedding_model)
vectorDB.add_documents(texts)
retriever = vectorDB.as_retriever()

# Load LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
pipeline_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipeline_model)

# Prompt
prompt_template = PromptTemplate.from_template("""
Use the context below to answer the question.
If you don’t know the answer, say “I don’t know.”

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

search_tool = TavilySearchResults()

def hybrid_rag(query):
    answer = rag_chain.invoke(query)

    if "I don't know" in answer or len(answer.strip()) < 15:
        web_docs = search_tool.run(query)
        if not web_docs:
            return "🤖 Sorry, I couldn’t find an answer."

        combined_content = "\n".join([doc["content"] for doc in web_docs])
        new_chunks = text_splitter.create_documents([combined_content])
        vectorDB.add_documents(new_chunks)
        answer = rag_chain.invoke(query)

    return answer
