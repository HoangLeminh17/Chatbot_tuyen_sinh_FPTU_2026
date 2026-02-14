from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"

# Embedding
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

# FREE LLM (OpenRouter)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-r1-0528:free",
    temperature=0.3,
)

# Vector DB
vector_store = Chroma(
    collection_name="rag_fptu_2026",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Retriever (giảm k)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "fetch_k":6}
)

def stream_response(message, history):

    docs = retriever.invoke(message)

    if not docs:
        yield "Mình không thấy thông tin trong cơ sở dữ liệu"
        return

    # chỉ giữ history gần
    history = history[-2:]
    history_text = ""

    for msg in history:
        role = "Người dùng" if msg["role"]=="user" else "Trợ lý"
        history_text += f"{role}: {msg['content']}\n"

    # build knowledge (không nhân đôi + cắt ngắn)
    knowledge = ""

    for i, doc in enumerate(docs[:3], 1):
        source = os.path.basename(doc.metadata.get("source","unknown"))
        content = doc.page_content[:900]
        knowledge += f"[Tài liệu {i} - {source}]\n{content}\n\n"

    rag_prompt = f"""
Bạn là trợ lý tuyển sinh của Trường Đại học FPT.

Chỉ trả lời dựa trên tri thức được cung cấp.
Nếu không có thông tin thì nói không biết.
Trả lời bằng tiếng Việt.

Câu hỏi: {message}

Lịch sử:
{history_text}

Tri thức:
{knowledge}
"""

    partial = ""
    for chunk in llm.stream(rag_prompt):
        partial += chunk.content
        yield partial

chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(
        placeholder="Hỏi về học phí, học bổng, ngành học...",
        container=False,
        scale=7
    ),
)

chatbot.launch(share=True)
