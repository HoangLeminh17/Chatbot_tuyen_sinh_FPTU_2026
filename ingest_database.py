from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from collections import Counter

# Load file env - API gọi model gemini / OpenAI
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# 1. Khởi tạo model embedding - đa ngôn ngữ, 1024 chiều output
embeddings_model = HuggingFaceEmbeddings(model_name = 'BAAI/bge-m3')

# 2. Load file.txt, tham số glob là regex pattern - lấy toàn bộ file txt trong thư mục con
# show_progres = True: hiển thị tiến trình
# Load file txt
loader_2026 = DirectoryLoader(path = DATA_PATH, glob = "**/*.txt", 
                              loader_cls = TextLoader,
                              loader_kwargs = {"encoding": "utf-8","autodetect_encoding": True},
                              use_multithreading = True,
                              show_progress = True, ) # use_multithreading = True : load song song
# Load file pdf
loader_2025 = DirectoryLoader(path = DATA_PATH, glob = "**/*.pdf",
                              loader_cls = PyMuPDFLoader,
                              show_progress = True)

# Tổng hợp loader - in ra số chunk của mỗi file
raw_doc = loader_2026.load() + loader_2025.load()
print(f"Đã tìm thấy và load {len(raw_doc)} tài liệu")

sources = [d.metadata["source"] for d in raw_doc]
counter = Counter(sources)
print("\n=== SỐ CHUNK CHO MỖI FILE ===\n")
print(f"{'File':60} | {'Chunks'}")
print("-"*75)

for file, count in counter.items():
    print(f"{file:60} | {count}")   

# 3. Chia nhỏ các doc 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 800, # cũ là 300
    chunk_overlap= 150, # cũ là 100
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n","\n","."," "] # split theo đoạn
)

# tạo các đoạn (chunks)
chunks = text_splitter.split_documents(raw_doc)

# 4. Lưu vào ChromaDB
vector_store = Chroma(
    collection_name="rag_fptu_2026", # có nhiều collection độc lập trong chroma_db
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# VectorDB luôn lưu (vector embedding, metadata, id)
# Tạo ID random duy nhất cho mỗi chunk
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Embed từng chunk + lưu vector vào DB + gắn ID cho mỗi vector, khi đã có vector db thì không cần chạy lại nữa
vector_store.add_documents(documents=chunks, ids=uuids)