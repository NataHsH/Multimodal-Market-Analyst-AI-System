from dotenv import load_dotenv
load_dotenv()

import os
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import open_clip
from PIL import Image
import torch
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from uuid import uuid4


# === CLIP model ===
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# === Settings ===
PDF_PATH = "apple_2024_10k.pdf"
CHROMA_TEXT_DIR = "chroma_text"
CHROMA_IMAGE_DIR = "chroma_image"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
text_embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# === CLIP wrapper ===
class CLIPEmbeddingWrapper:
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess

    def embed_documents(self, docs):
        embeddings = []
        for doc in docs:
            try:
                # image is passed outside of metadata
                image = doc.metadata["image_data"]
                image_tensor = self.preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    vector = self.model.encode_image(image_tensor).cpu().numpy()[0]
                embeddings.append(vector.tolist())
            except Exception as e:
                print(f"Error embedding image: {e}")
                embeddings.append([0.0] * 512)
        return embeddings

    def embed_query(self, text):
        tokens = clip_tokenizer([text]).to(device)
        with torch.no_grad():
            vec = self.model.encode_text(tokens).cpu().numpy()[0]
        return vec.tolist()


clip_embedder = CLIPEmbeddingWrapper(clip_model, preprocess)


# === PDF extraction ===
def extract_pdf(pdf_path):
    text_chunks, table_chunks, header_chunks, image_docs, image_data = [], [], [], [], []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # === Images
            page_image = page.to_image(resolution=300)
            base_image = page_image.original

            for img_dict in page.images:
                try:
                    x0, top, x1, bottom = img_dict["x0"], img_dict["top"], img_dict["x1"], img_dict["bottom"]
                    cropped_image = base_image.crop((x0, top, x1, bottom))

                    doc = Document(
                        page_content="[IMAGE]",
                        metadata={"type": "image", "page": page_num, "image_data": cropped_image}  # image_data used internally
                    )
                    image_docs.append(doc)
                except Exception as e:
                    print(f"Image extract error: {e}")

            # === Headers
            bold_headers = set()
            for char in page.chars:
                if "Bold" in char.get("fontname", "") and len(char["text"].strip()) > 2:
                    bold_headers.add(char["text"].strip())

            for header in bold_headers:
                if len(header) < 120:
                    header_chunks.append(Document(page_content=header, metadata={"type": "header"}))

            # === Tables
            for table in page.extract_tables():
                table_text = "\n".join(
                    [" | ".join(cell if cell else "" for cell in row) for row in table if row]
                )
                if table_text.strip():
                    table_chunks.append(Document(page_content=table_text, metadata={"type": "table"}))

            # === Text
            text = page.extract_text()
            if text:
                text_chunks.append(text)

    # === Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_docs = splitter.create_documents(text_chunks, metadatas=[{"type": "text"}] * len(text_chunks))

    # === Remove image_data from metadata
    for doc in image_docs:
        doc.metadata = {k: v for k, v in doc.metadata.items() if k != "image_data"}

    return text_docs + table_chunks + header_chunks, image_docs


# === Extract documents
text_docs, image_docs = extract_pdf(PDF_PATH)

# === Save text index
if not os.path.exists(CHROMA_TEXT_DIR) or not os.listdir(CHROMA_TEXT_DIR):
    text_store = Chroma.from_documents(
        documents=text_docs,
        embedding=text_embedding,
        persist_directory=CHROMA_TEXT_DIR
    )
    text_store.persist()
    print(f"✅ Text index created with {len(text_docs)} docs.")
else:
    text_store = Chroma(persist_directory=CHROMA_TEXT_DIR, embedding_function=text_embedding)
    print("📁 Loaded existing text index.")

# === Save image index
"""
image_docs_filtered = [
    Document(page_content=doc.page_content, metadata=filter_complex_metadata(doc.metadata))
    for doc in image_docs
]
"""



if not os.path.exists(CHROMA_IMAGE_DIR) or not os.listdir(CHROMA_IMAGE_DIR):
    image_vectors = clip_embedder.embed_documents(image_docs)
    image_store = Chroma.from_documents(
    documents=image_docs, #_filtered,
    embedding=clip_embedder,  #  CLIPEmbeddingWrapper
    persist_directory=CHROMA_IMAGE_DIR
)
    image_store.persist()

    #
    print(f"🖼️  Image index created with {len(image_docs)} images.")
else:
    image_store = Chroma(persist_directory=CHROMA_IMAGE_DIR, embedding_function=clip_embedder)
    print("📁 Loaded existing image index.")

# === Sample query
query = "chart"
results = image_store.similarity_search_by_vector(clip_embedder.embed_query(query), k=3)

print("\n🔍 Top matching images:")
for r in results:
    print(f"[{r.metadata.get('type')}] page {r.metadata.get('page')}")

# Storage for comparison of additional vectors
vectorstore_backing_store = InMemoryStore()

# creat retriever
retriever = MultiVectorRetriever(
    vectorstore=text_store,
    docstore=vectorstore_backing_store,
    id_key="doc_id",
)


# === Подготовка: привязка уникальных ID
for doc in text_docs:
    if "doc_id" not in doc.metadata:
        doc.metadata["doc_id"] = str(uuid4())

# === 1. Инициализация общего docstore
docstore = InMemoryStore()
docstore.mset([(doc.metadata["doc_id"], doc) for doc in text_docs])

# 4. Извлекаем эмбеддинги документов (как в vectorstore.similarity_search)
text_relevant_docs = text_store.similarity_search(query, k=5)
image_relevant_docs = image_store.similarity_search_by_vector(
    clip_embedder.embed_query(query), k=5
)

# 5. Собираем уникальные doc_id (по связке `page`)
combined_docs = []
pages_seen = set()

for doc in text_relevant_docs + image_relevant_docs:
    page = doc.metadata.get("page")
    if page not in pages_seen:
        # найдём основной текстовый документ по page
        for main_doc in text_docs:
            if main_doc.metadata.get("page") == page:
                combined_docs.append(main_doc)
                pages_seen.add(page)
                break

# 6. Печать результата
print("\n🔍 MultiModal Search Results:")
for i, doc in enumerate(combined_docs, 1):
    print(f"{i}. Page {doc.metadata.get('page')} — {doc.page_content[:300]}")

#query = "Show me charts about financial performance"
#results = retriever.get_relevant_documents(query)

#print("\n🔍 MultiVectorRetriever results:")
#for i, doc in enumerate(results, 1):
 #   print(f"{i}. [{doc.metadata.get('type')}] {doc.page_content[:300]}")
