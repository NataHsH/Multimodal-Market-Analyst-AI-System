from dotenv import load_dotenv
load_dotenv()

import os
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 📌 Settings
PDF_PATH = "apple_2024_10k.pdf"
CHROMA_DIR = "chroma_multivector"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# 🧠 Embedding model
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 📖 read PDF
text_chunks, table_chunks, header_chunks = [], [], []
with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
       
        # --- headers 
        bold_headers = set()
        chars = page.chars
        for char in chars:
            font = char.get("fontname", "")
            if "Bold" in font and len(char["text"].strip()) > 2:
                bold_headers.add(char["text"].strip())

        # Combine similar titles
        for header in bold_headers:
            if len(header) < 120:
                header_chunks.append(Document(page_content=header, metadata={"type": "header"}))

        # --- tables
        tables = page.extract_tables()
        for table in tables:
            cleaned_table = []
            for row in table:
                cleaned_row = [cell if cell is not None else "" for cell in row]
                cleaned_table.append(" | ".join(cleaned_row))
            table_text = "\n".join(cleaned_table)
            
            if table_text.strip():
                table_chunks.append(Document(page_content=table_text, metadata={"type": "table"}))

        # --- taxt
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)

# ✂️ text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_documents = splitter.create_documents(text_chunks, metadatas=[{"type": "text"}] * len(text_chunks))

# 🔗 combine all
all_documents = text_documents + table_chunks + header_chunks

# 💾 save db
if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
else:
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()

print(f"✅ hohgrladen: {len(all_documents)} Documents")

# 🔍 Example request
query = "What is the total revenue in the latest quarter?"
results = vectorstore.similarity_search(query, k=5)

print("\n📄 beste ansver:")
for doc in results:
    print(f"[{doc.metadata.get('type')}] {doc.page_content[:300]}\n")


