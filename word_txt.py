import os
import json
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------- استخراج النص من ملفات Word --------
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return "\n".join(full_text)

# -------- تقطيع النص إلى chunks ذكية --------
def split_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# -------- معالجة كل المجلدات --------
def process_all_folders(root_folder):
    all_chunks = []
    
    # Process files in root folder first
    print(f"\n📂 معالجة ملفات في المجلد الرئيسي")
    for file_name in os.listdir(root_folder):
        file_path = os.path.join(root_folder, file_name)
        if os.path.isfile(file_path) and (file_name.endswith(".docx") or file_name.endswith(".doc")):
            print(f"  📄 معالجة: {file_name}")
            
            text = extract_text_from_docx(file_path)
            chunks = split_into_chunks(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"root_{file_name}_{i}",
                    "category": "root",
                    "source": file_name,
                    "text": chunk
                })
    
    # Process subdirectories
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        print(f"\n📂 معالجة مجلد: {folder_name}")
        
        for file_name in os.listdir(folder_path):
            if not (file_name.endswith(".docx") or file_name.endswith(".doc")):
                continue
                
            file_path = os.path.join(folder_path, file_name)
            print(f"  📄 معالجة: {file_name}")
            
            text = extract_text_from_docx(file_path)
            chunks = split_into_chunks(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{folder_name}_{file_name}_{i}",
                    "category": folder_name,
                    "source": file_name,
                    "text": chunk
                })
    
    return all_chunks

# -------- توليد الـ Embeddings وحفظها --------
def generate_embeddings(chunks):
    print(f"\n⚙️ جاري توليد embeddings لـ {len(chunks)} chunk...")
    
    knowledge_base = []
    
    for i, chunk in enumerate(chunks):
        response = client_openai.embeddings.create(
            input=chunk["text"],
            model="text-embedding-3-small"
        )
        
        knowledge_base.append({
            **chunk,
            "embedding": response.data[0].embedding
        })
        
        if (i + 1) % 10 == 0:
            print(f"  ✅ تم معالجة {i + 1} من {len(chunks)}")
    
    return knowledge_base

# -------- حفظ الـ Knowledge Base --------
def save_knowledge_base(knowledge_base, output_file="knowledge_base.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    print(f"\n💾 تم حفظ الـ Knowledge Base في {output_file}")
    print(f"📊 إجمالي الـ chunks: {len(knowledge_base)}")

# -------- التشغيل --------
if __name__ == "__main__":
    ROOT_FOLDER = "./docs"  # ← ضع هنا مسار مجلداتك
    
    print("🚀 بدء معالجة الملفات...")
    chunks = process_all_folders(ROOT_FOLDER)
    
    print(f"\n📝 إجمالي الـ chunks المستخرجة: {len(chunks)}")
    
    knowledge_base = generate_embeddings(chunks)
    save_knowledge_base(knowledge_base)
    
    print("\n✅ اكتمل بناء الـ Knowledge Base!")
