import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from rank_bm25 import BM25Okapi
from elevenlabs import ElevenLabs
import re

def normalize_arabic(text):
    text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', text)
    text = re.sub(r'[\u0640]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text) 
    text = re.sub(r'[يى]', 'ي', text)
    # text = re.sub(r'[ةه]', 'ه', text)
    return text


load_dotenv()

client_openai  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client_eleven  = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
BOND_VOICE_ID  = os.getenv("bondVoiceId")

# -------- إعدادات الصفحة --------
st.set_page_config(
    page_title="التوأم التقني - بندق",
    page_icon="🎙️",
    layout="centered"
)

# -------- تحميل الـ Knowledge Base --------
@st.cache_resource
def load_knowledge_base(file_path="knowledge_base.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------- بناء BM25 Index --------
@st.cache_resource
def build_bm25_index(_knowledge_base):
    # tokenized_corpus = [chunk["text"].split() for chunk in _knowledge_base]
    tokenized_corpus = [normalize_arabic(chunk["text"]).split() for chunk in _knowledge_base]
    return BM25Okapi(tokenized_corpus)

# -------- حساب التشابه --------
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# -------- تحسين السؤال --------
def rewrite_query(question, chat_history):
    history_text = ""
    if chat_history:
        last_exchanges = chat_history[-4:]
        history_text = "\n".join([
            f"{'المستخدم' if msg['role'] == 'user' else 'المساعد'}: {msg['content'][:200]}"
            for msg in last_exchanges
        ])

    prompt = f"""بناءً على سياق المحادثة التالية:
{history_text if history_text else "لا يوجد سياق سابق"}

السؤال الحالي: {question}

أعد صياغة السؤال ليكون سؤالاً واحداً واضحاً ومستقلاً يحتوي على كل الكلمات المفتاحية المهمة للبحث.
أعطني فقط السؤال المُعاد صياغته بدون أي شرح أو مقدمة."""

    response = client_anthropic.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# -------- البحث الهجين Hybrid Search + RRF --------
def hybrid_search(question, knowledge_base, bm25_index, top_k=5, rrf_k=60):
    response = client_openai.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    question_embedding = response.data[0].embedding

    semantic_scores = [
        cosine_similarity(question_embedding, chunk["embedding"])
        for chunk in knowledge_base
    ]
    semantic_ranked = np.argsort(semantic_scores)[::-1].tolist()

    # tokenized_query = question.split()
    tokenized_query = normalize_arabic(question).split()
    bm25_scores     = bm25_index.get_scores(tokenized_query)
    bm25_ranked     = np.argsort(bm25_scores)[::-1].tolist()

    rrf_scores = {}
    for rank, idx in enumerate(semantic_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)

    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for idx in sorted_indices[:top_k]:
        chunk = knowledge_base[idx].copy()
        chunk["score"]      = semantic_scores[idx]
        chunk["bm25_score"] = float(bm25_scores[idx])
        chunk["rrf_score"]  = rrf_scores[idx]
        results.append(chunk)
        results = [r for r in results if r["rrf_score"] > 0.01] # فلترة النتائج ذات RRF منخفض جداً

    return results

# -------- تحويل الصوت إلى نص - Whisper --------
def speech_to_text(audio_file):
    transcript = client_openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ar"
    )
    return transcript.text

# -------- تحويل النص إلى صوت بندق - ElevenLabs --------
def text_to_speech(text):
    audio_generator = client_eleven.text_to_speech.convert(
        voice_id=BOND_VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    audio_bytes = b"".join(audio_generator)
    return audio_bytes

# -------- الـ System Prompt --------
SYSTEM_PROMPT = """أنت نموذج رقمي يحاكي تفكير وأسلوب صاحب هذه المعرفة بدقة تامة.

قواعد صارمة جداً:
- أجب فقط من المحتوى المُقدم لك في كل رسالة تحت "السياق المتاح"
- إذا لم يكن الجواب موجوداً حرفياً في السياق المُقدم قل: "هذا الموضوع ليس في المحتوى المتاح لديّ"
- ممنوع منعاً باتاً استخدام أي معلومة من خارج السياق المُقدم حتى لو كنت متأكداً منها
- ممنوع التخمين أو الاستنتاج بما يتجاوز ما هو مكتوب في السياق
- حافظ على أسلوب وطريقة تفكير صاحب المحتوى
- إذا كان السؤال بالعربية فأجب بالعربية، وإذا كان بالإنجليزية فأجب بالإنجليزية
- تذكر سياق المحادثة السابقة واربط الأسئلة ببعضها بشكل طبيعي
- اجعل إجاباتك مناسبة للاستماع الصوتي - جمل واضحة وطبيعية بدون رموز أو تنسيق خاص"""

# -------- توليد الإجابة مع Streaming --------
def generate_answer_streaming(question, relevant_chunks, chat_history):
    context = "\n\n---\n\n".join([
        f"المصدر: {chunk['source']} | التصنيف: {chunk['category']}\n{chunk['text']}"
        for chunk in relevant_chunks
    ])

    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"""السياق المتاح من قاعدة المعرفة:
{context}

السؤال الحالي: {question}"""
    })

    with client_anthropic.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            yield text

# ======== الواجهة الرئيسية ========
st.title("🎙️ التوأم التقني - بندق")
st.caption("🎙️ صوت → رد بالصوت | ⌨️ نص → رد بالنص")
st.divider()

knowledge_base = load_knowledge_base()
bm25_index     = build_bm25_index(knowledge_base)
st.success(f"✅ Knowledge Base محمّل - {len(knowledge_base)} chunk جاهزة")

# زر تفعيل الصوت
col1, col2 = st.columns([3, 1])
with col2:
    voice_enabled = st.toggle("🔊 الصوت", value=True)

# تهيئة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.messages:
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()

# عرض المحادثة السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 المصادر"):
                for i, chunk in enumerate(message["sources"], 1):
                    st.caption(
                        f"{i}. {chunk['source']} | "
                        f"دلالي: {chunk['score']:.0%} | "
                        f"RRF: {chunk['rrf_score']:.4f}"
                    )

# -------- إدخال الصوت --------
st.markdown("**🎙️ سجّل سؤالك:**")
audio_input = st.audio_input("اضغط للتسجيل")

question = None
input_was_voice = False  # ← هنا نتتبع نوع الإدخال

if audio_input:
    with st.spinner("🎧 جاري تحويل الصوت إلى نص..."):
        question = speech_to_text(audio_input)
    st.info(f"📝 تم التعرف على: **{question}**")
    input_was_voice = True  # ← الإدخال كان صوتاً

# -------- إدخال النص --------
text_input = st.chat_input("أو اكتب سؤالك هنا...")
if text_input:
    question = text_input
    input_was_voice = False  # ← الإدخال كان نصاً

# -------- معالجة السؤال --------
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔍 جاري التحليل والبحث..."):
        chat_history_for_rewrite = st.session_state.messages[:-1]
        rewritten_query  = rewrite_query(question, chat_history_for_rewrite)
        relevant_chunks  = hybrid_search(rewritten_query, knowledge_base, bm25_index)

    with st.chat_message("assistant"):
        chat_history = st.session_state.messages[:-1]

        full_answer = st.write_stream(
            generate_answer_streaming(question, relevant_chunks, chat_history)
        )

        # تشغيل صوت بندق فقط إذا كان الإدخال صوتاً ← التعديل الرئيسي
        if voice_enabled and input_was_voice:
            with st.spinner("🎙️ بندق يتكلم..."):
                audio_bytes = text_to_speech(full_answer)
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        with st.expander("📚 المصادر والتفاصيل التقنية"):
            st.caption(f"🔄 **السؤال المُحسَّن:** {rewritten_query}")
            st.divider()
            for i, chunk in enumerate(relevant_chunks, 1):
                st.caption(
                    f"{i}. {chunk['source']} | "
                    f"دلالي: {chunk['score']:.0%} | "
                    f"BM25: {chunk['bm25_score']:.2f} | "
                    f"RRF: {chunk['rrf_score']:.4f}"
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": relevant_chunks
    })
