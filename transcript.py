import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MP3_DIR = "./mp3"
TXT_DIR = "./txt"


def transcribe_arabic(file_path, output_dir=TXT_DIR):
    """تحويل ملف صوتي إلى نص عربي باستخدام Whisper"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ar",
                response_format="text"
            )

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"✅ تم: {file_path} → {output_path}")
        return transcript

    except FileNotFoundError:
        print(f"❌ الملف غير موجود: {file_path}")
        return None
    except Exception as e:
        print(f"❌ خطأ في معالجة {file_path}: {e}")
        return None


if __name__ == "__main__":
    # اكتشاف تلقائي لجميع ملفات MP3/MPEG في مجلد mp3
    if not os.path.isdir(MP3_DIR):
        print(f"❌ مجلد {MP3_DIR} غير موجود")
        exit(1)

    audio_extensions = (".mp3", ".mpeg", ".wav", ".m4a")
    files = sorted([
        os.path.join(MP3_DIR, f)
        for f in os.listdir(MP3_DIR)
        if f.lower().endswith(audio_extensions)
    ])

    if not files:
        print(f"⚠️ لا توجد ملفات صوتية في {MP3_DIR}")
        exit(0)

    print(f"🚀 بدء تحويل {len(files)} ملف صوتي...")
    success = 0
    for f in files:
        result = transcribe_arabic(f)
        if result:
            success += 1

    print(f"\n✅ اكتمل: {success}/{len(files)} ملف تم تحويله بنجاح")