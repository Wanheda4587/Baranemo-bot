import json
import os
from datetime import datetime
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_anthropic import ChatAnthropic

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GECMIS_DOSYA = "sohbet_gecmisi.json"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./veritabani", embedding_function=embeddings)
llm = ChatAnthropic(model="claude-sonnet-5", anthropic_api_key=ANTHROPIC_KEY, max_tokens=1000)

def gecmisi_yukle():
    if os.path.exists(GECMIS_DOSYA):
        with open(GECMIS_DOSYA, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def gecmisi_kaydet(gecmis):
    with open(GECMIS_DOSYA, "w", encoding="utf-8") as f:
        json.dump(gecmis, f, ensure_ascii=False, indent=2)

def metni_al(response):
    if isinstance(response.content, str):
        return response.content
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block["text"]
    return str(response.content)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    simdi = datetime.now().strftime("%d.%m.%Y %H:%M")

    await update.message.reply_text("⏳ Düşünüyorum...")

    docs = vectordb.similarity_search(user_message, k=4)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    gecmis = gecmisi_yukle()
    son_10 = gecmis[-10:] if len(gecmis) > 10 else gecmis
    gecmis_metni = "\n".join([
        f"[{m['tarih']}] Kullanıcı: {m['soru']}\n[{m['tarih']}] Sen: {m['cevap']}"
        for m in son_10
    ])

    prompt = f"""Sen Barış Baranemo'nun eğitimlerini derinlemesine özümsemiş bir kişisel gelişim ve psikoloji asistanısın.

Kuralların:
- Her zaman Türkçe konuş
- Baranemo'nun samimi, doğrudan ve pratik üslubunu kullan
- Sana verilen Baranemo içeriğinden cevap ver, içerikte yoksa dürüstçe söyle
- Gereksiz teselli etme, somut ve uygulanabilir şeyler söyle
- Kullanıcının geçmiş mesajlarını ve tarihlerini dikkate al
- Şu anki tarih ve saat: {simdi}

Geçmiş sohbetler:
{gecmis_metni if gecmis_metni else "Henüz geçmiş sohbet yok."}

Baranemo içeriği:
{context_text}

Kullanıcının şu anki mesajı ({simdi}): {user_message}"""

    messages = [{"role": "user", "content": prompt}]
    response = llm.invoke(messages)
    cevap = metni_al(response)

    gecmis.append({
        "tarih": simdi,
        "soru": user_message,
        "cevap": cevap
    })
    gecmisi_kaydet(gecmis)

    await update.message.reply_text(cevap)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot çalışıyor...")
    app.run_polling()

if __name__ == "__main__":
    main()