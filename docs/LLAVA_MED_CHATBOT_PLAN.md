# LLaVA-Med Medical VQA Chatbot — Uygulama Planı

Bu döküman, DiffuVQA reposundan yeni bir branch açarak LLaVA-Med tabanlı bir Medical VQA chatbot oluşturmak için gereken tüm adımları içerir. Sıfırdan başlayıp çalışan bir Gradio demo'ya ulaşmak için bu dökümanı takip et.

---

## Genel Bilgi

- **Model:** `katielink/llava-med-7b-slake-delta`
- **Mimari:** LLaVA-7B (Transformer tabanlı, diffusion değil) + SLAKE fine-tune
- **VRAM:** 4-bit quantization ile ~6-7GB (8GB GPU yeterli)
- **Arayüz:** Gradio (share=True ile public demo linki)
- **Database:** SQLite (soru-cevap geçmişi)
- **Disk:** ~15-20GB (base model + delta weights)

---

## ADIM 0: Branch

Branch zaten açıldı: `ChatBotPipeline`

```bash
# Repoyu klonla
git clone -b ChatBotPipeline https://github.com/panzerofthelake03/DiffuVQA.git
cd DiffuVQA

# veya zaten klonladıysan branch'e geç
git fetch origin
git checkout ChatBotPipeline
```

---

## ADIM 1: Klasör Yapısını Oluştur

```bash
mkdir -p chatbot
cd chatbot
```

Oluşacak yapı:

```
DiffuVQA/
└── chatbot/
    ├── app.py            # Ana uygulama (Gradio arayüzü)
    ├── model.py          # Model yükleme ve inference
    ├── database.py       # SQLite işlemleri
    ├── requirements.txt  # Bağımlılıklar
    └── chat_history.db   # Otomatik oluşur, commitleme
```

`.gitignore`'a ekle:
```
chatbot/chat_history.db
```

---

## ADIM 2: Ortam Kurulumu

Python 3.10+ ve CUDA 11.8+ gerekli. Sanal ortam oluştur:

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

`chatbot/requirements.txt` dosyasını oluştur:

```
torch==2.1.0
torchvision==0.16.0
transformers==4.37.0
peft==0.7.1
bitsandbytes==0.41.3
accelerate==0.25.0
gradio==4.19.2
Pillow==10.2.0
sqlalchemy==2.0.27
```

Kur:

```bash
# PyTorch CUDA 11.8 için
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Diğer paketler
pip install -r chatbot/requirements.txt
```

> **Not:** `bitsandbytes` Windows'ta sorun çıkarabilir. Linux veya WSL2 kullan.

---

## ADIM 3: Model Dosyası

`chatbot/model.py` dosyasını oluştur:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image

MODEL_ID = "katielink/llava-med-7b-slake-delta"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = None
model = None


def load_model():
    global tokenizer, model
    if model is not None:
        return  # Zaten yüklü, tekrar yükleme

    print("Model yükleniyor... (ilk seferinde 10-15 dakika sürebilir)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print("Model yüklendi.")


def ask(image_path: str, question: str) -> str:
    if model is None:
        load_model()

    image = Image.open(image_path).convert("RGB")

    prompt = (
        "You are a medical AI assistant. "
        "Answer the following question about the medical image briefly and accurately.\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1].strip()
    return answer if answer else "Model cevap üretemedi."
```

---

## ADIM 4: Database Dosyası

`chatbot/database.py` dosyasını oluştur:

```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///chatbot/chat_history.db", echo=False)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String(512))
    question   = Column(Text)
    answer     = Column(Text)


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def save(image_path: str, question: str, answer: str):
    session = Session()
    try:
        record = ChatHistory(
            image_path=image_path,
            question=question,
            answer=answer,
        )
        session.add(record)
        session.commit()
    finally:
        session.close()


def get_recent(limit: int = 20):
    session = Session()
    try:
        records = (
            session.query(ChatHistory)
            .order_by(ChatHistory.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "question": r.question,
                "answer": r.answer,
                "image_path": r.image_path,
            }
            for r in records
        ]
    finally:
        session.close()


def get_all_count():
    session = Session()
    try:
        return session.query(ChatHistory).count()
    finally:
        session.close()
```

---

## ADIM 5: Gradio Arayüzü

`chatbot/app.py` dosyasını oluştur:

```python
import gradio as gr
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.model import load_model, ask
from chatbot import database as db


def on_submit(image, question):
    if image is None:
        return "Lütfen bir görüntü yükleyin."
    if not question or not question.strip():
        return "Lütfen bir soru girin."

    answer = ask(image, question.strip())
    db.save(image, question.strip(), answer)
    return answer


def on_history():
    records = db.get_recent(20)
    if not records:
        return "Henüz soru sorulmamış."
    lines = []
    for r in records:
        lines.append(f"[{r['timestamp']}]")
        lines.append(f"Soru : {r['question']}")
        lines.append(f"Cevap: {r['answer']}")
        lines.append("-" * 50)
    return "\n".join(lines)


def on_stats():
    total = db.get_all_count()
    return f"Toplam soru sayısı: {total}"


# Model başlangıçta yükle
print("Uygulama başlatılıyor...")
load_model()

with gr.Blocks(title="Medical VQA — LLaVA-Med", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Medical Visual Question Answering\nLLaVA-Med (SLAKE fine-tuned)")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="Tıbbi Görüntü (JPG/PNG)",
            )
            question_input = gr.Textbox(
                label="Soru",
                placeholder="What modality is shown in this image?",
                lines=2,
            )
            submit_btn = gr.Button("Sor", variant="primary")

        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Cevap",
                lines=4,
                interactive=False,
            )

    gr.Markdown("---")

    with gr.Row():
        history_btn = gr.Button("Son 20 Soruyu Göster")
        stats_btn = gr.Button("İstatistik")

    history_output = gr.Textbox(label="Geçmiş", lines=15, interactive=False)

    submit_btn.click(
        fn=on_submit,
        inputs=[image_input, question_input],
        outputs=answer_output,
    )
    history_btn.click(fn=on_history, outputs=history_output)
    stats_btn.click(fn=on_stats, outputs=history_output)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,   # Public demo linki üretir
    )
```

---

## ADIM 6: Çalıştır

```bash
# Proje kökünden çalıştır
cd DiffuVQA
python chatbot/app.py
```

Terminalde şunu görmelisin:
```
Uygulama başlatılıyor...
Model yükleniyor...
Model yüklendi.
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxx.gradio.live
```

`share=True` ile üretilen `gradio.live` linki 72 saat geçerlidir. Demo için bu linki paylaş.

---

## ADIM 7: Test

Arayüz açıldıktan sonra şu sorularla test et:

| Görüntü Tipi | Test Sorusu | Beklenen Cevap |
|---|---|---|
| MRI | What modality is shown? | MRI |
| X-Ray | Is this a chest X-ray? | yes/no |
| CT | Which organ is visible? | liver / lung vb. |
| Herhangi | Does this image look normal? | yes/no |

---

## ADIM 8: Commit ve Push

```bash
git add chatbot/
git add .gitignore
git commit -m "Add LLaVA-Med chatbot with Gradio UI and SQLite history"
git push origin llava-med-chatbot
```

---

## Sık Karşılaşılan Hatalar

### `CUDA out of memory`
```python
# bnb_config'e ekle:
bnb_4bit_use_double_quant=True
# veya max_new_tokens'ı 64'e düşür
```

### `bitsandbytes not found`
```bash
pip install bitsandbytes --upgrade
# Windows'ta çalışmıyorsa WSL2 kullan
```

### `Delta weights not found`
Model ilk indirmede base LLaVA-7B'yi de çeker (~13GB). İnternet bağlantısı stabil olmalı.

### `Port 7860 already in use`
```python
demo.launch(server_port=7861, share=True)
```

---

## Notlar

- Bu model **diffusion tabanlı değil**, transformer tabanlıdır (LLaVA-7B + Mistral)
- SLAKE dataset'i üzerinde fine-tune edilmiştir, radyoloji sorularında iyi performans verir
- Demo amacıyla tasarlanmıştır, production ortamı için ek güvenlik katmanları eklenmelidir
- DiffuVQA (Bert branch) eğitimi tamamlanınca bu chatbot DiffuVQA modeli ile değiştirilebilir
