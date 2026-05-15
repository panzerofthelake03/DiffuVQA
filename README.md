# DiffGenMed-VQA

Mezuniyet projesi — tıbbi görüntüler üzerinde diffusion tabanlı Visual Question Answering.

## Proje Yapısı

```
PubMedBERT/
├── chatbot/          # LLaVA-1.5 tabanlı Gradio chatbot
├── shared/           # Ortak yardımcı modüller ve Colab notebook
├── samples/          # Eğitim denemelerinden üretilen örnek çıktılar
├── reports/          # Değerlendirme raporları
└── docs/             # Tüm dökümanlar
```

## Kurulum

```bash
# 1. Sanal ortam oluştur ve aktifleştir
python -m venv venv
venv\Scripts\activate             # Windows
# source venv/bin/activate        # Mac/Linux

# 2. Python bağımlılıklarını yükle
pip install -r chatbot/requirements.txt

# 3. HuggingFace token ayarla (.env dosyası repo kökünde)
# .env içeriği:
# HF_TOKEN=hf_...
```

> **Not:** CPU'da ~14 GB RAM gerekir; CUDA GPU varsa otomatik 4-bit quantization devreye girer (RTX 4060 ile test edildi).

---

## Servisleri Çalıştırma

Her servis ayrı bir terminal penceresinde çalıştırılır.

### 1 — FastAPI (LLaVA Backend)

Next.js frontend ile konuşan REST API. Model ilk istekte yüklenir (~1-2 dk).

```bash
# Repo kökünden çalıştır
uvicorn chatbot.api:app --host 127.0.0.1 --port 8000 --reload
# → http://127.0.0.1:8000
# → http://127.0.0.1:8000/docs  (Swagger UI)
```

Endpoints:
| Method | Path | Açıklama |
|--------|------|----------|
| `POST` | `/infer` | Görüntü + soru gönder, cevap al |
| `GET`  | `/history` | Son N soruyu listele (`?limit=20`) |
| `GET`  | `/stats` | Toplam soru sayısı |

### 2 — Next.js Frontend

```bash
cd frontend
npm install          # ilk kurulumda
npm run dev
# → http://localhost:3000
```

Frontend, `/api/chat` route'u üzerinden `http://127.0.0.1:8000/infer` adresine istek atar. FastAPI'nin çalışıyor olması gerekir.

### 3 — Gradio Arayüzü (Bağımsız, isteğe bağlı)

Next.js gerektirmeden modeli doğrudan test etmek için kullanılır.

```bash
# Repo kökünden çalıştır
python -m chatbot.app
# → http://127.0.0.1:7860
```

## Dökümanlar

| Dosya | İçerik |
|---|---|
| [docs/todo.md](docs/todo.md) | Yapılacaklar, aktif plan ve ilerleme |
| [docs/decisions.md](docs/decisions.md) | Mimari kararlar ve çözülen hatalar |
| [docs/sampling-mask-issue.md](docs/sampling-mask-issue.md) | Sampling/inference mask leakage problemi |

## Aktif Branch

`ChatBotPipeline` — LLaVA-1.5 Gradio chatbot geliştirmesi
