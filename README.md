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

## Hızlı Başlangıç (Chatbot)

```bash
# 1. Sanal ortam oluştur ve aktifleştir
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 2. Bağımlılıkları yükle
pip install -r chatbot/requirements.txt

# 3. HuggingFace token ayarla (model gated ise)
echo "HF_TOKEN=hf_..." > .env

# 4. Uygulamayı başlat
python -m chatbot.app
# → http://localhost:7860
```

> **Not:** CPU'da ~14 GB RAM gerekir, GPU (CUDA) varsa otomatik olarak 4-bit quantization devreye girer.

## Dökümanlar

| Dosya | İçerik |
|---|---|
| [docs/todo.md](docs/todo.md) | Yapılacaklar, aktif plan ve ilerleme |
| [docs/decisions.md](docs/decisions.md) | Mimari kararlar ve çözülen hatalar |
| [docs/sampling-mask-issue.md](docs/sampling-mask-issue.md) | Sampling/inference mask leakage problemi |

## Aktif Branch

`ChatBotPipeline` — LLaVA-1.5 Gradio chatbot geliştirmesi
