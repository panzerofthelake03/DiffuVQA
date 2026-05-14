# Yapılacaklar / İlerleme

---

## Sonraki Adımlar (Aktif Plan)

- [ ] **Frontend Migrasyonu — Panacea → Next.js + FastAPI**
  - Hedef: Gradio kaldırılır, panacea-alpha reposunun Next.js arayüzü buraya taşınır
  - Stack: Next.js (TypeScript) + shadcn/ui + Tailwind + FastAPI (Python backend)
  - Demo ortamı: Windows PC, RTX 4060 Laptop 8GB VRAM, CUDA 4-bit quantization
  - Adımlar:
    - [ ] `chatbot/api.py` — FastAPI ile `/ask` endpoint'i yaz (model.py'yi sarar)
    - [ ] `chatbot/requirements.txt`'e `fastapi`, `uvicorn`, `python-multipart` ekle
    - [ ] `frontend/` klasörü oluştur, Next.js projesi kur
    - [ ] Panacea'nın `page.tsx`, `globals.css`, `layout.tsx` ve shadcn/ui bileşenlerini taşı
    - [ ] Gemini API çağrısını FastAPI `/ask` endpoint'ine yönlendir
    - [ ] `app.py` (Gradio) kaldır ya da devre dışı bırak
    - [ ] Windows demo PC'de uçtan uca test et

- [ ] **Windows Demo PC Kurulumu**
  - CUDA 12.x sürücüsü kurulu olmalı (RTX 4060 Laptop, driver 581.29)
  - PyTorch CUDA build:
    ```powershell
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```
  - `.env` dosyasına `HF_TOKEN` ekle

---

## Tamamlananlar

- [x] **Branch** — `ChatBotPipeline` branch'i açıldı ve remote'a push edildi.

- [x] **Klasör Yapısı** — `chatbot/` dizini oluşturuldu:
  - `app.py` — Gradio arayüzü (ileride kaldırılacak)
  - `model.py` — LLaVA-1.5-7b-hf model yükleme & inference
  - `database.py` — SQLite geçmiş yönetimi
  - `requirements.txt` — Bağımlılıklar
  - `__init__.py` — Python paketi

- [x] **Ortam Kurulumu** — Paketler kuruldu:
  - `gradio 6.14.0`, `sqlalchemy 2.0.49`, `peft 0.19.1`, `accelerate 1.13.0`
  - `bitsandbytes 0.49.2`, `torch`, `transformers 5.5.4`, `Pillow 12.2.0`

- [x] **model.py** — 4-bit BitsAndBytes quantization; CPU/GPU otomatik algılama eklendi.

- [x] **database.py** — SQLAlchemy + SQLite; `save`, `get_recent`, `get_all_count`.

- [x] **app.py** — Gradio Blocks arayüzü, Gradio 6.x uyumu sağlandı.

- [x] **Proje düzeni** — Tüm MD dosyaları `docs/` altında toplandı, kök `README.md` yazıldı.

- [x] **requirements.txt** — Python 3.14 uyumlu güncel sürümlere güncellendi, `bitsandbytes` Mac'te kaldırıldı.

---

## DiffuVQA Ana Proje — Aşamalar

### Aşama 1: Model Eğitimi

- [ ] Eğitim ortamı kurulumu (hyperparameter, dataloader, GPU)
  - SLAKE, Kvasir-VQA, Med-VQA-2019 datasetleri
  - `diff_steps=2000`, `lr=0.00001`, `hidden_dim=64`
- [ ] 5.000 adım eğitim çalıştırması
- [ ] Checkpoint export ve doğrulama

### Aşama 2: Inference

- [ ] Checkpoint'ten model yükleme scripti
- [ ] Görüntü + soru ön işleme pipeline'ı (eğitimle aynı format)
- [ ] Diffusion sampling yoluyla cevap üretimi
- [ ] Çıktı temizleme ve loglama

### Aşama 3: UI

- [ ] Next.js arayüzü (Panacea'dan uyarlama — Aktif Plan ile birleşiyor)

### Aşama 4: Entegrasyon

- [ ] FastAPI `/ask` endpoint'i (Aktif Plan ile birleşiyor)
- [ ] Frontend ↔ Backend REST bağlantısı
- [ ] Uçtan uca test

---

## Bilinen Riskler

| Konu | Önem | Durum |
|---|---|---|
| `bitsandbytes` Windows GPU desteği | Orta | GPU makinede test edilmeli |
| `llava-1.5-7b-hf` tıbbi fine-tune değil | Düşük | Kabul edildi; genel VQA yeterli |
| Model boyutu ilk indirme (~15-20 GB) | Bilgi | Stabil internet gerekli |
| Symlink uyarısı (HF cache, Windows) | Bilgi | `HF_HUB_DISABLE_SYMLINKS_WARNING=1` ile susturulabilir |
