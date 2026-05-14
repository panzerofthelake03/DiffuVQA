# Kararlar ve Hata Kayıtları

---

## Mimari Kararlar

### A1. Frontend Migrasyonu: Gradio → Next.js + FastAPI
**Tarih:** 2026-05-14  
**Karar:** Gradio arayüzü kaldırılır. Panacea-alpha reposunun Next.js + shadcn/ui arayüzü projeye taşınır. Python backend FastAPI ile sarılır.  
**Gerekçe:** Demo bilgisayarı Windows PC (RTX 4060 Laptop, 8GB VRAM). Gradio arayüzü görsel olarak yetersiz; Panacea'nın arayüzü zaten çalışan, test edilmiş bir tasarım.  
**Sonuç:**
- `chatbot/api.py` → FastAPI, `POST /infer` + `GET /history` + `GET /stats` endpoint'leri ✅ Tamamlandı (2026-05-15)
- `frontend/` → Next.js projesi (Panacea kaynaklı) — bekliyor
- `chatbot/app.py` (Gradio) devre dışı bırakılacak — bekliyor

### A4. FastAPI Endpoint Adı: `/ask` → `/infer`
**Tarih:** 2026-05-15  
**Karar:** Endpoint adı `/ask` yerine `/infer` olarak belirlendi.  
**Gerekçe:** `/infer` model inference'ını daha doğru tanımlıyor; `/ask` sohbet/NLP çağrılarıyla karışabilir. Ayrıca `/history` ve `/stats` endpoint'leri de eklendi — Gradio'nun history/stats butonlarının REST karşılıkları.

---

### A2. Model Değişikliği: `katielink/llava-med-7b-slake-delta` → `llava-hf/llava-1.5-7b-hf`
**Tarih:** 2026-05-14  
**Hata:**
```
ValueError: Unrecognized model in katielink/llava-med-7b-slake-delta.
Should have a `model_type` key in its config.json.
```
**Kök Neden:** `katielink/llava-med-7b-slake-delta` bir delta modeldir. `apply_delta.py` scripti ile base LLaMA-7B ağırlıklarına elle uygulanması gerekiyor; `from_pretrained` doğrudan yükleyemiyor.  
**Karar:** Resmi transformers desteği olan `llava-hf/llava-1.5-7b-hf` kullanıldı.  
**Değişiklik:**
```python
# Önce
from transformers import LlamaTokenizer, AutoModelForCausalLM
MODEL_ID = "katielink/llava-med-7b-slake-delta"

# Sonra
from transformers import LlavaForConditionalGeneration, LlavaProcessor
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
```

---

### A3. CPU / GPU Otomatik Algılama
**Tarih:** 2026-05-14  
**Hata:**
```
ValueError: Some modules are dispatched on the CPU or the disk.
Make sure you have enough GPU RAM to fit the quantized model.
```
**Sebep:** `bitsandbytes` 4-bit quantization CUDA gerektiriyor; Mac'te `quantization_config` + `device_map="auto"` çakışıyor.  
**Karar:** Runtime'da iki yol ayrıldı:
- **GPU varsa:** `quantization_config=bnb_config`, `device_map="auto"`, `torch.float16`
- **GPU yoksa:** `quantization_config` yok, `device_map="cpu"`, `torch.float32`

---

## Çözülen Hatalar

### H1. Gradio Sürüm Uyumsuzluğu
**Tarih:** 2026-05-14  
**Sorun:** Plan `gradio==4.19.2` öngörüyordu, kurulumda `gradio 6.14.0` geldi.  
**Kırılma noktaları:**
- `gr.themes.Soft()` → `theme="soft"` string
- `theme` parametresi `gr.Blocks()`'tan `launch()`'a taşındı
```python
# Önce
with gr.Blocks(title="...", theme=gr.themes.Soft()) as demo:
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

# Sonra
with gr.Blocks(title="...") as demo:
    demo.launch(server_name="127.0.0.1", server_port=7860, theme="soft")
```

---

### H2. Eksik `sentencepiece` Paketi
**Tarih:** 2026-05-14  
**Hata:**
```
ValueError: Couldn't instantiate the backend tokenizer ...
You need to have sentencepiece or tiktoken installed.
```
**Çözüm:** `pip install sentencepiece`

---

### H3. HF Token Güvenliği
**Tarih:** 2026-05-14  
**Sorun:** Token chat penceresine yazıldı, güvensiz hale geldi; hemen iptal edildi.  
**Çözüm:** `.env` dosyası + `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
```
`.gitignore`'a `.env` eklendi.

---

### H4. `torch` CPU-Only → CUDA Build
**Tarih:** 2026-05-14  
**Sorun:** `torch.cuda.is_available()` = False dönüyordu. `nvidia-smi` ile RTX 4060 Laptop (8GB) görünüyordu; pip CPU-only index'ten kurulum yapmıştı.  
**Çözüm:**
```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
**Sonuç:** `torch 2.6.0+cu124`, `CUDA available: True`.

---

### H5. Python 3.14 + requirements.txt Sürüm Uyumsuzluğu
**Tarih:** 2026-05-14  
**Sorun:** `torch==2.1.0` Python 3.14 için dağıtım paketi yok.  
**Çözüm:** `requirements.txt` sabit sürümlerden `>=` formatına güncellendi, `bitsandbytes` Mac için listeden çıkarıldı.

---

## Açık Riskler

| Risk | Önem | Durum |
|---|---|---|
| `llava-1.5-7b-hf` tıbbi fine-tune değil | Düşük | Kabul edildi |
| `bitsandbytes` Windows GPU uyumu | Orta | Demo PC'de test edilmeli |
| Symlink uyarısı (HF cache, Windows) | Bilgi | `HF_HUB_DISABLE_SYMLINKS_WARNING=1` ile susturulabilir |
