# Chatbot — Karar ve Hata Kayıtları

## 1. Gradio Sürüm Uyumsuzluğu

**Durum:** Çözüldü  
**Tarih:** 2026-05-14

**Sorun:** Plan `gradio==4.19.2` öngörüyordu. Kurulumda `gradio 6.14.0` geldi. `gr.themes.Soft()` objesi 6.x'te `theme="soft"` string'ine dönüştü.

**Değişiklik:** [app.py](app.py)
```python
# Önce
with gr.Blocks(title="...", theme=gr.themes.Soft()) as demo:

# Sonra
with gr.Blocks(title="...", theme="soft") as demo:
```

---

## 2. Eksik `sentencepiece` Paketi

**Durum:** Çözüldü  
**Tarih:** 2026-05-14

**Hata:**
```
ValueError: Couldn't instantiate the backend tokenizer ...
You need to have sentencepiece or tiktoken installed.
```

**Sebep:** LLaMA tabanlı modelin tokenizer'ı `sentencepiece` kütüphanesine bağımlı; `.venv`'de yoktu.

**Çözüm:**
```powershell
pip install sentencepiece
# → sentencepiece 0.2.1 kuruldu
```

---

## 3. HF Token Güvenliği

**Durum:** Çözüldü  
**Tarih:** 2026-05-14

**Sorun:** Token chat penceresine yazıldı, güvensiz hale geldi. Hemen iptal edildi.

**Çözüm:** `.env` dosyası + `python-dotenv` kurulumu yapıldı. Token artık kod içinde veya ortam değişkeninde açık değil.

**Değişiklik:** [model.py](model.py)
```python
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
# from_pretrained(..., token=hf_token)
```

`.gitignore`'a eklendi:
```
.env
```

---

## 4. `AutoTokenizer` → `LlamaTokenizer`

**Durum:** Test Aşamasında  
**Tarih:** 2026-05-14

**Hata:**
```
ValueError: Couldn't instantiate the backend tokenizer from one of:
(1) a `tokenizers` library serialization file,
(2) a slow tokenizer instance to convert or
(3) an equivalent slow tokenizer class to instantiate and convert.
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.
```

**Sebep:** `transformers 5.5.4`, `use_fast=False` parametresine rağmen bu eski LLaMA tabanlı modelde `AutoTokenizer` fast tokenizer path'ine giriyor. `sentencepiece` kurulu olmasına rağmen hata devam etti.

**Karar:** `AutoTokenizer` yerine `LlamaTokenizer` (slow tokenizer) doğrudan kullanılıyor.

**Değişiklik:** [model.py](model.py)
```python
# Önce
from transformers import AutoTokenizer, ...
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, token=hf_token)

# Sonra
from transformers import LlamaTokenizer, ...
tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, token=hf_token)
```

**Açık Risk:** `katielink/llava-med-7b-slake-delta` bir **delta model**'dir — base LLaMA-7B ağırlıklarına delta apply edilerek elde edilmesi gerekir. Doğrudan `from_pretrained` ile yüklenip yüklenmeyeceği henüz doğrulanmadı. Sorun devam ederse model değişikliği gerekebilir.

---

## Açık Riskler

| Risk | Önem | Durum |
|---|---|---|
| Delta model doğrudan yüklenemiyor olabilir | Yüksek | Araştırılıyor |
| `transformers 5.5.4` ↔ model (2023) API uyumsuzluğu | Orta | Gözlemleniyor |
| `bitsandbytes` Windows CUDA desteği | Orta | Test edilmedi |
| CUDA yok — model GPU olmadan çalışmaz | Yüksek | Bekliyor |
