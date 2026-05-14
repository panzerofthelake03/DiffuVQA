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

## 4. `AutoTokenizer` → `LlamaTokenizer` (Geçici, iptal edildi)

**Durum:** Geçersiz — #5 ile değiştirildi  
**Tarih:** 2026-05-14

**Hata:**
```
ValueError: Couldn't instantiate the backend tokenizer ...
You need to have sentencepiece or tiktoken installed.
```

**Sebep:** `transformers 5.5.4`, `use_fast=False` parametresine rağmen fast tokenizer path'ine giriyor. `LlamaTokenizer` geçici çözüm olarak denendi ancak #5'teki hata nedeniyle zaten gereksiz kaldı.

---

## 5. Model Değişikliği — `katielink/llava-med-7b-slake-delta` → `llava-hf/llava-1.5-7b-hf`

**Durum:** Uygulandı  
**Tarih:** 2026-05-14

**Hata:**
```
ValueError: Unrecognized model in katielink/llava-med-7b-slake-delta.
Should have a `model_type` key in its config.json.
```

**Kök Neden:** `katielink/llava-med-7b-slake-delta` gerçek bir **delta model**'dir. Bu model tek başına yüklenemez — base LLaMA-7B ağırlıklarına LLaVA'nın `apply_delta.py` scripti çalıştırılarak elle uygulanması gerekiyor. `config.json`'da `model_type` tanımı yok, `from_pretrained` bunu algılayamıyor.

**Karar:** Model, resmi transformers 5.x desteği olan `llava-hf/llava-1.5-7b-hf` ile değiştirildi. Bu model:
- Doğrudan `from_pretrained` ile yüklenir
- `LlavaForConditionalGeneration` + `LlavaProcessor` kullanır (görüntüyü de prompt'a dahil eder)
- `transformers 5.x` ile tam uyumlu
- Genel VQA görevlerinde güçlü; tıbbi görüntülerde de kullanılabilir

**Değişiklik:** [model.py](model.py)
```python
# Önce
from transformers import LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
MODEL_ID = "katielink/llava-med-7b-slake-delta"
tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)

# Sonra
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(MODEL_ID, token=hf_token)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, ...)
```

`ask()` fonksiyonu da güncellendi: görüntü artık `processor` üzerinden chat template ile prompt'a dahil ediliyor, yalnızca yeni üretilen token'lar decode ediliyor.

---

## Açık Riskler

| Risk | Önem | Durum |
|---|---|---|
| `bitsandbytes` Windows CUDA desteği | Orta | Test edilmedi |
| CUDA yok — model GPU olmadan çalışmaz | Yüksek | Bekliyor (Colab önerilir) |
| `llava-1.5-7b-hf` tıbbi fine-tune değil | Düşük | Kabul edildi; genel VQA yeterli |
