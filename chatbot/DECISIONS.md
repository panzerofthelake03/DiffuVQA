# Chatbot — Karar ve Hata Kayıtları

## 1. Gradio Sürüm Uyumsuzluğu

**Durum:** Çözüldü  
**Tarih:** 2026-05-14

**Sorun:** Plan `gradio==4.19.2` öngörüyordu. Kurulumda `gradio 6.14.0` geldi. İki kırılma değişikliği:
- `gr.themes.Soft()` → `theme="soft"` string'i
- Gradio 6.0'da `theme` parametresi `gr.Blocks()`'tan `launch()`'a taşındı

**Değişiklik:** [app.py](app.py)
```python
# Önce
with gr.Blocks(title="...", theme=gr.themes.Soft()) as demo:
    ...
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

# Sonra
with gr.Blocks(title="...") as demo:
    ...
demo.launch(server_name="127.0.0.1", server_port=7860, theme="soft")
```

`server_name` ayrıca `0.0.0.0`'dan `127.0.0.1`'e alındı; tarayıcıda `http://localhost:7860` ile açılıyor.

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

## 6. CPU Fallback — CUDA Olmadan Yükleme

**Durum:** Uygulandı  
**Tarih:** 2026-05-14

**Hata:**
```
ValueError: Some modules are dispatched on the CPU or the disk.
Make sure you have enough GPU RAM to fit the quantized model.
```

**Sebep:** `bitsandbytes` 4-bit quantization CUDA gerektiriyor. Yerel makinede GPU (`torch.cuda.is_available() = False`) yok, model CPU'ya yüklenmeye çalışıldı ama `quantization_config` + `device_map="auto"` çakıştı.

**Karar:** `CUDA_AVAILABLE` flag'i ile runtime'da iki yol ayrıldı:
- **GPU varsa:** `quantization_config=bnb_config`, `device_map="auto"`, `torch.float16`
- **GPU yoksa:** `quantization_config` yok, `device_map="cpu"`, `torch.float32`

**Uyarı:** CPU modunda ~14GB RAM gerekir ve her inference birkaç dakika sürer. Gerçek kullanım için GPU (Colab T4) önerilir.

---

## 7. `torch` CPU-Only → CUDA Build

**Durum:** Çözüldü  
**Tarih:** 2026-05-14

**Sorun:** `torch.cuda.is_available()` = False dönüyordu. `nvidia-smi` ile GPU'nun var olduğu doğrulandı (RTX 4060 Laptop, 8GB, driver 581.29 / CUDA 12.x). Kurulu `torch 2.11.0` CPU-only index'ten gelmişti.

**Çözüm:**
```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# → torch 2.6.0+cu124 kuruldu
```

**Doğrulama:**
```
CUDA available: True
Torch version: 2.6.0+cu124
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

Artık `model.py`'deki `CUDA_AVAILABLE = True` path'i devreye giriyor: 4-bit quantization + `device_map="auto"` + `torch.float16`.

---

## Açık Riskler

| Risk | Önem | Durum |
|---|---|---|
| `llava-1.5-7b-hf` tıbbi fine-tune değil | Düşük | Kabul edildi; genel VQA yeterli |
| Symlink uyarısı (HF cache, Windows) | Bilgi | `HF_HUB_DISABLE_SYMLINKS_WARNING=1` ile susturulabilir |
