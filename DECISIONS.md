# Decisions & Concerns

Proje boyunca alınan teknik kararlar ve dikkat edilmesi gereken noktalar.

---

## 2026-05-06

### [KARAR] `efficient_sample.py` silindi
Dosya gerçek bir implementasyon içermiyordu, sadece placeholder koduydu. Tüm sampling işlemleri `sample_vqa_GPU.py` üzerinden yürütülecek.

---

### [KARAR] `sample_vqa_GPU.py` — Pure noise başlatma
**Değişiklik:** `x_start` artık cevap embeddinglerinden değil, pure random noise + frozen image-fusion features'tan oluşturuluyor.

**Neden:** Model training'de cevap embeddinglerini hem input (x_start) hem de loss hedefi olarak alıyordu. Bu, modelin cevap üretmeyi değil sadece gürültüden temizlemeyi öğrenmesine yol açıyordu. Test setinde başarılı görünüyor ama bilinmeyen sorular için üretim yapamıyordu.

**Etki:** Mevcut checkpoint bu beklentiyle eğitilmedi. Test sonuçlarının düşmesi mümkün — yeniden eğitim gerekebilir.

---

### [KARAR] `sample_vqa_GPU.py` — Mask yapısı yeniden kuruldu
**Değişiklik:** Mask artık `[zeros(fuse_len) + ones(seq_len)]` şeklinde oluşturuluyor. Eskiden Q+A mask vektörü pad/truncate ile zorla hizalanıyordu.

**Neden:** Eski mask'ta image-fusion token sayısı gözetilmiyordu. `fuse_len` dinamik olarak hesaplanmadığı için mask ile sequence arasında semantik uyumsuzluk vardı. Image-fusion tokenları `mask=0` (dondurulmuş), cevap tokenları `mask=1` (diffuse edilecek) olmalı.

---

### [KARAR] `gaussian_diffusion.py` — Eğitimde mask hizalaması
**Değişiklik:** `mask.repeat()` yerine başa `zeros(fuse_token_len)` prepend ediyor.

**Neden:** Repeat stratejisi, soru/cevap mask değerlerini image-fusion tokenlarına yanlış eşliyordu. Doğru davranış: image-fusion tokenları her zaman `mask=0` (diffuse edilmez), cevap tokenları orijinal mask değerlerini korur.

**Endişe:** Eğitim sırasında model bu yeni mask yapısını görmediyse (eski repeat stratejisiyle eğitildiyse), bu değişiklik inference'ta davranış farkı yaratır. Yeniden eğitim bu endişeyi ortadan kaldırır.

---

---

### [KARAR] Proje dosya yapısı yeniden düzenlendi
**Yeni yapı:**
- `checkpoints/` — eski `config/` klasörü + model ağırlıkları
- `outputs/` — sampling sonucu üretilen `.jsonl` dosyaları
- `eval/` — tüm değerlendirme scriptleri (`eval_DiffuVQA.py`, `enhanced_eval_metrics.py`, `compare_samples.py`, `prepare_eval.py`)
- `docs/` — proje dokümanları, görsel, argüman açıklamaları
- `notebooks/` — Colab notebook'ları
- Root'ta sadece giriş noktaları: `train.py`, `sample_vqa_GPU.py`, `requirements.txt`, `README.md`

**Silinen kopyalar:** `basic_utils.py`, `train_util.py`, `excel_export_module.py`, `test_enhanced_metrics.py` (root'taki) — canonical versiyonlar `shared/` altında.

**Import düzeltmesi:** `eval/eval_DiffuVQA.py` ve `eval/test_enhanced_metrics.py` dosyalarına `sys.path` ile repo kökü eklendi.

**Endişe:** Diğer branch'lerde `eval_DiffuVQA.py` root'ta bekleniyorsa bu değişiklik import hatası verir. Her branch'e aynı yapı uygulanmalı.

---

---

### [KARAR] BUG 1 — `sample_vqa_GPU.py` sample slicing düzeltildi
`a_shape = sample.size(1) // 2` ifadesi `fuse_len` ile değiştirildi. Eski kod `seq_len == fuse_len` olduğu sürece şans eseri doğru çalışıyordu; farklı bir encoder veya seq_len kullanıldığında cevap tokenları yanlış slicelanırdı.

---

### [KARAR] BUG 2 — `bert_model.py` `BertLayer` import edildi
`BertEncoder` içinde kullanılan `BertLayer` sınıfı tanımlanmamıştı. `transformers.models.bert.modeling_bert`'ten import edildi. `init_pretrained='no'` dışında pretrained weight'ler yüklendiğinde bu kod aktif olmadığı için şimdiye kadar fark edilmemişti.

---

### [KARAR] BUG 3 — `vqa_model.py` feature_fusion çıktısı sabitlendi
`feature_fusion` çıktısı artık her zaman `seq_len` (= `question_feats.size(1)`) uzunluğunda. `image_feats` ve `f4` bu uzunluğa pool+expand ile hizalanıyor. Assert eklendi — uyumsuzluk sessizce geçmez. Eskiden image patch sayısına göre değişken uzunlukta çıktı üretiliyordu.

---

### [KARAR] BUG 4 — `vqa_model.py` hardcoded `145` kaldırıldı
Vision encoder init sırasında dummy forward pass ile gerçek kanal boyutu ölçülüyor. CLIP ViT-B/32 dışında encoder kullanılabilir hale geldi. Mevcut checkpoint'lerin `image_MLP` ağırlıkları bu değişiklikle uyumlu — eğer encoder değiştirilmezse `145` zaten ölçülecek.

---

### [KARAR] BUG 5 — `gaussian_diffusion.py` debug print'ler temizlendi
`x_start` tanımlanmadan önce referans alan try/except ile örtülmüş debug print kaldırıldı. `DEBUG_SHAPES` env var'a bağlı ikinci debug bloğu da kaldırıldı — ihtiyaç halinde `DVQA_DEBUG` ortam değişkeni feature_fusion'da hâlâ mevcut.

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — PubMedBERT test konfigürasyonu
**Değişiklik:** Notebook config hücresi PubMedBERT branch'i ve SLAKE dataseti için güncellendi.

**Parametreler:**
- `BRANCH = "PubMedBERT"`, `MODEL_NAME = "pubmedbert"`, `DATASET = "SLAKE"`
- `BATCH_SIZE = 4`, `LEARNING_STEPS = 500`, `DIFFUSION_STEPS = 50`, `SEQ_LEN = 32`, `SAMPLE_STEP = DIFFUSION_STEPS`
- `SAVE_INTERVAL = 100`, `LOG_INTERVAL = 25`

**Neden:** İlk Colab testi için hızlı iterasyon — düşük step sayısı pipeline'ın hata vermeden çalıştığını doğrular. SEQ_LEN=32, SLAKE'in kısa cevap yapısına uygun. Gerçek eğitim için LEARNING_STEPS 4000+, DIFFUSION_STEPS 100-2000 olmalı.

**Ayrıca:** Eval hücresi `python eval_DiffuVQA.py` → `python eval/eval_DiffuVQA.py` olarak düzeltildi (dosya yapısı yeniden düzenlemesinden kaynaklı). Hardcoded dosya adı kaldırılıp checkpoint'ten dinamik olarak türetilen `CURRENT_SAMPLE_FILENAME` kullanılmaya başlandı.

---

### [KARAR] `Pooler` sınıfı silindi
`diffuvqa/vqa_model.py`'daki `Pooler` sınıfı (eski lines 27-37) hiçbir yerde kullanılmıyordu. `CVAE` aktif olarak `feature_fusion` içinde kullanılırken `Pooler` dead code olarak kalmıştı. Kalabalık yaratmaması için silindi.

---

### [KARAR] Validation loop düzeltildi — BUG 6
**Sorun 1:** `next(self.eval_data)` doğrudan çağrılıyordu. Validation dataseti tükenince `StopIteration` ile eğitim çöküyordu. `eval_iter` ayrı tutulup `StopIteration` yakalanarak yeniden başlatıldı.

**Sorun 2:** `step=0`'da (`0 % eval_interval == 0`) hiçbir eğitim adımı atılmadan validation çalışıyordu. `self.step > 0` koşulu eklendi.

**Sorun 3:** `forward_only` içinde `del cond['image_name']` orijinal batch dict'ini mutate ediyordu. `micro_cond` dict comprehension sırasında `image_name` key'i filtrelenerek kopyalanır hale getirildi.

---

### [KARAR] `sample_vqa_GPU.py` — Bounded slice ile answer_len kontrolü
**Değişiklik:** Answer segment boyutu artık `args.seq_len` sabitinden değil, `input_ids_a.size(1)` gerçek uzunluğundan alınıyor. `ans_noise`, `ans_mask` ve son slice hepsi `answer_len` ile tanımlanıyor: `sample[:, fuse_len:fuse_len+answer_len, :]`

**Neden:** `fuse_len:` açık uç slice gelecekte segmente ek token eklenmesi durumunda fazla pozisyonu decode'a sokar. Bounded slice `fuse_len:fuse_len+answer_len` her zaman tam olarak üretilen cevap segmentini alır, dışarıya taşmaz.

**Chatbot notu:** Açık uçlu üretim için `answer_len` tanımını değiştirmek gerekir (GT uzunluğu yerine `max_new_tokens` gibi bir üretim limiti). Slice yapısı değişmez, sadece `answer_len`'in kaynağı değişir.

---

### [KARAR] Data Leakage tamamen kapatıldı — `gaussian_diffusion.py`
**Değişiklik:** `training_losses` içinde üç kritik düzeltme yapıldı:

1. `x_start` artık `ans_emb + std*noise` (cevap embedding'i) değil, **pure `th.randn`** — inference ile tam tutarlı.
2. `target` artık `cond_x_start` (fuse+ans tümü) değil, sadece **`ans_emb` (answer segmenti)**. Fuse tokenları loss'a dahil edilmiyor.
3. `t0_loss`, `tT_loss`, `decoder_nll` tümü artık `x_start_mean` (ans_emb) üzerinden hesaplanıyor — `x_start` (pure noise) üzerinden değil.

**Neden:** Model eğitimde hem input hem hedef olarak cevap embeddingini görüyordu. Sadece gürültü temizlemeyi öğreniyor, sıfırdan üretim yapamıyordu. Bu düzeltmeyle training/inference davranışı tamamen hizalandı.

**Etki:** Mevcut checkpoint'ler bu değişiklikle uyumsuz — sıfırdan yeniden eğitim gerekiyor.
