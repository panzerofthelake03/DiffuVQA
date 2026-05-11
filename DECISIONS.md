# Decisions & Concerns

Proje boyunca alınan teknik kararlar ve dikkat edilmesi gereken noktalar.

---

## 2026-05-11

**Durum notu:** Bu başlıktaki değişiklikler Bio-Bert branch'inde kodda uygulanmış durumdadır.

### [KARAR] sample_vqa_GPU.py — Orta seviye token seçimi iyileştirmesi (Top-k rerank + minimum cevap uzunluğu)
**Sorun:** Üretilen cevaplarda boş string oranı yüksekti. Bunun ana nedeni top-1 greedy seçim ve ilk tokenlarda SEP/PAD gelince erken kesme davranışıydı.

**Uygulanan değişiklikler:**
- Decode tarafına yeni argümanlar eklendi:
    - `decode_top_k` (default: 5)
    - `min_answer_tokens` (default: 2)
    - `short_answer_penalty` (default: 1.0)
- `topk(logits, k=1)` yerine `topk(logits, k=decode_top_k)` ile adaylar toplandı.
- Her örnek için k adet aday cevap oluşturulup rerank edildi.
- `min_answer_tokens` pozisyonundan önce SEP/PAD adayları bastırıldı (erken boş cevap engeli).
- Confidence filtresi `min_answer_tokens` sonrasında uygulandı; düşük güven tokenlar PAD yapıldı.
- Aday skorlamasına kısa cevap cezası eklendi:
    - efektif uzunluk `< min_answer_tokens` ise ceza uygulanıyor.
- Decode kesme kuralı güncellendi:
    - SEP/PAD kesmesi sadece `min_answer_tokens` sonrasında geçerli.

**Neden bu çözüm seçildi:**
- Yeniden eğitim gerektirmeden inference kalitesini iyileştiren en dengeli yol.
- Küçük değişimden daha güçlü, LLM-rerank çözümünden daha hızlı ve operasyonel olarak ucuz.

**Beklenen etki:**
- Boş cevap oranında belirgin düşüş.
- Aşırı kısa/tek token cevaplarda azalma.
- Ortalama cevap tutarlılığında artış (özellikle yes/no ve kısa span sorularında).

**Risk/Trade-off:**
- `decode_top_k` arttıkça inference maliyeti (CPU-side aday değerlendirme) artar.
- `short_answer_penalty` fazla yüksek olursa bazı doğru kısa cevaplar (örn. "no", "2") gereksiz cezalanabilir.

**İzleme önerisi:**
- Aynı checkpoint için şu sweep önerilir:
    - `(decode_top_k, min_answer_tokens, confidence_threshold, short_answer_penalty)` =
    - `(5, 2, 0.25, 1.0)`, `(5, 2, 0.20, 1.0)`, `(7, 2, 0.20, 0.8)`
- Rapor metrikleri:
    - empty-answer rate
    - exact match
    - contains-match

---

## 2026-05-10

### [BUGFIX] sample_vqa_GPU.py — `decode_token()` tensor type hatasının düzeltilmesi
### [BUGFIX v2] shared/basic_utils.py — `decode_token()` 0D tensor hatasının köklü çözümü
**Hata:** `TypeError: object of type 'int' has no len()` satır 87'de (seq.squeeze(-1).tolist())
- seq_len=1 ise, seq_cut shape [1] → squeeze(-1) → shape [] (0D tensor)
- 0D tensor.tolist() → integer döner
- `while len(seq)>0` → INTEGER'a len() uygulanamaz → ERROR

**Root cause:** squeeze(-1) yapıldığında 0D tensor'dan integer çıkabiliyor

**Çözüm:** squeeze(-1) yerine flatten() kullan (her zaman liste döner):

    seq_list = seq.flatten().tolist()
    while len(seq_list)>0 and seq_list[-1] == self.pad_token_id:
        seq_list.pop()

**Beklenen etki:** seq_len=1 durumunda da sampling hatasız çalışır.

### [BUGFIX] sample_vqa_GPU.py — `decode_token()` tensor type hatasının düzeltilmesi
**Hata:** `TypeError: object of type 'int' has no len()` satır 363'te
- `seq_cut = seq_ids[:first_stop]` PyTorch tensor kalıyordu
- `tokenizer.decode_token(seq_cut)` fonksiyonu listeyi bekliyor ama tensor alıyordu
- `decode_token()` içinde `while len(seq)>0` çağrısı tensor'a uygulanıyordu

**Çözüm:** Üç yerde `.tolist()` eklendi:
1. Satır ~363: `seq_cut = seq_ids[:first_stop].tolist()`
2. Satır ~369: `tokenizer.decode_token(seq[:args.seq_len].tolist())`
3. Satır ~370: `tokenizer.decode_token(seq[args.seq_len:].tolist())`

**Beklenen etki:** Sampling hataları çözülür, JSONL output düzgün oluşturulur.

---

## 2026-05-10 (Önceki)

### [KARAR] sample_vqa_GPU.py — SEP/PAD kesme + confidence threshold + MBR
Üretilen sequence post-processing pipeline'ı eklendi:
- Seçenek 1: İlk [SEP]/[PAD] tokenına kadar kes
- Seçenek 3: SEP/PAD yoksa trailing confidence < 0.3 tokenları sil
- Seçenek 4: --num_samples N ile MBR decoding; N=1 mevcut davranışı korur, N>1 kaliteyi artırır (offline eval için N=3-5 önerilir)

Neden: JSONL analizi — doğru cevap %29 örnekte üretilmiş ama 15-20 token gürültü arasına gömülmüş. Exact match 0 iken contains %29.

### [BEKLEYEN KARAR] Seçenek 2 — Training'de padding mask ile loss masking
Henüz uygulanmadı. Sonraki training run başlamadan önce gaussian_diffusion.py training_losses içinde MSE loss'u gerçek cevap uzunluğuyla maskele:

    ans_len_mask = (token_ids != pad_id).float()
    terms["mse"] = mean_flat(
        (ans_emb - ans_output) ** 2 * ans_len_mask.unsqueeze(-1)
    )

Neden uygulanmadı: Mevcut run devam ediyor, yeniden eğitim gerektirir.
Beklenen etki: Model kısa cevaplar için SEP/PAD üretmeyi öğrenir, post-processing bağımlılığı azalır.

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

### [KARAR] Data Leakage kapatıldı — `gaussian_diffusion.py` (revize)
**Değişiklik:** `training_losses` içinde iki adımda düzeltme yapıldı:

**Adım 1 (önceki — geri alındı):** `x_start = pure randn`. Inference ile tutarlıydı ama diffusion'un öğrenme hedefini answer manifoldundan kopardı. `q_sample`'ın `add_information` dalı `f` üzerinden sinyal taşısa da `x_t → x_0` ters süreci anlamsız bir random latentten başka bir random latente gitmeyi öğrenebilirdi. Token-space ile latent-space hizalaması zayıflardı.

**Adım 2 (mevcut — doğru):**
- `x_start = _get_x_start(ans_emb, std)` — cevap embedding + küçük gürültü. Diffusion answer manifolduna bağlı kalır.
- `f = cond_x_start_mean` — clean `[fuse | ans_emb]`. `q_sample` içindeki `add_information` dalı temiz semantik sinyalle karışım yapar.
- MSE hedefi sadece answer segmenti: `mean_flat((ans_emb - ans_output)**2)`. Fuse tokenları loss'tan çıkarıldı.
- `t0_loss`, `tT_loss`, `decoder_nll` tümü `x_start_mean` (ans_emb) üzerinden.

**Neden doğru:** Leakage'ı yaratan şey `x_start`'ın random olmaması değil, `target = cond_x_start` (fuse+ans tümü) olmasıydı. Model hem fuse hem answer'ı yeniden üretmeyi öğreniyordu. Şimdi sadece answer segmentini öğreniyor; fuse tokenlar conditioning olarak kalıyor.

**Etki:** Mevcut checkpoint'ler uyumsuz — sıfırdan yeniden eğitim gerekiyor.

---

### [KARAR] `logger.py` — `dumpkvs()` çıktısı geri açıldı
**Değişiklik:** `dumpkvs()` içindeki `for fmt in self.output_formats: fmt.writekvs(d)` bloğu "LISA" yorumu adıyla yorum satırına alınmıştı. Bu yüzden logger her adımda değerleri biriktiriyor ama ne stdout'a ne dosyaya yazıyordu. Blok geri açıldı.

**Neden:** Eğitim boyunca terminal'de hiç loss görünmüyordu. `log_interval` adımda bir dumpkvs çağrılıyor ama çıktı tamamen susturulmuştu.

---

### [KARAR] `train_util.py` — tqdm postfix loss gösterimi düzeltildi
**Değişiklik:** `pbar.set_postfix` içindeki `logger.name2val['loss'].mean()` ifadesi `float(logger.get_current().name2val['loss'])` ile değiştirildi.

**Neden:** `name2val` değerleri `float` türünde — `.mean()` metodu yoktu ve sessizce `AttributeError` veriyordu. Tqdm progress bar'ında anlık loss görünmüyordu.

---

### [KARAR] Notebook sampling hücresi — `ls` path separator düzeltildi
**Değişiklik:** `!ls -lh {SAMPLE_FOLDER}*.jsonl` → `!ls -lh {SAMPLE_FOLDER}/*.jsonl`

**Neden:** Klasör adıyla glob pattern arasında `/` eksikti. Shell bunu `{klasör_adı}*.jsonl` şeklinde aynı dizinde arıyordu ve `No such file or directory` hatası veriyordu.

---

### [KARAR] Notebook — `compare_image_black_vectors` hücresi devre dışı bırakıldı
**Değişiklik:** Hücre içeriği `scripts.compare_image_black_vectors` modülünü çağırmak yerine bilgilendirici bir mesaj yazdırıyor.

**Neden:** Bu script repoda tanımlı değil — eski bir referans. `ModuleNotFoundError` veriyordu. Asıl eval `eval/eval_DiffuVQA.py` üzerinden yapılıyor.

---

### [KARAR] Notebook — 50k step eğitim konfigürasyonu
**Değişiklik:**
- `LEARNING_STEPS`: 6000 → 50000
- `DIFFUSION_STEPS`: 200 → 2000 (orijinal DiffuSeq konfigürasyonu)
- `SAMPLE_STEP`: 50 → 200 (DDIM, 2000 adımı 10x hızlandırır)
- `SAVE_INTERVAL`: 1000 → 5000 (50k'da 10 checkpoint)
- `LOG_INTERVAL`: 50 → 100

**Neden:** 6000 adım ile avg_nn_l2=416 — model embedding manifoldunu öğrenemedi. SLAKE için minimum 30k-50k adım gerekiyor. A100'de ~8-10 saat.

---

### [KARAR] Notebook eval hücresi — BERTScore LOAD REPORT susturuldu
**Değişiklik:** `bert_score_fn` çağrısı `warnings.catch_warnings()` + `logging.disable(logging.WARNING)` bloğuna alındı.

**Neden:** `bert_score` kütüphanesi `roberta-large` yüklerken transformers'ın ağırlık uyuşmazlığı loglarını (`lm_head.*` UNEXPECTED, `pooler.*` MISSING) her eval'da terminal'e yazdırıyordu. Bu bir hata değil — BERTScore sadece encoder katmanlarını kullanır. Hangi model seçilirse seçilsin LOAD REPORT çıkar; `model_type` değiştirmek çözmez, logging susturmak gerekir.

---

### [KARAR] `train.py` — Logger stdout tablosu kaldırıldı
**Değişiklik:** `logger.configure()` → `logger.configure(format_strs=["log", "csv"])`.

**Neden:** `log_interval` adımda bir basılan grad_norm/loss/mse/nll tablosu terminal çıktısını kalabalıklaştırıyordu. tqdm progress bar'ı zaten `loss=X.XXXX` gösteriyor. Metrikler `log.txt` ve `progress.csv` dosyalarına yazılmaya devam ediyor.

---

### [KARAR] BUG 11 — `train_util.py` `forward_backward` microbatch gradient accumulation düzeltildi
**Değişiklik:** `backward()`, `schedule_sampler.update_with_local_losses()` ve `log_loss_dict()` çağrıları microbatch döngüsü **dışına** taşınmıştı — sadece son microbatch'in gradyanı geriye yayılıyordu. Tüm bu çağrılar döngü **içine** alındı. Loss `/ num_microbatches` ile ölçeklendi, böylece gradient accumulation matematiksel olarak doğru.

**Neden:** batch=64, microbatch=16 iken 4 microbatch yerine 1 backward çalışıyordu. Eğitim hız olarak yapay biçimde hızlı görünüyordu ama öğrenme kalitesi 4x düşmüştü.

**Etki:** Mevcut checkpoint'ler bu hatalı davranışla eğitildi. Yeniden eğitim önerilir.

---

### [KARAR] BUG 12 — `vqa_datasets.py` DataLoader iterator hatası düzeltildi
**Değişiklik:** `next(data)` → `next(iter(data))`.

**Neden:** `DataLoader` nesnesi iterator değil, iterable. Doğrudan `next()` geçmek `TypeError: 'DataLoader' object is not an iterator` verir. `iter()` ile önce iterator oluşturulmalı.

---

### [KARAR] `eval/eval_DiffuVQA.py` — `punkt_tab` NLTK hatası düzeltildi
**Değişiklik:** `nltk.download('punkt_tab')` satırı eklendi.

**Neden:** NLTK 3.8+ sürümünde `punkt` tokenizer `punkt_tab` adını kullandı. Eski `punkt` resource'u bulunamadığında `LookupError` veriyordu.

---

### [KARAR] `shared/basic_utils.py` — Ölü model dalları temizlendi
**Değişiklik:** `transformer-pubmedbert`, `transformer-bio-bert`, `transformer-roberta` hardcoded branch'leri `create_model_and_diffusion`'dan silindi.

**Neden:** Bu dallar `hidden_t_dim=128` hardcode ediyordu, `args.hidden_t_dim` ile çakışıyordu. `transformer-bert` artık `--vocab pubmedbert` ile otomatik PubMedBERT seçiyor; ayrı bir dal gereksiz.

---

### [KARAR] `sample_vqa_GPU.py` — Model-family mismatch için fail-fast kontrolü eklendi
**Değişiklik:** `args.__dict__.update(training_args)` sonrasında `vocab`, `model`, `use_plm_init` key'leri için CLI argümanı ile checkpoint'teki değer karşılaştırılıyor; uyuşmazlıkta `ValueError` fırlatılıyor.

**Neden:** Farklı tokenizer ile kaydedilmiş bir checkpoint'i yanlış `--vocab` ile çalıştırmak decode'u sessizce bozar. Hata erken ve açıkça yakalanmalı.

---

### [KARAR] BUG 13 — `vqa_model.py` + `gaussian_diffusion.py` öğrenmeyi bloke eden 3 mimari hata düzeltildi
**Değişiklik:**

1. `lm_head` weight tying kaldırıldı — `bert`, `pubmedbert`, `roberta` init bloklarındaki `self.lm_head.weight = self.word_embedding.weight` satırları silindi. `lm_head` artık bağımsız `nn.Linear` olarak `__init__` içinde tanımlanıyor ve `normal_(0, 0.02)` ile başlatılıyor.

2. `feature_fusion.forward()` BERT preprocessing düzeltildi — `q_ids` → `language_encoder` → son olarak pozisyon ve token_type embedding'leri ekleniyor, `LayerNorm` + dropout uygulanıyor. Önceden sadece token embedding BERT encoder'a giriyordu; pozisyon bilgisi eksikti.

3. `gaussian_diffusion.py` `pre_answer_loss` kaldırıldı — `mean_flat((ans_emb_pre - ans_emb)**2)` loss toplamından çıkarıldı.

**Neden:**
- Tied lm_head: MSE loss lm_head'i bir yöne, NLL loss word_embedding üzerinden ters yöne çekiyordu. İkisi birbirini iptal ediyordu.
- Eksik BERT preprocessing: Pretrained pozisyon bilgisi encoder'a hiç ulaşmıyordu; model pozisyon bağlamı olmadan işliyordu.
- `pre_answer_loss`: `feature_fusion` görüntü+soru encode ediyor; cevap embedding'ine benzemesi istenmez. Bu terim yanlış gradient ekliyor, feature_fusion'ın conditioning kalitesini düşürüyordu.

**Etki:** Mevcut checkpoint'ler uyumsuz — sıfırdan yeniden eğitim gerekiyor.

---

### [KARAR] Resume (checkpoint'ten devam) desteği eklendi — `train_util.py`
**Değişiklik:**
- `_load_and_sync_parameters`: `pass` yerine `resume_checkpoint` verilince model ağırlıklarını yükler, `resume_step`'i checkpoint dosya adından parse eder.
- `_load_optimizer_state`: `opt{NNNNNN}.pt` dosyasını arar; varsa yükler, yoksa uyarı verip fresh optimizer ile devam eder.
- `save()`: her checkpoint'te `opt{NNNNNN}.pt` de kaydeder.
- `run_loop`: `step < learning_steps` → `step + resume_step < learning_steps` (total-step mantığı). tqdm `remaining = learning_steps - resume_step` adım gösterir.
- `__init__`: `_load_optimizer_state()` artık çağrılıyor (önceden yorum satırıydı).

**Notebook:** `RESUME_CHECKPOINT = None` config hücresine eklendi. Eğitim hücresine `resume_flag` değişkeni ile `--resume_checkpoint` geçiriliyor.
