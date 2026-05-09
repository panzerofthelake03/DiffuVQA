# Decisions & Concerns

Proje boyunca alınan teknik kararlar ve dikkat edilmesi gereken noktalar.

---

## 2026-05-09

### [KARAR] `train_util.py` — Resume init'teki hatalı LR hesabı kaldırıldı
**Değişiklik:** `__init__` içinde resume durumunda `lr = self.lr * (1 - frac_done)` ile yeni bir AdamW oluşturma bloğu kaldırıldı. Artık tek bir `AdamW(lr=self.lr)` oluşturuluyor, ardından `_load_optimizer_state()` checkpoint'teki optimizer state'i (LR dahil) yüklüyor.

**Neden:** `_load_optimizer_state()` optimizer'ın tüm state'ini (momentum, LR, param_groups) üzerine yazıyor. Önceki blok hem yanlış formül (lineer decay) kullanıyor hem de anlamsız bir AdamW ikinci kez oluşturuyordu. İlk `_anneal_lr` çağrısı zaten cosine LR'ı set edecek.

---

### [KARAR] `train_util.py` — LR: Warmup + Cosine Decay + Floor eklendi
**Değişiklik:** `_anneal_lr` üç bölgeli schedule'a geçirildi:
- **Warmup:** İlk `%3` adımda (150k için ~4500 adım) LR 0'dan `lr_base`'e lineer ısınma
- **Cosine decay:** Geriye kalan adımlarda `lr_min`'den `lr_base`'e cosine
- **Floor:** `lr_min = lr_base * 0.05` — LR sıfıra inmiyor, son adımlarda optimizer donmuyor

**Neden warmup:** Eğitim başında random init'li ağırlıklarla yüksek LR büyük gradyan patlamalarına yol açıyor. Embedding manifoldu yanlış yöne oturabilir — Med-VQA gibi küçük veri setlerinde bu özellikle zararlı.

**Neden floor:** Eski cosine `t/T=1`'de LR=0'a iniyor. Son adımlarda optimizer neredeyse güncelleme yapmıyordu. `lr_min = 0.05 * lr_base` ile son adımlarda da ince güncelleme devam ediyor.

**Endişe:** Resume durumunda `frac_done` checkpoint adımından hesaplanıyor. Warmup zaten geçilmişse otomatik olarak cosine bölgesine giriyor — resume davranışı tutarlı.

---

### [KARAR] `train_util.py` — Dinamik EMA rate (warmup) eklendi
**Değişiklik:** `_ema_rate(target_rate)` metodu eklendi. İlk 10k adımda `min(target_rate, 1 - 1/(step+1))` formülüyle EMA rate kademeli olarak hedef değere (`0.9999`) ısınıyor. 10k adım sonrası sabit `target_rate`.

**Neden:** `ema_rate=0.9999` ile step=1'de EMA neredeyse tamamen eski (random init) ağırlıklara ağırlık veriyor: `new_ema = 0.9999 * random_init + 0.0001 * updated`. İlk binlerce adımda EMA shadow modeli anlamlı güncellemeleri absorbe edemez.

**Formül davranışı:**
- step=1: rate = min(0.9999, 0.5) = 0.5 (50/50 mix)
- step=9: rate = min(0.9999, 0.9) = 0.9
- step=99: rate = min(0.9999, 0.99) = 0.99
- step=999: rate = min(0.9999, 0.999) = 0.999
- step≥10000: rate = 0.9999 (hedef)

**Endişe:** Checkpoint'e kaydedilen EMA ağırlıkları ilk 10k adımda daha az smooth. Bu özellikle erken checkpoint'lerde (5k, 10k) inference kalitesini etkileyebilir. 50k+ adım için önemsiz.

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

### [KARAR] Proje dosya yapısı yeniden düzenlendi
**Yeni yapı:**
- `checkpoints/` — model ağırlıkları ve `training_args.json`
- `outputs/` — sampling sonucu üretilen `.jsonl` dosyaları
- `eval/` — değerlendirme scriptleri (`eval_DiffuVQA.py`, `enhanced_eval_metrics.py`)
- `docs/` — proje dokümanları, görseller, argüman açıklamaları
- `notebooks/` — Colab notebook'ları
- Root'ta sadece giriş noktaları: `train.py`, `sample_vqa_GPU.py`, `requirements.txt`, `README.md`

**Silinen dosyalar:** `DiffuVQA_BGE_M3.ipynb` (kullanılmıyor), `eval/compare_samples.py`, `eval/prepare_eval.py`, `eval/test_enhanced_metrics.py`, `eval/enhanced_eval_DiffuVQA.py` (duplicate/kullanılmıyor).

**Import düzeltmesi:** `eval/eval_DiffuVQA.py` dosyasına `sys.path` ile repo kökü eklendi.

**Endişe:** Diğer branch'lerde `eval_DiffuVQA.py` root'ta bekleniyorsa bu değişiklik import hatası verir. Her branch'e aynı yapı uygulanmalı.

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

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — BERT test konfigürasyonu
**Değişiklik:** Notebook config hücresi Bert branch'i ve SLAKE dataseti için güncellendi.

**Parametreler:**
- `BRANCH = "Bert"`, `MODEL_NAME = "bert"`, `DATASET = "SLAKE"`
- `BATCH_SIZE = 4`, `LEARNING_STEPS = 500`, `DIFFUSION_STEPS = 50`, `SEQ_LEN = 32`, `SAMPLE_STEP = DIFFUSION_STEPS`
- `SAVE_INTERVAL = 100`, `LOG_INTERVAL = 25`

**Neden:** İlk Colab testi için hızlı iterasyon — düşük step sayısı pipeline'ın hata vermeden çalıştığını doğrular. SEQ_LEN=32, SLAKE'in kısa cevap yapısına uygun. Gerçek eğitim için LEARNING_STEPS 4000+, DIFFUSION_STEPS 100-2000 olmalı.

**Ayrıca:** Eval hücresi `python eval_DiffuVQA.py` → `python eval/eval_DiffuVQA.py` olarak düzeltildi (dosya yapısı yeniden düzenlemesinden kaynaklı). Hardcoded dosya adı kaldırılıp checkpoint'ten dinamik olarak türetilen `CURRENT_SAMPLE_FILENAME` kullanılmaya başlandı.

---

### [KARAR] `Pooler` sınıfı silindi
`diffuvqa/vqa_model.py`'daki `Pooler` sınıfı hiçbir yerde kullanılmıyordu. `CVAE` aktif olarak `feature_fusion` içinde kullanılırken `Pooler` dead code olarak kalmıştı. Kalabalık yaratmaması için silindi.

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

**Neden:** `bert_score` kütüphanesi model yüklerken transformers'ın ağırlık uyuşmazlığı loglarını her eval'da terminal'e yazdırıyordu. Bu bir hata değil — BERTScore sadece encoder katmanlarını kullanır. Logging susturmak gerekir.

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
**Değişiklik:** `__main__` test bloğunda `next(data)` → `next(iter(data))`.

**Neden:** `DataLoader` nesnesi doğrudan `next()` ile tüketilemez. `iter()` ile önce iterator'a dönüştürülmesi gerekir.

---

### [KARAR] BUG 13 — `vqa_model.py` lm_head weight tying kaldırıldı
**Değişiklik:** `self.lm_head.weight = self.word_embedding.weight` satırı tüm init bloklarından (bert, pubmedbert, roberta) kaldırıldı. `lm_head` bağımsız, rastgele ilklendirilmiş ağırlıklarla çalışıyor.

**Neden:** Weight tying, MSE loss (embeddingleri birbirinden uzaklaştırır) ile NLL loss (embeddingleri vocab matrisine çeker) arasında zıt gradyanlar oluşturuyor. Tying'siz lm_head kendi optimum yönünü öğrenebilir.

---

### [KARAR] BUG 13 — `vqa_model.py` `feature_fusion.forward()` BERT preprocessing düzeltildi
**Değişiklik:** Soru embeddingi artık tam BERT embedding pipeline'ından geçiyor: `token_emb + position_emb + token_type_emb → LayerNorm → dropout`.

**Neden:** Eskiden sadece `language_encoder(q_ids)` çıktısı (token embedding) kullanılıyordu. BERT'in beklediği pozisyon ve token-type bilgileri eksikti. Bu, BERT encoder katmanlarına girdiğinde attention pattern'larının bozulmasına yol açıyordu.

---

### [KARAR] BUG 13 — `gaussian_diffusion.py` `pre_answer_loss` kaldırıldı
**Değişiklik:** `terms["loss"]` hesabından `pre_answer_loss` terimi çıkarıldı.

**Neden:** `pre_answer_loss`, fuse tokenları için ek MSE cezası uyguluyordu. Ancak fuse tokenlar conditioner rolünde — loss'tan muaf tutulmalı. Bu terim, modeli fusion tokenlarını da yeniden üretmeye zorlayarak veri sızıntısını pekiştiriyordu.

---

### [KARAR] Resume checkpoint desteği eklendi
**Değişiklik:** `train_util.py` — `_load_and_sync_parameters` gerçek checkpoint yüklüyor, `_load_optimizer_state` eşleşen `opt{NNNNNN}.pt` dosyasını buluyor, `save()` optimizer state'i kaydediyor, `run_loop` total-step mantığıyla kaldığı yerden devam ediyor.

**Neden:** Colab'da eğitim çöktüğünde ya da session sona erdiğinde sıfırdan başlamak gerekmemeli. Resume step dosya adındaki `NNNNNN` pattern'ından regex ile çözümleniyor.

---

### [KARAR] Cosine LR decay eklendi
**Değişiklik:** `train_util.py` `_anneal_lr`: `lr * 0.5 * (1 + cos(π * frac_done))`.

**Neden:** Sabit LR ile son adımlarda loss platoya giriyordu. Cosine decay, öğrenme sonlarında daha küçük güncellemelerle fine-tuning etkisi sağlar. Diffusion modellerinde yaygın kullanılan strateji.

---

### [KARAR] `shared/basic_utils.py` — Ölü branch'ler silindi
**Değişiklik:** `transformer-bio-bert` ve `transformer-roberta` dalları kaldırıldı. BERT ve PubMedBERT init'i tek dinamik blokta birleştirildi: `_plm` değişkeni `args.vocab`'a göre seçiliyor.

**Neden:** BioBERT ve RoBERTa init blokları yıllardır güncellenmemiş, test edilmemiş dead code'du. Aktif vocab seçenekleri: `bert`, `pubmedbert`. RoBERTa/BioBERT gerekirse `vqa_model.py`'daki mevcut init bloğundan türetilebilir.

---