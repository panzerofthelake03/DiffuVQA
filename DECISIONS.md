# Decisions & Concerns

Proje boyunca alınan teknik kararlar ve dikkat edilmesi gereken noktalar.


---

## 2026-05-20

### [KARAR] `diffuvqa/gaussian_diffusion.py` — `decoder_nll` loss'a geri eklendi
**Değişiklik:** `terms["loss"] = terms["mse"] + terms["nll"] + pre_answer_loss` → `terms["loss"] = terms["mse"] + terms["nll"] + decoder_nll + pre_answer_loss`

**Neden:** Önceki kararda (2026-05-18) `decoder_nll` çifte sayım gerekçesiyle çıkarılmıştı. Ancak bu yanlıştı: `decoder_nll = _token_discrete_loss(x_start_mean, ...)` temiz cevap embedding'inin vocab'a ne kadar yakın olduğunu ölçer; `terms["nll"] = _token_discrete_loss(model_out_x_start, ...)` ise denoised çıktının vocab'a yakınlığını. İkisi farklı şeyleri hedefliyor. `decoder_nll` olmayınca word_embedding uzayı vocab'tan serbestçe kayabiliyor — 25K sampling'de avg_nn_l2=23.5 bunun kanıtıydı. Orijinal cloneiq tasarımında her iki terim de loss'ta.

**Etki:** Embedding uzayının vocab manifolduna bağlı kalması bekleniyor; avg_nn_l2'nin daha hızlı düşmesi gerekiyor. Sıfırdan eğitim gerekiyor.

---

### [KARAR] `diffuvqa/rounding.py` + `sample_vqa_GPU.py` — WordPiece `##` tokenları denoising ve logit seçiminden dışlandı
**Değişiklik 1 (Decision 22 karşılığı):** `get_efficient_knn`'e `subword_mask` parametresi eklendi. `##` ile başlayan tokenlara ait squared-L2 mesafeleri `inf` yapılıyor — her DDIM adımında bu tokenlar nearest-neighbour adayı olamıyor. `denoised_fn_round` da `subword_mask` alacak şekilde güncellendi; `partial()` ile inference'a iletiliyor.

**Değişiklik 2 (Decision 23 karşılığı):** `sample_vqa_GPU.py`'de `model.get_logits(sample)` sonrasına `logits.masked_fill(subword_mask, -inf)` eklendi — `topk` öncesinde `##` tokenlar logit uzayından tamamen çıkarılıyor.

**Neden:** BERT WordPiece tokenizer'da `##` ile başlayan tokenlar kelime ortası parçacıklardır ("playing" → `["play", "##ing"]`). Rounding sırasında gürültülü embedding bu tokenlara kilitlenirse trajectory tüm DDIM adımları boyunca `##OWzie`, `##sedel` gibi çıktılar üretir. Logit maskeleme ise rounding'den bağımsız olarak final çıktıdan `##` tokenları tamamen kaldırır — BioBERT branch'inde (Decision 22/23) doğrulanmış fix.

**Uygulama notu:** `subword_mask` tokenizer vocab'tan bir kez build ediliyor (~4K token), her batch'te GPU'ya taşınıyor, ek hesaplama maliyeti ihmal edilebilir.

---

### [KARAR] `diffuvqa/config.json` — `gradient_clipping` 0.5 → 1.0
**Değişiklik:** `gradient_clipping: 0.5` → `gradient_clipping: 1.0`

**Neden:** 25K analizi: loss %97.8 düştü (19.24 → 0.42) ama avg_nn_l2 = 23.75 — önceki run'dan sıfır fark. Grad norm progress.csv'de tüm 25K boyunca sabit ~0.5 — yani her adımda clip tetikleniyordu. `word_embedding` [30522, 768] boyutlu büyük bir matris; 0.5 normu bu matrisin gerçek gradient adımını her seferinde kesiyor. Embedding uzayı vocab manifolduna hiç yaklaşamıyor.

**Strateji:** Tek değişken izole edildi — önce clip artışının etkisini ölç, ardından gerekirse `use_noising_f=True` dene. 25K checkpoint'ten resume ederek 5-10K adım sonra avg_nn_l2'ye bakılacak.

**Beklenti:** avg_nn_l2'nin 25K'da olduğu 23.75'ten aşağıya doğru hareket etmesi. Grad norm'un artık zaman zaman 1.0'ın altında kalması.

---

### [KARAR] `tests/test_architecture.py` — Test hiperparametreleri gerçek training boyutlarına çekildi
**Değişiklik:** `B=2, Q_LEN=16, A_LEN=8` → `B=4, Q_LEN=32, A_LEN=32`

**Neden:** Küçük parametreler bazı bug'ları gizledi. Özellikle `sqrt(0)` NaN bug'ı B=2/Q=16/A=8 ile hiç tetiklenmiyordu — bu boyutlarda `dist<0` pozisyon sayısı tesadüfen sıfır çıkıyor. B=4/Q=32/A=32 (gerçek training seq_len) ile `[CLS]`, `[SEP]`, `[PAD]` tokenları yeterince çoğalınca bug step 1'de deterministik olarak patlıyor. `fake_cond()` de B'ye dinamik hale getirildi (önceden 2 sabit örnek vardı, `[:B]` slice B>2'de sessizce kısalıyordu).

**Kural:** Test hiperparametreleri `seq_len` ve tensor shape bakımından gerçek training değerleriyle eşleşmeli. Batch boyutu CPU'da makul süre için küçük (4) tutulabilir.

---

### [BUG FIX] `diffuvqa/vqa_model.py` — `get_logits` logits_mode=2: `sqrt(0)` NaN gradient patlaması
**Değişiklik:** `th.clamp(dist, 0.0, np.inf)` → `th.clamp(dist, 1e-12, np.inf)`

**Neden:** `decoder_nll = _token_discrete_loss(x_start_mean, ...)` fonksiyonu için `x_start_mean = get_embeds(input_ids_a)` — tam vocab satırları. `lm_head.weight` ve `word_embedding.weight` aynı tensor (tied). Bu nedenle bazı token pozisyonlarında `dist` floating point precision nedeniyle `~0` veya negatif (`-1.9e-6`) çıkıyor. `clamp(0)` sıfıra basıyor, ardından `sqrt(0)` backward'da `1/(2*sqrt(0)) = inf` → NaN gradient → tüm parametreler step 1'de NaN. Test'te bu `LR=1e-5` ile bile step 1'de 243 parametre NaN'a gidiyor şeklinde gözlemlendi.

**Etki:** `decoder_nll` loss'a eklendiği andan itibaren her training run'da step 1'de model patlıyor ve öğrenemiyor. **Bu en kritik bug.** Fix ile step 1'den itibaren NaN yok, loss düşüyor, avg_nn_l2 düşüyor.

---

### [BUG FIX] `diffuvqa/gaussian_diffusion.py` — `model_kwargs` pollution: `input_a_id` pop edilmiyordu
**Değişiklik:** `training_losses_seq2seq` içinde:
```python
# Önce (bug):
input_ids_a = model_kwargs['input_a_id']   # dict access, silinmedi

# Sonra (fix):
input_ids_a = model_kwargs.pop('input_a_id')  # pop ile temizlendi
model_kwargs.pop('image_name', None)           # image_name de temizlendi
```

**Neden:** `input_ids`, `input_mask` pop edilirken `input_a_id` dict'te kalıyordu. Ardından `model(x_t, t, **model_kwargs)` çağrısında `TransformerNetModel.forward(self, x, timesteps)` imzasına beklenmedik `input_a_id=tensor` kwarg olarak iletiliyordu.

**Öğrenmeye etkisi:** Training Colab'da `SpacedDiffusion._WrappedModel` path'inden geçiyor. `_WrappedModel.__call__(self, x, ts, **kwargs)` tanımı bu `**kwargs`'ı tamamen yutuyor — `return self.model(x, new_ts)` ile modele iletmiyor. Bu nedenle training crash olmadı ve öğrenme bozulmadı. Ancak `model_kwargs` state'i kirli kalıyordu; fix kod doğruluğu ve future-proof güvenlik için gerekli.

---

## 2026-05-19

### [KARAR] `diffuvqa/vqa_model.py` — BERT language encoder freeze edildi
**Değişiklik:** `feature_fusion.__init__` içinde `self.language_encoder = language_encoder` satırından hemen sonra:
```python
for p in self.language_encoder.parameters():
    p.requires_grad_(False)
```

**Neden:** BERT-base-uncased 110M parametreden oluşuyor. SLAKE'de 14K eğitim örneği var. Fully trainable BERT, CLIP freeze + diffusion + fusion katmanlarıyla aynı anda optimize edilince optimizer serbestlik derecesi fazla oluyor ve `the`/`in` collapse'a yol açıyor — bu en sık görünen yüksek-frekanslı tokenlar loss'u minimize etmek için yeterli. CLIP freeze'de aynı gerekçe kullanıldı (151M param, BUG 13 sonrası eklendi). Freeze ile sadece fusion + diffusion katmanları (≈50M param) güncelleniyor.

**Etki:** `RESUME_CHECKPOINT = None` — sıfırdan eğitim gerekiyor.

---

### [KARAR] `the`/`in` token collapse analizi — seq_len=16 reddedildi
**Bağlam:** 30k checkpoint'te avg_nn_l2=562, üretilen cevapların %100'ü `the`/`in` token'larından oluşuyor.

**İncelenen hipotez:** seq_len=32'den 16'ya düşürmek collapse'ı çözebilir mi?

**Analiz:** Collapse inference'ta oluyor — diffusion modeli hangi token distribüsyonunu üretmesi gerektiğini öğrenemiyor. Padding mask loss masking halihazırda uygulanmış durumda (2026-05-11 kararı). seq_len=16 yapsak bile model 16 pozisyon için `the`/`in` üretir. Kök sebep gereksiz parametre sayısı, seq_len değil.

**Karar:** seq_len değiştirilmedi (32'de kalıyor). Yalnızca BERT freeze uygulandı.

---

### [KARAR] Tüm codebase yorum temizliği yapıldı
**Değişiklik:** 10 dosyada gereksiz, açıklayıcı olmayan, debug amaçlı ve Türkçe/Çince inline yorumlar kaldırıldı. Toplam 1100+ satır silindi.

**Etkilenen dosyalar:** `gaussian_diffusion.py`, `sample_vqa_GPU.py`, `shared/basic_utils.py`, `diffuvqa/rounding.py`, `diffuvqa/vqa_datasets.py`, `diffuvqa/step_sample.py`, `diffuvqa/vqa_model.py`, `shared/train_util.py`, `train.py`.

**Kaldırılanlar:** Debug `print()` çağrıları, Türkçe/Çince inline yorumlar, `# ---` section bannerları, obvious docstring'ler, commentted-out dead code blokları (betas/alphas hesabı, `betas_for_alpha_bar` fonksiyonu, `__main__` test bloğu, `dist.all_gather` bloğu).

**Korunanlar:** Mimarinin neden öyle yapıldığını açıklayan yorumlar (non-contiguous tensor notu, EMA warmup formülü, mask semantiği).

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — Tam temizlik
**Değişiklik:** Notebook 1463 satır küçüldü. Emoji (`📂`, `✅`, `⚠️`, `💡`), verbose print başlıkları, `# opsiyonel` ipucu satırları, gereksiz inline yorum blokları kaldırıldı. `vqa_datasets` test hücresi (dead code) silindi.

**Ek düzeltme:** `dataset_local_imgs` path üçlemesi (`SLAKE/imgs/imgs`) kaldırıldı — `IMAGEFOLDER_NAME = "SLAKE/imgs"` zaten tam yolu içerdiğinden ek `"imgs"` append'i gereksizdi. Kök neden: `DATASET_IMG_PATHS` listesindeki ekstra `"imgs"` candidate.

---

## 2026-05-18

### [KARAR] `diffuvqa/vqa_model.py` — `get_logits` `.view()` → `.reshape()`
**Değişiklik:** `logits_mode=2` dalında `text_emb.view(-1, ...)` ve `(text_emb**2).sum(-1).view(-1,1)` → `.reshape()`.

**Neden:** CLIP freeze sonrası cross-attention çıktısı non-contiguous bellek layout'ına düşüyor. `.view()` contiguous tensor zorunluluğu var, `.reshape()` değil. `logits_mode=2`'ye geçişimizle birlikte bu kod yolu aktif hale geldi ve her eğitim adımında `RuntimeError` verdi.

---

### [KARAR] Drive gereksiz yazımları temizlendi
**Değişiklikler:**
- `train.py`: `import wandb` ve `wandb.init()` bloğu kaldırıldı. `diffuvqa/utils/logger.py`: `import wandb` kaldırıldı.
- `notebooks` Cell 5: GitHub→`/content`→Drive copytree bloğu kaldırıldı. Kod artık `/content/DiffuVQA`'da çalışıyor; sadece `checkpoints/` Drive'a yazılıyor.
- `notebooks` Cell 24: `shutil.copy(OUTPUT_CSV, DRIVE_RESULTS_PATH)` kaldırıldı, sadece `files.download()` bırakıldı.

**Neden:** wandb offline run klasörleri her session'da Drive'da birikiyordu (~yüzlerce MB). copytree her hücre çalıştırmasında binlerce dosyayı Drive'a kopyalıyordu — hem yavaş hem Drive/GitHub senkronizasyon riski. Drive'da zaten çalışan projede CSV'yi ayrı bir klasöre kopyalamak anlamsızdı.

**Drive'da kalanlar (gerekli):** `ema_*.pt`, `opt*.pt`, `training_args.json`, `progress.csv`, `log.txt` — Colab session kapanınca kaybolmaması için şart.

---

### [KARAR] `notebooks` — `SAMPLE_STEP` 2000 → 200, `NUM_SAMPLES` 3 → 1
**Değişiklik:** Config hücresinde `SAMPLE_STEP=200`, `NUM_SAMPLES=1`.

**Neden:** `SAMPLE_STEP == DIFFUSION_STEPS` (her ikisi de 2000) koşulunda `use_ddim=False` devreye giriyor ve tüm 2000 adım sırayla çalışıyor. Her adımda `denoised_fn_round` → `get_efficient_knn` çağrısı → [30522×2048] GPU matris → OOM. `SAMPLE_STEP=200` ile DDIM aktif, 10x daha az adım. Önceki başarılı sampling zaten `samplestep200` ile yapılmıştı (dosya adında görünüyor). `NUM_SAMPLES=3` MBR 3x bellek kullanımı gerektiriyor, mevcut GPU baskısında gereksiz.

---

### [KARAR] `notebooks` Cell 7 — `dataset_local_imgs` path üçlemesi düzeltildi
**Değişiklik:** `dataset_local_imgs = os.path.join(dataset_local_root, "imgs")` → `dataset_local_imgs = dataset_local_root`.

**Neden:** `IMAGEFOLDER_NAME = "SLAKE/imgs"` zaten `/imgs` ile bitiyor. Cell 7 bunun üstüne `"imgs"` ekliyordu → `SLAKE/imgs/imgs`. ACTIVE_IMAGE_DIR buraya set edilince JSONL'deki `imgs/xmlab102/source.jpg` ile birleşip `SLAKE/imgs/imgs/imgs/xmlab102/source.jpg` üçlemesi çıkıyordu. Tüm örnekler placeholder (siyah görüntü) ile üretildi.

---

### [KARAR] 100k checkpoint analizi — avg_nn_l2=558, lm_head tying bug tespit edildi
**Bulgular:** Yeni mimariyle (decoder_nll/tT_loss kaldırıldı, CLIP freeze) 100k eğitim sonrası:
- avg_nn_l2=558 (min=538, max=581) — 200k eski checkpoint ile neredeyse aynı, hiç düşmemiş
- exact_match=%0.18 (2/1088)
- contains=%45.7 — referans kelimeler üretiliyor ama 20-30 token uzunluğunda kelime akışı içine gömülü
- confidence=0.053 — düşük, model kararsız

**Kök neden:** `lm_head` weight tying `bert`/`pubmedbert`/`roberta` dallarında eksikti.
`TransformerNetModel.__init__` başında `word_embedding` random init ile oluşturuluyor, `lm_head` buna tied ediliyor. Sonra bert dalında `self.word_embedding = temp_bert.embeddings.word_embeddings` ile pretrained ağırlık atanıyor — ama Python'da bu atama `lm_head`'in referansını kopardı, `lm_head` random init haliyle kaldı.

`get_efficient_knn(model_emb=lm_head.weight)` ve `get_logits(logits_mode=2)` random matrise karşı L2 mesafesi hesaplıyordu. Denoised embedding pretrained BERT uzayında, lm_head random uzayda → avg_nn_l2 anlamsız, hiç düşmüyor.

**Karar:** Her 3 pretrained dalın sonuna (`word_embedding` set edildikten sonra) `lm_head.weight = word_embedding.weight` tying eklendi. Sıfırdan eğitim gerekiyor — mevcut checkpointler yanlış uzayda eğitildi.

---

### [KARAR] `diffuvqa/utils/logger.py` + `train.py` — `progress.csv` fresh-start
**Değişiklik:** `CSVOutputFormat.__init__(append=False)` parametresi zaten mevcut. `train.py`'de `resume_checkpoint=None` ise `append=False` (fresh) geçiliyor. `configure()` ve `make_output_format()` zincirine `append` parametresi eklendi.

**Neden:** Sıfırdan eğitim başlatılınca eski eğitimin satırları progress.csv'de kalıyordu. Örnek: 0→52k temiz eğitim + 117.500 (eski bozuk eğitim) aynı dosyada karışmış haldeydi. Analizi yanıltıyordu. Resume'da (`resume_checkpoint` set) dosya korunuyor — bu davranış doğru.

---

### [KARAR] `diffuvqa/vqa_model.py` — CLIP vision encoder freeze edildi
**Değişiklik:** `feature_fusion.__init__` içinde `build_model(...)` çağrısının hemen ardından `for p in self.vision_encoder.parameters(): p.requires_grad_(False)` eklendi.

**Neden:** CLIP ViT-B/32 ~151M parametreden oluşuyor ve medical VQA için zaten zengin visual features üretiyor. Gradyan akışına açık bırakıldığında: (1) eğitim belleği ~2GB artar, (2) SLAKE gibi küçük veri setlerinde overfitting riski yükselir, (3) CLIP'in genel visual representation'ını bozabilir. Freeze ile sadece fusion+diffusion katmanları güncelleniyor — toplam trainable parametre ~151M azalıyor.

---

### [KARAR] `diffuvqa/gaussian_diffusion.py` — `decoder_nll` ve `tT_loss` loss formülünden çıkarıldı
**Değişiklik:** `terms["loss"] = terms["mse"] + tT_loss + terms["nll"] + decoder_nll + pre_answer_loss` → `terms["loss"] = terms["mse"] + terms["nll"] + pre_answer_loss`.

**Neden:** `tT_loss` = `mean_flat(q_mean_variance(x_start, T)**2)` — forward process sonundaki gaussian'ın sıfıra ne kadar yakın olduğunu ölçüyor. Bu terim gereksiz kısıt koyuyor, diffusion'ın answer manifoldunu öğrenmesini zorlaştırıyor. `decoder_nll` = `_token_discrete_loss(x_start_mean, ...)` — clean embedding'den NLL. `terms["nll"]` = `_token_discrete_loss(model_out, ...)` zaten var ve denoised çıktıyı hedefliyor — bu daha doğru. decoder_nll çift sayma yapıyordu.

**Risk:** `terms["nll"]` tek başına yeterince baskın olabilir. İzleme: 50k'da NLL/MSE oranı <5x kalmazsa `nll_weight` parametresi eklenecek.

---

### [KARAR] `diffuvqa/config.json` — `logits_mode: 2` eklendi
**Değişiklik:** `"logits_mode": 2` config'e eklendi.

**Neden:** `logits_mode=1` (dot-product) ile `denoised_fn_round` L2-NN tutarsızlığı giderildi. `logits_mode=2` L2 tabanlı logit hesaplar — `get_efficient_knn` ile aynı metrik. Sampling sırasında hangi token'ın "en yakın" olduğu konusunda eğitim/inference tutarlılığı sağlandı.

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — REPO_URL Aliekinozcetin'e güncellendi
**Değişiklik:** `REPO_URL = "https://github.com/panzerofthelake03/DiffuVQA.git"` → `"https://github.com/Aliekinozcetin/DiffuVQA.git"`.

**Neden:** Bundan itibaren aktif geliştirme Aliekinozcetin reposunda devam edecek. Panzerofthelake reposu ara ara sync için kullanılacak.

---

## 2026-05-11

### [KARAR] 100k checkpoint analizi — Mimari kriz teşhisi ve kararlar
**Bulgular:** 100k BERT checkpoint (seed102, step200, bsize64): BLEU-1=0, ROUGE-L=0, exact match=0, avg_nn_l2=552.
50k'da avg_nn_l2=416'dan 100k'da 552'ye çıkması — embedding manifoldu iyileşmek yerine geriliyor.

**Kök nedenler:**
1. **lm_head weight tying kaldırılması (BUG 13):** MSE loss embeddingleri iterasyonla geri itiyor, NLL loss onları vocab matrisine çekiyor. Tying'siz bu zıt gradyanlar ortak sabitleyici nokta bulamıyor → avg_nn_l2 divergence.
2. **Padding mask loss eksikliği:** Model 64 token output üretmek zorunda ama cevap ortalaması ~3 token. Sonraki 61 pozisyon için da MSE kaybı hesaplanıyor — model SEP/PAD üretmeyi hiç öğrenemiyor. 40+ token gürültü streamleri bu yüzden.
3. **Training/inference mismatch:** `f=cond_x_start_mean` ile forward process training'de answer bilgisine bağlı ama inference'ta bu bilgi yok — model iki farklı hedef öğreniyor.

---

### [KARAR] `diffuvqa/vqa_model.py` — lm_head weight tying GERİ YÜKLENDİ
**Değişiklik:** `self.lm_head.weight = self.word_embedding.weight` satırı tüm init bloklarına (bert, pubmedbert, roberta) geri eklendi.

**Neden:** BUG 13 kararı (tying kaldırmak) yanlıştı. avg_nn_l2 metriğinin 50k→100k arasında 416→552'ye gerilemesi doğrudan bu değişiklikten kaynaklanıyor. Tying'siz MSE+NLL zıt gradyanlar embedding manifoldunu çöküştürdü. Tying ile lm_head ve embedding uzayı hizalı kalır — avg_nn_l2 gerilemesi önlenir.

---

### [KARAR] `diffuvqa/vqa_model.py` — `feature_fusion` question_emb residual eklendi
**Değişiklik:** fusion çıktısı: `f = alpha * f4 + beta * image_feats + theta * (q_for_image + question_emb)` — `question_emb` (raw token embedding) residual olarak eklendi.

**Neden:** Baseline DiffuVQA (cloneiq/DiffuVQA) bu pattern'ı kullanıyor. `question_feats` encoder'ı geçmiş yüksek-seviye temsil; `question_emb` token-level semantik detayı. İki seviyeyi birlikte fusion'a vermek conditioning kalitesini artırıyor.

---

### [KARAR] `diffuvqa/gaussian_diffusion.py` — Seçenek 2: Padding mask loss masking UYGULANDI
**Değişiklik:** `training_losses_seq2seq` içinde MSE loss yalnızca gerçek cevap token'larında hesaplanıyor:
```python
pad_id = getattr(_model_args, 'pad_token_id', 0)
ans_len_mask = (input_ids_a != pad_id).float()  # [B, seq_len]
ans_den = ans_len_mask.sum(dim=-1).clamp(min=1.0)
mse_per_token = ((ans_emb - ans_output) ** 2).mean(dim=-1)
terms["mse"] = (mse_per_token * ans_len_mask).sum(dim=-1) / ans_den
```
`t0_loss`, `decoder_nll`, `terms["nll"]` da aynı `ans_len_mask` ile maskeli.

**Neden:** 100k analizinde model doğru medikal terimleri öğrenmişti ama 40+ token gürültü streamine gömmüştü — exact match 0. PAD pozisyonları MSE'ye dahil edilince model bu pozisyonlarda "bir şey üretmek" zorunda kalıyor. Maske ile PAD öğrenimi doğal — model SEP/PAD üretip durmayı öğreniyor. Yeniden eğitim gerektiriyor.

---

### [KARAR] `diffuvqa/gaussian_diffusion.py` — `use_noising_f` bayrağı eklendi
**Değişiklik:** `f = cond_x_start_mean if use_noising_f else None`. Default: `False`.

**Neden:** `f` her zaman `cond_x_start_mean` (answer bilgisi içeriyor) olurken inference'ta bu bilgi yok → training/inference mismatch. `use_noising_f=False` ile training da inference gibi pure noise'tan başlıyor. `use_noising_f=True` ablasyon için saklandı.

---

### [KARAR] `diffuvqa/gaussian_diffusion.py` — `pre_answer_loss` koşullu hale getirildi
**Değişiklik:** `pre_answer_loss_weight=0.0` (config default) iken `pre_answer_loss = zeros` — hesaplama ve gradient yok. `> 0.0` iken padding mask ile ağırlıklı hesaplanıyor.

**Neden:** Önceki `BUG 13` kararıyla pre_answer_loss tamamen kaldırılmıştı. Ablasyon için geri alındı ama güvenli default (0.0) ile. `0.05` ile fusion supervision etkisi test edilebilir.

---

### [KARAR] `diffuvqa/config.json` — Yeni bayraklar eklendi
**Değişiklik:** `"use_noising_f": false, "pre_answer_loss_weight": 0.0` eklendi.

**Neden:** Argümanlar `argparse` ile `train.py`'ye geçiliyor; `config.json` training_args.json olarak kaydediliyor. Inference sırasında `sample_vqa_GPU.py` bu dosyadan okuduğu için flagların burada da olması gerekiyor.

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — 200k eğitim konfigürasyonu
**Değişiklik:** Config hücresi:
- `LEARNING_STEPS = 200000`, `SAVE_INTERVAL = 5000` (40 checkpoint toplam)
- `RESUME_CHECKPOINT = None` (sıfırdan eğitim — yeni mimari önceki checkpoint'lerle uyumsuz)
- `USE_NOISING_F = False`, `PRE_ANSWER_LOSS_WEIGHT = 0.0` eklendi

Training hücresi: `--use_noising_f {USE_NOISING_F} --pre_answer_loss_weight {PRE_ANSWER_LOSS_WEIGHT}` argümanları eklendi.

**Neden:** Padding mask loss masking + lm_head tying restore + use_noising_f=False üçlüsü birlikte yeniden eğitim gerektiriyor. 100k analizi ışığında 200k hedeflendi; her 5k'da checkpoint ile converging noktası takip edilebilir.

---

### [KARAR] `eval/eval_DiffuVQA.py` — BERTScore `int too big to convert` hatası düzeltildi
**Değişiklik:** `bert_score()` çağrısından önce tüm kandidat ve referans stringler 512 karaktere truncate ediliyor (`r[:512]`). `verbose=False` ayarlandı; çağrı `warnings.catch_warnings()` + `logging.disable(WARNING)` bloğuna alındı.

**Neden:** 50k Bert analizinde model ~34 kelimelik stringler üretiyordu. `microsoft/deberta-xlarge-mnli` tokenizer bu uzunlukta stringleri int32 aralığını aşan token ID'lere dönüştürüyor. 512 karakter BERT token limitinin (~512 token) güvenli altında, kısa tıbbi cevaplarda anlam kaybı yok.

**Ayrıca:** `verbose=True` → `verbose=False` ile transformers weight loading sırasında her adımı iki kez basan tqdm duplicate satır sorunu giderildi. Logging suppress ile LOAD REPORT gürültüsü de susturuldu.

---

### [KARAR] `sample_vqa_GPU.py` — Top-k rerank + minimum cevap uzunluğu iyileştirmesi
**Değişiklik:** Yeni argümanlar eklendi: `--decode_top_k` (default: 5), `--min_answer_tokens` (default: 2), `--short_answer_penalty` (default: 1.0).

**Mekanizma:**
- `topk(logits, k=decode_top_k)` ile her pozisyon için k aday token toplanıyor.
- Her batch örneği için k farklı aday sequence oluşturulup rerank ediliyor.
- SEP/PAD kesme yalnızca `min_answer_tokens` sonrasında geçerli — erken boş cevap engelleniyor.
- Confidence filtresi (`0.1`) `min_answer_tokens` sonrasında uygulanıyor.
- Her aday için `avg_log_prob - short_answer_penalty` skoru hesaplanıyor; en yüksek skorlu aday seçiliyor.
- `short_answer_penalty`: efektif uzunluk `< min_answer_tokens` ise uygulanır.

**Neden:** Yeniden eğitim gerektirmeden boş cevap oranını düşüren en dengeli yol. Top-1 greedy seçimde ilk tokenda gelen SEP/PAD tüm cevabı boşaltıyordu.

**İzleme sweep önerisi:** `(decode_top_k=5, min_answer_tokens=2, conf=0.25, penalty=1.0)`, `(5, 2, 0.20, 1.0)`, `(7, 2, 0.20, 0.8)`

**Risk:** `decode_top_k` arttıkça CPU-side aday değerlendirme maliyeti artar. `short_answer_penalty` fazla yüksek olursa doğru kısa cevaplar ("no", "2") gereksiz cezalanabilir.

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — `RESUME_CHECKPOINT` 50k güncellendi
**Değişiklik:** `ema_0.9999_045000.pt` → `ema_0.9999_050000.pt`

**Neden:** 50k checkpoint ile devam ediliyor.

---


---

## 2026-05-10

### [KARAR] `sample_vqa_GPU.py` — Confidence threshold 0.3 → 0.1 düşürüldü
**Değişiklik:** `conf_threshold = 0.3` → `conf_threshold = 0.1`

**Neden:** PubMedBERT 95k checkpoint analizi (500 sample step, 1024 örnek): cevapların %59'u tamamen boştu. `0.3` eşiği trailing noise'u temizlemek yerine neredeyse tüm token'ları kesiyordu. `0.1`'de anlamlı token'lar korunurken aşırı gürültü hâlâ filtrelenir.

**Bulgular (95k):** BLEU-1=0.017, exact match=%0.98, entity_overlap=0.885, clinical_similarity=0.693 — model medikal terimleri öğrenmiş ama henüz doğru bağlamı oturtamamış. 150k'da anlamlı iyileşme bekleniyor.

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — `copytree` veri kaybı düzeltildi
**Değişiklik:** Clone hücresindeki `shutil.copytree(LOCAL_CLONE_PATH, DRIVE_PROJECT_PATH, dirs_exist_ok=True)` kaldırıldı. Yerine yalnızca kod dosyalarını kopyalayan SKIP_DIRS mekanizması eklendi: `datasets`, `checkpoints`, `samples`, `outputs`, `reports`, `.git` klasörleri atlanıyor.

**Neden:** Git clone `datasets/` ve `checkpoints/` getirmiyor (`.gitignore`'da). `copytree` bu klasörleri Drive'da boş olarak yaratıyor, mevcut içeriği (checkpoint .pt dosyaları, SLAKE images) siliyordu.

**Ayrıca:** Hücrenin başına `os.chdir("/content")` eklendi — cwd Drive içindeyken `rmtree(LOCAL_CLONE_PATH)` yapılınca `getcwd` crash veriyordu.

---

### [KARAR] `sample_vqa_GPU.py` — SEP/PAD kesme + confidence threshold + MBR eklendi
**Değişiklik:** Üretilen sequence post-processing pipeline'ı eklendi:

**Seçenek 1 — SEP/PAD kesme:** Her üretilen sequence'te ilk `[SEP]` veya `[PAD]` token'ına kadar kes. Model seq_len kadar token üretmek zorunda olduğu için gereksiz trailing noise bu şekilde temizlenir.

**Seçenek 3 — Confidence threshold:** SEP/PAD bulunamadıysa (model henüz bunu öğrenmemişse) trailing token'ları arasında confidence < 0.3 olanları sil. Gürültüyü kısmen temizler, yeniden eğitim gerektirmez.

**Seçenek 4 — MBR (Minimum Bayes Risk):** `--num_samples N` ile N adet sample üretilip ortalama embedding'e L2 mesafesi en düşük olanı seç. N=1'de mevcut davranış korunur. N>1'de kalite artar, latency N katına çıkar — chatbot için N=1, offline eval için N=3-5 önerilir.

**Neden:** JSONL analizi gösterdi ki doğru cevap token'ı örneklerin %29'unda üretilmiş ama 15-20 token'lık gürültü arasına gömülmüş. Exact match 0 ama contains %29. Bu post-processing pipeline exact match'i artırmayı hedefliyor.

**Endişe:** SEP/PAD kesme, model bu token'ları üretmeyi öğrenmemişse etkisiz kalır. 150k checkpoint'te test edilmeli. Kalıcı çözüm Seçenek 2 (aşağıda).

---

### [BEKLEYEN KARAR] Seçenek 2 — Training'de padding mask ile loss masking (Sonraki Run)
**Değişiklik (henüz uygulanmadı):** `gaussian_diffusion.py` `training_losses` içinde MSE loss'u gerçek cevap uzunluğuyla maskele — padding token pozisyonlarına loss ağırlığı 0 ver:
```python
ans_len_mask = (token_ids != pad_id).float()  # [B, seq_len]
terms["mse"] = mean_flat((ans_emb - ans_output) ** 2 * ans_len_mask.unsqueeze(-1))
```

**Neden uygulanmadı:** Mevcut 150k run devam ediyor. Bu değişiklik yeniden eğitim gerektirir — mevcut checkpoint'lerle uyumsuz.

**Ne zaman uygulanmalı:** 150k run tamamlanıp sonuçlar değerlendirildikten sonra, bir sonraki training run başlamadan önce uygulanmalı.

**Beklenen etki:** Model kısa cevaplar için doğal olarak erken durur, SEP/PAD üretmeyi öğrenir. Post-processing pipeline'ına olan bağımlılık azalır.


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