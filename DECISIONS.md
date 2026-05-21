# Decisions & Concerns

Proje boyunca alınan teknik kararlar ve dikkat edilmesi gereken noktalar.
En son alınan karar en üstte yer alır.

---

## 2026-05-21

### [KARAR] LR 5e-5 → 7e-5, sıfırdan eğitim
**Değişiklik:** `diffuvqa/config.json` → `lr: 0.00007`. Notebook `LR = 0.00007`, `CHECKPOINT_PATH` → `lr7e-05`, `RESUME_CHECKPOINT = None`.

**Bağlam:** `use_noising_f=True` ile 22.5K adım sonucu: loss %98.8 düştü (19.24 → 0.28) ama sampling %100 garbled. MSE step 8K'dan sonra ~0.006'da dondu — diffusion backbone hiç ilerlemedi, sadece NLL düşüyordu. avg_nn_l2 JSONL'e yazılmamış. Token collapse devam etti (`the`, `in`, `?` hakimiyeti). Grad norm 0.81'e düştü — backbone yeterli sinyal almıyor.

**Hipotez:** `5e-5` LR backbone'u hareket ettirmek için yetersiz. NLL loss vocab'ı ezberliyor ama MSE (diffusion trajectory) öğrenilemiyor. `7e-5` ile backbone gradient adımı büyüyecek, MSE'nin de düşmeye başlaması bekleniyor.

**Neden `1e-4` değil `7e-5`:** NLL öğrenmesini bozmamak için konservatif artış. `1e-4` mevcut NLL ilerlemesini unstable hale getirebilir.

**Beklenti:** 10K'da MSE'nin 0.006'nın altına inmesi. 25K sampling'de token collapse azalması.

**Alternatif (gerekirse):** `1e-4` dene, veya `mse_weight > 1.0` ile MSE loss'unu direkt ağırlıklandır.

---

## 2026-05-20

### [KARAR] `use_noising_f=True` ile sıfırdan eğitim — CIGN aktif
**Değişiklik:** `diffuvqa/config.json` → `use_noising_f: true`. Notebook `RESUME_CHECKPOINT = None`, `USE_NOISING_F = True`.

**Bağlam:** 25K checkpoint (gradient_clipping=0.5): avg_nn_l2=23.751, %100 garbled çıktı. 30K resume (gradient_clipping=1.0): avg_nn_l2=23.770 — daha da kötüleşti. Grad norm 25K'da her adımda tam 0.500 (clip her zaman tetikleniyordu). 30K'da norm peak ~1.0'a çıktı, settled ~0.86 — iyileşme var ama embedding uzayı vocab manifolduna sıfır hareket.

**Hipotez:** `decoder_nll` ve `terms["nll"]` aynı tied tensor (`lm_head.weight = word_embedding.weight`) üzerinden zıt yönlerde gradient itiyor olabilir → net embedding hareketi ≈ 0. Ya da ağırlıklı olarak: CIGN (`use_noising_f=True`) olmaksızın forward process koşulsuz Gaussian gürültüsü ekliyor — diffusion trajectory vocab manifoldundan tamamen kopuk başlıyor.

**CIGN mekanizması:** `use_noising_f=True` ile `x_start_mean` (answer embedding) noising başlangıcı olarak kullanılıyor; saf Gaussian yerine cevap manifolduna yakın bir noktadan başlanıyor. Hem training hem inference'ta birlikte kullanılmalı.

**Beklenti:** avg_nn_l2'nin 25K'da ~23.75'ten belirgin şekilde aşağıya (<15 hedef) inmesi.

**Alternatif (gerekirse):** `decoder_nll` weight'ini düşür veya `pre_answer_loss_weight > 0` dene.

---

### [KARAR] `diffuvqa/config.json` — `gradient_clipping` 0.5 → 1.0
**Değişiklik:** `gradient_clipping: 0.5` → `gradient_clipping: 1.0`

**Neden:** 25K analizi: loss %97.8 düştü (19.24 → 0.42) ama avg_nn_l2 = 23.75 — önceki run'dan sıfır fark. Grad norm progress.csv'de tüm 25K boyunca sabit ~0.5 — yani her adımda clip tetikleniyordu. `word_embedding` [30522, 768] boyutlu büyük bir matris; 0.5 normu bu matrisin gerçek gradient adımını her seferinde kesiyor. Embedding uzayı vocab manifolduna hiç yaklaşamıyor.

**Strateji:** Tek değişken izole edildi — önce clip artışının etkisini ölç, ardından gerekirse `use_noising_f=True` dene. 25K checkpoint'ten resume ederek 5-10K adım sonra avg_nn_l2'ye bakılacak.

**Beklenti:** avg_nn_l2'nin 25K'da olduğu 23.75'ten aşağıya doğru hareket etmesi. Grad norm'un artık zaman zaman 1.0'ın altında kalması.

---

### [BUG FIX] `diffuvqa/vqa_model.py` — `get_logits` logits_mode=2: `sqrt(0)` NaN gradient patlaması
**Değişiklik:** `th.clamp(dist, 0.0, np.inf)` → `th.clamp(dist, 1e-12, np.inf)`

**Neden:** `decoder_nll = _token_discrete_loss(x_start_mean, ...)` fonksiyonu için `x_start_mean = get_embeds(input_ids_a)` — tam vocab satırları. `lm_head.weight` ve `word_embedding.weight` aynı tensor (tied). Bu nedenle bazı token pozisyonlarında `dist` floating point precision nedeniyle `~0` veya negatif (`-1.9e-6`) çıkıyor. `clamp(0)` sıfıra basıyor, ardından `sqrt(0)` backward'da `1/(2*sqrt(0)) = inf` → NaN gradient → tüm parametreler step 1'de NaN. Test'te bu `LR=1e-5` ile bile step 1'de 243 parametre NaN'a gidiyor şeklinde gözlemlendi.

**Etki:** `decoder_nll` loss'a eklendiği andan itibaren her training run'da step 1'de model patlıyor ve öğrenemiyor. **Bu en kritik bug.** Fix ile step 1'den itibaren NaN yok, loss düşüyor, avg_nn_l2 düşüyor.

---

### [KARAR] `tests/test_architecture.py` — Test hiperparametreleri gerçek training boyutlarına çekildi
**Değişiklik:** `B=2, Q_LEN=16, A_LEN=8` → `B=4, Q_LEN=32, A_LEN=32`

**Neden:** Küçük parametreler bazı bug'ları gizledi. Özellikle `sqrt(0)` NaN bug'ı B=2/Q=16/A=8 ile hiç tetiklenmiyordu — bu boyutlarda `dist<0` pozisyon sayısı tesadüfen sıfır çıkıyor. B=4/Q=32/A=32 (gerçek training seq_len) ile `[CLS]`, `[SEP]`, `[PAD]` tokenları yeterince çoğalınca bug step 1'de deterministik olarak patlıyor. `fake_cond()` de B'ye dinamik hale getirildi (önceden 2 sabit örnek vardı, `[:B]` slice B>2'de sessizce kısalıyordu).

**Kural:** Test hiperparametreleri `seq_len` ve tensor shape bakımından gerçek training değerleriyle eşleşmeli. Batch boyutu CPU'da makul süre için küçük (4) tutulabilir.

---

### [KARAR] `diffuvqa/rounding.py` + `sample_vqa_GPU.py` — WordPiece `##` tokenları denoising ve logit seçiminden dışlandı
**Değişiklik 1:** `get_efficient_knn`'e `subword_mask` parametresi eklendi. `##` ile başlayan tokenlara ait squared-L2 mesafeleri `inf` yapılıyor — her DDIM adımında bu tokenlar nearest-neighbour adayı olamıyor. `denoised_fn_round` da `subword_mask` alacak şekilde güncellendi; `partial()` ile inference'a iletiliyor.

**Değişiklik 2:** `sample_vqa_GPU.py`'de `model.get_logits(sample)` sonrasına `logits.masked_fill(subword_mask, -inf)` eklendi — `topk` öncesinde `##` tokenlar logit uzayından tamamen çıkarılıyor.

**Neden:** BERT WordPiece tokenizer'da `##` ile başlayan tokenlar kelime ortası parçacıklardır ("playing" → `["play", "##ing"]`). Rounding sırasında gürültülü embedding bu tokenlara kilitlenirse trajectory tüm DDIM adımları boyunca `##OWzie`, `##sedel` gibi çıktılar üretir. Logit maskeleme ise rounding'den bağımsız olarak final çıktıdan `##` tokenları tamamen kaldırır.

**Uygulama notu:** `subword_mask` tokenizer vocab'tan bir kez build ediliyor (~4K token), her batch'te GPU'ya taşınıyor, ek hesaplama maliyeti ihmal edilebilir.

---

### [KARAR] `diffuvqa/gaussian_diffusion.py` — `decoder_nll` loss'a geri eklendi
**Değişiklik:** `terms["loss"] = terms["mse"] + terms["nll"] + pre_answer_loss` → `terms["loss"] = terms["mse"] + terms["nll"] + decoder_nll + pre_answer_loss`

**Neden:** Önceki kararda (2026-05-18) `decoder_nll` çifte sayım gerekçesiyle çıkarılmıştı. Ancak bu yanlıştı: `decoder_nll = _token_discrete_loss(x_start_mean, ...)` temiz cevap embedding'inin vocab'a ne kadar yakın olduğunu ölçer; `terms["nll"] = _token_discrete_loss(model_out_x_start, ...)` ise denoised çıktının vocab'a yakınlığını. İkisi farklı şeyleri hedefliyor. `decoder_nll` olmayınca word_embedding uzayı vocab'tan serbestçe kayabiliyor — 25K sampling'de avg_nn_l2=23.5 bunun kanıtıydı.

**Etki:** Embedding uzayının vocab manifolduna bağlı kalması bekleniyor; avg_nn_l2'nin daha hızlı düşmesi gerekiyor. Sıfırdan eğitim gerekiyor.

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

**Neden:** BERT-base-uncased 110M parametreden oluşuyor. SLAKE'de 14K eğitim örneği var. Fully trainable BERT, CLIP freeze + diffusion + fusion katmanlarıyla aynı anda optimize edilince optimizer serbestlik derecesi fazla oluyor ve `the`/`in` collapse'a yol açıyor. CLIP freeze'de aynı gerekçe kullanıldı (151M param). Freeze ile sadece fusion + diffusion katmanları (≈50M param) güncelleniyor.

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

**Kaldırılanlar:** Debug `print()` çağrıları, Türkçe/Çince inline yorumlar, `# ---` section bannerları, obvious docstring'ler, commentted-out dead code blokları.

**Korunanlar:** Mimarinin neden öyle yapıldığını açıklayan yorumlar (non-contiguous tensor notu, EMA warmup formülü, mask semantiği).

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — Tam temizlik
**Değişiklik:** Notebook 1463 satır küçüldü. Emoji, verbose print başlıkları, gereksiz inline yorum blokları kaldırıldı. `vqa_datasets` test hücresi (dead code) silindi.

**Ek düzeltme:** `dataset_local_imgs` path üçlemesi (`SLAKE/imgs/imgs`) kaldırıldı — `IMAGEFOLDER_NAME = "SLAKE/imgs"` zaten tam yolu içerdiğinden ek `"imgs"` append'i gereksizdi.

---

## 2026-05-18

### [KARAR] `diffuvqa/vqa_model.py` — `get_logits` `.view()` → `.reshape()`
**Değişiklik:** `logits_mode=2` dalında `text_emb.view(-1, ...)` ve `(text_emb**2).sum(-1).view(-1,1)` → `.reshape()`.

**Neden:** CLIP freeze sonrası cross-attention çıktısı non-contiguous bellek layout'ına düşüyor. `.view()` contiguous tensor zorunluluğu var, `.reshape()` değil. `logits_mode=2`'ye geçişimizle birlikte bu kod yolu aktif hale geldi ve her eğitim adımında `RuntimeError` verdi.

---

### [KARAR] Drive gereksiz yazımları temizlendi
**Değişiklikler:**
- `train.py`: `import wandb` ve `wandb.init()` bloğu kaldırıldı. `diffuvqa/utils/logger.py`: `import wandb` kaldırıldı.
- `notebooks` Clone hücresi: `copytree` kaldırıldı; sadece kod dosyaları Drive'a kopyalanıyor.
- `notebooks` İndirme hücresi: `shutil.copy(OUTPUT_CSV, DRIVE_RESULTS_PATH)` kaldırıldı, sadece `files.download()` bırakıldı.

**Neden:** wandb offline run klasörleri her session'da Drive'da birikiyordu. copytree her çalıştırmada binlerce dosyayı Drive'a kopyalıyordu.

**Drive'da kalanlar (gerekli):** `ema_*.pt`, `opt*.pt`, `training_args.json`, `progress.csv`, `log.txt`.

---

### [KARAR] `notebooks` — `SAMPLE_STEP` 2000 → 200, `NUM_SAMPLES` 3 → 1
**Değişiklik:** Config hücresinde `SAMPLE_STEP=200`, `NUM_SAMPLES=1`.

**Neden:** `SAMPLE_STEP == DIFFUSION_STEPS` koşulunda `use_ddim=False` devreye giriyor ve tüm 2000 adım sırayla çalışıyor. Her adımda `denoised_fn_round` → `get_efficient_knn` çağrısı → [30522×2048] GPU matris → OOM. `SAMPLE_STEP=200` ile DDIM aktif, 10x daha az adım.

---

### [KARAR] `notebooks` Cell 7 — `dataset_local_imgs` path üçlemesi düzeltildi
**Değişiklik:** `dataset_local_imgs = os.path.join(dataset_local_root, "imgs")` → `dataset_local_imgs = dataset_local_root`.

**Neden:** `IMAGEFOLDER_NAME = "SLAKE/imgs"` zaten `/imgs` ile bitiyor. Cell 7 bunun üstüne `"imgs"` ekliyordu → `SLAKE/imgs/imgs/imgs/...` üçlemesi. Tüm örnekler placeholder (siyah görüntü) ile üretildi.

---

### [KARAR] 100k checkpoint analizi — avg_nn_l2=558, lm_head tying bug tespit edildi
**Bulgular:** Yeni mimariyle 100k eğitim sonrası: avg_nn_l2=558, exact_match=%0.18, confidence=0.053.

**Kök neden:** `lm_head` weight tying `bert`/`pubmedbert`/`roberta` dallarında eksikti. `TransformerNetModel.__init__` başında `word_embedding` random init ile oluşturuluyor, `lm_head` buna tied ediliyor. Sonra bert dalında `self.word_embedding = temp_bert.embeddings.word_embeddings` ile pretrained ağırlık atanıyor — ama Python'da bu atama `lm_head`'in referansını kopardı, `lm_head` random init haliyle kaldı.

`get_efficient_knn(model_emb=lm_head.weight)` ve `get_logits(logits_mode=2)` random matrise karşı L2 mesafesi hesaplıyordu. Denoised embedding pretrained BERT uzayında, lm_head random uzayda → avg_nn_l2 anlamsız, hiç düşmüyor.

**Karar:** Her 3 pretrained dalın sonuna `lm_head.weight = word_embedding.weight` tying eklendi. Sıfırdan eğitim gerekiyor.

---

### [KARAR] `diffuvqa/utils/logger.py` + `train.py` — `progress.csv` fresh-start
**Değişiklik:** `resume_checkpoint=None` ise `append=False` (fresh) geçiliyor. Resume'da dosya korunuyor.

**Neden:** Sıfırdan eğitim başlatılınca eski eğitimin satırları progress.csv'de kalıyordu. Analizi yanıltıyordu.

---

### [KARAR] `diffuvqa/vqa_model.py` — CLIP vision encoder freeze edildi
**Değişiklik:** `feature_fusion.__init__` içinde `build_model(...)` çağrısının hemen ardından `for p in self.vision_encoder.parameters(): p.requires_grad_(False)` eklendi.

**Neden:** CLIP ViT-B/32 ~151M parametre. Medical VQA için zaten zengin visual features üretiyor. Freeze ile sadece fusion+diffusion katmanları güncelleniyor — toplam trainable parametre ~151M azalıyor.

---

### [KARAR] `diffuvqa/gaussian_diffusion.py` — `decoder_nll` ve `tT_loss` loss formülünden çıkarıldı
**Değişiklik:** `terms["loss"] = terms["mse"] + tT_loss + terms["nll"] + decoder_nll + pre_answer_loss` → `terms["loss"] = terms["mse"] + terms["nll"] + pre_answer_loss`.

**Neden:** `tT_loss` gereksiz kısıt koyuyor. `decoder_nll` ile `terms["nll"]` çift sayma yapıyordu (o dönemki değerlendirme — 2026-05-20'de geri alındı).

**Risk:** `terms["nll"]` tek başına yeterince baskın olabilir. İzleme: 50k'da NLL/MSE oranı <5x kalmazsa `nll_weight` parametresi eklenecek.

---

### [KARAR] `diffuvqa/config.json` — `logits_mode: 2` eklendi
**Değişiklik:** `"logits_mode": 2` config'e eklendi.

**Neden:** `logits_mode=1` (dot-product) ile `denoised_fn_round` L2-NN tutarsızlığı giderildi. `logits_mode=2` L2 tabanlı logit hesaplar — `get_efficient_knn` ile aynı metrik. Sampling sırasında eğitim/inference tutarlılığı sağlandı.

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — REPO_URL Aliekinozcetin'e güncellendi
**Değişiklik:** `REPO_URL = "https://github.com/panzerofthelake03/DiffuVQA.git"` → `"https://github.com/Aliekinozcetin/DiffuVQA.git"`.

**Neden:** Aktif geliştirme Aliekinozcetin reposunda devam edecek. Panzerofthelake reposu ara ara sync için kullanılacak.

---

## 2026-05-11

### [KARAR] 100k checkpoint analizi — Mimari kriz teşhisi ve kararlar
**Bulgular:** 100k BERT checkpoint: BLEU-1=0, ROUGE-L=0, exact match=0, avg_nn_l2=552. 50k'da 416'dan 100k'da 552'ye çıkması — embedding manifoldu iyileşmek yerine geriliyor.

**Kök nedenler:**
1. **lm_head weight tying kaldırılması (BUG 13):** MSE ve NLL loss zıt gradyanlar oluşturuyor → avg_nn_l2 divergence.
2. **Padding mask loss eksikliği:** 64 token output için tüm pozisyonlarda MSE hesaplanıyor — model SEP/PAD üretmeyi öğrenemiyor.
3. **Training/inference mismatch:** `f=cond_x_start_mean` ile forward process training'de answer bilgisine bağlı ama inference'ta bu bilgi yok.

---

### [KARAR] `diffuvqa/vqa_model.py` — lm_head weight tying GERİ YÜKLENDİ
**Değişiklik:** `self.lm_head.weight = self.word_embedding.weight` satırı tüm init bloklarına (bert, pubmedbert, roberta) geri eklendi.

**Neden:** BUG 13 kararı (tying kaldırmak) yanlıştı. avg_nn_l2 metriğinin 50k→100k arasında 416→552'ye gerilemesi doğrudan bu değişiklikten kaynaklanıyor. Tying ile lm_head ve embedding uzayı hizalı kalır.

---

### [KARAR] `diffuvqa/vqa_model.py` — `feature_fusion` question_emb residual eklendi
**Değişiklik:** fusion çıktısı: `f = alpha * f4 + beta * image_feats + theta * (q_for_image + question_emb)` — `question_emb` (raw token embedding) residual olarak eklendi.

**Neden:** Baseline DiffuVQA (cloneiq/DiffuVQA) bu pattern'ı kullanıyor. `question_feats` encoder'ı geçmiş yüksek-seviye temsil; `question_emb` token-level semantik detayı.

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

**Neden:** 100k analizinde model doğru medikal terimleri öğrenmişti ama 40+ token gürültü streamine gömmüştü — exact match 0. Maske ile PAD öğrenimi doğal — model SEP/PAD üretip durmayı öğreniyor.

---

### [KARAR] `diffuvqa/gaussian_diffusion.py` — `use_noising_f` bayrağı eklendi
**Değişiklik:** `f = cond_x_start_mean if use_noising_f else None`. Default: `False`.

**Neden:** `f` her zaman `cond_x_start_mean` (answer bilgisi içeriyor) olurken inference'ta bu bilgi yok → training/inference mismatch. `use_noising_f=False` ile training da inference gibi pure noise'tan başlıyor.

---

### [KARAR] `diffuvqa/gaussian_diffusion.py` — `pre_answer_loss` koşullu hale getirildi
**Değişiklik:** `pre_answer_loss_weight=0.0` (config default) iken `pre_answer_loss = zeros`. `> 0.0` iken padding mask ile ağırlıklı hesaplanıyor.

**Neden:** Önceki `BUG 13` kararıyla pre_answer_loss tamamen kaldırılmıştı. Ablasyon için geri alındı ama güvenli default (0.0) ile.

---

### [KARAR] `diffuvqa/config.json` — Yeni bayraklar eklendi
**Değişiklik:** `"use_noising_f": false, "pre_answer_loss_weight": 0.0` eklendi.

**Neden:** Argümanlar `argparse` ile `train.py`'ye geçiliyor; `config.json` training_args.json olarak kaydediliyor. Inference sırasında `sample_vqa_GPU.py` bu dosyadan okuduğu için flagların burada da olması gerekiyor.

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — 200k eğitim konfigürasyonu
**Değişiklik:** `LEARNING_STEPS = 200000`, `SAVE_INTERVAL = 5000`, `RESUME_CHECKPOINT = None`, `USE_NOISING_F = False`, `PRE_ANSWER_LOSS_WEIGHT = 0.0` eklendi.

**Neden:** Padding mask loss masking + lm_head tying restore + use_noising_f=False üçlüsü birlikte yeniden eğitim gerektiriyor.

---

### [KARAR] `eval/eval_DiffuVQA.py` — BERTScore `int too big to convert` hatası düzeltildi
**Değişiklik:** `bert_score()` çağrısından önce tüm kandidat ve referans stringler 512 karaktere truncate ediliyor. `verbose=False` ayarlandı; `warnings.catch_warnings()` + `logging.disable(WARNING)` bloğuna alındı.

**Neden:** 50k Bert analizinde model ~34 kelimelik stringler üretiyordu. `microsoft/deberta-xlarge-mnli` tokenizer bu uzunlukta stringleri int32 aralığını aşan token ID'lere dönüştürüyor.

---

### [KARAR] `sample_vqa_GPU.py` — Top-k rerank + minimum cevap uzunluğu iyileştirmesi
**Değişiklik:** Yeni argümanlar: `--decode_top_k` (default: 5), `--min_answer_tokens` (default: 2), `--short_answer_penalty` (default: 1.0).

**Mekanizma:** `topk(logits, k=decode_top_k)` ile k aday token toplanıyor. SEP/PAD kesme yalnızca `min_answer_tokens` sonrasında geçerli. Confidence filtresi `min_answer_tokens` sonrasında uygulanıyor.

**İzleme sweep önerisi:** `(decode_top_k=5, min_answer_tokens=2, conf=0.25)`, `(5, 2, 0.20)`, `(7, 2, 0.20)`

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — `RESUME_CHECKPOINT` 50k güncellendi
**Değişiklik:** `ema_0.9999_045000.pt` → `ema_0.9999_050000.pt`

---

## 2026-05-10

### [KARAR] `sample_vqa_GPU.py` — Confidence threshold 0.3 → 0.1 düşürüldü
**Değişiklik:** `conf_threshold = 0.3` → `conf_threshold = 0.1`

**Neden:** PubMedBERT 95k checkpoint analizi (500 sample step, 1024 örnek): cevapların %59'u tamamen boştu. `0.3` eşiği trailing noise'u temizlemek yerine neredeyse tüm token'ları kesiyordu.

---

### [KARAR] `notebooks/run_diffuvqa_colab.ipynb` — `copytree` veri kaybı düzeltildi
**Değişiklik:** Clone hücresindeki `shutil.copytree` kaldırıldı. Yerine SKIP_DIRS mekanizması eklendi: `datasets`, `checkpoints`, `samples`, `outputs`, `reports`, `.git` atlanıyor.

**Neden:** `copytree` Drive'daki mevcut checkpoint .pt dosyalarını ve SLAKE images'ı siliyordu.

**Ayrıca:** Hücrenin başına `os.chdir("/content")` eklendi — cwd Drive içindeyken `rmtree` yapılınca `getcwd` crash veriyordu.

---

### [KARAR] `sample_vqa_GPU.py` — SEP/PAD kesme + confidence threshold + MBR eklendi
**Değişiklik:** Üretilen sequence post-processing pipeline'ı eklendi:
- **SEP/PAD kesme:** İlk `[SEP]` veya `[PAD]` token'ına kadar kes.
- **Confidence threshold:** Trailing token'lar arasında confidence < 0.3 olanları sil.
- **MBR:** `--num_samples N` ile N adet sample üretilip ortalama embedding'e L2 mesafesi en düşük olanı seç.

**Neden:** JSONL analizi gösterdi ki doğru cevap token'ı örneklerin %29'unda üretilmiş ama 15-20 token'lık gürültü arasına gömülmüş. Exact match 0 ama contains %29.

---

## 2026-05-09

### [KARAR] `train_util.py` — Resume init'teki hatalı LR hesabı kaldırıldı
**Değişiklik:** Resume durumunda `lr = self.lr * (1 - frac_done)` ile yeni AdamW oluşturma bloğu kaldırıldı. `_load_optimizer_state()` checkpoint'teki optimizer state'i (LR dahil) yüklüyor.

---

### [KARAR] `train_util.py` — LR: Warmup + Cosine Decay + Floor eklendi
**Değişiklik:** `_anneal_lr` üç bölgeli schedule:
- **Warmup:** İlk `%3` adımda LR 0'dan `lr_base`'e lineer ısınma.
- **Cosine decay:** Geriye kalan adımlarda cosine.
- **Floor:** `lr_min = lr_base * 0.05` — LR sıfıra inmiyor.

**Neden warmup:** Eğitim başında random init'li ağırlıklarla yüksek LR büyük gradyan patlamalarına yol açıyor. **Neden floor:** Son adımlarda optimizer neredeyse güncelleme yapmıyordu.

---

### [KARAR] `train_util.py` — Dinamik EMA rate (warmup) eklendi
**Değişiklik:** İlk 10k adımda `min(target_rate, 1 - 1/(step+1))` formülüyle EMA rate kademeli olarak `0.9999`'a ısınıyor.

**Formül davranışı:** step=1 → 0.5, step=99 → 0.99, step≥10000 → 0.9999.

**Neden:** `ema_rate=0.9999` ile step=1'de EMA neredeyse tamamen random init ağırlıklara ağırlık veriyor.

---

## 2026-05-06

### [KARAR] `efficient_sample.py` silindi
Dosya gerçek bir implementasyon içermiyordu. Tüm sampling işlemleri `sample_vqa_GPU.py` üzerinden yürütülecek.

---

### [KARAR] `sample_vqa_GPU.py` — Pure noise başlatma
**Değişiklik:** `x_start` artık cevap embeddinglerinden değil, pure random noise + frozen image-fusion features'tan oluşturuluyor.

**Neden:** Model training'de cevap embeddinglerini hem input (x_start) hem de loss hedefi olarak alıyordu. Bu, modelin cevap üretmeyi değil sadece gürültüden temizlemeyi öğrenmesine yol açıyordu.

---

### [KARAR] `sample_vqa_GPU.py` — Mask yapısı yeniden kuruldu
**Değişiklik:** Mask artık `[zeros(fuse_len) + ones(seq_len)]` şeklinde oluşturuluyor.

**Neden:** Eski mask'ta image-fusion token sayısı gözetilmiyordu. Image-fusion tokenları `mask=0` (dondurulmuş), cevap tokenları `mask=1` (diffuse edilecek) olmalı.

---

### [KARAR] `gaussian_diffusion.py` — Eğitimde mask hizalaması
**Değişiklik:** `mask.repeat()` yerine başa `zeros(fuse_token_len)` prepend ediyor.

**Neden:** Repeat stratejisi, soru/cevap mask değerlerini image-fusion tokenlarına yanlış eşliyordu.

---

### [KARAR] Proje dosya yapısı yeniden düzenlendi
**Yeni yapı:** `checkpoints/`, `outputs/`, `eval/`, `docs/`, `notebooks/`. Root'ta sadece giriş noktaları: `train.py`, `sample_vqa_GPU.py`.

**Silinen dosyalar:** `DiffuVQA_BGE_M3.ipynb`, `eval/compare_samples.py`, `eval/prepare_eval.py`, `eval/test_enhanced_metrics.py`, `eval/enhanced_eval_DiffuVQA.py`.

---

### [KARAR] BUG 1 — `sample_vqa_GPU.py` sample slicing düzeltildi
`a_shape = sample.size(1) // 2` → `fuse_len`. Eski kod `seq_len == fuse_len` olduğu sürece şans eseri doğru çalışıyordu.

---

### [KARAR] BUG 2 — `bert_model.py` `BertLayer` import edildi
`BertEncoder` içinde kullanılan `BertLayer` sınıfı tanımlanmamıştı. `transformers.models.bert.modeling_bert`'ten import edildi.

---

### [KARAR] BUG 3 — `vqa_model.py` feature_fusion çıktısı sabitlendi
`feature_fusion` çıktısı artık her zaman `seq_len` uzunluğunda. Assert eklendi.

---

### [KARAR] BUG 4 — `vqa_model.py` hardcoded `145` kaldırıldı
Vision encoder init sırasında dummy forward pass ile gerçek kanal boyutu ölçülüyor.

---

### [KARAR] BUG 5 — `gaussian_diffusion.py` debug print'ler temizlendi
`x_start` tanımlanmadan önce referans alan debug print kaldırıldı.

---

### [KARAR] Validation loop düzeltildi — BUG 6
- `next(self.eval_data)` → `eval_iter` ayrı tutulup `StopIteration` yakalanarak yeniden başlatıldı.
- `step=0`'da erken validation: `self.step > 0` koşulu eklendi.
- `del cond['image_name']` orijinal dict'i mutate ediyordu: `micro_cond` dict comprehension ile kopyalanır hale getirildi.

---

### [KARAR] `sample_vqa_GPU.py` — Bounded slice ile answer_len kontrolü
**Değişiklik:** `sample[:, fuse_len:, :]` → `sample[:, fuse_len:fuse_len+answer_len, :]`.

**Neden:** Açık uç slice gelecekte segmente ek token eklenmesi durumunda fazla pozisyonu decode'a sokar.

---

### [KARAR] Data Leakage kapatıldı — `gaussian_diffusion.py`
**Mevcut (doğru):**
- `x_start = _get_x_start(ans_emb, std)` — cevap embedding + küçük gürültü.
- `f = cond_x_start_mean` — clean `[fuse | ans_emb]`.
- MSE hedefi sadece answer segmenti: `mean_flat((ans_emb - ans_output)**2)`.

**Neden doğru:** Leakage'ı yaratan şey `target = cond_x_start` (fuse+ans tümü) olmasıydı. Model hem fuse hem answer'ı yeniden üretmeyi öğreniyordu. Şimdi sadece answer segmentini öğreniyor.

---

### [KARAR] `logger.py` — `dumpkvs()` çıktısı geri açıldı
`dumpkvs()` içindeki `for fmt in self.output_formats: fmt.writekvs(d)` bloğu yorum satırına alınmıştı. Geri açıldı.

**Neden:** Eğitim boyunca terminal'de hiç loss görünmüyordu.

---

### [KARAR] `train_util.py` — tqdm postfix loss gösterimi düzeltildi
`logger.name2val['loss'].mean()` → `float(logger.get_current().name2val['loss'])`.

**Neden:** `name2val` değerleri `float` türünde — `.mean()` metodu yoktu, sessizce `AttributeError` veriyordu.

---

### [KARAR] Notebook sampling hücresi — `ls` path separator düzeltildi
`!ls -lh {SAMPLE_FOLDER}*.jsonl` → `!ls -lh {SAMPLE_FOLDER}/*.jsonl`

---

### [KARAR] Notebook — `compare_image_black_vectors` hücresi devre dışı bırakıldı
Script repoda tanımlı değil — eski bir referans. `ModuleNotFoundError` veriyordu.

---

### [KARAR] Notebook — 50k step eğitim konfigürasyonu
`LEARNING_STEPS`: 6000 → 50000, `DIFFUSION_STEPS`: 200 → 2000, `SAMPLE_STEP`: 50 → 200, `SAVE_INTERVAL`: 1000 → 5000.

**Neden:** 6000 adım ile avg_nn_l2=416 — model embedding manifoldunu öğrenemedi. SLAKE için minimum 30k-50k adım gerekiyor.

---

### [KARAR] Notebook eval hücresi — BERTScore LOAD REPORT susturuldu
`bert_score_fn` çağrısı `warnings.catch_warnings()` + `logging.disable(logging.WARNING)` bloğuna alındı.

---

### [KARAR] `train.py` — Logger stdout tablosu kaldırıldı
`logger.configure()` → `logger.configure(format_strs=["log", "csv"])`.

**Neden:** tqdm progress bar'ı zaten `loss=X.XXXX` gösteriyor. Metrikler `log.txt` ve `progress.csv`'ye yazılmaya devam ediyor.

---

### [KARAR] BUG 11 — `train_util.py` microbatch gradient accumulation düzeltildi
`backward()`, `schedule_sampler.update_with_local_losses()`, `log_loss_dict()` microbatch döngüsü **içine** alındı. Loss `/ num_microbatches` ile ölçeklendi.

**Neden:** batch=64, microbatch=16 iken 4 microbatch yerine 1 backward çalışıyordu. Öğrenme kalitesi 4x düşmüştü.

---

### [KARAR] BUG 12 — `vqa_datasets.py` DataLoader iterator hatası düzeltildi
`next(data)` → `next(iter(data))`.

---

### [KARAR] BUG 13 — `vqa_model.py` lm_head weight tying kaldırıldı *(2026-05-11'de geri alındı)*
`self.lm_head.weight = self.word_embedding.weight` satırı kaldırıldı.

**Neden (o dönemki değerlendirme):** MSE ve NLL loss zıt gradyanlar oluşturuyor. *(Bu karar yanlış çıktı — 2026-05-11'de tying geri yüklendi.)*

---

### [KARAR] BUG 13 — `vqa_model.py` `feature_fusion.forward()` BERT preprocessing düzeltildi
Soru embeddingi artık tam BERT embedding pipeline'ından geçiyor: `token_emb + position_emb + token_type_emb → LayerNorm → dropout`.

---

### [KARAR] BUG 13 — `gaussian_diffusion.py` `pre_answer_loss` kaldırıldı *(2026-05-11'de koşullu olarak geri alındı)*
`pre_answer_loss` loss toplamından çıkarıldı. Fuse tokenlar conditioner rolünde — loss'tan muaf tutulmalı.

---

### [KARAR] Resume checkpoint desteği eklendi
`train_util.py` — `_load_and_sync_parameters`, `_load_optimizer_state`, `save()` güncellendi; `run_loop` total-step mantığıyla kaldığı yerden devam ediyor.

---

### [KARAR] Cosine LR decay eklendi
`train_util.py` `_anneal_lr`: `lr * 0.5 * (1 + cos(π * frac_done))`.

---

### [KARAR] `shared/basic_utils.py` — Ölü branch'ler silindi
`transformer-bio-bert` ve `transformer-roberta` dalları kaldırıldı. Aktif vocab seçenekleri: `bert`, `pubmedbert`.

---
