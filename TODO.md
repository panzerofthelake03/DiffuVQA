# TODO

## Aktif Görevler

- [ ] Bert 150k eğitimi tamamlanınca sampling yap ve sonuçları değerlendir
- [ ] Eval script düzeltmesi — `yes_no_acc`, `f1_score`, `bert_score` sıfır hatası giderilecek
- [ ] Chatbot arayüzü: `inference.py` wrapper + Gradio endpoint (150k sonrası)
- [ ] Sonraki training run için Seçenek 2 uygula — `gaussian_diffusion.py` training_losses'da padding mask ile loss masking

## Tamamlanan Görevler

- [x] `sample_vqa_GPU.py`: Cevap embeddingleri yerine pure noise ile başlat
- [x] `sample_vqa_GPU.py`: Image patch + answer token maskesini doğru kur
- [x] `gaussian_diffusion.py`: Eğitimde mask repeat/pad yerine doğru hizalama (zeros prepend)
- [x] `efficient_sample.py`: Kullanılmadığı için silindi
- [x] CLAUDE.md oluşturuldu ve güncellendi
- [x] Proje dosya yapısı yeniden düzenlendi (checkpoints/, outputs/, eval/, docs/, notebooks/)
- [x] Kullanılmayan dosyalar temizlendi — `DiffuVQA_BGE_M3.ipynb`, `eval/compare_samples.py`, `eval/prepare_eval.py`, `eval/test_enhanced_metrics.py`, `eval/enhanced_eval_DiffuVQA.py`
- [x] BUG 1: `sample_vqa_GPU.py` — `a_shape = sample.size(1)//2` yerine `fuse_len` kullanıldı
- [x] BUG 2: `bert_model.py` — Tanımsız `BertLayer` transformers'dan import edildi
- [x] BUG 3: `vqa_model.py` — `feature_fusion` çıktısı her zaman `seq_len` uzunluğunda garantilendi
- [x] BUG 4: `vqa_model.py` — Hardcoded `145` kanal boyutu dummy forward ile dinamik hesaplamaya çevrildi
- [x] BUG 5: `gaussian_diffusion.py` — Tanımsız değişkeni referans alan debug print'ler kaldırıldı
- [x] BUG 6: `train_util.py` — Validation loop 3 hata düzeltildi (StopIteration çökmesi, step=0 erken eval, cond dict mutation)
- [x] BUG 7: `train_util.py` — `dist.get_world_size()` single GPU'da crash, `dist.is_initialized()` guard eklendi
- [x] BUG 8: `train_util.py` — `forward_backward`'da `del cond['image_name']` KeyError düzeltildi
- [x] Data Leakage tamamen kapatıldı — `gaussian_diffusion.py` training_losses: x_start pure noise, target sadece ans_emb
- [x] BUG 9: `logger.py` — `dumpkvs()` içindeki `writekvs` bloğu yorum satırına alınmıştı, geri açıldı (loss terminal'e hiç yazılmıyordu)
- [x] BUG 10: `train_util.py` — tqdm postfix'te `logger.name2val['loss'].mean()` → `float(logger.get_current().name2val['loss'])` düzeltildi
- [x] `vqa_model.py` — Kullanılmayan `Pooler` sınıfı silindi
- [x] `notebooks/run_diffuvqa_colab.ipynb` — BERT + SLAKE test parametreleri güncellendi
- [x] `notebooks/run_diffuvqa_colab.ipynb` — Eval hücresi `eval/eval_DiffuVQA.py` yolu ve dinamik dosya adı kullanacak şekilde düzeltildi
- [x] `notebooks/run_diffuvqa_colab.ipynb` — Sampling hücresinde `ls` path separator hatası düzeltildi (`{FOLDER}*.jsonl` → `{FOLDER}/*.jsonl`)
- [x] `notebooks/run_diffuvqa_colab.ipynb` — `compare_image_black_vectors` hücresi devre dışı bırakıldı (script repoda yok)
- [x] `notebooks/run_diffuvqa_colab.ipynb` — 50k step eğitim config: LEARNING_STEPS=50000, DIFFUSION_STEPS=2000, SAMPLE_STEP=200, SAVE_INTERVAL=5000
- [x] `notebooks/run_diffuvqa_colab.ipynb` — BERTScore LOAD REPORT gürültüsü `warnings` + `logging.disable` ile susturuldu
- [x] `train.py` — `logger.configure(format_strs=["log", "csv"])` ile stdout tablosu kaldırıldı, tqdm loss satırı korundu
- [x] BUG 11: `train_util.py` — `forward_backward` microbatch gradient accumulation düzeltildi (backward döngü içine alındı, loss/num_microbatches ile ölçeklendi)
- [x] Resume desteği — `train_util.py` `_load_and_sync_parameters`, `_load_optimizer_state`, `save()` güncellendi; `run_loop` total-step mantığına geçirildi
- [x] `notebooks/run_diffuvqa_colab.ipynb` — `RESUME_CHECKPOINT` config değişkeni ve `resume_flag` eğitim hücresi eklendi
- [x] BUG 12: `vqa_datasets.py` — `next(data)` → `next(iter(data))` DataLoader iterator hatası düzeltildi
- [x] `eval/eval_DiffuVQA.py` — `nltk.download('punkt_tab')` eklendi (NLTK 3.8+ LookupError)
- [x] `shared/basic_utils.py` — Ölü `transformer-bio-bert`, `transformer-roberta` dalları silindi
- [x] `sample_vqa_GPU.py` — Model-family mismatch için fail-fast `ValueError` kontrolü eklendi
- [x] LR scheduler cosine decay'e geçirildi — `train_util.py` `_anneal_lr`: `lr * 0.5 * (1 + cos(π * frac_done))`
- [x] `train.py` + `train_util.py` — `logger.configure(dir=args.checkpoint_path)` ile `progress.csv` ve `log.txt` Drive'a kaydediliyor
- [x] BUG 13: `vqa_model.py` — `lm_head` weight tying kaldırıldı (bert/pubmedbert/roberta init bloklarında)
- [x] BUG 13: `vqa_model.py` — `feature_fusion.forward()` BERT preprocessing düzeltildi (pozisyon + token_type embedding + LayerNorm + dropout)
- [x] BUG 13: `gaussian_diffusion.py` — `pre_answer_loss` loss toplamından kaldırıldı
- [x] `train_util.py` — Resume init'teki hatalı lineer LR hesabı kaldırıldı (optimizer state zaten doğru LR'ı yükleyecek)
- [x] `train_util.py` — LR scheduler: warmup (ilk %3) + cosine decay + lr_min floor (lr*0.05) eklendi
- [x] `train_util.py` — Dinamik EMA rate: ilk 10k adımda `1 - 1/(step+1)` ile kademeli ısınma
- [x] `train.py` — `parse_known_args` ile bilinmeyen argümanlar warning olarak loglanıyor, crash yok
- [x] `notebooks/run_diffuvqa_colab.ipynb` — 150k full training config güncellendi (LR=0.000283, batch=64, diffusion=2000, seq_len=64)
- [x] `notebooks/run_diffuvqa_colab.ipynb` — Resume checkpoint yapısı PubMedBERT ile birebir hizalandı
- [x] `sample_vqa_GPU.py` — Post-processing pipeline eklendi: SEP/PAD kesme (Seçenek 1) + confidence threshold (Seçenek 3) + MBR `--num_samples` (Seçenek 4)
- [x] `sample_vqa_GPU.py` — Confidence threshold 0.3 → 0.1 düşürüldü (95k PubMedBERT analizinde %59 boş cevap tespit edildi)
- [x] `notebooks/run_diffuvqa_colab.ipynb` — `copytree` veri kaybı düzeltildi: `datasets/`, `checkpoints/` Drive'a kopyalanmıyor artık
- [x] `notebooks/run_diffuvqa_colab.ipynb` — Clone hücresine `os.chdir("/content")` eklendi (getcwd crash düzeltildi)