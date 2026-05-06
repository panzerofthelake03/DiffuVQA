# TODO

## Aktif Görevler

- [ ] Sıfırdan yeniden eğitim başlat (data leakage kapatıldı, eski checkpoint'ler geçersiz)
- [ ] Chatbot arayüzü tasarla ve implemente et

## Tamamlanan Görevler

- [x] `sample_vqa_GPU.py`: Cevap embeddingleri yerine pure noise ile başlat
- [x] `sample_vqa_GPU.py`: Image patch + answer token maskesini doğru kur
- [x] `gaussian_diffusion.py`: Eğitimde mask repeat/pad yerine doğru hizalama (zeros prepend)
- [x] `efficient_sample.py`: Kullanılmadığı için silindi
- [x] CLAUDE.md oluşturuldu ve güncellendi
- [x] Proje dosya yapısı yeniden düzenlendi (checkpoints/, outputs/, eval/, docs/, notebooks/)
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
- [x] `notebooks/run_diffuvqa_colab.ipynb` — PubMedBERT + SLAKE test parametreleri güncellendi
- [x] `notebooks/run_diffuvqa_colab.ipynb` — Eval hücresi `eval/eval_DiffuVQA.py` yolu ve dinamik dosya adı kullanacak şekilde düzeltildi
