# LLaVA-Med Chatbot — Yapılacaklar / İlerleme

## Tamamlananlar

- [x] **ADIM 0 — Branch** `ChatBotPipeline` branch'i açıldı ve remote'a push edildi.

- [x] **ADIM 1 — Klasör Yapısı** `chatbot/` dizini ve içindeki tüm dosyalar oluşturuldu:
  - `app.py` — Gradio 6.x arayüzü
  - `model.py` — Model yükleme & inference
  - `database.py` — SQLite geçmiş yönetimi
  - `requirements.txt` — Bağımlılıklar
  - `__init__.py` — Python paketi tanımı

- [x] **ADIM 2 — Ortam Kurulumu** Mevcut `.venv` (Python 3.13) üzerine eksik paketler kuruldu:
  - `gradio 6.14.0`
  - `sqlalchemy 2.0.49`
  - `peft 0.19.1`
  - `accelerate 1.13.0`
  - `bitsandbytes 0.49.2`
  - `torch 2.11.0`, `transformers 5.5.4`, `Pillow 12.2.0` zaten mevcuttu.

- [x] **ADIM 3 — model.py** 4-bit BitsAndBytes quantization ile LLaVA-Med inference kodu yazıldı.

- [x] **ADIM 4 — database.py** SQLAlchemy ile SQLite geçmiş kaydı; `save`, `get_recent`, `get_all_count` fonksiyonları.

- [x] **ADIM 5 — app.py** Gradio Blocks arayüzü: görüntü yükleme, soru-cevap, geçmiş ve istatistik panelleri. Gradio 6.x için `theme="soft"` uyumu yapıldı.

- [x] **.gitignore** `chatbot/chat_history.db` ve `.venv/` hariç tutuldu.

---

## Bekleyenler

- [ ] **ADIM 6 — İlk Çalıştırma** Uygulamayı GPU makinede başlat:
  ```bash
  .venv\Scripts\python.exe chatbot\app.py
  ```
  İlk çalıştırmada model (~15-20 GB) Hugging Face Hub'dan indirilir.

- [ ] **ADIM 7 — Test** Arayüzde aşağıdaki senaryoları dene:
  - MRI görüntüsü → `"What modality is shown?"`
  - X-Ray → `"Is this a chest X-ray?"`
  - CT → `"Which organ is visible?"`
  - Geçmiş butonu ve istatistik butonu çalışıyor mu?

- [ ] **ADIM 8 — Commit & Push**
  ```bash
  git add chatbot/ .gitignore
  git commit -m "Add LLaVA-Med chatbot with Gradio UI and SQLite history"
  git push origin ChatBotPipeline
  ```

---

## Bilinen Riskler / Notlar

| Konu | Durum | Açıklama |
|---|---|---|
| `bitsandbytes` Windows GPU | Belirsiz | Import başarılı ama gerçek CUDA desteği GPU makinede test edilmeli |
| `transformers` sürüm farkı | Düşük risk | Plan 4.37.0 önerdi, 5.5.4 kurulu — API değişiklikleri olabilir |
| `gradio` sürüm farkı | Çözüldü | Plan 4.19.2, kurulu 6.14.0 — `theme="soft"` string ile uyumluluk sağlandı |
| Model boyutu | Bekliyor | İlk indirmede ~15-20 GB disk ve stabil internet gerekli |
| Windows vs Linux | Düşük risk | `bitsandbytes` Linux'ta daha güvenilir; sorun çıkarsa WSL2 dene |
