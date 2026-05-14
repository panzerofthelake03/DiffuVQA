import gradio as gr
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.model import load_model, ask
from chatbot import database as db


def on_submit(image, question):
    if image is None:
        return "Lütfen bir görüntü yükleyin."
    if not question or not question.strip():
        return "Lütfen bir soru girin."

    answer = ask(image, question.strip())
    db.save(image, question.strip(), answer)
    return answer


def on_history():
    records = db.get_recent(20)
    if not records:
        return "Henüz soru sorulmamış."
    lines = []
    for r in records:
        lines.append(f"[{r['timestamp']}]")
        lines.append(f"Soru : {r['question']}")
        lines.append(f"Cevap: {r['answer']}")
        lines.append("-" * 50)
    return "\n".join(lines)


def on_stats():
    total = db.get_all_count()
    return f"Toplam soru sayısı: {total}"


print("Uygulama başlatılıyor...")
load_model()

with gr.Blocks(title="Medical VQA — LLaVA-Med") as demo:
    gr.Markdown("# Medical Visual Question Answering\nLLaVA-Med (SLAKE fine-tuned)")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="Tıbbi Görüntü (JPG/PNG)",
            )
            question_input = gr.Textbox(
                label="Soru",
                placeholder="What modality is shown in this image?",
                lines=2,
            )
            submit_btn = gr.Button("Sor", variant="primary")

        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Cevap",
                lines=4,
                interactive=False,
            )

    gr.Markdown("---")

    with gr.Row():
        history_btn = gr.Button("Son 20 Soruyu Göster")
        stats_btn = gr.Button("İstatistik")

    history_output = gr.Textbox(label="Geçmiş", lines=15, interactive=False)

    submit_btn.click(
        fn=on_submit,
        inputs=[image_input, question_input],
        outputs=answer_output,
    )
    history_btn.click(fn=on_history, outputs=history_output)
    stats_btn.click(fn=on_stats, outputs=history_output)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme="soft",
    )
