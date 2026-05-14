import os
import torch
from dotenv import load_dotenv
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig
from PIL import Image

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = None
model = None


def load_model():
    global processor, model
    if model is not None:
        return

    print("Model yükleniyor... (ilk seferinde 10-15 dakika sürebilir)")
    processor = LlavaProcessor.from_pretrained(MODEL_ID, token=hf_token)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    model.eval()
    print("Model yüklendi.")


def ask(image_path: str, question: str) -> str:
    if model is None:
        load_model()

    image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"You are a medical AI assistant. Answer briefly and accurately.\n{question}"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return answer if answer else "Model cevap üretemedi."
