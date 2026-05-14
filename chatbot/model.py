import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image

MODEL_ID = "katielink/llava-med-7b-slake-delta"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = None
model = None


def load_model():
    global tokenizer, model
    if model is not None:
        return

    print("Model yükleniyor... (ilk seferinde 10-15 dakika sürebilir)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print("Model yüklendi.")


def ask(image_path: str, question: str) -> str:
    if model is None:
        load_model()

    image = Image.open(image_path).convert("RGB")

    prompt = (
        "You are a medical AI assistant. "
        "Answer the following question about the medical image briefly and accurately.\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1].strip()
    return answer if answer else "Model cevap üretemedi."
