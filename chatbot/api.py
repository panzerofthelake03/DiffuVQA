import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from chatbot.model import load_model, ask
import chatbot.database as db

app = FastAPI(title="DiffuVQA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    load_model()


@app.post("/infer")
async def infer(image: UploadFile = File(...), question: str = Form(...)):
    suffix = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name
    try:
        answer = ask(tmp_path, question.strip())
        db.save(tmp_path, question.strip(), answer)
    finally:
        os.unlink(tmp_path)
    return {"answer": answer}


@app.get("/history")
def history(limit: int = 20):
    return db.get_recent(limit)


@app.get("/stats")
def stats():
    return {"total": db.get_all_count()}
