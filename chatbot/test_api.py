"""
Quick endpoint tests using FastAPI TestClient.
Patches model loading so tests run instantly without GPU/download.
Run: python -m chatbot.test_api
"""
import sys
import os
import io
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch model functions BEFORE importing the app
with patch("chatbot.model.load_model", return_value=None), \
     patch("chatbot.model.ask", return_value="CT scan showing abdominal region."), \
     patch("chatbot.model.model", MagicMock()):

    from fastapi.testclient import TestClient
    from chatbot.api import app

    client = TestClient(app)

    results = []

    # --- /stats ---
    r = client.get("/stats")
    ok = r.status_code == 200 and "total" in r.json()
    results.append(("GET /stats", r.status_code, r.json(), "PASS" if ok else "FAIL"))

    # --- /history ---
    r = client.get("/history?limit=5")
    ok = r.status_code == 200 and isinstance(r.json(), list)
    results.append(("GET /history", r.status_code, f"{len(r.json())} records", "PASS" if ok else "FAIL"))

    # --- /infer ---
    img_bytes = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    with patch("chatbot.model.ask", return_value="CT scan showing abdominal region."):
        r = client.post(
            "/infer",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"question": "What modality is shown?"},
        )
    ok = r.status_code == 200 and "answer" in r.json()
    results.append(("POST /infer", r.status_code, r.json(), "PASS" if ok else "FAIL"))

    # --- Print results ---
    print("\n" + "=" * 60)
    print("  DiffuVQA FastAPI Endpoint Test Results")
    print("=" * 60)
    all_pass = True
    for endpoint, status, body, verdict in results:
        print(f"  {verdict}  {endpoint}")
        print(f"        status: {status}")
        print(f"        body  : {body}")
        print()
        if verdict == "FAIL":
            all_pass = False
    print("=" * 60)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 60 + "\n")
    sys.exit(0 if all_pass else 1)
