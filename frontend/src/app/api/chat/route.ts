import { NextResponse } from "next/server";

const BACKEND = "http://127.0.0.1:8000";

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") ?? "";

  // Image + question: forward as multipart to Python /infer
  if (contentType.includes("multipart/form-data")) {
    try {
      const form = await req.formData();
      const image = form.get("image") as File | null;
      const question = (form.get("question") as string | null) ?? "";

      if (!image) {
        return NextResponse.json({ text: "No image received." });
      }

      const backendForm = new FormData();
      backendForm.append("image", image, image.name);
      backendForm.append("question", question || "What does this medical image show?");

      const res = await fetch(`${BACKEND}/infer`, {
        method: "POST",
        body: backendForm,
      });

      if (!res.ok) {
        console.error("Backend error:", res.status, await res.text().catch(() => ""));
        return NextResponse.json({
          text: "The model backend returned an error. Is the Python server running on port 8000?",
        });
      }

      const data = await res.json();
      return NextResponse.json({ text: data.answer ?? "No answer returned." });

    } catch (err: any) {
      console.error("Infer route error:", err.message);
      return NextResponse.json({
        text: "Could not reach the model backend. Start it with: uvicorn chatbot.api:app --port 8000",
      });
    }
  }

  // Text-only: guide user to use medical mode with an image
  return NextResponse.json({
    text: "Please enable Medical Analysis Mode and upload a medical image to use DiffuVQA.",
  });
}
