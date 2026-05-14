import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";

const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));

// 1. Protokol Koruması: Geçmiş her zaman 'user' ile başlamalı
function cleanHistoryForChat(history: any[]): any[] {
  if (!history || history.length === 0) return [];
  
  let cleaned = [...history];
  
  // Gemini 'user' ile başlanmasını şart koşar
  while (cleaned.length > 0 && cleaned[0]?.role !== "user") {
    console.warn(`⚠️ Protokol düzeltildi: '${cleaned[0]?.role}' mesajı atlanıyor.`);
    cleaned.shift();
  }
  
  return cleaned;
}

export async function POST(req: Request) {
  const apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY?.trim();
  if (!apiKey) return NextResponse.json({ error: "Key missing" }, { status: 500 });

  const genAI = new GoogleGenerativeAI(apiKey);
  
  try {
    const { message, history } = await req.json();

    // 2. KRİTİK: Temizlenmiş geçmişi kullanıyoruz
    const finalHistory = cleanHistoryForChat(history);

    const MAX_RETRIES = 3;
    let lastError: any = null;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        // İsmi senin terminal listendeki tam haliyle kullanıyoruz
        const model = genAI.getGenerativeModel({ model: "models/gemini-flash-latest" });
        
        const chat = model.startChat({
          history: finalHistory,
        });

        const result = await chat.sendMessage(message);
        const response = await result.response;
        
        return NextResponse.json({ text: response.text() });

      } catch (error: any) {
        lastError = error;
        // 429 (Kota) veya 503 (Meşgul) ise bekle ve tekrar dene
        if ((error.message?.includes("429") || error.message?.includes("503")) && attempt < MAX_RETRIES) {
          const waitTime = attempt * 1500; // Biraz daha uzun bekleme
          console.warn(`🔄 Kota/Yoğunluk hatası. Deneme ${attempt}, ${waitTime}ms bekleniyor...`);
          await delay(waitTime);
          continue;
        }
        break; 
      }
    }

    throw lastError;

  } catch (error: any) {
    console.error("❌ Gemini Final Error:", error.message);
    
    // Hata tipine göre tutarlı status döndürme
    let status = 500;
    if (error.message?.includes("429")) status = 429;
    if (error.message?.includes("404")) status = 404;

    return NextResponse.json(
      { error: "Şu an bağlantı kurulamadı, lütfen bir saniye sonra tekrar dene." },
      { status }
    );
  }
}