from typing import Dict, List
import os
import google.generativeai as genai

MODEL_NAME = "gemini-1.5-flash"


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def generate_quick_notes(text: str) -> Dict:
    model = _get_client()
    prompt = (
        "Create quick study notes from the text:\n"
        "1) 5 flashcards (Q/A)\n"
        "2) One-page cheat sheet\n"
        "3) 5 MCQs with answers\n"
        "4) 5 interview questions\n\n"
        f"Text:\n{text[:4000]}"
    )
    response = model.generate_content(prompt)
    return {"notes": response.text}
