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


def summarize_text_levels(text: str) -> Dict:
    model = _get_client()
    prompt = (
        "Summarize the text at three levels:\n"
        "1) TL;DR (1-2 sentences)\n"
        "2) Concept summary (3-5 bullets)\n"
        "3) Beginner-friendly (short paragraph)\n\n"
        f"Text:\n{text[:4000]}"
    )
    response = model.generate_content(prompt)
    return {"summary": response.text}


def summarize_sections(sections: List[Dict]) -> List[Dict]:
    model = _get_client()
    summaries: List[Dict] = []

    for section in sections:
        content = section.get("content", "")
        prompt = (
            "Summarize this section in 2-3 sentences."
            f"\n\n{content[:2000]}"
        )
        response = model.generate_content(prompt)
        summaries.append({
            "heading": section.get("heading"),
            "summary": response.text
        })

    return summaries
