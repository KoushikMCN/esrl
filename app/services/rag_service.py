from typing import Dict, List
import os
from google import genai

from app.services.embedding_service import query_similar

MODEL_NAME = "gemini-1.5-flash"


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def retrieve_context(query: str, top_k: int = 5) -> Dict:
    return query_similar(query, top_k=top_k)


def generate_answer(query: str, context: Dict) -> str:
    client = _get_client()
    documents: List[str] = context.get("documents", [[]])[0]
    prompt = (
        "Answer the question using only the context. If unsure, say you do not know.\n\n"
        "Context:\n"
        + "\n---\n".join(documents[:5])
        + "\n\nQuestion: "
        + query
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text
