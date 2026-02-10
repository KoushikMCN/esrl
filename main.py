from fastapi import FastAPI, UploadFile, File
from app.services.pdf_service import save_pdf, extract_text_from_pdf, extract_images_from_pdf, generate_document_id
from app.services.text_processing_service import clean_text, structure_pages
from app.services.discourse_service import classify_discourse
from app.services.chunk_service import chunk_sections
from app.services.embedding_service import upsert_chunks, query_similar
from app.services.rag_service import generate_answer
from app.services.notes_service import generate_quick_notes

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    path = await save_pdf(file)

    document_id = generate_document_id(path)
    full_text, pages_text = extract_text_from_pdf(path)
    cleaned = clean_text(full_text)
    sections = structure_pages(pages_text)
    sections = classify_discourse(sections)

    for section in sections:
        section["document_id"] = document_id

    chunks = chunk_sections(sections, document_id)
    upsert_chunks(chunks)

    images = extract_images_from_pdf(path, document_id)

    return {
        "message": "PDF processed",
        "document_id": document_id,
        "characters_extracted": len(cleaned),
        "chunks": len(chunks),
        "images": len(images)
    }


@app.post("/rag")
async def rag_query(payload: dict):
    query = payload.get("query", "")
    context = query_similar(query, top_k=5)
    answer = generate_answer(query, context)
    return {"answer": answer, "context": context}


@app.post("/notes")
async def notes_query(payload: dict):
    text = payload.get("text", "")
    notes = generate_quick_notes(text)
    return notes
