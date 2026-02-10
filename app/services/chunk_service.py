def chunk_sections(sections, document_id):
    chunks = []
    chunk_id = 0

    for section in sections:
        paragraphs = section["content"].split(". ")

        for para in paragraphs:
            if len(para.strip()) < 30:
                continue

            chunks.append({
                "id": f"{document_id}_chunk_{chunk_id}",
                "text": para.strip(),
                "heading": section["heading"],
                "document_id": document_id,
                "page": section.get("page"),
                "discourse_type": section.get("discourse_type", "unknown"),
                "difficulty": section.get("difficulty", "unknown")
            })
            chunk_id += 1

    return chunks
