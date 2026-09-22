from fastapi import FastAPI

app = FastAPI(title="NER Service")

@app.get("/")
async def root():
    return {"service": "NER Service", "status": "running"}

@app.get("/entities/{document_id}")
async def get_entities(document_id: str):
    return {"document_id": document_id, "entities": []}
