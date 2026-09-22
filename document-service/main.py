from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Document Service")

@app.get("/")
async def root():
    return {"service": "Document Service", "status": "running"}

@app.post("/process")
async def process_document(document_id: str, file: UploadFile = File(...)):
    return {"document_id": document_id, "status": "received"}
