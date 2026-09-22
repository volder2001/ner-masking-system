from fastapi import FastAPI

app = FastAPI(title="Masking Service")

@app.get("/")
async def root():
    return {"service": "Masking Service", "status": "running"}

@app.post("/mask")
async def mask_document(request: dict):
    return {"masked_document_id": "test-id", "status": "success"}
