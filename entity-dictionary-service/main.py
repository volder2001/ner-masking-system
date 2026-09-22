from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
from typing import List, Dict
import redis
import json

app = FastAPI(title="Entity Dictionary Service")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/4")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

DICTIONARY_PATH = os.getenv("DICTIONARY_PATH", "/app/dictionaries/entities.txt")

class Entity(BaseModel):
    prefix: str
    definition: str

class EntityDictionary:
    def __init__(self):
        self.entities: Dict[str, List[str]] = {}
        self.load_dictionary()
    
    def load_dictionary(self):
        if not os.path.exists(DICTIONARY_PATH):
            os.makedirs(os.path.dirname(DICTIONARY_PATH), exist_ok=True)
            with open(DICTIONARY_PATH, 'w', encoding='utf-8') as f:
                f.write("# Справочник сущностей\n# Формат: PREFIX->definition\n")
            return
        
        with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '->' in line:
                    prefix, definition = line.split('->', 1)
                    prefix = prefix.strip()
                    definition = definition.strip()
                    if prefix not in self.entities:
                        self.entities[prefix] = []
                    if definition not in self.entities[prefix]:
                        self.entities[prefix].append(definition)
        self._cache_to_redis()
    
    def _cache_to_redis(self):
        try:
            redis_client.set("entity_dictionary", json.dumps(self.entities))
        except:
            pass
    
    def get_all_entities(self) -> Dict[str, List[str]]:
        try:
            cached = redis_client.get("entity_dictionary")
            if cached:
                return json.loads(cached)
        except:
            pass
        return self.entities
    
    def add_entity(self, prefix: str, definition: str):
        if prefix not in self.entities:
            self.entities[prefix] = []
        if definition not in self.entities[prefix]:
            self.entities[prefix].append(definition)
            with open(DICTIONARY_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{prefix}->{definition}\n")
            self._cache_to_redis()

dictionary = EntityDictionary()

@app.get("/")
async def root():
    return {"service": "Entity Dictionary Service", "status": "running"}

@app.get("/entities")
async def get_entities():
    return dictionary.get_all_entities()

@app.post("/entities")
async def add_entity(entity: Entity):
    dictionary.add_entity(entity.prefix, entity.definition)
    return {"status": "success", "entity": {"prefix": entity.prefix, "definition": entity.definition}}

@app.post("/entities/upload")
async def upload_dictionary(file: UploadFile = File(...)):
    content = await file.read()
    with open(DICTIONARY_PATH, 'wb') as f:
        f.write(content)
    dictionary.load_dictionary()
    return {"status": "success", "message": "Dictionary uploaded and reloaded"}
