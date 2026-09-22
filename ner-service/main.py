"""
ner-service/main.py
Production-ready NER-сервис (Natasha + Yargy)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# Natasha экстракторы
from natasha import DateExtractor, MoneyExtractor
# Yargy для морфологического поиска
from yargy import Parser
from yargy.pipelines import morph_pipeline

app = FastAPI(title="NER Service", version="1.0.0")

# ==========================================
# МОДЕЛИ ДАННЫХ
# ==========================================
class Entity(BaseModel):
    text: str
    type: str  # DATE, MONEY, ACT, SUBJECT
    start_pos: int
    end_pos: int
    confidence: float = 1.0

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: List[Entity]
    act_money_pairs: List[Dict]

# ==========================================
# ИНИЦИАЛИЗАЦИЯ ЭКСТРАКТОРОВ
# ==========================================
date_extractor = DateExtractor()
money_extractor = MoneyExtractor()

# Yargy пайплайны (находят слова в любой морфологической форме)
act_pipeline = Parser(morph_pipeline([
    'акт', 'акта', 'акту', 'актом', 'акте', 'акты', 'актов'
]))

subject_pipeline = Parser(morph_pipeline([
    'квартира', 'автомобиль', 'машина', 'дом', 'земля', 'участок',
    'гараж', 'дача', 'офис', 'помещение'
]))

# ==========================================
# ЛОГИКА ИЗВЛЕЧЕНИЯ
# ==========================================
def extract_entities(text: str) -> List[Entity]:
    entities = []

    # 1. Даты (Natasha)
    for match in date_extractor(text):
        entities.append(Entity(
            text=match.text, type='DATE',
            start_pos=match.start, end_pos=match.stop, confidence=0.95
        ))

    # 2. Деньги (Natasha)
    for match in money_extractor(text):
        entities.append(Entity(
            text=match.text, type='MONEY',
            start_pos=match.start, end_pos=match.stop, confidence=0.95
        ))

    # 3. Акты (Yargy)
    for match in act_pipeline.findall(text):
        entities.append(Entity(
            text=text[match.span.start:match.span.stop], type='ACT',
            start_pos=match.span.start, end_pos=match.span.stop, confidence=0.95
        ))

    # 4. Предметы взыскания (Yargy)
    for match in subject_pipeline.findall(text):
        entities.append(Entity(
            text=text[match.span.start:match.span.stop], type='SUBJECT',
            start_pos=match.span.start, end_pos=match.span.stop, confidence=0.85
        ))

    # Удаляем дубликаты по позициям (если Natasha и Yargy нашли одно и то же)
    unique_entities = {}
    for e in entities:
        key = (e.start_pos, e.end_pos)
        if key not in unique_entities:
            unique_entities[key] = e

    return list(unique_entities.values())

def link_act_money_pairs(entities: List[Entity], text: str) -> List[Dict]:
    pairs = []
    sentences = text.replace('\n', ' ').split('.')
    act_entities = [e for e in entities if e.type == 'ACT']
    money_entities = [e for e in entities if e.type == 'MONEY']

    for act in act_entities:
        for money in money_entities:
            act_idx = money_idx = None
            current_pos = 0
            for idx, sentence in enumerate(sentences):
                start, end = current_pos, current_pos + len(sentence)
                if start <= act.start_pos < end: act_idx = idx
                if start <= money.start_pos < end: money_idx = idx
                current_pos = end + 1

            # Если в одном предложении и деньги идут после акта (не дальше 100 символов)
            if (act_idx == money_idx and money.start_pos > act.start_pos and
                (money.start_pos - act.end_pos) < 100):
                pairs.append({
                    'act': act.model_dump(),
                    'money': money.model_dump(),
                    'distance': money.start_pos - act.end_pos,
                    'context': text[act.start_pos:min(money.end_pos + 30, len(text))]
                })
                break # Берем только ближайшее MONEY
    return pairs

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {"status": "ok", "service": "ner-service", "version": "1.0.0"}

@app.post("/extract", response_model=NERResponse)
def extract(request: NERRequest):
    try:
        entities = extract_entities(request.text)
        act_money_pairs = link_act_money_pairs(entities, request.text)
        return NERResponse(entities=entities, act_money_pairs=act_money_pairs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)