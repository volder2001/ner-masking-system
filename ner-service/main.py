"""
ner-service/main.py
Production-ready NER: Словарь из JSON + Natasha (MorphVocab) + Yargy + Pymorphy3
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json
from pathlib import Path
import pymorphy3

# ПРАВИЛЬНЫЕ импорты Natasha
from natasha import MorphVocab, MoneyExtractor, DatesExtractor, NamesExtractor
# Yargy для поиска по словарю
from yargy import Parser
from yargy.pipelines import morph_pipeline

app = FastAPI(title="NER Service", version="1.0.0")

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ (Ваш правильный подход)
# ==========================================
# 1.1. Морфологический словарь Natasha (ОБЯЗАТЕЛЕН для экстракторов)
morph_vocab = MorphVocab()

# 1.2. Инициализируем экстракторы с morph_vocab
money_extractor = MoneyExtractor(morph_vocab)
date_extractor = DatesExtractor(morph_vocab)
names_extractor = NamesExtractor(morph_vocab) # Для поиска ФИО

# 1.3. Pymorphy3 оставляем для лемматизации слов из нашего кастомного словаря
morph = pymorphy3.MorphAnalyzer()

# 1.4. Загрузка нашего JSON-словаря
DICT_PATH = Path("/app/data/dictionary.json")
def load_dictionary():
    if DICT_PATH.exists():
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"ACT": ["акт", "акта", "актов"], "SUBJECT": ["квартира", "гараж"]}

dictionary = load_dictionary()

# Динамическое создание Yargy-пайплайнов из словаря
def build_pipeline(words):
    return Parser(morph_pipeline(words))

act_pipeline = build_pipeline(dictionary.get("ACT", []))
not_act_pipeline = build_pipeline(dictionary.get("NOT_ACT", []))
subject_pipeline = build_pipeline(dictionary.get("SUBJECT", []))

# ==========================================
# 2. МОДЕЛИ ДАННЫХ
# ==========================================
class Entity(BaseModel):
    text: str
    normal_form: str
    type: str         # DATE, MONEY, NAME, ACT, NOT_ACT, SUBJECT
    start_pos: int
    end_pos: int
    confidence: float = 1.0

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: List[Entity]
    act_money_pairs: List[Dict]

# ==========================================
# 3. ЛОГИКА ИЗВЛЕЧЕНИЯ
# ==========================================
def extract_entities(text: str) -> List[Entity]:
    entities = []

    def add_entity(word: str, entity_type: str, start: int, end: int, conf: float):
        normal = morph.parse(word)[0].normal_form
        entities.append(Entity(
            text=word, normal_form=normal, type=entity_type,
            start_pos=start, end_pos=end, confidence=conf
        ))

    # 1. ДЕНЬГИ (Ваш метод: проверка на RUB)
    for match in money_extractor(text):
        if hasattr(match.fact, 'currency') and match.fact.currency == 'RUB':
            word = text[match.start:match.stop]
            add_entity(word, 'MONEY', match.start, match.stop, 0.95)

    # 2. ДАТЫ
    for match in date_extractor(text):
        word = text[match.start:match.stop]
        add_entity(word, 'DATE', match.start, match.stop, 0.95)

    # 3. ИМЕНА (ФИО)
    for match in names_extractor(text):
        word = text[match.start:match.stop]
        add_entity(word, 'NAME', match.start, match.stop, 0.90)

    # 4. СУЩНОСТИ ИЗ СЛОВАРЯ (Yargy)
    for match in act_pipeline.findall(text):
        word = text[match.span.start:match.span.stop]
        add_entity(word, 'ACT', match.span.start, match.span.stop, 0.95)

    for match in not_act_pipeline.findall(text):
        word = text[match.span.start:match.span.stop]
        add_entity(word, 'NOT_ACT', match.span.start, match.span.stop, 0.95)

    for match in subject_pipeline.findall(text):
        word = text[match.span.start:match.span.stop]
        add_entity(word, 'SUBJECT', match.span.start, match.span.stop, 0.85)

    # Удаляем дубликаты по позициям
    unique_entities = {}
    for e in entities:
        key = (e.start_pos, e.end_pos)
        if key not in unique_entities:
            unique_entities[key] = e

    return list(unique_entities.values())

def link_act_money_pairs(entities: List[Entity], text: str) -> List[Dict]:
    pairs = []
    sentences = text.replace('\n', ' ').split('.')
    act_entities = [e for e in entities if e.type in ['ACT', 'NOT_ACT']]
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

            if (act_idx == money_idx and money.start_pos > act.start_pos and
                (money.start_pos - act.end_pos) < 150):
                pairs.append({
                    'act': act.model_dump(),
                    'money': money.model_dump(),
                    'distance': money.start_pos - act.end_pos,
                    'context': text[act.start_pos:min(money.end_pos + 40, len(text))]
                })
                break
    return pairs

# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "ner-service",
        "version": "1.0.0",
        "dict_loaded": len(dictionary)
    }

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