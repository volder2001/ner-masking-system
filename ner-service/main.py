"""
ner-service/main.py
Production-ready NER: Строгие имена (Yargy) + Валюта + Словарь (Yargy morph_pipeline)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from pathlib import Path
import pymorphy3

from natasha import MorphVocab, MoneyExtractor, DatesExtractor
from yargy import Parser, rule, or_
from yargy.pipelines import morph_pipeline
from yargy.predicates import gram

app = FastAPI(title="NER Service", version="1.0.0")

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ
# ==========================================
morph_vocab = MorphVocab()
money_extractor = MoneyExtractor(morph_vocab)
date_extractor = DatesExtractor(morph_vocab)
morph = pymorphy3.MorphAnalyzer()

# --- СТРОГИЙ ПАРСЕР ИМЕН (Yargy) ---
RULE_FIO = rule(gram('Surn'), gram('Name'), gram('Patr'))
RULE_FIO_INIT = rule(gram('Surn'), gram('Abbr'), gram('Abbr'))
RULE_IOF = rule(gram('Name'), gram('Patr'), gram('Surn'))
NAME_PARSER = Parser(or_(RULE_FIO, RULE_FIO_INIT, RULE_IOF))

# Загрузка словаря
DICT_PATH = Path("/app/data/dictionary.json")
def load_dictionary():
    if DICT_PATH.exists():
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"ACT": ["акт"], "SUBJECT": ["квартира"]}

dictionary = load_dictionary()

# Yargy morph_pipeline НАТУРАЛЬНО поддерживает словосочетания!
# Он сам разобьет "кадастровый номер" и просклоняет каждое слово.
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
    type: str
    currency: Optional[str] = None
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

    def add_entity(word: str, entity_type: str, start: int, end: int, conf: float, currency: str = None):
        normal = word if entity_type.startswith('MONEY') or entity_type == 'DATE' else morph.parse(word)[0].normal_form
        entities.append(Entity(
            text=word, normal_form=normal, type=entity_type,
            currency=currency, start_pos=start, end_pos=end, confidence=conf
        ))

    # 1. ДЕНЬГИ
    for match in money_extractor(text):
        if hasattr(match.fact, 'currency') and match.fact.currency:
            add_entity(text[match.start:match.stop], f"MONEY_{match.fact.currency}", match.start, match.stop, 0.95, currency=match.fact.currency)

    # 2. ДАТЫ
    for match in date_extractor(text):
        add_entity(text[match.start:match.stop], 'DATE', match.start, match.stop, 0.95)

    # 3. ИМЕНА (СТРОГО ФИО / Ф.И.О. / И.О.Ф.)
    for match in NAME_PARSER.findall(text):
        add_entity(text[match.span.start:match.span.stop], 'NAME', match.span.start, match.span.stop, 0.95)

    # 4. СЛОВАРЬ (Yargy morph_pipeline - нативная поддержка словосочетаний)
    for match in act_pipeline.findall(text):
        add_entity(text[match.span.start:match.span.stop], 'ACT', match.span.start, match.span.stop, 0.95)

    for match in not_act_pipeline.findall(text):
        add_entity(text[match.span.start:match.span.stop], 'NOT_ACT', match.span.start, match.span.stop, 0.95)

    for match in subject_pipeline.findall(text):
        add_entity(text[match.span.start:match.span.stop], 'SUBJECT', match.span.start, match.span.stop, 0.85)

    # Дедупликация: более длинные совпадения (биграммы) имеют приоритет над короткими
    unique_entities = {}
    for e in sorted(entities, key=lambda x: (x.end_pos - x.start_pos), reverse=True):
        covered = False
        for start in range(e.start_pos, e.end_pos):
            if start in unique_entities:
                covered = True
                break
        if not covered:
            for start in range(e.start_pos, e.end_pos):
                unique_entities[start] = e

    return list(unique_entities.values())

def link_act_money_pairs(entities: List[Entity], text: str) -> List[Dict]:
    pairs = []
    sentences = text.replace('\n', ' ').split('.')
    act_entities = [e for e in entities if e.type in ['ACT', 'NOT_ACT']]
    money_entities = [e for e in entities if e.type.startswith('MONEY')]

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
    return {"status": "ok", "service": "ner-service", "dict_loaded": len(dictionary)}

@app.post("/extract", response_model=NERResponse)
def extract(request: NERRequest):
    try:
        entities = extract_entities(request.text)
        return NERResponse(entities=entities, act_money_pairs=link_act_money_pairs(entities, request.text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)