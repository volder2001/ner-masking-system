"""
ner-service/main.py
Полноценный NER-сервис для извлечения сущностей (Natasha + Yargy).
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from pathlib import Path

# NER библиотеки
from natasha import (
    Segmenter,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    NewsEmbedding, Doc
)
from yargy import Parser, or_
from yargy.predicates import eq  # <-- eq перенесли сюда
from yargy.pipelines import morph_pipeline

app = FastAPI(title="NER Service", version="1.0.0")


# ==========================================
# МОДЕЛИ ДАННЫХ
# ==========================================
class Entity(BaseModel):
    text: str
    type: str  # PERSON, DATE, MONEY, ACT, NOT_ACT, SUM, SUBJECT
    start_pos: int
    end_pos: int
    confidence: float = 1.0


class NERRequest(BaseModel):
    text: str
    dictionary_path: Optional[str] = "/app/data/dictionary.json"


class NERResponse(BaseModel):
    entities: List[Entity]
    act_money_pairs: List[Dict]


# ==========================================
# ИНИЦИАЛИЗАЦИЯ NATASHA
# ==========================================
embeddings = NewsEmbedding()
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(embeddings)
syntax_parser = NewsSyntaxParser(embeddings)
ner_tagger = NewsNERTagger(embeddings)


def extract_with_natasha(text: str) -> List[Entity]:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    entities = []
    for span in doc.spans:
        if span.type in ['PER', 'DATE', 'MONEY']:
            entities.append(Entity(
                text=span.text, type=span.type,
                start_pos=span.start, end_pos=span.stop, confidence=0.9
            ))
    return entities


# ==========================================
# YARGY: ПОИСК ПО СЛОВАРЮ И МОРФОЛОГИИ
# ==========================================
def extract_with_yargy(text: str, dictionary: dict) -> List[Entity]:
    entities = []
    for entity_type, variants in dictionary.items():
        rule_pattern = or_(*[eq(variant.lower()) for variant in variants]).label(entity_type)
        parser = Parser(rule_pattern)
        for match in parser.finditer(text.lower()):
            entities.append(Entity(
                text=match.text, type=entity_type,
                start_pos=match.start, end_pos=match.start + len(match.text), confidence=0.95
            ))
    return entities


def extract_subjects_morph(text: str) -> List[Entity]:
    subjects = ['квартира', 'автомобиль', 'машина', 'дом', 'земля', 'участок', 'гараж', 'дача', 'офис']
    parser = Parser(morph_pipeline(subjects).label('SUBJECT'))
    entities = []
    for match in parser.finditer(text):
        entities.append(Entity(
            text=match.text, type='SUBJECT',
            start_pos=match.start, end_pos=match.start + len(match.text), confidence=0.85
        ))
    return entities


# ==========================================
# СВЯЗЫВАНИЕ ACT-MONEY
# ==========================================
def link_act_money_pairs(entities: List[Entity], text: str) -> List[Dict]:
    pairs = []
    sentences = text.replace('\n', ' ').split('.')
    act_entities = [e for e in entities if e.type in ['ACT', 'NOT_ACT']]
    money_entities = [e for e in entities if e.type in ['MONEY', 'SUM']]

    for act in act_entities:
        for money in money_entities:
            act_idx = money_idx = None
            current_pos = 0
            for idx, sentence in enumerate(sentences):
                start, end = current_pos, current_pos + len(sentence)
                if start <= act.start_pos < end: act_idx = idx
                if start <= money.start_pos < end: money_idx = idx
                current_pos = end + 1

            if act_idx == money_idx and money.start_pos > act.start_pos and (money.start_pos - act.end_pos) < 100:
                pairs.append({
                    'act': act.dict(), 'money': money.dict(),
                    'distance': money.start_pos - act.end_pos,
                    'context': text[act.start_pos:min(money.end_pos + 30, len(text))]
                })
                break
    return pairs


# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {"status": "ok", "service": "ner-service", "version": "1.0.0"}


@app.post("/extract", response_model=NERResponse)
def extract_entities(request: NERRequest):
    try:
        natasha_entities = extract_with_natasha(request.text)

        dictionary = {}
        if request.dictionary_path and Path(request.dictionary_path).exists():
            with open(request.dictionary_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)

        dict_entities = extract_with_yargy(request.text, dictionary)
        subject_entities = extract_subjects_morph(request.text)

        # Объединяем и удаляем дубликаты по позиции
        all_entities = {}
        for entity in natasha_entities + dict_entities + subject_entities:
            key = (entity.start_pos, entity.end_pos)
            if key not in all_entities:
                all_entities[key] = entity

        entities_list = list(all_entities.values())
        act_money_pairs = link_act_money_pairs(entities_list, request.text)

        return NERResponse(entities=entities_list, act_money_pairs=act_money_pairs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)