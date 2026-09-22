"""
ner-service/main.py
Production-ready NER-сервис с лемматизацией (Natasha NER + Yargy + Pymorphy3)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pymorphy3

# Правильные импорты Natasha
from natasha import (
    Segmenter, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, NewsEmbedding, Doc
)
# Yargy для морфологического поиска
from yargy import Parser
from yargy.pipelines import morph_pipeline

app = FastAPI(title="NER Service", version="1.0.0")

# Инициализируем морфологический анализатор (лемматизация)
morph = pymorphy3.MorphAnalyzer()

# Инициализируем пайплайн Natasha
embeddings = NewsEmbedding()
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(embeddings)
syntax_parser = NewsSyntaxParser(embeddings)
ner_tagger = NewsNERTagger(embeddings)

# Yargy пайплайны
act_pipeline = Parser(morph_pipeline([
    'акт', 'акта', 'акту', 'актом', 'акте', 'акты', 'актов'
]))

subject_pipeline = Parser(morph_pipeline([
    'квартира', 'автомобиль', 'машина', 'дом', 'земля', 'участок',
    'гараж', 'дача', 'офис', 'помещение'
]))

# ==========================================
# МОДЕЛИ ДАННЫХ
# ==========================================
class Entity(BaseModel):
    text: str
    normal_form: str  # Нормальная форма слова (лемма)
    type: str         # DATE, MONEY, ACT, SUBJECT
    start_pos: int
    end_pos: int
    confidence: float = 1.0

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: List[Entity]
    act_money_pairs: List[Dict]

# ==========================================
# ЛОГИКА ИЗВЛЕЧЕНИЯ
# ==========================================
def extract_entities(text: str) -> List[Entity]:
    entities = []

    # 1. Natasha NER (находит DATE и MONEY)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    for span in doc.spans:
        if span.type in ['DATE', 'MONEY']:
            word = span.text
            normal = morph.parse(word)[0].normal_form
            entities.append(Entity(
                text=word, normal_form=normal, type=span.type,
                start_pos=span.start, end_pos=span.stop, confidence=0.95
            ))

    # 2. Yargy (находит ACT и SUBJECT)
    for match in act_pipeline.findall(text):
        word = text[match.span.start:match.span.stop]
        normal = morph.parse(word)[0].normal_form
        entities.append(Entity(
            text=word, normal_form=normal, type='ACT',
            start_pos=match.span.start, end_pos=match.span.stop, confidence=0.95
        ))

    for match in subject_pipeline.findall(text):
        word = text[match.span.start:match.span.stop]
        normal = morph.parse(word)[0].normal_form
        entities.append(Entity(
            text=word, normal_form=normal, type='SUBJECT',
            start_pos=match.span.start, end_pos=match.span.stop, confidence=0.85
        ))

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