"""
ner-service/main.py
FINAL PRODUCTION: Глобальная минимальная дистанция для пар + Фильтр латиницы в именах
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import json
import re
from pathlib import Path
import pymorphy3

from natasha import MorphVocab, MoneyExtractor, DatesExtractor, NamesExtractor
from yargy import Parser
from yargy.pipelines import morph_pipeline
from yargy.interpretation import fact

app = FastAPI(title="NER Service", version="1.0.0")

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ
# ==========================================
morph_vocab = MorphVocab()
money_extractor = MoneyExtractor(morph_vocab)
date_extractor = DatesExtractor(morph_vocab)
names_extractor = NamesExtractor(morph_vocab)
morph = pymorphy3.MorphAnalyzer()

ADDRESS_MARKERS: Set[str] = {
    'ул.', 'улица', 'пр.', 'проспект', 'д.', 'дом', 'кв.', 'квартира',
    'г.', 'город', 'пер.', 'переулок', 'бул.', 'бульвар', 'ш.', 'шоссе',
    'ул', 'пр', 'д', 'кв', 'г', 'пер', 'бул', 'ш',
    'обл.', 'область', 'р-н', 'район', 'с.', 'село', 'п.', 'поселок',
    'мкр.', 'микрорайон', 'наб.', 'набережная', 'туп.', 'тупик'
}

PREPOSITIONS: Set[str] = {
    'за', 'по', 'на', 'в', 'с', 'к', 'у', 'о', 'об', 'от', 'до',
    'из', 'под', 'над', 'через', 'между', 'при', 'без', 'для', 'про',
    'а', 'и', 'но', 'или', 'же', 'бы', 'ли', 'то'
}

DICT_PATH = Path("/app/data/dictionary.json")
def load_dictionary():
    if DICT_PATH.exists():
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"SUBJECT": ["квартира"], "NOT_SUBJECT": ["отсутствует акт"]}

dictionary = load_dictionary()

phrase_to_category = {}
all_phrases = []
for category, phrases in dictionary.items():
    for phrase in phrases:
        phrase_to_category[phrase.lower()] = category
        all_phrases.append(phrase)

DictEntity = fact('DictEntity', ['text'])
DICT_RULE = morph_pipeline(all_phrases).interpretation(DictEntity.text.normalized())
DICT_PARSER = Parser(DICT_RULE)

DATE_REGEXES = [
    re.compile(r'\b\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}\b'),
    re.compile(r'["\']?\d{1,2}["\']?\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{2,4}\s*г\.?', re.IGNORECASE)
]

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

class SubjectMoneyPair(BaseModel):
    subject: Entity
    money: Optional[Entity] = None
    distance: Optional[int] = None
    context: Optional[str] = None

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: List[Entity]
    subject_money_pairs: List[SubjectMoneyPair]

# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def is_address_context(text: str, start_pos: int) -> bool:
    context_start = max(0, start_pos - 15)
    context = text[context_start:start_pos].lower().strip()
    for marker in ADDRESS_MARKERS:
        if context.endswith(marker) or context.endswith(marker + '. ') or context.endswith(marker + ' '):
            return True
    return False

def get_phrase_normal_form(phrase: str) -> str:
    words = phrase.split()
    normal_words = []
    for word in words:
        parsed = morph.parse(word)[0]
        if 'PREP' in parsed.tag.grammemes or 'CONJ' in parsed.tag.grammemes:
            continue
        normal_words.append(parsed.normal_form)
    return ' '.join(normal_words) if normal_words else phrase

# ==========================================
# 4. ЛОГИКА ИЗВЛЕЧЕНИЯ
# ==========================================
def extract_entities(text: str) -> List[Entity]:
    entities = []

    def add_entity(word: str, entity_type: str, start: int, end: int, conf: float, currency: str = None, normal_form: str = None):
        if normal_form is None:
            normal_form = word if entity_type.startswith('MONEY') or entity_type == 'DATE' else morph.parse(word)[0].normal_form
        entities.append(Entity(
            text=word, normal_form=normal_form, type=entity_type,
            currency=currency, start_pos=start, end_pos=end, confidence=conf
        ))

    # 1. ДЕНЬГИ
    for match in money_extractor(text):
        if hasattr(match.fact, 'currency') and match.fact.currency:
            word = text[match.start:match.stop]
            add_entity(word, f"MONEY_{match.fact.currency}", match.start, match.stop, 0.95, currency=match.fact.currency)

    # 2. ДАТЫ
    for match in date_extractor(text):
        word = text[match.start:match.stop]
        add_entity(word, 'DATE', match.start, match.stop, 0.95)
    for pattern in DATE_REGEXES:
        for match in pattern.finditer(text):
            add_entity(match.group(0), 'DATE', match.start(), match.end(), 0.95)

    # 3. ИМЕНА (С ФИЛЬТРОМ АДРЕСА И ФИЛЬТРОМ ЛАТИНИЦЫ)
    for match in names_extractor(text):
        word = text[match.start:match.stop]
        if len(word.split()) >= 2:
            if not is_address_context(text, match.start):
                # НОВОЕ: Отбрасываем, если есть латинские буквы (защита от OCR-мусора типа "Qe Л.И.")
                if not any(char.isascii() and char.isalpha() for char in word):
                    add_entity(word, 'NAME', match.start, match.stop, 0.95)

    # 4. СЛОВАРЬ
    for match in DICT_PARSER.findall(text):
        original_text = text[match.span.start:match.span.stop]
        normalized_text = match.fact.text.lower() if hasattr(match.fact, 'text') else str(match.fact).lower()
        category = phrase_to_category.get(normalized_text, 'SUBJECT')
        normal_form = get_phrase_normal_form(original_text)
        add_entity(word=original_text, entity_type=category, start=match.span.start, end=match.span.stop, conf=0.9, normal_form=normal_form)

    # 5. ДЕДУПЛИКАЦИЯ
    unique_entities = []
    sorted_entities = sorted(entities, key=lambda x: (x.start_pos, -(x.end_pos - x.start_pos)))
    for e in sorted_entities:
        is_overlapping = False
        for existing in unique_entities:
            if (existing.start_pos <= e.start_pos < existing.end_pos) or (existing.start_pos < e.end_pos <= existing.end_pos):
                is_overlapping = True
                break
        if not is_overlapping:
            unique_entities.append(e)

    return unique_entities


def link_subject_money_pairs(entities: List[Entity], text: str) -> List[SubjectMoneyPair]:
    """
    Алгоритм "Глобальной минимальной дистанции".
    Отдает приоритет самым близким парам, решая проблему перехвата денег общими фразами.
    """
    subjects = [e for e in entities if e.type == 'SUBJECT']
    money_entities = [e for e in entities if e.type.startswith('MONEY')]
    WINDOW_SIZE = 150

    # 1. Генерируем все возможные валидные пары в пределах окна
    candidates = []
    for subj in subjects:
        for money in money_entities:
            distance = abs(money.start_pos - subj.end_pos)
            if distance <= WINDOW_SIZE:
                candidates.append({
                    'subject': subj,
                    'money': money,
                    'distance': distance
                })

    # 2. Сортируем кандидатов по дистанции (от самых близких к дальним)
    candidates.sort(key=lambda x: x['distance'])

    # 3. Жадно выбираем лучшие непересекающиеся пары
    used_subjects = set()
    used_moneys = set()
    final_pairs = []

    for candidate in candidates:
        subj_id = (candidate['subject'].start_pos, candidate['subject'].end_pos)
        money_id = (candidate['money'].start_pos, candidate['money'].end_pos)

        if subj_id not in used_subjects and money_id not in used_moneys:
            used_subjects.add(subj_id)
            used_moneys.add(money_id)

            context_start = candidate['subject'].start_pos
            context_end = min(candidate['money'].end_pos + 40, len(text))
            context = text[context_start:context_end]

            final_pairs.append(SubjectMoneyPair(
                subject=candidate['subject'],
                money=candidate['money'],
                distance=candidate['distance'],
                context=context
            ))

    # 4. Добавляем SUBJECT-ы, которые остались без пар (money = null)
    for subj in subjects:
        subj_id = (subj.start_pos, subj.end_pos)
        if subj_id not in used_subjects:
            final_pairs.append(SubjectMoneyPair(
                subject=subj,
                money=None,
                distance=None,
                context=None
            ))

    # Сортируем итоговые пары по порядку появления SUBJECT в тексте
    final_pairs.sort(key=lambda x: x.subject.start_pos)
    return final_pairs

# ==========================================
# 5. API ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "ner-service",
        "dict_loaded_total": len(dictionary),
        "phrases_in_pipeline": all_phrases
    }

@app.post("/extract", response_model=NERResponse)
def extract(request: NERRequest):
    try:
        entities = extract_entities(request.text)
        subject_money_pairs = link_subject_money_pairs(entities, request.text)
        return NERResponse(entities=entities, subject_money_pairs=subject_money_pairs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)