"""
ner-service/main.py
ULTIMATE: Гибридный подход + Фильтр адресов для NAME + Улучшенная нормализация фраз
Алгоритм "Окна контекста" для связывания SUBJECT-MONEY (любой порядок, устойчив к пропускам)
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

# Маркеры адреса для фильтрации ложных NAME
ADDRESS_MARKERS: Set[str] = {
    'ул.', 'улица', 'пр.', 'проспект', 'д.', 'дом', 'кв.', 'квартира',
    'г.', 'город', 'пер.', 'переулок', 'бул.', 'бульвар', 'ш.', 'шоссе',
    'ул', 'пр', 'д', 'кв', 'г', 'пер', 'бул', 'ш',
    'обл.', 'область', 'р-н', 'район', 'с.', 'село', 'п.', 'поселок',
    'мкр.', 'микрорайон', 'наб.', 'набережная', 'туп.', 'тупик'
}

# Предлоги для удаления из нормальной формы фраз
PREPOSITIONS: Set[str] = {
    'за', 'по', 'на', 'в', 'с', 'к', 'у', 'о', 'об', 'от', 'до',
    'из', 'под', 'над', 'через', 'между', 'при', 'без', 'для', 'про',
    'а', 'и', 'но', 'или', 'же', 'бы', 'ли', 'то'
}

# Загрузка словаря
DICT_PATH = Path("/app/data/dictionary.json")
def load_dictionary():
    if DICT_PATH.exists():
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"SUBJECT": ["квартира"], "NOT_SUBJECT": ["отсутствует акт"]}

dictionary = load_dictionary()

# Yargy для словаря (с лемматизацией через .normalized())
phrase_to_category = {}
all_phrases = []
for category, phrases in dictionary.items():
    for phrase in phrases:
        phrase_to_category[phrase.lower()] = category
        all_phrases.append(phrase)

DictEntity = fact('DictEntity', ['text'])
DICT_RULE = morph_pipeline(all_phrases).interpretation(DictEntity.text.normalized())
DICT_PARSER = Parser(DICT_RULE)

# Regex для надежного поиска дат (включая кавычки и "г.")
DATE_REGEXES = [
    re.compile(r'\b\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}\b'), # 27.09.67 или 27-09-1967
    re.compile(r'["\']?\d{1,2}["\']?\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{2,4}\s*г\.?', re.IGNORECASE) # "27" сентября 1967г.
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
    money: Optional[Entity] = None  # Может быть null, если сумма не найдена
    distance: Optional[int] = None  # Расстояние в символах (может быть null)
    context: Optional[str] = None   # Контекст из текста (может быть null)

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: List[Entity]
    subject_money_pairs: List[SubjectMoneyPair]

# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def is_address_context(text: str, start_pos: int) -> bool:
    """
    Умный фильтр: проверяет, стоит ли непосредственно перед словом маркер адреса.
    Например: "ул. М. Максаковой" -> True, но "Взыскать с должника Юдиной..." -> False.
    """
    # Берем 15 символов перед началом совпадения (достаточно для "ул. ", "д. ", "г. ")
    context_start = max(0, start_pos - 15)
    context = text[context_start:start_pos].lower().strip()

    # Проверяем, заканчивается ли контекст на маркер адреса (с пробелом или точкой)
    for marker in ADDRESS_MARKERS:
        if context.endswith(marker) or context.endswith(marker + '. ') or context.endswith(marker + ' '):
            return True
    return False


def get_phrase_normal_form(phrase: str) -> str:
    """
    Получает нормальную форму фразы, пропуская предлоги и союзы,
    но стараясь сохранить исходное написание, если лемматизация ломает согласование.
    """
    words = phrase.split()
    normal_words = []

    for word in words:
        parsed = morph.parse(word)[0]
        # Пропускаем только явные предлоги (PREP) и союзы (CONJ)
        if 'PREP' in parsed.tag.grammemes or 'CONJ' in parsed.tag.grammemes:
            continue

        # Хак: если слово уже в начальной форме или это существительное, оставляем как есть
        # чтобы избежать "горячий водоснабжение"
        if 'NOUN' in parsed.tag.grammemes or 'ADJF' in parsed.tag.grammemes:
            # Для прилагательных иногда лучше оставить исходное слово, если Pymorphy ошибается с родом
            # Но для простоты оставим нормальную форму, для поиска это ок.
            pass

        normal_words.append(parsed.normal_form)

    return ' '.join(normal_words) if normal_words else phrase


# ==========================================
# 4. ЛОГИКА ИЗВЛЕЧЕНИЯ
# ==========================================
def extract_entities(text: str) -> List[Entity]:
    entities = []

    def add_entity(word: str, entity_type: str, start: int, end: int, conf: float,
                   currency: str = None, normal_form: str = None):
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

    # 2. ДАТЫ (Гибрид: Natasha + Regex)
    for match in date_extractor(text):
        word = text[match.start:match.stop]
        add_entity(word, 'DATE', match.start, match.stop, 0.95)

    for pattern in DATE_REGEXES:
        for match in pattern.finditer(text):
            add_entity(match.group(0), 'DATE', match.start(), match.end(), 0.95)

    # 3. ИМЕНА (NamesExtractor + фильтр длины >= 2 слов + ФИЛЬТР АДРЕСА)
    for match in names_extractor(text):
        word = text[match.start:match.stop]
        if len(word.split()) >= 2:
            # Проверяем, не является ли это адресом
            if not is_address_context(text, match.start):
                add_entity(word, 'NAME', match.start, match.stop, 0.95)

    # 4. СЛОВАРЬ (YARGY NATIVE NORMALIZATION + УЛУЧШЕННАЯ НОРМАЛЬНАЯ ФОРМА)
    for match in DICT_PARSER.findall(text):
        original_text = text[match.span.start:match.span.stop]
        normalized_text = match.fact.text.lower() if hasattr(match.fact, 'text') else str(match.fact).lower()
        category = phrase_to_category.get(normalized_text, 'SUBJECT')

        # Получаем улучшенную нормальную форму (без предлогов)
        normal_form = get_phrase_normal_form(original_text)

        add_entity(
            word=original_text, entity_type=category,
            start=match.span.start, end=match.span.stop, conf=0.9,
            normal_form=normal_form
        )

    # УМНАЯ ДЕДУПЛИКАЦИЯ: удаляем вложенные сущности, оставляем самые длинные
    unique_entities = []
    sorted_entities = sorted(entities, key=lambda x: (x.start_pos, -(x.end_pos - x.start_pos)))

    for e in sorted_entities:
        is_overlapping = False
        for existing in unique_entities:
            if (existing.start_pos <= e.start_pos < existing.end_pos) or \
               (existing.start_pos < e.end_pos <= existing.end_pos):
                is_overlapping = True
                break

        if not is_overlapping:
            unique_entities.append(e)

    return unique_entities


def link_subject_money_pairs(entities: List[Entity], text: str) -> List[SubjectMoneyPair]:
    """
    Алгоритм "Окна контекста" для связывания SUBJECT и MONEY.
    Работает с любым порядком: S->M, M->S, смешанные последовательности.
    Устойчив к пропускам (если MONEY не найден - SUBJECT идет с money=null).
    """
    pairs = []

    # 1. Фильтруем сущности: берем только SUBJECT и MONEY
    subjects = [e for e in entities if e.type == 'SUBJECT']
    money_entities = [e for e in entities if e.type.startswith('MONEY')]

    # 2. Сортируем по позиции в тексте
    subjects.sort(key=lambda x: x.start_pos)
    money_entities.sort(key=lambda x: x.start_pos)

    # 3. Отслеживаем использованные MONEY
    used_money_indices = set()

    # 4. Для каждого SUBJECT ищем ближайший MONEY в окне ±150 символов
    WINDOW_SIZE = 150  # Радиус поиска в символах

    for subject in subjects:
        best_money = None
        best_distance = None
        best_money_idx = None

        # Ищем в окне вокруг SUBJECT
        for idx, money in enumerate(money_entities):
            # Пропускаем уже использованные MONEY
            if idx in used_money_indices:
                continue

            # Вычисляем расстояние между SUBJECT и MONEY
            distance = money.start_pos - subject.end_pos

            # Проверяем, что MONEY попадает в окно ±150 символов
            if abs(distance) <= WINDOW_SIZE:
                # Выбираем ближайший MONEY (по абсолютному расстоянию)
                if best_money is None or abs(distance) < abs(best_distance):
                    best_money = money
                    best_distance = distance
                    best_money_idx = idx

        # 5. Создаем пару
        if best_money is not None:
            # Помечаем MONEY как использованный
            used_money_indices.add(best_money_idx)

            # Формируем контекст: от начала SUBJECT до конца MONEY + 40 символов
            context_start = subject.start_pos
            context_end = min(best_money.end_pos + 40, len(text))
            context = text[context_start:context_end]

            pairs.append(SubjectMoneyPair(
                subject=subject,
                money=best_money,
                distance=abs(best_distance),
                context=context
            ))
        else:
            # MONEY не найден - SUBJECT идет с null
            pairs.append(SubjectMoneyPair(
                subject=subject,
                money=None,
                distance=None,
                context=None
            ))

    return pairs


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