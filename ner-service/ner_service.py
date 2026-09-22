"""
ner-service/ner_service.py
Сервис для извлечения именованных сущностей (NER) из текста.
Использует Natasha для ФИО, дат, сумм и Yargy для предметов взыскания.
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
    NewsEmbeddings,
    Doc
)
from yargy import (
    Parser,
    rule,
    or_
)
from yargy.pipelines import morph_pipeline
from yargy.predicates import eq, in_

app = FastAPI(title="NER Service", version="1.0.0")

# ==========================================
# МОДЕЛИ ДАННЫХ
# ==========================================

class Entity(BaseModel):
    """Модель одной сущности"""
    text: str
    type: str  # ACT, NOT_ACT, SUM, PERSON, DATE, SUBJECT
    start_pos: int
    end_pos: int
    confidence: float = 1.0


class NERRequest(BaseModel):
    """Запрос на извлечение сущностей"""
    text: str
    dictionary_path: Optional[str] = None


class NERResponse(BaseModel):
    """Ответ с извлеченными сущностями"""
    entities: List[Entity]
    act_money_pairs: List[Dict]  # Связанные пары ACT-MONEY


# ==========================================
# ЗАГРУЗКА СЛОВАРЯ
# ==========================================

def load_dictionary(dictionary_path: str) -> Dict[str, List[str]]:
    """
    Загружает словарь сущностей из JSON файла.

    Ожидаемый формат:
    {
        "ACT": ["акт", "акта", "акту"],
        "NOT_ACT": ["не акт", "отказ"],
        "SUM": ["руб", "рублей", "тыс", "млн"],
        "SUBJECT": ["квартира", "автомобиль", "дом"]
    }
    """
    path = Path(dictionary_path)
    if not path.exists():
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==========================================
# ИНИЦИАЛИЗАЦИЯ NATASHA
# ==========================================

embeddings = NewsEmbeddings()
segmenter = Segmenter()
morph_tagger = NewsMorphTagger(embeddings)
syntax_parser = NewsSyntaxParser(embeddings)
ner_tagger = NewsNERTagger(embeddings)


def extract_with_natasha(text: str) -> List[Entity]:
    """
    Извлекает сущности с помощью Natasha:
    - PERSON (ФИО)
    - DATE (даты)
    - MONEY (суммы)
    """
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    entities = []

    for span in doc.spans:
        if span.type in ['PER', 'DATE', 'MONEY']:
            entities.append(Entity(
                text=span.text,
                type=span.type,
                start_pos=span.start,
                end_pos=span.stop,
                confidence=0.9  # Natasha обычно дает хорошую точность
            ))

    return entities


# ==========================================
# YARGY: ПОИСК ПО СЛОВАРЮ
# ==========================================

def create_dictionary_parser(dictionary: Dict[str, List[str]]) -> Parser:
    """
    Создает парсер Yargy для поиска по словарю.
    Поддерживает морфологические варианты.
    """
    rules = []

    for entity_type, variants in dictionary.items():
        # Создаем правило для каждого типа сущности
        rule_pattern = or_(
            *[eq(variant.lower()) for variant in variants]
        )
        rules.append(rule_pattern.label(entity_type))

    return or_(*rules)


def extract_with_yargy(text: str, dictionary: Dict[str, List[str]]) -> List[Entity]:
    """
    Извлекает сущности из текста с помощью Yargy по словарю.
    """
    if not dictionary:
        return []

    parser = create_dictionary_parser(dictionary)

    entities = []
    for match in parser.finditer(text.lower()):
        entities.append(Entity(
            text=match.text,
            type=match.type,
            start_pos=match.start,
            end_pos=match.start + len(match.text),
            confidence=0.95  # Точное совпадение со словарем
        ))

    return entities


# ==========================================
# YARGY: МОРФОЛОГИЧЕСКИЙ ПОИСК (ПРЕДМЕТЫ ВЗЫСКАНИЯ)
# ==========================================

def create_subject_pipeline() -> Parser:
    """
    Создает морфологический пайплайн для поиска предметов взыскания.
    Ищет слова в любой форме (именительный, родительный, множественное число и т.д.)
    """
    # Список базовых форм предметов
    subjects = [
        'квартира', 'автомобиль', 'машина', 'дом', 'земля', 'участок',
        'гараж', 'дача', 'комната', 'офис', 'помещение', 'склад'
    ]

    # morph_pipeline автоматически найдет все морфологические формы
    return morph_pipeline(subjects).label('SUBJECT')


def extract_subjects_morph(text: str) -> List[Entity]:
    """
    Извлекает предметы взыскания с учетом всех морфологических форм.
    Например: "квартира", "квартиры", "квартире", "квартирами" - все будут найдены.
    """
    parser = create_subject_pipeline()

    entities = []
    for match in parser.finditer(text):
        entities.append(Entity(
            text=match.text,
            type='SUBJECT',
            start_pos=match.start,
            end_pos=match.start + len(match.text),
            confidence=0.85  # Немного ниже из-за возможной омонимии
        ))

    return entities


# ==========================================
# СВЯЗЫВАНИЕ ACT-MONEY ПАРОЧЕК
# ==========================================

def link_act_money_pairs(entities: List[Entity], text: str) -> List[Dict]:
    """
    Связывает акты (ACT) с денежными суммами (MONEY/SUM).

    Логика связывания:
    1. Ищем ACT и MONEY в одном предложении
    2. Если MONEY находится в пределах 100 символов после ACT - считаем их парой
    3. Приоритет: ближайшее MONEY после ACT
    """
    pairs = []

    # Разбиваем текст на предложения (простая эвристика)
    sentences = text.replace('\n', ' ').split('.')

    act_entities = [e for e in entities if e.type in ['ACT', 'NOT_ACT']]
    money_entities = [e for e in entities if e.type in ['MONEY', 'SUM']]

    for act in act_entities:
        # Ищем ближайшее MONEY в том же предложении или рядом
        for money in money_entities:
            # Проверяем, находятся ли они в одном предложении
            act_sentence_idx = None
            money_sentence_idx = None

            current_pos = 0
            for idx, sentence in enumerate(sentences):
                sentence_start = current_pos
                sentence_end = current_pos + len(sentence)

                if sentence_start <= act.start_pos < sentence_end:
                    act_sentence_idx = idx
                if sentence_start <= money.start_pos < sentence_end:
                    money_sentence_idx = idx

                current_pos = sentence_end + 1  # +1 для точки

            # Если в одном предложении и MONEY идет после ACT
            if (act_sentence_idx == money_sentence_idx and
                    money.start_pos > act.start_pos and
                    money.start_pos - act.end_pos < 100):  # Не дальше 100 символов

                pairs.append({
                    'act': act.dict(),
                    'money': money.dict(),
                    'distance': money.start_pos - act.end_pos,
                    'context': text[act.start_pos:min(money.end_pos + 20, len(text))]
                })
                break  # Берем только первое подходящее MONEY

    return pairs


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    """Health check"""
    return {"status": "ok", "service": "ner-service", "version": "1.0.0"}


@app.post("/extract", response_model=NERResponse)
def extract_entities(request: NERRequest):
    """
    Извлекает все сущности из текста и связывает ACT-MONEY пары.

    Возвращает:
    - Все найденные сущности (PERSON, DATE, MONEY, ACT, NOT_ACT, SUM, SUBJECT)
    - Связанные пары ACT-MONEY
    """
    try:
        # 1. Извлекаем сущности с помощью Natasha
        natasha_entities = extract_with_natasha(request.text)

        # 2. Загружаем словарь (если указан путь)
        dictionary = {}
        if request.dictionary_path:
            dictionary = load_dictionary(request.dictionary_path)

        # 3. Извлекаем сущности по словарю (Yargy)
        dictionary_entities = extract_with_yargy(request.text, dictionary)

        # 4. Извлекаем предметы взыскания (морфологический поиск)
        subject_entities = extract_subjects_morph(request.text)

        # 5. Объединяем все сущности (удаляем дубликаты)
        all_entities = {}
        for entity in natasha_entities + dictionary_entities + subject_entities:
            key = (entity.start_pos, entity.end_pos)
            if key not in all_entities:
                all_entities[key] = entity

        entities_list = list(all_entities.values())

        # 6. Связываем ACT-MONEY пары
        act_money_pairs = link_act_money_pairs(entities_list, request.text)

        return NERResponse(
            entities=entities_list,
            act_money_pairs=act_money_pairs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-simple")
def extract_simple(text: str):
    """
    Упрощенный endpoint для быстрого тестирования.
    Просто передайте текст как query параметр.
    """
    request = NERRequest(text=text)
    return extract_entities(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) # <-- МЕНЯЕМ ЗДЕСЬ