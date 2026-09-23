"""
ner-service/main.py
FINAL PRODUCTION v3: Разделение на шапку (суд, судья, номер дела) и резолютивную часть (долги)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Set, Tuple
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
# КАСТОМНЫЕ REGEX-ПАТТЕРНЫ
# ==========================================
CUSTOM_PATTERNS = {
    'INN': re.compile(r'\bИНН\s*(\d{10}|\d{12})\b'),
    'KPP': re.compile(r'\b(?:КПП|КИП)\s*(\d{9})\b'),
    'OGRN': re.compile(r'\bОГРН\s*(\d{13}|\d{15})\b'),
    'BIK': re.compile(r'\bБИК\s*(04\d{7})\b'),
    'BANK_ACCOUNT': re.compile(r'(?:р/с|расч[её]т\.?|корр\.?\s*сч[её]т\.?)\s*(\d{20})\b'),
    'PASSPORT': re.compile(r'\bпаспорт\s+(?:гражданина\s+РФ\s+)?(?:серия\s+)?(\d{4})\s*(?:№\s*)?(\d{6})\b', re.IGNORECASE),
    'CONTRACT_NUMBER': re.compile(r'(?:договор|соглашение)\s+(?:№\s*)?([A-Za-zА-Яа-я0-9\-/\.]+)', re.IGNORECASE),
}

# Regex для адресов
ADDRESS_REGEX = re.compile(
    r'(?:адрес регистрации|адрес|проживающий по\s+адресу|юридический адрес|почтовый адрес)\s*:\s*([^\n]+)',
    re.IGNORECASE
)

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
    case_info: Optional[Dict] = None  # Новая секция для данных дела

# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def split_document(text: str) -> Tuple[str, str, int]:
    """
    Разделяет документ на шапку и резолютивную часть.
    Возвращает: (header_text, resolution_text, split_position)
    """
    # Ищем маркеры начала резолютивной части
    markers = [
        r'\bРЕШИЛ\b',
        r'\bПОСТАНОВИЛ\b',
        r'\bОПРЕДЕЛИЛ\b',
        r'\bПРИКАЗЫВАЮ\b',
        r'\bРЕШЕНИЕ\b',
    ]

    combined_pattern = '|'.join(markers)
    match = re.search(combined_pattern, text, re.IGNORECASE)

    if match:
        split_pos = match.start()
        return text[:split_pos].strip(), text[split_pos:].strip(), split_pos
    else:
        # Если маркер не найден, считаем весь текст резолютивной частью
        return "", text, 0

def extract_case_info(header_text: str) -> Dict:
    """Извлекает информацию о деле из шапки документа"""
    case_info = {
        'case_number': None,
        'case_date': None,
        'court_name': None,
        'court_address': None,
        'judge_name': None
    }

    # 1. Номер дела (ищем паттерны типа "Дело 2-1968/2806/2023" или просто цифры с дефисами и слешами)
    case_number_match = re.search(r'(?:Дело\s+|дело\s+№?\s*)?(\d+[-/]\d+[/\d]*)', header_text)
    if case_number_match:
        case_info['case_number'] = case_number_match.group(1)

    # 2. Дата дела (ищем даты в шапке)
    date_patterns = [
        r'(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4})\s+года?',
        r'(\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4})'
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, header_text, re.IGNORECASE)
        if date_match:
            case_info['case_date'] = date_match.group(1)
            break

    # 3. Наименование суда (обычно в первых строках до адреса)
    court_name_match = re.search(
        r'^(.+?)(?:\d{6}|\bадрес:|\b628|\b121|\b101)',
        header_text,
        re.DOTALL
    )
    if court_name_match:
        case_info['court_name'] = court_name_match.group(1).strip()

    # 4. Адрес суда (ищем после маркеров адреса)
    court_address_match = re.search(
        r'(?:\d{6},\s*.+?(?:ул\.|улица|пр\.|проспект|д\.|дом).+?)(?=\n\n|e-mail:|$)',
        header_text,
        re.IGNORECASE
    )
    if court_address_match:
        case_info['court_address'] = court_address_match.group(0).strip()

    # 5. ФИО судьи (ищем после слов "судья" или в конце шапки)
    judge_patterns = [
        r'судья\s+([А-Яа-яЁё]+\s+[А-Я]\.?[А-Я]\.?\s*[А-Яа-яЁё]*)',
        r'([А-Яа-яЁё]+\s+[А-Я]\.\s*[А-Яа-яЁё]+)\s*$',  # В конце текста (подпись)
    ]
    for pattern in judge_patterns:
        judge_match = re.search(pattern, header_text)
        if judge_match:
            case_info['judge_name'] = judge_match.group(1).strip()
            break

    return case_info

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
def extract_entities(text: str) -> Tuple[List[Entity], Dict]:
    entities = []

    # Разделяем документ на части
    header_text, resolution_text, split_pos = split_document(text)

    # Извлекаем информацию о деле из шапки
    case_info = extract_case_info(header_text)

    # Работаем с резолютивной частью (или со всем текстом, если шапки нет)
    working_text = resolution_text if resolution_text else text
    text_offset = split_pos if resolution_text else 0

    def add_entity(word: str, entity_type: str, start: int, end: int, conf: float, currency: str = None, normal_form: str = None):
        if normal_form is None:
            normal_form = word if entity_type.startswith('MONEY') or entity_type == 'DATE' else morph.parse(word)[0].normal_form
        entities.append(Entity(
            text=word, normal_form=normal_form, type=entity_type,
            currency=currency, start_pos=start, end_pos=end, confidence=conf
        ))

    # 1. ДЕНЬГИ
    for match in money_extractor(working_text):
        if hasattr(match.fact, 'currency') and match.fact.currency:
            word = working_text[match.start:match.stop]
            add_entity(word, f"MONEY_{match.fact.currency}", match.start + text_offset, match.stop + text_offset, 0.95, currency=match.fact.currency)

    # 2. ДАТЫ
    for match in date_extractor(working_text):
        word = working_text[match.start:match.stop]
        add_entity(word, 'DATE', match.start + text_offset, match.stop + text_offset, 0.95)
    for pattern in DATE_REGEXES:
        for match in pattern.finditer(working_text):
            add_entity(match.group(0), 'DATE', match.start() + text_offset, match.end() + text_offset, 0.95)

    # 3. ИМЕНА
    for match in names_extractor(working_text):
        word = working_text[match.start:match.stop]
        if len(word.split()) >= 2:
            if not is_address_context(working_text, match.start):
                if not any(char.isascii() and char.isalpha() for char in word):
                    add_entity(word, 'NAME', match.start + text_offset, match.stop + text_offset, 0.95)

    # 4. СЛОВАРЬ
    for match in DICT_PARSER.findall(working_text):
        original_text = working_text[match.span.start:match.span.stop]
        normalized_text = match.fact.text.lower() if hasattr(match.fact, 'text') else str(match.fact).lower()
        category = phrase_to_category.get(normalized_text, 'SUBJECT')
        normal_form = get_phrase_normal_form(original_text)
        add_entity(word=original_text, entity_type=category, start=match.span.start + text_offset, end=match.span.stop + text_offset, conf=0.9, normal_form=normal_form)

    # 5. КАСТОМНЫЕ РЕКВИЗИТЫ
    for entity_type, pattern in CUSTOM_PATTERNS.items():
        for match in pattern.finditer(working_text):
            if entity_type == 'PASSPORT':
                full_text = f"{match.group(1)} {match.group(2)}"
                add_entity(full_text, 'PASSPORT', match.start() + text_offset, match.end() + text_offset, 0.95, normal_form=full_text)
            elif entity_type == 'BANK_ACCOUNT':
                add_entity(match.group(1), entity_type, match.start() + text_offset, match.end() + text_offset, 0.95, normal_form=match.group(1))
            else:
                add_entity(match.group(1), entity_type, match.start() + text_offset, match.end() + text_offset, 0.95, normal_form=match.group(1))

    # 6. АДРЕСА
    for match in ADDRESS_REGEX.finditer(working_text):
        address_text = match.group(1).strip()
        if any(m in address_text.lower() for m in ADDRESS_MARKERS):
            add_entity(address_text, 'ADDRESS', match.start(1) + text_offset, match.end(1) + text_offset, 0.9, normal_form=address_text)

    # 7. ДЕДУПЛИКАЦИЯ
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

    return unique_entities, case_info


def link_subject_money_pairs(entities: List[Entity], text: str) -> List[SubjectMoneyPair]:
    subjects = [e for e in entities if e.type == 'SUBJECT']
    money_entities = [e for e in entities if e.type.startswith('MONEY')]
    WINDOW_SIZE = 150

    candidates = []
    for subj in subjects:
        for money in money_entities:
            distance = money.start_pos - subj.end_pos
            if distance < 0:
                distance = 9999
            if distance <= WINDOW_SIZE:
                candidates.append({
                    'subject': subj,
                    'money': money,
                    'distance': distance
                })

    candidates.sort(key=lambda x: x['distance'])

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

    for subj in subjects:
        subj_id = (subj.start_pos, subj.end_pos)
        if subj_id not in used_subjects:
            final_pairs.append(SubjectMoneyPair(
                subject=subj,
                money=None,
                distance=None,
                context=None
            ))

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
        entities, case_info = extract_entities(request.text)
        subject_money_pairs = link_subject_money_pairs(entities, request.text)
        return NERResponse(entities=entities, subject_money_pairs=subject_money_pairs, case_info=case_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)