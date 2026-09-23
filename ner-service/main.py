"""
ner-service/main.py
FINAL PRODUCTION v5: Укрощение regex денег, усиление паспорта и адреса суда
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
# КАСТОМНЫЕ REGEX-ПАТТЕРНЫ (БЕЗОПАСНЫЕ)
# ==========================================
CUSTOM_PATTERNS = {
    'INN': re.compile(r'\bИНН\s*(\d{10}|\d{12})\b'),
    'KPP': re.compile(r'\b(?:КПП|КИП)\s*(\d{9})\b'),
    'OGRN': re.compile(r'\bОГРН\s*(\d{13}|\d{15})\b'),
    'BIK': re.compile(r'\bБИК\s*(04\d{7})\b'),
    'BANK_ACCOUNT': re.compile(r'(?:р/с|расч[её]т\.?|корр\.?\s*сч[её]т\.?)\s*(\d{20})\b'),
    # Паспорт: .*? позволяет игнорировать переносы строк и мусор OCR между словами
    'PASSPORT': re.compile(r'паспорт.*?(?:серии?\s+)?(\d{4}).*?(?:номер\s+)?(\d{6})\b', re.IGNORECASE | re.DOTALL),
    'CONTRACT_NUMBER': re.compile(r'(?:договор|соглашение)\s+(?:№\s*)?([A-Za-zА-Яа-я0-9\-/\.]+)', re.IGNORECASE),
}

# БЕЗОПАСНЫЕ fallback-правила для денег (только целевые!)
MONEY_FALLBACKS = [
    # 1. Строгое требование валюты
    re.compile(r'\b(\d{1,3}(?:\s?\d{3})*(?:,\d{2})?)\s+(?:руб\.|рублей|коп\.|копеек)\b', re.IGNORECASE),
    # 2. Специфично для "пени [число]" (как в примере "пени 22023,65,")
    re.compile(r'пени\s+(\d{1,3}(?:\s?\d{3})*(?:,\d{2})?)', re.IGNORECASE),
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
    case_info: Optional[Dict] = None

# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def extract_case_info(text: str) -> Dict:
    case_info = {
        'case_number': None, 'case_date': None, 'court_name': None,
        'court_address': None, 'judge_name': None
    }

    # 1. Номер дела
    case_number_match = re.search(r'(?:Дело\s+|дело\s+№?\s*)?(\d+[-/]\d+[/\d]*)', text)
    if case_number_match:
        case_info['case_number'] = case_number_match.group(1)

    # 2. Дата дела
    date_match = re.search(r'(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4})', text, re.IGNORECASE)
    if date_match:
        case_info['case_date'] = date_match.group(1)
    else:
        date_match2 = re.search(r'(\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4})', text)
        if date_match2:
            case_info['case_date'] = date_match2.group(1)

    # 3. Наименование суда (до индекса или e-mail)
    court_name_match = re.search(r'^(.+?)(?=\d{6},|e-mail:|Именем)', text, re.DOTALL | re.IGNORECASE)
    if court_name_match:
        case_info['court_name'] = re.sub(r'\s+', ' ', court_name_match.group(1)).strip()

    # 4. Адрес суда (от индекса до e-mail, игнорируя переносы строк)
    court_address_match = re.search(r'(\d{6},\s*.*?)(?=e-mail:|$)', text, re.DOTALL | re.IGNORECASE)
    if court_address_match:
        case_info['court_address'] = re.sub(r'\s+', ' ', court_address_match.group(1)).strip()

    # 5. ФИО судьи
    judge_patterns = [
        r'(?:Югры|области|края|республики)\s+([А-Яа-яA-Za-z]+\s+[A-Za-zА-Яа-я]\.?\s*[A-Za-zА-Яа-я]\.?)',
        r'Мировой судья\s+([А-Яа-яA-Za-z]+\s+[A-Za-zА-Яа-я]\.?\s*[A-Za-zА-Яа-я]\.?)'
    ]
    for pattern in judge_patterns:
        judge_match = re.search(pattern, text)
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
    case_info = extract_case_info(text)

    def add_entity(word: str, entity_type: str, start: int, end: int, conf: float, currency: str = None, normal_form: str = None):
        if normal_form is None:
            normal_form = word if entity_type.startswith('MONEY') or entity_type == 'DATE' else morph.parse(word)[0].normal_form
        entities.append(Entity(
            text=word, normal_form=normal_form, type=entity_type,
            currency=currency, start_pos=start, end_pos=end, confidence=conf
        ))

    # 1. ДЕНЬГИ (Natasha)
    for match in money_extractor(text):
        if hasattr(match.fact, 'currency') and match.fact.currency:
            word = text[match.start:match.stop]
            add_entity(word, f"MONEY_{match.fact.currency}", match.start, match.stop, 0.95, currency=match.fact.currency)

    # 1.1 ДЕНЬГИ (Безопасные Fallback Regex)
    for pattern in MONEY_FALLBACKS:
        for match in pattern.finditer(text):
            # Проверяем, не перекрывается ли уже найденное Natasha
            overlap = any(e.start_pos <= match.start() < e.end_pos for e in entities if e.type.startswith('MONEY'))
            if not overlap:
                # Для паттерна "пени [число]" добавляем слово "пени" в текст для контекста
                full_match_text = match.group(0)
                if not re.search(r'руб|коп', full_match_text, re.IGNORECASE):
                    full_match_text = f"пени {match.group(1)}" # Восстанавливаем контекст

                add_entity(full_match_text, 'MONEY_RUB', match.start(), match.end(), 0.90, currency='RUB')

    # 2. ДАТЫ
    for match in date_extractor(text):
        word = text[match.start:match.stop]
        add_entity(word, 'DATE', match.start, match.stop, 0.95)
    for pattern in DATE_REGEXES:
        for match in pattern.finditer(text):
            add_entity(match.group(0), 'DATE', match.start(), match.end(), 0.95)

    # 3. ИМЕНА
    for match in names_extractor(text):
        word = text[match.start:match.stop]
        if len(word.split()) >= 2:
            if not is_address_context(text, match.start):
                if not any(char.isascii() and char.isalpha() for char in word):
                    add_entity(word, 'NAME', match.start, match.stop, 0.95)

    # 4. СЛОВАРЬ
    for match in DICT_PARSER.findall(text):
        original_text = text[match.span.start:match.span.stop]
        normalized_text = match.fact.text.lower() if hasattr(match.fact, 'text') else str(match.fact).lower()
        category = phrase_to_category.get(normalized_text, 'SUBJECT')
        normal_form = get_phrase_normal_form(original_text)
        add_entity(word=original_text, entity_type=category, start=match.span.start, end=match.span.stop, conf=0.9, normal_form=normal_form)

    # 5. КАСТОМНЫЕ РЕКВИЗИТЫ
    for entity_type, pattern in CUSTOM_PATTERNS.items():
        for match in pattern.finditer(text):
            if entity_type == 'PASSPORT':
                full_text = f"{match.group(1)} {match.group(2)}"
                add_entity(full_text, 'PASSPORT', match.start(), match.end(), 0.95, normal_form=full_text)
            elif entity_type == 'BANK_ACCOUNT':
                add_entity(match.group(1), entity_type, match.start(), match.end(), 0.95, normal_form=match.group(1))
            else:
                add_entity(match.group(1), entity_type, match.start(), match.end(), 0.95, normal_form=match.group(1))

    # 6. ДЕДУПЛИКАЦИЯ
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
    WINDOW_SIZE = 200

    candidates = []
    for subj in subjects:
        for money in money_entities:
            distance = money.start_pos - subj.end_pos
            if distance < 0:
                distance = 9999
            if distance <= WINDOW_SIZE:
                candidates.append({'subject': subj, 'money': money, 'distance': distance})

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

            final_pairs.append(SubjectMoneyPair(
                subject=candidate['subject'],
                money=candidate['money'],
                distance=candidate['distance'],
                context=text[context_start:context_end]
            ))

    for subj in subjects:
        subj_id = (subj.start_pos, subj.end_pos)
        if subj_id not in used_subjects:
            final_pairs.append(SubjectMoneyPair(subject=subj, money=None, distance=None, context=None))

    final_pairs.sort(key=lambda x: x.subject.start_pos)
    return final_pairs

# ==========================================
# 5. API ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {"status": "ok", "service": "ner-service", "dict_loaded_total": len(dictionary)}

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