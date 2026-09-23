"""
ner-service/main.py
FINAL PRODUCTION v6: OCR-устойчивый словарь (сохраняет длинные уникальные SUBJECTы) + фикс case_info
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
    'мкр.', 'микрорайон', 'наб.', 'набережная', 'туп.', 'тупик',
    'республика', 'республики', 'край', 'края', 'автономный округ' # <-- Защита от ложных NAME
}

DICT_PATH = Path("/app/data/dictionary.json")
def load_dictionary():
    if DICT_PATH.exists():
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"SUBJECT": ["квартира"], "NOT_SUBJECT": ["отсутствует акт"]}

dictionary = load_dictionary()

# ==========================================
# 2. ГЕНЕРАТОР OCR-УСТОЙЧИВЫХ REGEX ДЛЯ СЛОВАРЯ
# ==========================================
COMPILED_DICT_REGEXES = []

def build_ocr_resilient_regex(phrase: str) -> re.Pattern:
    """
    Превращает фразу из словаря в гибкий regex, устойчивый к пропущенным предлогам
    и мелким опечаткам OCR (например, "нсустойка" или "связи" вместо "в связи с").
    """
    # Убираем пунктуацию для анализа слов
    clean_phrase = re.sub(r'[^\w\s]', ' ', phrase)
    words = clean_phrase.split()

    significant_words = []
    for w in words:
        p = morph.parse(w)[0]
        # Оставляем только знаменательные части речи
        if not ('PREP' in p.tag.grammemes or 'CONJ' in p.tag.grammemes or 'PRCL' in p.tag.grammemes):
            significant_words.append(w)

    regex_parts = []
    for w in significant_words:
        w_lower = w.lower()
        # Хак 1: Разрешаем пропуск 'е' или 'ё' после 'н' (неустойка -> нсустойка)
        if w_lower.startswith('не'):
            w_regex = r'н[её]?' + w[2:]
        else:
            w_regex = w

        # Хак 2: Делаем последнюю гласную опциональной (связи -> связ, кредита -> кредит)
        w_regex = re.sub(r'([аеёиоуыэюя])$', r'[\1]?', w_regex, flags=re.IGNORECASE)
        regex_parts.append(w_regex)

    # Соединяем ключевые слова через .*? (допускает любые символы, пробелы, переносы строк между ними)
    pattern_str = r'\b' + r'.*?'.join(regex_parts) + r'\b'
    return re.compile(pattern_str, re.IGNORECASE | re.DOTALL)

# Компилируем все фразы из словаря
for category, phrases in dictionary.items():
    for phrase in phrases:
        regex = build_ocr_resilient_regex(phrase)
        # Сохраняем оригинальную фразу как нормальную форму, чтобы не терять смысл!
        COMPILED_DICT_REGEXES.append((regex, category, phrase))

# ==========================================
# 3. ОСТАЛЬНЫЕ ПАТТЕРНЫ
# ==========================================
CUSTOM_PATTERNS = {
    'INN': re.compile(r'\bИНН\s*(\d{10}|\d{12})\b'),
    'KPP': re.compile(r'\b(?:КПП|КИП)\s*(\d{9})\b'),
    'OGRN': re.compile(r'\bОГРН\s*(\d{13}|\d{15})\b'),
    'BIK': re.compile(r'\bБИК\s*(04\d{7})\b'),
    'BANK_ACCOUNT': re.compile(r'(?:р/с|расч[её]т\.?|корр\.?\s*сч[её]т\.?)\s*(\d{20})\b'),
    'PASSPORT': re.compile(r'паспорт.*?(?:серии?\s+)?(\d{4}).*?(?:номер\s+)?(\d{6})\b', re.IGNORECASE | re.DOTALL),
    'CONTRACT_NUMBER': re.compile(r'(?:договор|соглашение)\s+(?:№\s*)?([A-Za-zА-Яа-я0-9\-/\.]+)', re.IGNORECASE),
}

MONEY_FALLBACKS = [
    re.compile(r'\b(\d+(?:[.,]\d{2})?)\s+(?:руб\.|рублей|коп\.|копеек)\b', re.IGNORECASE),
    re.compile(r'пени\s+(\d+(?:[.,]\d{2})?)', re.IGNORECASE),
]

# ==========================================
# 4. МОДЕЛИ ДАННЫХ
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
# 5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def extract_case_info(text: str) -> Dict:
    case_info = {'case_number': None, 'case_date': None, 'court_name': None, 'court_address': None, 'judge_name': None}

    case_number_match = re.search(r'(?:Дело\s+|дело\s+№?\s*|Производство\s+)?(\d+[-/]\d+[/\d]*)', text)
    if case_number_match:
        case_info['case_number'] = case_number_match.group(1)

    date_match = re.search(r'(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4})', text, re.IGNORECASE)
    if date_match:
        case_info['case_date'] = date_match.group(1)
    else:
        date_match2 = re.search(r'(\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4})', text)
        if date_match2:
            case_info['case_date'] = date_match2.group(1)

    court_name_match = re.search(r'^(.+?)(?=\d{6},|Именем)', text, re.DOTALL | re.IGNORECASE)
    if court_name_match:
        case_info['court_name'] = re.sub(r'\s+', ' ', court_name_match.group(1)).strip()

    # Останавливаем адрес на "сайт", "@" или "Именем"
    court_address_match = re.search(r'(\d{6},\s*.*?)(?:сайт|e-mail|@|Именем|$)', text, re.DOTALL | re.IGNORECASE)
    if court_address_match:
        case_info['court_address'] = re.sub(r'\s+', ' ', court_address_match.group(1)).strip()

    # Поддержка форматов "Фамилия И.О." и "И.О.Фамилия"
    judge_patterns = [
        r'Мировой судья\s+([А-Яа-яA-Za-z]+\s+[А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.?)',
        r'Мировой судья\s+([А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.\s*[А-Яа-яA-Za-z]+)', # Ю.Т.Шакирьянова
        r'судья\s+([А-Яа-яA-Za-z]+\s+[А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.?)'
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

# ==========================================
# 6. ЛОГИКА ИЗВЛЕЧЕНИЯ
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
            add_entity(text[match.start:match.stop], f"MONEY_{match.fact.currency}", match.start, match.stop, 0.95, currency=match.fact.currency)

    # 1.1 ДЕНЬГИ (Fallback Regex)
    for pattern in MONEY_FALLBACKS:
        for match in pattern.finditer(text):
            if pattern.pattern.startswith('пени'):
                text_match, start, end = match.group(1), match.start(1), match.end(1)
            else:
                text_match, start, end = match.group(0), match.start(), match.end()

            overlap = any(e.start_pos <= start < e.end_pos for e in entities if e.type.startswith('MONEY'))
            if not overlap:
                add_entity(text_match, 'MONEY_RUB', start, end, 0.90, currency='RUB', normal_form=text_match)

    # 2. ДАТЫ
    for match in date_extractor(text):
        add_entity(text[match.start:match.stop], 'DATE', match.start, match.stop, 0.95)
    for pattern in [re.compile(r'\b\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}\b'), re.compile(r'["\']?\d{1,2}["\']?\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{2,4}\s*г\.?', re.IGNORECASE)]:
        for match in pattern.finditer(text):
            add_entity(match.group(0), 'DATE', match.start(), match.end(), 0.95)

    # 3. ИМЕНА
    for match in names_extractor(text):
        word = text[match.start:match.stop]
        if len(word.split()) >= 2:
            if not is_address_context(text, match.start):
                if not any(char.isascii() and char.isalpha() for char in word):
                    add_entity(word, 'NAME', match.start, match.stop, 0.95)

    # 4. СЛОВАРЬ (OCR-УСТОЙЧИВЫЙ ПОИСК)
    # Сортируем по длине фразы (desc), чтобы длинные уникальные фразы матчились первыми и блокировали короткие
    sorted_dict_regexes = sorted(COMPILED_DICT_REGEXES, key=lambda x: len(x[2]), reverse=True)

    for regex, category, original_phrase in sorted_dict_regexes:
        for match in regex.finditer(text):
            matched_text = match.group(0)
            # Проверяем, не перекрыто ли это совпадение уже найденной более длинной/ранней сущностью
            overlap = any(e.start_pos <= match.start() < e.end_pos or e.start_pos < match.end() <= e.end_pos for e in entities)
            if not overlap:
                add_entity(matched_text, category, match.start(), match.end(), 0.9, normal_form=original_phrase)

    # 5. КАСТОМНЫЕ РЕКВИЗИТЫ
    for entity_type, pattern in CUSTOM_PATTERNS.items():
        for match in pattern.finditer(text):
            if entity_type == 'PASSPORT':
                add_entity(f"{match.group(1)} {match.group(2)}", 'PASSPORT', match.start(), match.end(), 0.95, normal_form=f"{match.group(1)} {match.group(2)}")
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
    subjects = [e for e in entities if e.type in ['SUBJECT', 'NOT_SUBJECT']] # Включаем оба типа для связывания, если нужно
    # Но по логике задачи мы связываем только SUBJECT с деньгами
    subjects = [e for e in entities if e.type == 'SUBJECT']
    money_entities = [e for e in entities if e.type.startswith('MONEY')]
    WINDOW_SIZE = 250 # Чуть увеличили для длинных фраз

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
# 7. API ENDPOINTS
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