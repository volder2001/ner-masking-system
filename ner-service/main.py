"""
ner-service/main.py
FINAL PRODUCTION v21: Фикс денег с переносом строки (242 637,21\nруб.), чистка judge_name и court_address
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
    'республика', 'республики', 'край', 'края', 'автономный округ'
}

DICT_PATH = Path("/app/data/dictionary.json")
def load_dictionary():
    if DICT_PATH.exists():
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"SUBJECT": ["квартира"], "NOT_SUBJECT": ["отсутствует акт"]}

dictionary = load_dictionary()

# ==========================================
# 2. YARGY + SMART FILTER
# ==========================================
phrase_to_category = {}
all_phrases = []

for category, phrases in dictionary.items():
    for phrase in phrases:
        clean_phrase = " ".join([w.strip(".,;:-") for w in phrase.split()])
        all_phrases.append(clean_phrase)
        phrase_to_category[clean_phrase.lower()] = category

DICT_RULE = morph_pipeline(all_phrases)
DICT_PARSER = Parser(DICT_RULE)

STOP_WORDS = {'в', 'на', 'по', 'за', 'с', 'к', 'у', 'о', 'об', 'от', 'до', 'из', 'под', 'над', 'и', 'а', 'но', 'или', 'же', 'бы', 'ли', 'то'}

def is_valid_dict_match(matched_text: str, clean_phrase: str) -> bool:
    if len(matched_text) < len(clean_phrase) * 0.60:
        return False
    matched_words = set(re.findall(r'\w+', matched_text.lower()))
    phrase_words = set(re.findall(r'\w+', clean_phrase.lower()))
    matched_sig = matched_words - STOP_WORDS
    phrase_sig = phrase_words - STOP_WORDS
    if not phrase_sig:
        return True
    overlap_ratio = len(matched_sig & phrase_sig) / len(phrase_sig)
    return overlap_ratio >= 0.70

def normalize_matched_text(matched_text: str) -> str:
    norm_words = []
    for w in matched_text.split():
        clean_w = re.sub(r'[^\w]', '', w)
        if clean_w:
            p = morph.parse(clean_w)[0]
            if 'PREP' not in p.tag.grammemes and 'CONJ' not in p.tag.grammemes:
                norm_words.append(p.normal_form)
    return " ".join(norm_words) if norm_words else matched_text

# ==========================================
# 3. ОСТАЛЬНЫЕ ПАТТЕРНЫ (С ФИКСАМИ)
# ==========================================
CUSTOM_PATTERNS = {
    'INN': re.compile(r'\bИНН\s*(\d{10}|\d{12})\b'),
    'KPP': re.compile(r'\b(?:КПП|КИП)\s*(\d{9})\b'),
    'OGRN': re.compile(r'\bОГРН\s*(\d{13}|\d{15})\b'),
    'BIK': re.compile(r'\bБИК\s*(04\d{7})\b'),
    'BANK_ACCOUNT': re.compile(r'(?:р/с|расч[её]т\.?|корр\.?\s*сч[её]т\.?)\s*(\d{20})\b'),
    'PASSPORT': re.compile(r'паспорт.*?(?:серии?\s+)?(\d{2}\s?\d{2}|\d{4}).*?(?:номер\s+)?(\d{6})\b', re.IGNORECASE | re.DOTALL),
    'CONTRACT_NUMBER': re.compile(r'(?:договор|соглашение)\s+(?:№\s*)?([A-Za-zА-Яа-я0-9\-/\.]+)', re.IGNORECASE),
}

# ФИКС: Корректно ловит "242 637,21\nруб.", "57 329 руб. 92 коп.", "727,36 руб."
MONEY_FALLBACKS = [
    re.compile(r'(\d{1,3}(?:\s?\d{3})*(?:[.,]\d{2})?)\s+((?:руб\.?(?:\s*\d{1,2})?)|рублей|(?:коп\.?(?:\s*\d{1,2})?)|копеек|(?:py6\.?(?:\s*\d{1,2})?))', re.IGNORECASE),
    re.compile(r'пени\s+(\d{1,3}(?:\s?\d{3})*(?:[.,]\d{2})?)', re.IGNORECASE),
]

SUBJECT_FALLBACKS = [
    (re.compile(r'задолженност\w*\s+по\s+кредит\w*\s+договор\w*', re.IGNORECASE | re.DOTALL), "задолженность кредитный договор"),
    (re.compile(r'задолженност\w*\s+по\s+уплат\w*\s+процент\w*', re.IGNORECASE | re.DOTALL), "задолженность уплата процент"),
    (re.compile(r'расход\w*\s+по\s+оплат\w*\s+(?:государствен\w*\s+)?пошлин\w*', re.IGNORECASE | re.DOTALL), "расход оплата государственная пошлина"),
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
    header_match = re.search(r'^(.*?)(?:РЕШИЛ:|ПОСТАНОВИЛ:|ОПРЕДЕЛИЛ:|ПРИКАЗЫВАЮ:)', text, re.IGNORECASE | re.DOTALL)
    header = header_match.group(1) if header_match else ""

    case_info = {'case_number': None, 'case_date': None, 'court_name': None, 'court_address': None, 'judge_name': None}
    if not header.strip():
        return case_info

    cn_match = re.search(r'(?:Дело|производство|дело|производство)\s*№?\s*([A-Za-zА-Яа-я0-9\-/\.\s]+?)(?=\n|$)', header, re.IGNORECASE)
    if cn_match:
        case_info['case_number'] = cn_match.group(1).strip()
    else:
        cn_match2 = re.search(r'\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}\s+([0-9\-/\s]+?)(?=\n|$)', header, re.IGNORECASE)
        if cn_match2:
            case_info['case_number'] = cn_match2.group(1).strip()
        else:
            cn_match3 = re.search(r'№\s*(\d+[-/\s\d]+)', header)
            if cn_match3:
                case_info['case_number'] = cn_match3.group(1).strip()

    date_match = re.search(r'(\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4})', header, re.IGNORECASE)
    if date_match:
        case_info['case_date'] = date_match.group(1)
    else:
        date_match2 = re.search(r'(\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4})', header)
        if date_match2:
            case_info['case_date'] = date_match2.group(1)

    lines = header.strip().split('\n')
    court_lines = [line.strip() for line in lines[:5] if line.strip() and len(line.strip()) > 10 and not re.search(r'\d{6}', line)]
    if court_lines:
        case_info['court_name'] = ' '.join(court_lines).strip()

    # ФИКС: Добавлены стоп-слова должнику, взыскателя, РЕШИЛ: чтобы не захватывать лишний текст
    addr_match = re.search(r'(\d{6},\s*.*?(?:ул\.|улица|г\.|город|пр\.|проспект).*?)(?=\n\n|Именем|сайт|e-mail|@|должнику|взыскателя|РЕШИЛ:|$)', header, re.IGNORECASE | re.DOTALL)
    if addr_match:
        case_info['court_address'] = re.sub(r'\s+', ' ', addr_match.group(1)).strip()

    # ФИКС: Паттерн с инициалами (И.О. Фамилия) стоит ПЕРВЫМ, чтобы перехватить чистую подпись в конце
    judge_patterns = [
        r'([А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.\s*[А-Яа-яA-Za-z]+)', # И.О. Фамилия (наивысший приоритет)
        r'Мировой судья\s+(?:.*?\s+)?([А-Я][а-я]+)(?:\s+рассмотрев|\n|$)', # Фамилия после "Мировой судья"
        r'судья\s+([А-Яа-яA-Za-z]+\s+[А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.?)',
    ]
    for pattern in judge_patterns:
        judge_match = re.search(pattern, header)
        if judge_match:
            case_info['judge_name'] = re.sub(r'\s+', ' ', judge_match.group(1)).strip()
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

    resolution_match = re.search(r'(?:РЕШИЛ:|ПОСТАНОВИЛ:|ОПРЕДЕЛИЛ:|ПРИКАЗЫВАЮ:)', text, re.IGNORECASE)
    resolution_start = resolution_match.end() if resolution_match else 0
    resolution_text = text[resolution_start:]

    def add_entity(word: str, entity_type: str, start: int, end: int, conf: float, currency: str = None, normal_form: str = None):
        if normal_form is None:
            normal_form = word if entity_type.startswith('MONEY') or entity_type == 'DATE' else morph.parse(word)[0].normal_form
        entities.append(Entity(
            text=word, normal_form=normal_form, type=entity_type,
            currency=currency, start_pos=start, end_pos=end, confidence=conf
        ))

    # А. ГЛОБАЛЬНЫЙ ПОИСК
    for match in names_extractor(text):
        word = text[match.start:match.stop]
        if len(word.split()) >= 2:
            if not is_address_context(text, match.start):
                if not any(char.isascii() and char.isalpha() for char in word):
                    add_entity(word, 'NAME', match.start, match.stop, 0.95)

    for match in date_extractor(text):
        add_entity(text[match.start:match.stop], 'DATE', match.start, match.stop, 0.95)
    for pattern in [re.compile(r'\b\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}\b'), re.compile(r'["\']?\d{1,2}["\']?\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{2,4}\s*г\.?', re.IGNORECASE)]:
        for match in pattern.finditer(text):
            add_entity(match.group(0), 'DATE', match.start(), match.end(), 0.95)

    for entity_type, pattern in CUSTOM_PATTERNS.items():
        for match in pattern.finditer(text):
            if entity_type == 'PASSPORT':
                series = match.group(1).replace(' ', '').replace('-', '')
                number = match.group(2)
                add_entity(f"{series} {number}", 'PASSPORT', match.start(), match.end(), 0.95, normal_form=f"{series} {number}")
            elif entity_type == 'BANK_ACCOUNT':
                add_entity(match.group(1), entity_type, match.start(), match.end(), 0.95, normal_form=match.group(1))
            else:
                add_entity(match.group(1), entity_type, match.start(), match.end(), 0.95, normal_form=match.group(1))

    # Б. ЛОКАЛЬНЫЙ ПОИСК (РЕЗОЛЮЦИЯ)
    for match in money_extractor(resolution_text):
        if hasattr(match.fact, 'currency') and match.fact.currency:
            add_entity(resolution_text[match.start:match.stop], f"MONEY_{match.fact.currency}",
                       match.start + resolution_start, match.stop + resolution_start, 0.95, currency=match.fact.currency)

    for pattern in MONEY_FALLBACKS:
        for match in pattern.finditer(resolution_text):
            # Группа 0 - это всё совпадение целиком (например, "242 637,21\nруб.")
            text_match = match.group(0)
            start, end = match.start(), match.end()

            text_match = re.sub(r'\s+', ' ', text_match).strip()
            real_start = start + resolution_start
            real_end = end + resolution_start

            overlap = any(e.start_pos <= real_start < e.end_pos for e in entities if e.type.startswith('MONEY'))
            if not overlap:
                add_entity(text_match, 'MONEY_RUB', real_start, real_end, 0.90, currency='RUB', normal_form=text_match)

    sorted_phrases = sorted(phrase_to_category.keys(), key=len, reverse=True)
    for match in DICT_PARSER.findall(resolution_text):
        matched_text = resolution_text[match.span.start:match.span.stop]
        for clean_phrase in sorted_phrases:
            if is_valid_dict_match(matched_text, clean_phrase):
                category = phrase_to_category[clean_phrase]
                normal_form = normalize_matched_text(matched_text)
                add_entity(matched_text, category, match.span.start + resolution_start, match.span.stop + resolution_start, 0.9, normal_form=normal_form)
                break

    for pattern, normal_form in SUBJECT_FALLBACKS:
        for match in pattern.finditer(resolution_text):
            matched_text = match.group(0).strip()
            clean_matched = re.sub(r'\s+', ' ', matched_text)
            real_start = match.start() + resolution_start
            real_end = match.end() + resolution_start

            overlap = any(e.start_pos <= real_start < e.end_pos or real_start < e.end_pos <= real_end for e in entities if e.type == 'SUBJECT')
            if not overlap:
                add_entity(clean_matched, 'SUBJECT', real_start, real_end, 0.9, normal_form=normal_form)

    # В. ДЕДУПЛИКАЦИЯ
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

    subjects.sort(key=lambda x: x.start_pos)
    money_entities.sort(key=lambda x: x.start_pos)

    final_pairs = []
    money_idx = 0
    WINDOW_SIZE = 300

    for subj in subjects:
        best_money = None
        best_distance = 9999
        best_money_idx = -1

        for i in range(money_idx, len(money_entities)):
            money = money_entities[i]
            distance = money.start_pos - subj.end_pos
            if distance >= 0 and distance < best_distance:
                best_money = money
                best_distance = distance
                best_money_idx = i

        if best_money is None:
            for i in range(money_idx):
                money = money_entities[i]
                distance = subj.start_pos - money.end_pos
                if distance >= 0 and distance < best_distance:
                    best_money = money
                    best_distance = distance
                    best_money_idx = i

        if best_money is not None and best_distance <= WINDOW_SIZE:
            if best_distance >= 0:
                money_idx = best_money_idx + 1

            context_start = subj.start_pos
            context_end = min(best_money.end_pos + 40, len(text))

            final_pairs.append(SubjectMoneyPair(
                subject=subj,
                money=best_money,
                distance=best_distance,
                context=text[context_start:context_end]
            ))
        else:
            final_pairs.append(SubjectMoneyPair(subject=subj, money=None, distance=None, context=None))

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