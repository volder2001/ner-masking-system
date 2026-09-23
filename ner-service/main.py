"""
ner-service/main.py
FINAL PRODUCTION v14: Bounded Gap без цифр (решает опечатки OCR и ложные срабатывания) + фикс шапки
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Set, Tuple
import json
import re
from pathlib import Path
import pymorphy3

from natasha import MorphVocab, MoneyExtractor, DatesExtractor, NamesExtractor

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
# 2. BOUNDED GAP REGEX БЕЗ ЦИФР В РАЗРЫВЕ
# ==========================================
COMPILED_DICT_REGEXES = []

def build_ocr_resilient_regex(phrase: str) -> Optional[re.Pattern]:
    words = phrase.split()
    significant_parts = []
    for w in words:
        clean_w = re.sub(r'[^\w\s]', '', w)
        if not clean_w: continue
        p = morph.parse(clean_w)[0]
        if 'PREP' in p.tag.grammemes or 'CONJ' in p.tag.grammemes or 'PRCL' in p.tag.grammemes:
            continue
        lemma = p.normal_form.lower()
        # Хак для OCR: неустойка -> н[её]?устойка
        lemma = re.sub(r'^н([её])', r'н[её]?', lemma)
        significant_parts.append(lemma + r'\w*')

    if not significant_parts:
        return None

    # РАЗРЫВ: до 35 символов, НО без цифр ([^\d]). Это блокирует "долга 12345 договор",
    # но пропускает "неустойка, начисленная связи возврата кредита"
    gap = r'[^\d]{0,35}?'
    pattern_str = r'\b' + gap.join(significant_parts) + r'\b'
    return re.compile(pattern_str, re.IGNORECASE | re.DOTALL)

# Компилируем и сортируем по длине фразы (desc), чтобы длинные матчились первыми
compiled_list = []
for category, phrases in dictionary.items():
    for phrase in phrases:
        regex = build_ocr_resilient_regex(phrase)
        if regex:
            compiled_list.append((regex, category, phrase))

COMPILED_DICT_REGEXES = sorted(compiled_list, key=lambda x: len(x[2]), reverse=True)

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
    header_match = re.search(r'^(.*?)(?:РЕШИЛ:|ПОСТАНОВИЛ:|ОПРЕДЕЛИЛ:|ПРИКАЗЫВАЮ:)', text, re.IGNORECASE | re.DOTALL)
    header = header_match.group(1) if header_match else ""

    case_info = {'case_number': None, 'case_date': None, 'court_name': None, 'court_address': None, 'judge_name': None}
    if not header.strip():
        return case_info

    # Номер дела (разрешаем пробелы и слэши, например "2- /2022")
    cn_match = re.search(r'(?:Дело|производство|дело|производство)\s*№?\s*([A-Za-zА-Яа-я0-9\-/\.\s]+?)(?=\n|$)', header, re.IGNORECASE)
    if cn_match:
        case_info['case_number'] = cn_match.group(1).strip()
    else:
        cn_match2 = re.search(r'№\s*(\d+[-/\s\d]+)', header)
        if cn_match2:
            case_info['case_number'] = cn_match2.group(1).strip()

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

    addr_match = re.search(r'(\d{6},\s*.*?(?:ул\.|улица|г\.|город|пр\.|проспект).*?)(?=\n\n|Именем|сайт|e-mail|@|$)', header, re.IGNORECASE | re.DOTALL)
    if addr_match:
        case_info['court_address'] = re.sub(r'\s+', ' ', addr_match.group(1)).strip()

    # ФИО судьи (учитываем запятые и разные форматы)
    judge_patterns = [
        r'Мировой судья\s+([А-Яа-яA-Za-z]+\s+[А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.?)',
        r'Мировой судья\s+([А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.\s*[А-Яа-яA-Za-z]+)',
        r'судья\s+([А-Яа-яA-Za-z]+\s+[А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.?)',
        r'([А-Яа-яA-Za-z]+\s+[А-ЯA-Za-z]\.\s*[А-ЯA-Za-z]\.?)[\s,]*рассмотрев',
    ]
    for pattern in judge_patterns:
        judge_match = re.search(pattern, header)
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

    # 4. СЛОВАРЬ (BOUNDED GAP БЕЗ ЦИФР В РАЗРЫВЕ)
    for regex, category, original_phrase in COMPILED_DICT_REGEXES:
        for match in regex.finditer(text):
            matched_text = match.group(0).strip()
            # Доп. проверка: длина найденного должна быть хотя бы 50% от длины фразы в словаре
            if len(matched_text) >= len(original_phrase) * 0.5:
                overlap = any(e.start_pos <= match.start() < e.end_pos or e.start_pos < match.end() <= e.end_pos for e in entities)
                if not overlap:
                    normal_form = normalize_matched_text(matched_text)
                    add_entity(matched_text, category, match.start(), match.end(), 0.9, normal_form=normal_form)

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