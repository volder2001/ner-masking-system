"""
ner-service/main.py
FINAL PRODUCTION v29: Чистый Yargy morph_pipeline (без ручных костылей морфологии) + Гибридный алгоритм связывания
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
# 2. ЧИСТЫЙ YARGY ДЛЯ СЛОВАРЯ (БЕЗ КОСТЫЛЕЙ)
# ==========================================
# Собираем все фразы из словаря в один список для morph_pipeline
all_phrases = []
phrase_to_category = {}

for category, phrases in dictionary.items():
    for phrase in phrases:
        # Очищаем фразу от лишней пунктуации для yargy
        clean_phrase = " ".join([w.strip(".,;:-") for w in phrase.split()])
        all_phrases.append(clean_phrase)
        phrase_to_category[clean_phrase.lower()] = category

# morph_pipeline САМ знает все формы слов! "госпошлина" найдет "госпошлину", "госпошлиной" и т.д.
DICT_PARSER = Parser(morph_pipeline(all_phrases))

def get_category_for_matched_text(matched_text: str) -> Optional[str]:
    """
    Определяет категорию (SUBJECT/NOT_SUBJECT) для найденного текста.
    Поскольку yargy нашел фразу в любой форме, мы лемматизируем найденный текст
    и сравниваем с лемматизированными фразами из словаря для точного определения категории.
    """
    # Лемматизируем найденный текст
    matched_lemmas = set()
    for w in re.findall(r'\w+', matched_text.lower()):
        p = morph.parse(w)[0]
        if 'PREP' not in p.tag.grammemes and 'CONJ' not in p.tag.grammemes:
            matched_lemmas.add(p.normal_form)

    if not matched_lemmas:
        return None

    # Ищем совпадение в словаре
    for category, phrases in dictionary.items():
        for phrase in phrases:
            clean_phrase = " ".join([w.strip(".,;:-") for w in phrase.split()])
            phrase_lemmas = set()
            for w in re.findall(r'\w+', clean_phrase.lower()):
                p = morph.parse(w)[0]
                if 'PREP' not in p.tag.grammemes and 'CONJ' not in p.tag.grammemes:
                    phrase_lemmas.add(p.normal_form)

            # Если леммы совпадают (или одна содержит другую), это наша категория
            if matched_lemmas & phrase_lemmas:
                return category

    return None

# ==========================================
# 3. ОСТАЛЬНЫЕ ПАТТЕРНЫ
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

MONEY_FALLBACKS = [
    re.compile(r'(\d{1,3}(?:\s?\d{3})*(?:[.,]\d{2})?)\s+(?:руб\.?|рублей|коп\.?|копеек|py6\.?|kon\.?)', re.IGNORECASE),
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

    addr_match = re.search(r'((?:ул\.|улица|г\.|гор\.|город|пр\.|проспект|д\.|дом|обл\.|область).*?\d{6})', header, re.IGNORECASE | re.DOTALL)
    if addr_match:
        case_info['court_address'] = re.sub(r'\s+', ' ', addr_match.group(1)).strip()

    judge_patterns = [
        r'([А-Я]\.\s*[А-Я]\.\s*[А-Я][а-я]+)',
        r'Мировой\s+судья.*?([А-Я][а-я]{2,})\s+(?:рассмотрев|подписал|вынес)',
        r'судья\s+([А-Я][а-я]+\s+[А-Я]\.\s*[А-Я]\.)',
    ]
    for pattern in judge_patterns:
        judge_match = re.search(pattern, header)
        if judge_match:
            case_info['judge_name'] = re.sub(r'\s+', ' ', judge_match.group(1)).strip()
            break

    if not case_info['judge_name']:
        footer_match = re.search(r'([А-Я]\.\s*[А-Я]\.\s*[А-Я][а-я]+)', text)
        if footer_match:
            case_info['judge_name'] = re.sub(r'\s+', ' ', footer_match.group(1)).strip()

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
            text_match = match.group(0)
            start, end = match.start(), match.end()

            text_match = re.sub(r'\s+', ' ', text_match).strip()
            real_start = start + resolution_start
            real_end = end + resolution_start

            overlap = any(e.start_pos <= real_start < e.end_pos for e in entities if e.type.startswith('MONEY'))
            if not overlap:
                add_entity(text_match, 'MONEY_RUB', real_start, real_end, 0.90, currency='RUB', normal_form=text_match)

    # 1. SUBJECT / NOT_SUBJECT через ЧИСТЫЙ YARGY
    for match in DICT_PARSER.findall(resolution_text):
        matched_text = resolution_text[match.span.start:match.span.stop]

        # Определяем категорию через лемматизацию (это надежно и использует силу yargy для поиска)
        category = get_category_for_matched_text(matched_text)

        if category:
            # Нормализуем найденный текст для normal_form
            norm_words = []
            for w in matched_text.split():
                clean_w = re.sub(r'[^\w]', '', w)
                if clean_w:
                    p = morph.parse(clean_w)[0]
                    if 'PREP' not in p.tag.grammemes and 'CONJ' not in p.tag.grammemes:
                        norm_words.append(p.normal_form)
            normal_form = " ".join(norm_words) if norm_words else matched_text

            add_entity(matched_text, category, match.span.start + resolution_start, match.span.stop + resolution_start, 0.9, normal_form=normal_form)

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

    # ГИБРИДНЫЙ АЛГОРИТМ
    if len(subjects) == len(money_entities) and len(subjects) > 0:
        final_pairs = []
        for i, subj in enumerate(subjects):
            money = money_entities[i]
            context_start = min(subj.start_pos, money.start_pos)
            context_end = max(subj.end_pos, money.end_pos) + 40
            distance = abs(money.start_pos - subj.end_pos)

            final_pairs.append(SubjectMoneyPair(
                subject=subj,
                money=money,
                distance=distance,
                context=text[context_start:context_end]
            ))
        return final_pairs

    final_pairs = []
    used_moneys = set()
    MAX_DISTANCE = 150

    for subj in subjects:
        best_money = None
        best_distance = 9999

        for money in money_entities:
            if id(money) in used_moneys:
                continue

            dist_forward = money.start_pos - subj.end_pos
            dist_backward = subj.start_pos - money.end_pos

            if 0 <= dist_forward <= MAX_DISTANCE and dist_forward < best_distance:
                best_money = money
                best_distance = dist_forward
            elif 0 <= dist_backward <= MAX_DISTANCE and dist_backward < best_distance:
                best_money = money
                best_distance = dist_backward

        if best_money is not None:
            used_moneys.add(id(best_money))
            context_start = min(subj.start_pos, best_money.start_pos)
            context_end = max(subj.end_pos, best_money.end_pos) + 40

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