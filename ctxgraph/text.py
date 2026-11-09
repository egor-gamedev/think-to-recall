"""Токенизация, леммы и канонические имена (русский — pymorphy3, английский — простая нормализация)."""

from __future__ import annotations

import re
from functools import lru_cache

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_'-]*")
_CYR = re.compile(r"[А-Яа-яЁё]")
# Граница camelCase — перед заглавной, за которой строчная: parseConfig → parse|Config,
# а PostgreSQL, iOS, JSON — целиком.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z][a-z])")

STOP_RU = set(
    [
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "с",
        "со",
        "как",
        "а",
        "то",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "по",
        "только",
        "ее",
        "мне",
        "было",
        "вот",
        "от",
        "меня",
        "еще",
        "нет",
        "о",
        "из",
        "ему",
        "теперь",
        "когда",
        "даже",
        "ну",
        "вдруг",
        "ли",
        "если",
        "уже",
        "или",
        "ни",
        "быть",
        "был",
        "него",
        "до",
        "вас",
        "нибудь",
        "опять",
        "уж",
        "вам",
        "ведь",
        "там",
        "потом",
        "себя",
        "ничего",
        "ей",
        "может",
        "они",
        "тут",
        "где",
        "есть",
        "надо",
        "ней",
        "для",
        "мы",
        "тебя",
        "их",
        "чем",
        "была",
        "сам",
        "чтоб",
        "без",
        "будто",
        "чего",
        "раз",
        "тоже",
        "себе",
        "под",
        "будет",
        "ж",
        "тогда",
        "кто",
        "этот",
        "того",
        "потому",
        "этого",
        "какой",
        "совсем",
        "ним",
        "здесь",
        "этом",
        "один",
        "почти",
        "мой",
        "тем",
        "чтобы",
        "нее",
        "сейчас",
        "были",
        "куда",
        "зачем",
        "всех",
        "никогда",
        "можно",
        "при",
        "наконец",
        "два",
        "об",
        "другой",
        "хоть",
        "после",
        "над",
        "больше",
        "тот",
        "через",
        "эти",
        "нас",
        "про",
        "всего",
        "них",
        "какая",
        "много",
        "разве",
        "три",
        "эту",
        "моя",
        "впрочем",
        "хорошо",
        "свою",
        "этой",
        "перед",
        "иногда",
        "лучше",
        "чуть",
        "том",
        "нельзя",
        "такой",
        "им",
        "более",
        "всегда",
        "конечно",
        "всю",
        "между",
        "это",
        "весь",
        "свой",
        "который",
        "наш",
        "также",
        "каждый",
        "очень",
        "нужно",
        "мочь",
    ]
)
STOP_EN = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "me",
        "my",
        "our",
        "your",
        "their",
        "his",
        "her",
        "them",
        "us",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "not",
        "no",
        "yes",
        "so",
        "than",
        "too",
        "very",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "just",
        "also",
        "now",
        "new",
        "use",
        "used",
        "using",
        "via",
        "per",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "while",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
    ]
)
#: Служебные слова языков программирования: как понятия бесполезны.
STOP_CODE = set(
    [
        "def",
        "class",
        "import",
        "return",
        "self",
        "none",
        "null",
        "true",
        "false",
        "function",
        "const",
        "var",
        "let",
        "async",
        "await",
        "elif",
        "lambda",
        "yield",
        "pass",
        "raise",
        "try",
        "except",
        "finally",
        "public",
        "private",
        "static",
        "void",
        "int",
        "str",
        "string",
        "bool",
        "dict",
        "list",
        "println",
        "printf",
        "print",
        "begin",
        "end",
        "select",
        "insert",
        "update",
        "delete",
        "where",
    ]
)
#: Наречия и служебные слова, которых не хватало в списках выше (из разметки рёбер, раунд 1).
STOP_EXTRA = set(
    "currently plus minus already still yet never always often usually maybe perhaps really "
    "actually basically just only even else etc via per like unlike within without whether "
    "дважды трижды однажды сейчас потом затем снова уже ещё еще очень просто примерно около "
    "весьма вообще именно лишь тоже также поэтому однако хотя".split()
)
STOP_ALL = STOP_RU | STOP_EN | STOP_CODE | STOP_EXTRA

#: Слишком общие слова: как понятие-подсказка ничего не дают (опыт, якоря).
GENERIC = set(
    "data year time way thing value number part case type example item kind stuff something "
    "info information result results lot amount use user users new old first last "
    "год время способ вещь часть случай пример тип вид штука данные информация результат "
    "количество раз дело вопрос момент".split()
)


def is_cyrillic(s: str) -> bool:
    return bool(_CYR.search(s))


def detect_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    cyr = sum(1 for c in letters if _CYR.match(c))
    return "ru" if cyr / len(letters) > 0.3 else "en"


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    w = word.lower().replace("ё", "е")
    if is_cyrillic(w):
        return _morph().parse(w)[0].normal_form.replace("ё", "е")
    return _en_norm(w)


#: Служебные имена колонок и полей: в опыте это не понятия («created at — default»).
STOP_COLUMNS = set(
    "created at|updated at|deleted at|modified at|inserted at|default|auto|null|pk|fk|id|uuid|"
    "timestamp|key|value|meta|metadata|index|enabled|flag|count|total|ordering|related name".split(
        "|"
    )
)


@lru_cache(maxsize=50_000)
def en_noun_ok(word: str) -> bool:
    """Английское слово — существительное по WordNet: смысл-существительное есть и он не реже
    других частей речи (частота по SemCor, при нуле — число смыслов). Слова, которых нет в
    WordNet (redis, idempotency, login), считаются терминами и проходят."""
    try:
        from ctxgraph.sources.wordnet import _wn

        wn = _wn()
    except LookupError:  # данных WordNet нет — не фильтруем
        return True
    w = word.lower()
    scores = {}
    for p in ("n", "v", "a", "s", "r"):
        forms = set(wn._morphy(w, p)) | {w}
        syns = [s for f in forms for s in wn.synsets(f, p)]
        cnt = sum(lm.count() for s in syns for lm in s.lemmas() if lm.name().lower() in forms)
        scores[p] = (cnt, len(syns))
    if not any(n for _, n in scores.values()):
        return True
    noun = scores.pop("n")
    return noun[1] > 0 and noun >= max(scores.values())


#: Слова на -s, которые не множественное число.
EN_NO_STRIP = set(
    "kubernetes postgres redis aws https news series species status alias analysis canvas atlas "
    "sales jenkins nginx ios macos windows always perhaps bus gas yes its this has was does".split()
)
SQL_KW = set(
    "select create table insert update delete from where join alter drop index primary foreign "
    "references values into returning group order having limit offset distinct union".split()
)


def _en_norm(w: str) -> str:
    if w in EN_NO_STRIP:
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "shes", "ches", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def pos(word: str) -> str | None:
    if not is_cyrillic(word):
        return None
    return _morph().parse(word.lower())[0].tag.POS


def split_identifier(token: str) -> list[str]:
    """`parseConfig` / `parse_config` / `parse-config` → ['parse', 'config']."""
    parts: list[str] = []
    for chunk in re.split(r"[_\-.]+", token):
        parts.extend(p for p in _CAMEL.split(chunk) if p)
    return [p.lower() for p in parts]


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def content_lemmas(text: str) -> list[str]:
    """Леммы значимых слов (без стоп-слов и коротких)."""
    out = []
    for w in words(text):
        for part in split_identifier(w) if not is_cyrillic(w) else [w]:
            lem = lemma(part)
            if len(lem) < 3 or lem in STOP_ALL or lem.isdigit():
                continue
            out.append(lem)
    return out


def _ru_phrase(ws: list[str]) -> str:
    """Русская именная группа → начальная форма с согласованием.

    «головной боли» → «головная боль», «возвраты средств» → «возврат средств»:
    прилагательные до главного существительного согласуются с ним в именительном падеже,
    зависимые слова после него остаются в своей форме.
    """
    m = _morph()
    parses = [m.parse(w.lower())[0] for w in ws]
    head = next((i for i, p in enumerate(parses) if p.tag.POS == "NOUN"), None)
    if head is None:
        return " ".join(lemma(w) for w in ws)
    hp = parses[head]
    pltm = "Pltm" in hp.tag
    norm = hp.inflect({"nomn", "plur" if pltm else "sing"}) or hp
    out: list[str] = []
    for i, (w, p) in enumerate(zip(ws, parses, strict=True)):
        if i == head:
            out.append(norm.word)
        elif i < head and p.tag.POS in ("ADJF", "PRTF"):
            feats = (
                {"nomn", "plur"}
                if pltm or norm.tag.gender is None
                else {"nomn", "sing", norm.tag.gender}
            )
            infl = p.inflect(feats)
            out.append(infl.word if infl else p.normal_form)
        elif i < head:
            out.append(lemma(w))
        else:
            out.append(w.lower())
    return " ".join(out).replace("ё", "е")


def canonical(phrase: str) -> str:
    """Каноническое имя фразы. `parse_config` = `parseConfig` = `parse config`;
    русская фраза из нескольких слов — с согласованием (см. `_ru_phrase`)."""
    ws = words(phrase)
    if len(ws) > 1 and all(is_cyrillic(w) for w in ws):
        return _ru_phrase(ws)
    n_cyr = _cyr_prefix(ws)
    if len(ws) > 1 and n_cyr and all(not is_cyrillic(w) for w in ws[n_cyr:]):
        # «резервное копирование PostgreSQL»: русская группа + латинский зависимый в конце
        ru = _ru_phrase(ws[:n_cyr]) if n_cyr > 1 else lemma(ws[0])
        return " ".join([ru] + [w.lower() for w in ws[n_cyr:]])
    toks: list[str] = []
    for w in ws:
        parts = split_identifier(w) if not is_cyrillic(w) else [w]
        toks.extend(lemma(p) for p in parts)
    return " ".join(t for t in toks if t)


def _cyr_prefix(ws: list[str]) -> int:
    n = 0
    while n < len(ws) and is_cyrillic(ws[n]):
        n += 1
    return n


def is_noun_phrase(phrase: str) -> bool:
    """Похожа ли фраза на понятие: именная группа без глаголов, предлогов и повторов.

    Русский: (прил.)* сущ. (сущ./прил. в род. падеже)*; прилагательное согласовано
    с существительным по падежу. «литров начинающим» — нет. Английский: нет стоп-слов,
    однобуквенных токенов и повторов («pod is alive», «Shopline Shopline» — нет).
    """
    ws = words(phrase)
    if not ws or len(ws) > 4 or "." in phrase:
        return False
    low = [w.lower() for w in ws]
    if len(set(low)) < len(low) or any(len(w) < 2 for w in low):
        return False
    if any(w.isupper() and w.lower() in SQL_KW for w in ws):
        return False  # «CREATE TABLE products» — синтаксис SQL, не понятие
    if not any(is_cyrillic(w) for w in ws):
        return not any(lemma(w) in STOP_ALL for w in ws)
    n_cyr = _cyr_prefix(ws)
    if n_cyr < len(ws):
        # Смешанная фраза: латиница допустима только в конце («таблица payments»).
        if n_cyr == 0 or any(is_cyrillic(w) for w in ws[n_cyr:]):
            return False
        if any(lemma(w) in STOP_ALL for w in ws[n_cyr:]):
            return False
        return is_noun_phrase(" ".join(ws[:n_cyr]))
    m = _morph()
    parses = [m.parse(w)[0] for w in low]
    tags = [p.tag.POS for p in parses]
    if "NOUN" not in tags:
        return False
    head = tags.index("NOUN")

    def cases(i: int) -> set:
        # Падеж по всем вариантам разбора этой части речи: «резервное» — и им., и вин.
        return {p.tag.case for p in m.parse(low[i]) if tags[i] == p.tag.POS}

    head_cases = cases(head)
    for i in range(len(low)):
        if i < head:
            if tags[i] not in ("ADJF", "PRTF") or not cases(i) & head_cases:
                return False
        elif i > head and (tags[i] not in ("NOUN", "ADJF", "PRTF") or "gent" not in cases(i)):
            return False
    return True


#: Поправка по usage.prompt_tokens DeepSeek V4.1 Flash на 10 меню (2026-10-18):
#: без неё оценка занижала на 17% (reports/data/token_calibration.json).
TOKEN_SCALE = 1.167


def approx_tokens(text: str) -> int:
    """Оценка числа токенов (BPE): ~4 символа на токен для латиницы, ~2.5 для кириллицы,
    умноженная на `TOKEN_SCALE`, откалиброванный по токенизатору DeepSeek.
    """
    cyr = len(_CYR.findall(text))
    other = len(text) - cyr
    return int(round((cyr / 2.5 + other / 4.0) * TOKEN_SCALE))
