# Английский путь и оценка времени построения графа для чужих бенчмарков (отладка, без API)

## 1. Английский путь: 30 опорных абзацев MuSiQue (seed 0)

«Ответ шага — понятие»: все леммы ответа подвопроса, к которому относится абзац (мост многохопового вопроса), есть среди извлечённого из абзаца.

| | spaCy (en_core_web_sm 3.8.0, MIT) | старый извлекатель (правила, YAKE) |
|---|---|---|
| понятий на абзац | 23.6 | 26.6 |
| ответ шага — понятие | 26/30 | 25/30 |
| секунд на абзац (CPU) | 0.206 | 0.231 |

<details><summary>По абзацам</summary>

| абзац | ответ шага | spaCy | правила | понятий spaCy | понятий правила |
|---|---|---|---|---|---|
| Tonight You're Mine | Eric Carmen | да | да | 15 | 21 |
| Kentucky Tavern | Sazerac Company | да | да | 28 | 29 |
| Pfrang Association | Nanjing | да | да | 18 | 19 |
| United States Army | the Army | да | да | 33 | 32 |
| Standing Rules of the United States Sena | The Senate | да | да | 15 | 17 |
| Papa Roach | California | да | да | 13 | 28 |
| St. Peter's Basilica | Vatican City | да | да | 23 | 23 |
| Opening Day | Thursday, March 29 | да | — | 43 | 23 |
| WWNQ | Forest Acres | да | да | 10 | 22 |
| Nintendo Entertainment System | NES | да | да | 29 | 35 |
| Anwer Ali | Muhammad Shah | да | да | 16 | 26 |
| Yoshi (video game) | Nintendo Entertainment System | да | да | 18 | 18 |
| Indonesia–Timor Leste Commission of Trut | East Timor | да | да | 36 | 32 |
| Vatican City | 11 February 1929 | да | да | 19 | 25 |
| On and On and Beyond | Mac Miller | да | да | 25 | 19 |
| Galicia (Spain) | 406 | — | — | 48 | 39 |
| Tanzania | 54.7% | — | — | 36 | 33 |
| Adult contemporary music | Walter Sabo | да | да | 50 | 52 |
| Tucson, Arizona | 7,531 | — | — | 25 | 21 |
| Dazed and Confused (film) | Fred O'Bannion | да | да | 45 | 64 |
| Mickey Mouse | Walt Disney | да | да | 12 | 22 |
| Denmark national futsal team | FIFA | да | да | 10 | 21 |
| Ask the Fish | Leftover Salmon | да | да | 7 | 15 |
| Josip Broz Tito | Yugoslavia | да | да | 23 | 26 |
| Arizona | Phoenix | да | да | 34 | 35 |
| Olga Kuragina | USSR | да | да | 12 | 20 |
| Katsuaki Watanabe | Toyota | да | да | 8 | 11 |
| John Phan | Da Nang | да | да | 14 | 25 |
| History of Saudi Arabia | 1930 | — | — | 29 | 27 |
| List of municipalities in Georgia (U.S.  | Atlanta | да | да | 14 | 19 |

</details>

## 2. Оценка времени построения графа

Замер на этом CPU: spaCy ≈ 135 предложений/с, e5-base ≈ 13 предложений/с (≈ 0.083 с на предложение вместе). GPU Kaggle: e5-base 500–1000 предложений/с — типичная оценка, проверить первым запуском; spaCy остаётся на CPU. Запись в SQLite и PPR не учтены (секунды).

| набор | единиц | предложений | CPU | GPU Kaggle (e5) |
|---|---|---|---|---|
| MuSiQue | 11656 абзацев | 43693 | 1.0 ч | 6 мин–7 мин |
| 2Wiki | 6119 абзацев | 23071 | 32 мин | 3 мин–4 мин |
| LongMemEval-S, без кэша | 500 вопросов, 246750 реплик | 3213012 | 74.0 ч | 7.5 ч–8.4 ч |
| LongMemEval-S, кэш по сессиям | 19195 уникальных сессий | 2589930 | 59.6 ч | 6.0 ч–6.8 ч |

Кэш по сессиям для LongMemEval-S сокращает объём в 1.2 раза. Граф вопроса = слияние подграфов его сессий (дозапись); рёбра «синоним» и «subphrase» между сессиями достраиваются при слиянии.

**Поправка к ARCHITECTURE_V2 §6:** там для LongMemEval-S без кэша стояло ≈ 250 тыс. предложений — занижено; по данным 3213012.

Recall@5 по абзацам (как в HippoRAG 2) для MuSiQue / 2Wiki считается без вызовов LLM; переранжирование bge-reranker-v2-m3 — на GPU (на CPU ≈ 5 пар/с: 1 000 вопросов × 60 пар ≈ 3,3 ч).
