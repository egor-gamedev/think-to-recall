# Диалог загрузки: Telegram-бот записи в стоматологию «Белая линия» (aiogram 3 + учётная система «СтомаОфис»)

## S1

**[S1.0] Пользователь:**

Привет. Взял заказ на аутсорсе: Telegram-бот записи для стоматологии «Белая линия». Договор с фиксированной суммой, сдача 15 декабря, затем 3 месяца гарантийной поддержки. Бот на Python и aiogram 3, работает через вебхук, VPS уже взял в Selectel (Москва). Подскажи нормальную структуру проекта, чтобы потом не было каши.

**[S1.1] Ассистент:**

Для бота такого размера хватит плоской структуры по слоям: хэндлеры отдельно, работа с учётной системой отдельно, клавиатуры отдельно.

```
clinic_bot/
  __main__.py        # запуск: aiohttp-приложение + вебхук
  config.py          # настройки из .env (pydantic-settings)
  handlers/
    start.py         # /start, опознание пациента
    booking.py       # сценарий записи
    my_visits.py     # мои записи, отмена
  keyboards/
    calendar.py
    common.py
  clinic/            # всё, что касается учётной системы
    repo.py
    models.py
  services/
    reminders.py
```

Запуск через вебхук на aiohttp:

```python
# clinic_bot/__main__.py
from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from .config import settings
from .handlers import start, booking, my_visits

async def on_startup(bot: Bot):
    await bot.set_webhook(
        f"{settings.base_url}/tg/webhook",
        secret_token=settings.webhook_secret,
        drop_pending_updates=True,
    )

def main():
    bot = Bot(settings.bot_token)
    dp = Dispatcher()
    dp.include_routers(start.router, booking.router, my_visits.router)
    dp.startup.register(on_startup)

    app = web.Application()
    SimpleRequestHandler(dp, bot, secret_token=settings.webhook_secret).register(app, path="/tg/webhook")
    setup_application(app, dp, bot=bot)
    web.run_app(app, host="127.0.0.1", port=8080)

if __name__ == "__main__":
    main()
```

Наружу — nginx с сертификатом Let's Encrypt, проксирует `/tg/webhook` на `127.0.0.1:8080`. `secret_token` обязателен: Telegram кладёт его в заголовок `X-Telegram-Bot-Api-Secret-Token`, и `SimpleRequestHandler` отбрасывает чужие запросы.

**[S1.2] Пользователь:**

Ок. Теперь самое интересное. Учётная система у клиники — «СтомаОфис», стоит на сервере в самой клинике, под ней база Firebird. Договорились так: расписание читаем и записи создаём напрямую в базе, через туннель WireGuard от VPS до сервера клиники. Туннель айтишник клиники уже поднял. Как к Firebird подключаться из асинхронного aiogram?

**[S1.3] Ассистент:**

Нормальных асинхронных драйверов для Firebird нет, поэтому берём официальный синхронный `firebird-driver` и выносим вызовы в поток, чтобы не блокировать event loop.

```python
# clinic_bot/clinic/repo.py
import asyncio
from contextlib import contextmanager
from firebird.driver import connect, driver_config

from ..config import settings

driver_config.server_defaults.host.value = settings.fb_host   # адрес сервера внутри WireGuard, например 10.8.0.2
driver_config.server_defaults.port.value = "3050"

@contextmanager
def _conn():
    con = connect(settings.fb_database, user=settings.fb_user, password=settings.fb_password, charset="WIN1251")
    try:
        yield con
    finally:
        con.close()

def _fetch(sql: str, params: tuple = ()):
    with _conn() as con:
        cur = con.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

async def fetch(sql: str, params: tuple = ()):
    return await asyncio.to_thread(_fetch, sql, params)
```

Пара замечаний:

- Кодировка: старые учётные системы почти всегда в `WIN1251`. Если увидите кракозябры в ФИО — дело в `charset`.
- Соединение на каждый запрос через туннель — это 30–80 мс накладных. Для начала нормально; если станет узким местом, сделаем небольшой пул на `queue.Queue` из 3–4 соединений.
- Проверьте с VPS: `nc -vz 10.8.0.2 3050`. Если порт закрыт — брандмауэр Windows на сервере клиники, туда нужно правило для интерфейса WireGuard.

**[S1.4] Пользователь:**

Подключился, ФИО читаются. Ещё деталь: у клиники два филиала — Калининград (ул. Горького) и Зеленоградск. Часть врачей работает в обоих, например терапевт Соколова: пн–ср на Горького, чт–пт в Зеленоградске. В таблице SCHEDULE филиал — поле BRANCH_ID.

**[S1.5] Ассистент:**

Тогда филиал — полноценный параметр каждого запроса к расписанию, а не свойство врача. Модель слота:

```python
# clinic_bot/clinic/models.py
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Slot:
    doctor_id: int
    branch_id: int
    start: datetime
    minutes: int
```

Запрос свободных слотов врача за день:

```python
SLOTS_SQL = """
    SELECT s.DOCTOR_ID, s.BRANCH_ID, s.START_TIME, s.DURATION
    FROM SCHEDULE s
    WHERE s.DOCTOR_ID = ? AND s.BRANCH_ID = ?
      AND s.START_TIME >= ? AND s.START_TIME < ?
      AND s.PATIENT_ID IS NULL AND s.IS_BLOCKED = 0
    ORDER BY s.START_TIME
"""
```

Имена колонок я пишу по вашему описанию — сверьте с реальной схемой (`SHOW TABLE SCHEDULE` в isql). Адреса филиалов держите в конфиге, а не в базе:

```python
BRANCHES = {
    1: "Калининград, ул. Горького",
    2: "Зеленоградск",
}
```

**[S1.6] Пользователь:**

Дальше — как опознавать пациента. Решили: по номеру из кнопки «Поделиться контактом», и ищем карту в учётке по телефону. Набросай клавиатуру и хэндлер.

**[S1.7] Ассистент:**

```python
# clinic_bot/handlers/start.py
from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove

from ..clinic import patients

router = Router()

contact_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="📱 Поделиться контактом", request_contact=True)]],
    resize_keyboard=True, one_time_keyboard=True,
)

@router.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Здравствуйте! Чтобы записаться, нажмите кнопку ниже — по номеру мы найдём вашу карту.",
        reply_markup=contact_kb,
    )

@router.message(F.contact)
async def got_contact(message: Message):
    if message.contact.user_id != message.from_user.id:
        await message.answer("Пожалуйста, отправьте свой контакт кнопкой, а не чужой.")
        return
    cards = await patients.find_by_phone(message.contact.phone_number)
    ...
```

Проверка `contact.user_id == from_user.id` важна: иначе можно переслать чужой контакт и увидеть чужие записи. Кнопка `request_contact` работает только в личном чате с ботом и только в мобильных клиентах и десктопе — в веб-версии Telegram она иногда не отображается, на это стоит дать текстовую подсказку.

**[S1.8] Пользователь:**

Сделал поиск WHERE PHONE = ? — находит меньше половины. Полез смотреть: телефоны в учётке записаны вразнобой — 8…, +7…, с пробелами и скобками, где-то вообще «8 (4012) …». Договорились с регистратурой сравнивать по последним 10 цифрам.

**[S1.9] Ассистент:**

Сравнение по последним 10 цифрам — разумно, но в SQL это делать неудобно: в Firebird нет нормальной регулярки, а вычисляемый индекс на таблицу вендора добавлять не стоит — это их схема, при обновлении учётки может слететь или сломать что-то у них.

Проще держать индекс в памяти бота: при старте и раз в час читать `ID, PHONE` всех карт и строить словарь.

```python
# clinic_bot/clinic/patients.py
import re
from collections import defaultdict

from .repo import fetch

_index: dict[str, list[int]] = defaultdict(list)

def norm(phone: str) -> str | None:
    digits = re.sub(r"\D", "", phone or "")
    return digits[-10:] if len(digits) >= 10 else None

async def rebuild_index():
    rows = await fetch("SELECT ID, PHONE FROM PATIENTS WHERE PHONE IS NOT NULL")
    new = defaultdict(list)
    for r in rows:
        key = norm(r["PHONE"])
        if key:
            new[key].append(r["ID"])
    _index.clear()
    _index.update(new)

async def find_by_phone(phone: str) -> list[int]:
    key = norm(phone)
    return list(_index.get(key, [])) if key else []
```

Для ~8 тысяч карт это доли мегабайта памяти и один запрос в час. Городские номера вида «8 (4012) 12-34-56» тоже дадут 10 цифр, но с мобильным Telegram они не совпадут — это нормально.

**[S1.10] Пользователь:**

Работает, нашлись почти все. Теперь выбор даты. Заказчик просил «красивый календарь как в приложении». Я думал про Telegram Mini App на React для выбора слота, но отказались делать: после сдачи его некому поддерживать. Так что только inline-кнопки. Накидай календарь на inline-кнопках.

**[S1.11] Ассистент:**

Сделаем календарь на `CallbackData` — aiogram сам упакует и распакует данные кнопки.

```python
# clinic_bot/keyboards/calendar.py
import calendar
from datetime import date
from aiogram.filters.callback_data import CallbackData
from aiogram.utils.keyboard import InlineKeyboardBuilder

class DayCb(CallbackData, prefix="day"):
    y: int
    m: int
    d: int

class MonthCb(CallbackData, prefix="mon"):
    y: int
    m: int

MONTHS = ["", "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
          "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]

def month_kb(y: int, m: int, available: set[date]):
    kb = InlineKeyboardBuilder()
    kb.button(text=f"{MONTHS[m]} {y}", callback_data="noop")
    for wd in ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]:
        kb.button(text=wd, callback_data="noop")
    for week in calendar.monthcalendar(y, m):
        for d in week:
            if d and date(y, m, d) in available:
                kb.button(text=str(d), callback_data=DayCb(y=y, m=m, d=d))
            else:
                kb.button(text=" " if not d else "·", callback_data="noop")
    prev_y, prev_m = (y, m - 1) if m > 1 else (y - 1, 12)
    next_y, next_m = (y, m + 1) if m < 12 else (y + 1, 1)
    kb.button(text="◀", callback_data=MonthCb(y=prev_y, m=prev_m))
    kb.button(text="▶", callback_data=MonthCb(y=next_y, m=next_m))
    kb.adjust(1, 7, *[7] * len(calendar.monthcalendar(y, m)), 2)
    return kb.as_markup()
```

Дни без свободных слотов показываются точкой и не кликаются. Не забудьте пустой хэндлер на `noop`, иначе у пользователя будет крутиться часик:

```python
@router.callback_query(F.data == "noop")
async def noop(cb: CallbackQuery):
    await cb.answer()
```

**[S1.12] Пользователь:**

Глянул на телефоне — кнопок многовато, 31 день плюс заголовки, экран целиком. Пациенты у клиники в основном 40+, им мелко.

**[S1.13] Ассистент:**

Для аудитории 40+ месячная сетка действительно перебор. Лучше показывать неделю и только дни, где есть свободное время, — крупными кнопками по одной-две в ряд:

```python
WD = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

def days_kb(days: list[date], page: int):
    kb = InlineKeyboardBuilder()
    chunk = days[page * 6:(page + 1) * 6]
    for d in chunk:
        kb.button(text=f"{WD[d.weekday()]}, {d.day} {MONTHS_GEN[d.month]}",
                  callback_data=DayCb(y=d.year, m=d.month, d=d.day))
    if page > 0:
        kb.button(text="◀ Раньше", callback_data=PageCb(p=page - 1))
    if (page + 1) * 6 < len(days):
        kb.button(text="Позже ▶", callback_data=PageCb(p=page + 1))
    kb.adjust(2, 2, 2, 2)
    return kb.as_markup()
```

Подписи вида «чт, 14 ноября» читаются без напряжения, а пустые дни просто не попадают в список. Время внутри дня — тоже кнопками, по три в ряд: «09:00 10:30 12:00».

## S2

**[S2.0] Пользователь:**

Поговорил с главврачом, Олегом Викторовичем. Через бота записываются только к терапевтам и гигиенисту; к хирургам и имплантологам — через регистратуру, там надо сначала с человеком поговорить. Список врачей беру из таблицы DOCTORS, там есть поле SPEC с кодом специальности.

**[S2.1] Ассистент:**

Тогда фильтр по специальности — в одном месте, в репозитории, чтобы его не пришлось искать по хэндлерам:

```python
# clinic_bot/clinic/doctors.py
from .repo import fetch
from ..config import settings

# коды из справочника SPECIALITY; сверить с базой
BOOKABLE_SPECS = settings.bookable_specs   # например {"TER", "GIG"}

DOCTORS_SQL = """
    SELECT d.ID, d.FIO, d.SPEC
    FROM DOCTORS d
    WHERE d.IS_ACTIVE = 1
    ORDER BY d.FIO
"""

async def bookable_doctors(branch_id: int) -> list[dict]:
    rows = await fetch(DOCTORS_SQL)
    rows = [r for r in rows if r["SPEC"] in BOOKABLE_SPECS]
    # врач показывается в филиале, только если у него там есть сетка
    has_grid = {r["DOCTOR_ID"] for r in await fetch(
        "SELECT DISTINCT DOCTOR_ID FROM SCHEDULE WHERE BRANCH_ID = ? AND START_TIME >= CURRENT_DATE",
        (branch_id,),
    )}
    return [r for r in rows if r["ID"] in has_grid]
```

Коды специальностей вынесены в `.env` (`BOOKABLE_SPECS=TER,GIG`) — если клиника передумает, это правка настройки, а не кода. FSM записи тогда выглядит так:

```python
class Booking(StatesGroup):
    card = State()      # кого записываем
    branch = State()
    doctor = State()
    day = State()
    time = State()
    confirm = State()
```

**[S2.2] Пользователь:**

Нашёл засаду с картами: на один телефон в учётке бывает несколько карт — члены семьи, муж записан на телефон жены, пожилые родители на телефоне сына. Пусть бот спрашивает, кого записать.

**[S2.3] Ассистент:**

Да, `find_by_phone` и так возвращает список — показываем его кнопками. ФИО целиком в чате показывать не обязательно, достаточно имени и отчества с первой буквой фамилии — человек своих узнает:

```python
class CardCb(CallbackData, prefix="card"):
    id: int

def short_name(fio: str) -> str:
    parts = fio.split()
    if len(parts) >= 3:
        return f"{parts[1]} {parts[2]} {parts[0][0]}."
    return fio

async def ask_card(message: Message, state: FSMContext, card_ids: list[int]):
    cards = await patients.get_cards(card_ids)   # ID, FIO, BIRTH_DATE
    if len(cards) == 1:
        await state.update_data(card_id=cards[0]["ID"])
        await choose_branch(message, state)
        return
    kb = InlineKeyboardBuilder()
    for c in cards:
        kb.button(text=short_name(c["FIO"]), callback_data=CardCb(id=c["ID"]))
    kb.adjust(1)
    await state.set_state(Booking.card)
    await message.answer("На этот номер несколько карт. Кого записываем?", reply_markup=kb.as_markup())
```

В обработчике `CardCb` обязательно проверьте, что выбранный `id` действительно из списка этого телефона (храните список в `state`) — callback-данные можно подделать.

**[S2.4] Пользователь:**

Ок. Олег Викторович, кстати, по поводу карт сказал: всех, кому меньше 18, отправляем в «Карапуз» на Театральной, у нас даже детского кресла нет, и лицензию на детский приём не продлевали. BIRTH_DATE в карте есть, если что. Ладно, это потом. Сейчас другое: MemoryStorage для FSM теряет состояние при каждом рестарте, пациент посреди записи получает «не понимаю». Чем заменить?

**[S2.5] Ассистент:**

В aiogram 3 из коробки есть `RedisStorage` — это самый частый выбор:

```python
from aiogram.fsm.storage.redis import RedisStorage

storage = RedisStorage.from_url("redis://localhost:6379/1", state_ttl=3600, data_ttl=3600)
dp = Dispatcher(storage=storage)
```

`state_ttl` и `data_ttl` в час — чтобы брошенные на середине записи состояния сами чистились и не висели неделями. Redis на том же VPS: `apt install redis-server`, слушать только `127.0.0.1`.

И на случай, если состояние всё же потерялось, сделайте «мягкий» обработчик последнего уровня: любой непонятный callback отвечает «Сессия устарела, начнём заново» и показывает главное меню, а не молчит.

**[S2.6] Пользователь:**

Сделал. Следующее: новых пациентов, у кого карты нет, бот записывает только на первичную консультацию 30 минут. Карту регистратура заводит, когда человек пришёл. Как записывать без PATIENT_ID?

**[S2.7] Ассистент:**

Если в SCHEDULE есть поле под комментарий (часто `NOTE` или `COMMENT`), занимаем слот без карты и пишем туда, кто это:

```python
BOOK_NEW_SQL = """
    UPDATE SCHEDULE
    SET PATIENT_ID = NULL, VISIT_TYPE = ?, NOTE = ?, IS_BLOCKED = 1
    WHERE ID = ? AND PATIENT_ID IS NULL AND IS_BLOCKED = 0
"""

async def book_new_patient(slot_id: int, name: str, phone: str) -> bool:
    note = f"Бот: новый пациент {name}, {phone}"
    n = await execute(BOOK_NEW_SQL, (settings.first_visit_type, note, slot_id))
    return n == 1
```

Условие `PATIENT_ID IS NULL AND IS_BLOCKED = 0` в самом `UPDATE` — защита от гонки: если двое нажали на один слот, обновится строка только у первого, второму `n == 0` — показываем «время только что заняли».

Первичная консультация — 30 минут, а у врачей сетка может быть по 15 минут. Тогда новичку нужно два подряд идущих свободных слота; функция подбора:

```python
def pairs_for_30(slots: list[Slot]) -> list[Slot]:
    out = []
    for a, b in zip(slots, slots[1:]):
        if a.minutes == 15 and b.start == a.start + timedelta(minutes=15):
            out.append(a)
        elif a.minutes >= 30:
            out.append(a)
    return out
```

Как именно «СтомаОфис» хранит запись без карты, лучше посмотреть по уже существующим записям, которые регистратура вносила по телефону.

**[S2.8] Пользователь:**

Посмотрел, у них так и делают — NOTE с именем. Теперь напоминания. Пока делаем просто: одно сообщение в боте за 24 часа до приёма. Думал взять APScheduler и на каждую запись ставить job на «приём минус сутки».

**[S2.9] Ассистент:**

Можно, но разовые job'ы на каждую запись — хрупкая схема: запись перенесли или отменили в регистратуре, а job остался; бот упал — job'ы из памяти пропали. Надёжнее одна периодическая задача, которая сама смотрит в расписание:

```python
# clinic_bot/services/reminders.py
from datetime import datetime, timedelta

REMIND_SQL = """
    SELECT s.ID, s.PATIENT_ID, s.START_TIME, s.BRANCH_ID, d.FIO AS DOCTOR
    FROM SCHEDULE s JOIN DOCTORS d ON d.ID = s.DOCTOR_ID
    WHERE s.START_TIME BETWEEN ? AND ? AND s.PATIENT_ID IS NOT NULL
"""

async def tick(bot: Bot):
    now = datetime.now()
    rows = await fetch(REMIND_SQL, (now + timedelta(hours=23, minutes=50), now + timedelta(hours=24, minutes=10)))
    for r in rows:
        if await sent.was_sent(r["ID"], "24h"):
            continue
        tg_id = await links.tg_for_card(r["PATIENT_ID"])
        if not tg_id:
            continue
        await bot.send_message(tg_id, f"Напоминаем: завтра в {r['START_TIME']:%H:%M} приём у врача {r['DOCTOR']}.")
        await sent.mark(r["ID"], "24h")
```

Запуск — каждые 5 минут:

```python
scheduler = AsyncIOScheduler()
scheduler.add_job(tick, "interval", minutes=5, args=[bot])
scheduler.start()
```

Окно ±10 минут перекрывает интервал в 5 минут, а флаг «уже отправлено» не даёт продублировать.

**[S2.10] Пользователь:**

Логично. Пара деталей по времени: в SCHEDULE START_TIME хранится просто TIMESTAMP без пояса — регистратура вбивает его так, как видит на часах в холле. А APScheduler у меня берёт пояс системы, на VPS я ничего не трогал, как Selectel поставил, так и стоит. Вопрос не про это: флаг «уже отправлено» где хранить? Добавить колонку в SCHEDULE?

**[S2.11] Ассистент:**

Нет, в схему вендора колонок не добавляем: при обновлении «СтомаОфиса» их миграция может упасть на незнакомой колонке или молча её удалить, и вы окажетесь виноваты в сломанной учётке.

Своё состояние бота держим у себя на VPS. Хватит SQLite:

```python
# clinic_bot/storage/sent.py
import aiosqlite

DB = "/var/lib/clinic-bot/bot.sqlite3"

INIT = """
CREATE TABLE IF NOT EXISTS sent (
    schedule_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    sent_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (schedule_id, kind)
);
CREATE TABLE IF NOT EXISTS links (
    card_id INTEGER PRIMARY KEY,
    tg_id INTEGER NOT NULL
);
"""

async def was_sent(schedule_id: int, kind: str) -> bool:
    async with aiosqlite.connect(DB) as db:
        cur = await db.execute("SELECT 1 FROM sent WHERE schedule_id=? AND kind=?", (schedule_id, kind))
        return await cur.fetchone() is not None

async def mark(schedule_id: int, kind: str):
    async with aiosqlite.connect(DB) as db:
        await db.execute("INSERT OR IGNORE INTO sent(schedule_id, kind) VALUES (?, ?)", (schedule_id, kind))
        await db.commit()
```

Таблица `links` — связка карты с `telegram_id`, которую вы заполняете в момент опознания по контакту. Без неё бот не знает, кому слать напоминание. Раз в месяц можно чистить `sent` старше 30 дней.

## S3

**[S3.0] Пользователь:**

Плохие новости. Вендор «СтомаОфис» узнал, что мы пишем прямо в базу, и запретил прямую запись под угрозой снять клинику с поддержки. Без поддержки клиника не может, так что перешли на их облачный API: лимит 60 запросов в минуту, и чтение расписания, и запись — всё через API. Доку прислали, REST с токеном. Переделываем.

**[S3.1] Ассистент:**

Неприятно, но так даже чище: у вас будет один клиент с понятными методами, а не SQL по чужой схеме. Важно с самого начала уважать лимит, иначе первый же наплыв пациентов упрётся в 429.

```python
# clinic_bot/clinic/api.py
import asyncio
import httpx
from aiolimiter import AsyncLimiter

from ..config import settings

class StomaApi:
    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=settings.stoma_api_url,
            headers={"Authorization": f"Bearer {settings.stoma_api_token}"},
            timeout=httpx.Timeout(10, connect=5),
        )
        # 55, а не 60: запас на часы вендора и повторы
        self._limit = AsyncLimiter(55, 60)

    async def _request(self, method: str, url: str, **kw) -> dict:
        for attempt in range(4):
            async with self._limit:
                r = await self._http.request(method, url, **kw)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"StomaApi: 429 после повторов: {url}")

    async def free_slots(self, doctor_id: int, branch_id: int, day: str) -> list[dict]:
        return await self._request("GET", "/v1/slots", params={"doctor": doctor_id, "branch": branch_id, "date": day})

    async def create_visit(self, slot_id: int, patient_id: int | None, note: str = "") -> dict:
        return await self._request("POST", "/v1/visits", json={"slot": slot_id, "patient": patient_id, "note": note})

    async def cancel_visit(self, visit_id: int) -> None:
        await self._request("DELETE", f"/v1/visits/{visit_id}")
```

Лимитер один на весь процесс — создайте `StomaApi()` один раз в `__main__` и передавайте через `dp["api"]`, иначе у каждого хэндлера будет свой счётчик и лимит фактически умножится.

**[S3.2] Пользователь:**

Переделал. Сразу упёрся: показ слотов на неделю для одного врача — это 7 запросов, по одному на день. При десятке пациентов, которые одновременно листают календарь, это уже за 60 в минуту, бот подвисает.

**[S3.3] Ассистент:**

Тут два приёма, лучше оба.

1. Кэш слотов на короткое время. Слоты меняются не каждую секунду; кэш на 60–90 секунд снимает почти всю нагрузку от «листающих»:

```python
from cachetools import TTLCache

_slots_cache: TTLCache = TTLCache(maxsize=2000, ttl=90)

async def cached_slots(api: StomaApi, doctor_id: int, branch_id: int, day: str):
    key = (doctor_id, branch_id, day)
    if key not in _slots_cache:
        _slots_cache[key] = await api.free_slots(doctor_id, branch_id, day)
    return _slots_cache[key]
```

2. Сбрасывать кэш дня сразу после записи или отмены через бота, чтобы человек не видел только что занятое им же время.

Риск кэша понятен: слот могли занять в регистратуре за эти 90 секунд. Поэтому финальная проверка — сам `create_visit`: если API ответит, что слот занят, показываем «время только что заняли, выберите другое».

Если и с кэшем не хватит — спросите вендора, есть ли метод «слоты на диапазон дат»: одна неделя одним запросом вместо семи.

**[S3.4] Пользователь:**

Спрошу. Ещё момент, который я сразу не сказал: после гарантии бот сопровождает приходящий айтишник клиники, Вадим. Знает Windows и 1С, Python — нет. Так что Redis, который ты в прошлый раз предлагал для FSM, мне кажется лишним — ещё один сервис, который ему придётся понимать. Можно без него?

**[S3.5] Ассистент:**

Можно и, с учётом сопровождения, нужно: чем меньше движущихся частей, тем больше шансов, что человек без Python перезапустит и починит. У вас уже есть SQLite для флагов напоминаний — положим туда и FSM. Встроенного SQL-хранилища в aiogram нет, но интерфейс маленький:

```python
# clinic_bot/storage/fsm_sqlite.py
import json
import aiosqlite
from aiogram.fsm.storage.base import BaseStorage, StorageKey

class SqliteStorage(BaseStorage):
    def __init__(self, path: str):
        self.path = path

    async def _init(self, db):
        await db.execute("CREATE TABLE IF NOT EXISTS fsm (k TEXT PRIMARY KEY, state TEXT, data TEXT)")

    @staticmethod
    def _k(key: StorageKey) -> str:
        return f"{key.bot_id}:{key.chat_id}:{key.user_id}"

    async def set_state(self, key, state=None):
        st = state.state if hasattr(state, "state") else state
        async with aiosqlite.connect(self.path) as db:
            await self._init(db)
            await db.execute("INSERT INTO fsm(k, state, data) VALUES (?, ?, '{}') "
                             "ON CONFLICT(k) DO UPDATE SET state=excluded.state", (self._k(key), st))
            await db.commit()

    async def get_state(self, key):
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute("SELECT state FROM fsm WHERE k=?", (self._k(key),))
            row = await cur.fetchone()
            return row[0] if row else None

    async def set_data(self, key, data):
        async with aiosqlite.connect(self.path) as db:
            await self._init(db)
            await db.execute("INSERT INTO fsm(k, state, data) VALUES (?, NULL, ?) "
                             "ON CONFLICT(k) DO UPDATE SET data=excluded.data", (self._k(key), json.dumps(data)))
            await db.commit()

    async def get_data(self, key):
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute("SELECT data FROM fsm WHERE k=?", (self._k(key),))
            row = await cur.fetchone()
            return json.loads(row[0]) if row and row[0] else {}

    async def close(self):
        pass
```

Итого на VPS: один процесс бота под systemd, nginx и один файл SQLite. Бэкап — копия одного файла.

**[S3.6] Пользователь:**

Отлично, Redis снёс. Теперь отмена. Правило клиники: отменить запись в боте можно не позже чем за 12 часов до приёма; позже — только звонком в регистратуру. Сделай кнопку «Отменить» в «Моих записях».

**[S3.7] Ассистент:**

```python
# clinic_bot/handlers/my_visits.py
from datetime import datetime, timedelta

CANCEL_DEADLINE = timedelta(hours=12)

class CancelCb(CallbackData, prefix="cancel"):
    visit_id: int

@router.callback_query(CancelCb.filter())
async def cancel(cb: CallbackQuery, callback_data: CancelCb, api: StomaApi, state: FSMContext):
    visit = await my_visit_or_none(cb.from_user.id, callback_data.visit_id, api)
    if not visit:
        await cb.answer("Запись не найдена", show_alert=True)
        return
    if visit.start - datetime.now() < CANCEL_DEADLINE:
        await cb.message.answer(LATE_CANCEL_TEXT)
        await cb.answer()
        return
    await api.cancel_visit(visit.id)
    await cb.message.edit_text(f"Запись на {visit.start:%d.%m в %H:%M} отменена.")
    await cb.answer()
```

Проверка времени — на сервере в момент нажатия, а не при показе кнопки: список «Моих записей» могли открыть за 13 часов, а нажать через два. Функция `my_visit_or_none` должна проверять, что запись принадлежит карте этого Telegram-пользователя, — иначе подменой `visit_id` можно отменить чужую.

**[S3.8] Пользователь:**

А что писать в LATE_CANCEL_TEXT? Номер регистратуры дать, ок. Кстати, про регистратуру: у девочек на стойке личные телефоны, Олег Викторович запретил рабочие чаты на личных, поэтому рабочий телеграм у них только на компьютере у стойки. Работают с 8 до 21, в воскресенье закрыто.

**[S3.9] Ассистент:**

Текст лучше сделать коротким и с кликабельным номером — в Telegram номер в формате `+7…` сам становится ссылкой на звонок:

```python
LATE_CANCEL_TEXT = (
    "До приёма меньше 12 часов — отменить запись в боте уже нельзя.\n"
    "Пожалуйста, позвоните в регистратуру: +7 4012 00-00-00\n"
    "Мы работаем с 8:00 до 21:00, кроме воскресенья."
)
```

Номер и часы работы вынесите в настройки, а не в код: клиника их поменяет раньше, чем вы думаете. И в Telegram можно добавить кнопку-ссылку, но `tel:` в inline-кнопках Telegram не поддерживает, так что номер просто текстом.

**[S3.10] Пользователь:**

Заказчик спросил, нельзя ли сделать так, чтобы пациент писал свободным текстом «запишите на четверг к Соколовой», а нейросеть разбирала. LLM-ассистента для записи свободным текстом отказались делать — главврач боится ошибочных записей, и я с ним согласен. Но люди всё равно пишут текстом. Что им отвечать?

**[S3.11] Ассистент:**

Нужен обработчик «последнего шанса» — он регистрируется самым последним роутером и ловит любой текст, который не попал в сценарий:

```python
# clinic_bot/handlers/fallback.py
router = Router()

@router.message(F.text)
async def any_text(message: Message, state: FSMContext):
    await message.answer(
        "Я пока понимаю только кнопки 🙂\n"
        "Нажмите «Записаться», чтобы выбрать врача и время, или «Мои записи».",
        reply_markup=main_menu_kb(),
    )
```

Пара полезных деталей:

- Если человек находится в состоянии FSM (например, выбирает время) и пишет текст — не сбрасывайте сценарий, а повторите последний вопрос с кнопками: «Выберите время кнопкой ниже».
- Текстовые сообщения, которые бот не понял, полезно складывать в счётчик (без содержимого) — через месяц станет видно, каких кнопок людям не хватает.

## S4

**[S4.0] Пользователь:**

Клиника посмотрела на неявки за первые две недели тестового запуска — многие забывают. Решили: напоминаний два, за 24 часа и за 2 часа до приёма. Надо доделать.

**[S4.1] Ассистент:**

Периодическая задача у вас уже есть, так что второе напоминание — это второе окно и второй вид флага. Обобщим `tick`, чтобы не копировать код:

```python
# clinic_bot/services/reminders.py
REMINDERS = [
    ("24h", timedelta(hours=24), "Напоминаем: завтра в {t:%H:%M} приём у врача {doctor}.\n{address}"),
    ("2h",  timedelta(hours=2),  "Через два часа, в {t:%H:%M}, вас ждёт врач {doctor}.\n{address}"),
]
WINDOW = timedelta(minutes=10)

async def tick(bot: Bot, api: StomaApi):
    now = datetime.now()
    for kind, before, template in REMINDERS:
        visits = await upcoming_visits(api, now + before - WINDOW, now + before + WINDOW)
        for v in visits:
            if await sent.was_sent(v.id, kind):
                continue
            tg_id = await links.tg_for_card(v.patient_id)
            if not tg_id:
                continue
            text = template.format(t=v.start, doctor=v.doctor, address=BRANCHES[v.branch_id])
            try:
                await bot.send_message(tg_id, text)
            except TelegramForbiddenError:
                await links.forget(tg_id)   # пациент заблокировал бота
            await sent.mark(v.id, kind)
```

Обратите внимание: `upcoming_visits` теперь ходит в API, и это два запроса каждые 5 минут — в лимит укладывается спокойно, но если вендор даст метод «визиты за период», используйте его, а не перебор по врачам.

Ещё момент: запись, сделанная меньше чем за 24 часа до приёма, «перепрыгнет» первое окно — это нормально, человек и так только что записался.

**[S4.2] Пользователь:**

Ещё нужно: уведомления о новых записях и отменах бот шлёт в общий Telegram-чат регистратуры. Чат создали, бота туда добавили. Как узнать chat_id?

**[S4.3] Ассистент:**

Самый простой способ — временный хэндлер, который печатает `chat.id` любого сообщения в группе:

```python
@router.message(F.chat.type.in_({"group", "supergroup"}))
async def where_am_i(message: Message):
    logging.info("chat_id=%s title=%s", message.chat.id, message.chat.title)
```

Напишите что-нибудь в чат и посмотрите лог. Две ловушки:

1. **Privacy mode.** По умолчанию бот в группе видит только команды и ответы на свои сообщения. Для вывода `chat_id` достаточно написать в чат `/start@имя_бота`. Отключать privacy mode в BotFather для уведомлений не нужно — бот в чат только пишет.
2. **Миграция в супергруппу.** Если в группе включат историю для новых участников или сделают кого-то админом с особыми правами, Telegram превратит её в супергруппу с новым `chat_id` вида `-100…`. Старый id начнёт отдавать ошибку `migrate_to_chat_id`. Обработайте её:

```python
async def notify_registry(bot: Bot, text: str):
    try:
        await bot.send_message(settings.registry_chat_id, text)
    except TelegramMigrateToChat as e:
        logging.warning("registry chat migrated to %s", e.migrate_to_chat_id)
        await bot.send_message(e.migrate_to_chat_id, text)
```

И новый id — в `.env`, иначе после рестарта всё повторится.

**[S4.4] Пользователь:**

Нашёл id. Какой формат у уведомления, чтобы девочкам было удобно? Они смотрят его между звонками.

**[S4.5] Ассистент:**

Главное — чтобы суть читалась по первой строке, а детали были ниже. Эмодзи в начале помогают отличать типы на глаз:

```python
def fmt_new(v, patient) -> str:
    return (
        f"🟢 Новая запись · {BRANCHES_SHORT[v.branch_id]}\n"
        f"{v.start:%d.%m, %a, %H:%M} — {v.doctor}\n"
        f"{patient.fio}, {patient.phone}\n"
        f"{'Первичный, карты нет' if patient.is_new else 'Карта №' + str(patient.card_id)}"
    )

def fmt_cancel(v, patient) -> str:
    return (
        f"🔴 Отмена · {BRANCHES_SHORT[v.branch_id]}\n"
        f"{v.start:%d.%m, %a, %H:%M} — {v.doctor}\n"
        f"{patient.fio}, {patient.phone}"
    )
```

`%a` по умолчанию даст английские дни недели — либо поставьте локаль `ru_RU.UTF-8` на VPS, либо сделайте свой список «пн, вт…», как в календаре. Второе надёжнее: локаль на сервере кто-нибудь однажды поменяет.

**[S4.6] Пользователь:**

Скинул им пример, понравилось. Теперь описания услуг: «Гигиеническая чистка», «Консультация терапевта» и т. д., чтобы человек понимал, на что записывается. Ирина, управляющая, просила в описаниях ничего про стоимость не писать. Год назад у них в группе ВК висела старая сумма за отбеливание, пациентка пришла со скриншотом и написала жалобу в Роспотребнадзор — теперь у Олега Викторовича на этом пунктик. Ладно. Вопрос: где хранить эти тексты, чтобы Вадим мог их править без меня?

**[S4.7] Ассистент:**

Вадиму проще всего будет обычный текстовый файл, который открывается в Блокноте. YAML подходит, если формат максимально простой:

```yaml
# /etc/clinic-bot/services.yaml
# Тексты услуг для бота. После правки: sudo systemctl restart clinic-bot
TER_CONSULT:
  title: Консультация терапевта
  text: |
    Осмотр, при необходимости снимок, план лечения.
    Длительность — около 30 минут.
GIG_CLEAN:
  title: Гигиеническая чистка
  text: |
    Ультразвук, Air Flow, полировка.
    Длительность — около часа.
```

В коде — загрузка с проверкой, чтобы опечатка в файле не роняла бота:

```python
def load_services(path="/etc/clinic-bot/services.yaml") -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        return data
    except Exception:
        logging.exception("services.yaml сломан, оставляю прежние тексты")
        return _last_good
```

И скажите Вадиму сохранять файл в UTF-8: Блокнот в новых Windows так и делает, а вот старые редакторы могут сохранить в 1251.

**[S4.8] Пользователь:**

Сделаю. По календарю: админ в СтомаОфисе по пятницам выставляет докторам расписание на 14 дней. Вопрос по интерфейсу: когда у врача на выбранной неделе вообще ничего нет, что показывать — пустой список выглядит как поломка.

**[S4.9] Ассистент:**

Пустой экран действительно читается как «сломалось». Лучше всегда давать человеку следующий шаг:

```python
async def show_days(cb: CallbackQuery, doctor_id: int, branch_id: int, api: StomaApi):
    days = await days_with_slots(api, doctor_id, branch_id)
    if not days:
        kb = InlineKeyboardBuilder()
        kb.button(text="Другой врач", callback_data="back:doctor")
        kb.button(text="Другой филиал", callback_data="back:branch")
        kb.adjust(1)
        await cb.message.edit_text(
            "У этого врача сейчас нет свободного времени для записи через бот.\n"
            "Можно выбрать другого врача или позвонить в регистратуру.",
            reply_markup=kb.as_markup(),
        )
        return
    await cb.message.edit_text("Выберите день:", reply_markup=days_kb(days, page=0))
```

Текст без обещаний вроде «появится завтра» — вы не знаете, когда появится. Кнопки «Другой врач» и «Другой филиал» возвращают на предыдущий шаг FSM, а не в начало, чтобы не выбирать карту заново.

**[S4.10] Пользователь:**

Ещё: пациенты присылают в бот фото снимков и зубов, спрашивают «что это у меня». Приём фото и рентгеновских снимков через бот отказались делать — Олег Викторович сказал, медданные в Telegram не гоняем. Что бот должен отвечать на фото?

**[S4.11] Ассистент:**

Отдельный хэндлер на любые медиа, без скачивания файла:

```python
@router.message(F.photo | F.document | F.video)
async def media(message: Message):
    await message.answer(
        "Мы не принимаем снимки и фотографии через бот.\n"
        "Возьмите их с собой на приём — врач посмотрит.",
        reply_markup=main_menu_kb(),
    )
    try:
        await message.delete()
    except TelegramBadRequest:
        pass
```

Пояснения:

- `bot.download` не вызываем — файл не попадает на ваш сервер вообще.
- Удалить сообщение пользователя в личном чате бот может (в течение 48 часов) — так снимок не останется висеть в переписке, если телефон увидит кто-то ещё. Если клиника против удаления — просто уберите `delete`.
- Хэндлер ставьте выше fallback-роутера, иначе фото с подписью уйдёт в обработчик текста.

## S5

**[S5.0] Пользователь:**

Неделю гоняли в бою. Лимит 60 в минуту нас душит: в понедельник утром слоты показываются по 10 секунд, кэш не спасает, метода «слоты за неделю» у вендора нет и не будет до следующего года. Пациенты пишут, что бот тормозит.

**[S5.1] Ассистент:**

Тогда чтение — главный потребитель лимита, а запись — копейки: даже в пиковый час это несколько десятков созданий и отмен. Есть смысл говорить с вендором именно о чтении. Варианты, которые обычно соглашаются обсуждать:

1. Повышенный лимит для одного клиента (иногда за доплату).
2. Доступ только на чтение к базе — для вендора это безопаснее, чем кажется: бот не может ничего испортить, а нагрузка на их облако падает.
3. Выгрузка расписания по расписанию (раз в несколько минут) в файл или вебхук.

Для аргумента полезно показать цифры: сколько запросов в минуту в пике уходит на `free_slots`, а сколько — на `create_visit`/`cancel_visit`. Это легко достать из логов лимитера.

**[S5.2] Пользователь:**

Поговорил с вендором, показал цифры. Разрешили. Итог: вернулись к чтению расписания напрямую из Firebird — под read-only пользователем, с письменного разрешения вендора, письмо у клиники. А создание и отмена записей — только через API, как сейчас. Как это правильно оформить?

**[S5.3] Ассистент:**

Сначала пользователь в Firebird. Создавать его должен администратор сервера (SYSDBA), ваша часть — точный список прав:

```sql
CREATE USER BOT_RO PASSWORD 'длинный-пароль';
GRANT SELECT ON SCHEDULE TO USER BOT_RO;
GRANT SELECT ON DOCTORS  TO USER BOT_RO;
GRANT SELECT ON PATIENTS TO USER BOT_RO;
GRANT SELECT ON BRANCHES TO USER BOT_RO;
```

Никаких `UPDATE`/`INSERT`/`DELETE`, и не `GRANT ALL`. Проверьте с VPS, что запись действительно запрещена:

```bash
echo "UPDATE SCHEDULE SET NOTE = NOTE WHERE 1 = 0;" | isql -user BOT_RO -password '...' 10.8.0.2:/data/stoma.fdb
```

Должно быть `no permission for UPDATE access`.

В коде — два раздельных класса, чтобы никто случайно не начал писать через репозиторий чтения:

```python
class ScheduleReader:            # Firebird, только SELECT
    async def free_slots(self, doctor_id, branch_id, day_from, day_to) -> list[Slot]: ...
    async def visits_between(self, t_from, t_to) -> list[Visit]: ...

class VisitWriter:               # облачный API вендора
    def __init__(self, api: StomaApi): self.api = api
    async def create(self, slot_id, patient_id, note=""): ...
    async def cancel(self, visit_id): ...
```

Бонус: неделя слотов из Firebird — это один SQL-запрос с диапазоном дат вместо семи вызовов API.

**[S5.4] Пользователь:**

Нашёл подвох: облачное API пишет в базу клиники не сразу, а через их агент синхронизации, задержка до минуты. Пациент записался, бот перечитывает слоты из Firebird — а слот ещё свободен. Второй пациент может на него нажать.

**[S5.5] Ассистент:**

Классическая гонка между двумя источниками. Закрываем её у себя: после успешного `create` бот локально «держит» слот, пока его не увидит Firebird.

```python
# clinic_bot/storage/holds.py
HOLD_TTL = timedelta(minutes=5)

async def hold(slot_id: int):
    async with aiosqlite.connect(DB) as db:
        await db.execute("INSERT OR REPLACE INTO holds(slot_id, until) VALUES (?, ?)",
                         (slot_id, (datetime.utcnow() + HOLD_TTL).isoformat()))
        await db.commit()

async def held_ids() -> set[int]:
    async with aiosqlite.connect(DB) as db:
        cur = await db.execute("SELECT slot_id FROM holds WHERE until > ?", (datetime.utcnow().isoformat(),))
        return {r[0] for r in await cur.fetchall()}
```

```python
async def free_slots_for_user(reader, doctor_id, branch_id, d1, d2):
    slots = await reader.free_slots(doctor_id, branch_id, d1, d2)
    held = await held_ids()
    return [s for s in slots if s.id not in held]
```

Пять минут — с запасом к минуте синхронизации. Даже если второй пациент всё же успеет нажать на слот, API вендора при `create` откажет, и бот покажет «время только что заняли». Для отмены — то же в обратную сторону: освобождённый слот появится в списке, когда его увидит Firebird, это не страшно.

**[S5.6] Пользователь:**

Работает. Встал вопрос, как это всё тестировать. Для API есть песочница вендора, а копии боевой базы Firebird для тестов нет — главврач не разрешил выносить базу, там же все пациенты.

**[S5.7] Ассистент:**

И правильно, что не разрешил. Тогда разделяем тесты по тем же двум половинам:

- **Запись и отмена** — против песочницы вендора: создание, отмена, ответ на занятый слот, 429.
- **Чтение** — против своей тестовой Firebird. Нужна только схема, без данных. Её можно получить у вендора или снять с боевой базы командой, которая выгружает одни метаданные:

```bash
isql -x -user SYSDBA -password '...' 10.8.0.2:/data/stoma.fdb > schema.sql
```

Ключ `-x` выгружает только DDL — ни одной строки с пациентами. Но запускать это всё равно должен админ клиники, с вашего согласования.

Дальше Firebird в Docker на вашей машине:

```bash
docker run -d --name fb-test -e FIREBIRD_ROOT_PASSWORD=test -e FIREBIRD_DATABASE=stoma.fdb -p 3050:3050 firebirdsql/firebird:3
```

и накатить `schema.sql`.

**[S5.8] Пользователь:**

Схему без данных вендор дал сам, файлом. Docker поднял. Как наполнять — руками неохота.

**[S5.9] Ассистент:**

Генератор фикстур на pytest: несколько врачей, два филиала, сетка на две недели, часть слотов занята. Важно воспроизвести «грязь» из боевой базы — телефоны в разных форматах и несколько карт на номер, иначе тесты пройдут, а бой — нет.

```python
# tests/conftest.py
import random
from datetime import date, datetime, timedelta

PHONES = ["8 911 123-45-67", "+7(911)1234567", "89111234567", "8 (4012) 55-66-77"]

@pytest.fixture(scope="session")
def fb_seed(fb_conn):
    cur = fb_conn.cursor()
    cur.execute("INSERT INTO DOCTORS (ID, FIO, SPEC, IS_ACTIVE) VALUES (1, 'Соколова Анна Петровна', 'TER', 1)")
    cur.execute("INSERT INTO DOCTORS (ID, FIO, SPEC, IS_ACTIVE) VALUES (2, 'Белов Игорь Сергеевич', 'GIG', 1)")
    for i, ph in enumerate(PHONES, start=1):
        cur.execute("INSERT INTO PATIENTS (ID, FIO, PHONE, BIRTH_DATE) VALUES (?, ?, ?, ?)",
                    (i, f"Тестов{i} Тест Тестович", ph, date(1980, 1, i)))
    start = datetime.combine(date.today(), datetime.min.time()).replace(hour=9)
    sid = 1
    for day in range(14):
        for k in range(16):
            t = start + timedelta(days=day, minutes=30 * k)
            branch = 1 if t.weekday() < 3 else 2
            busy = random.random() < 0.4
            cur.execute("INSERT INTO SCHEDULE (ID, DOCTOR_ID, BRANCH_ID, START_TIME, DURATION, PATIENT_ID, IS_BLOCKED) "
                        "VALUES (?, 1, ?, ?, 30, ?, 0)", (sid, branch, t, 1 if busy else None))
            sid += 1
    fb_conn.commit()
```

`random` зафиксируйте `random.seed(42)`, чтобы тесты были воспроизводимыми.

**[S5.10] Пользователь:**

Новость от клиники: Олег Викторович передумал по хирургам. К хирургам и имплантологам в боте теперь можно записаться, но только на консультацию, не на операцию. Операции назначает врач после консультации.

**[S5.11] Ассистент:**

Тогда фильтр становится не «по специальности», а «по специальности и виду визита». В настройках:

```python
# .env
# BOOKABLE=TER:*,GIG:*,HIR:CONSULT,IMP:CONSULT
def parse_bookable(s: str) -> dict[str, set[str] | None]:
    out = {}
    for item in s.split(","):
        spec, kinds = item.split(":")
        out[spec] = None if kinds == "*" else set(kinds.split("|"))
    return out
```

`None` — любые визиты, иначе только перечисленные. В сценарии записи для хирургов и имплантологов шаг «выбор услуги» пропускается, сразу ставится вид визита «консультация», а в подтверждении стоит явная строка: «Это консультация. Лечение врач назначит на приёме».

Длительность консультации берите из сетки врача, не зашивайте 30 минут — у хирургов она бывает другой.

**[S5.12] Пользователь:**

И отзывы. Схема клиники: через 3 часа после приёма бот просит оценку от 1 до 5. На 5 — даёт ссылку на отзыв в 2ГИС, на 1–4 — пересылает отзыв управляющей, Ирине. Ей в личку, она сама разбирается.

**[S5.13] Ассистент:**

Встраивается в ту же периодическую задачу. Важно спрашивать только тех, кто реально пришёл, — иначе человек, который не явился, получит «Как прошёл приём?». Статус визита есть в SCHEDULE (обычно что-то вроде `STATUS = 'DONE'`):

```python
class RateCb(CallbackData, prefix="rate"):
    visit_id: int
    score: int

async def ask_ratings(bot, reader):
    now = datetime.now()
    for v in await reader.done_visits_between(now - timedelta(hours=3, minutes=10), now - timedelta(hours=3)):
        if await sent.was_sent(v.id, "rate"):
            continue
        tg_id = await links.tg_for_card(v.patient_id)
        if tg_id:
            kb = InlineKeyboardBuilder()
            for s in range(1, 6):
                kb.button(text="⭐" * s, callback_data=RateCb(visit_id=v.id, score=s))
            kb.adjust(1)
            await bot.send_message(tg_id, f"Как прошёл приём у врача {v.doctor}?", reply_markup=kb.as_markup())
        await sent.mark(v.id, "rate")

@router.callback_query(RateCb.filter())
async def rated(cb: CallbackQuery, callback_data: RateCb, bot: Bot):
    if callback_data.score == 5:
        await cb.message.edit_text(f"Спасибо! Будем рады отзыву: {settings.review_url}")
    else:
        await bot.send_message(settings.manager_tg_id, f"Оценка {callback_data.score} · визит {callback_data.visit_id}")
        await cb.message.edit_text("Спасибо за оценку. Мы обязательно разберёмся.")
    await cb.answer()
```

`settings.manager_tg_id` — Ирина должна хотя бы раз нажать `/start` у бота, иначе бот не сможет ей написать.

**[S5.14] Пользователь:**

Ирина просит, чтобы при оценке ниже пятёрки бот ещё спрашивал «что не так», и текст тоже ей.

**[S5.15] Ассистент:**

Добавим одно состояние FSM после низкой оценки:

```python
class Feedback(StatesGroup):
    text = State()

# в rated(), ветка score < 5:
await state.set_state(Feedback.text)
await state.update_data(visit_id=callback_data.visit_id, score=callback_data.score)
await cb.message.edit_text("Спасибо. Напишите, пожалуйста, что было не так — это прочитает управляющая.")

@router.message(Feedback.text, F.text)
async def feedback_text(message: Message, state: FSMContext, bot: Bot):
    data = await state.get_data()
    await bot.send_message(
        settings.manager_tg_id,
        f"Оценка {data['score']} · визит {data['visit_id']}\n«{message.text[:1500]}»",
    )
    await state.clear()
    await message.answer("Передали управляющей. Спасибо, что написали.")
```

Обрезка до 1500 символов — чтобы длинный текст не упёрся в лимит сообщения вместе с шапкой. Если человек вместо текста нажмёт меню — состояние сбросится, и это нормально.

## S6

**[S6.0] Пользователь:**

Опять напоминания. Пациенты жалуются, что сообщение за 2 часа приходит, когда они уже в пути, — раздражает, а неявок меньше не стало. Новая схема от клиники: напоминание одно — за 48 часов, с кнопкой «Подтверждаю». Кто не подтвердил за сутки до приёма, того бот не отменяет, а передаёт регистратуре на обзвон. Напоминание за 2 часа убрали совсем, за 24 — тоже.

**[S6.1] Ассистент:**

Тогда в `REMINDERS` остаётся одна запись, а за сутки работает уже не напоминание, а отдельная задача «собрать неподтверждённых».

```python
class ConfirmCb(CallbackData, prefix="ok"):
    visit_id: int

async def remind_48(bot, reader):
    now = datetime.now()
    for v in await reader.visits_between(now + timedelta(hours=47, minutes=50), now + timedelta(hours=48, minutes=10)):
        if await sent.was_sent(v.id, "48h"):
            continue
        tg_id = await links.tg_for_card(v.patient_id)
        if tg_id:
            kb = InlineKeyboardBuilder()
            kb.button(text="✅ Подтверждаю", callback_data=ConfirmCb(visit_id=v.id))
            await bot.send_message(
                tg_id,
                f"Напоминаем о приёме {v.start:%d.%m в %H:%M} у врача {v.doctor}.\n"
                f"{BRANCHES[v.branch_id]}\n\nПожалуйста, подтвердите, что придёте.",
                reply_markup=kb.as_markup(),
            )
        await sent.mark(v.id, "48h")

async def unconfirmed_to_registry(bot, reader):
    now = datetime.now()
    for v in await reader.visits_between(now + timedelta(hours=23, minutes=50), now + timedelta(hours=24, minutes=10)):
        if await sent.was_sent(v.id, "48h") and not await sent.was_sent(v.id, "confirmed"):
            if not await sent.was_sent(v.id, "to_registry"):
                await notify_registry(bot, fmt_unconfirmed(v))
                await sent.mark(v.id, "to_registry")
```

Учтите граничный случай: тем, кто записался меньше чем за 48 часов, напоминание не придёт, и в список неподтверждённых они попадать не должны — отсюда условие `was_sent(v.id, "48h")`. Старые флаги `24h` и `2h` в таблице `sent` можно оставить, они ничему не мешают.

**[S6.2] Пользователь:**

Кнопка «Подтверждаю» — хочу, чтобы подтверждение было видно и в самой учётке, а не только у меня в SQLite. Регистратура смотрит в СтомаОфис, а не в бот.

**[S6.3] Ассистент:**

Писать в учётку можем только через API, так что смотрите в доке вендора метод изменения визита. Обычно это `PATCH /v1/visits/{id}` со статусом или флагом подтверждения:

```python
@router.callback_query(ConfirmCb.filter())
async def confirmed(cb: CallbackQuery, callback_data: ConfirmCb, writer: VisitWriter):
    visit = await my_visit_or_none(cb.from_user.id, callback_data.visit_id)
    if not visit:
        await cb.answer("Запись не найдена", show_alert=True)
        return
    try:
        await writer.confirm(visit.id)        # PATCH /v1/visits/{id} {"confirmed": true}
    except httpx.HTTPError:
        logging.exception("confirm failed, visit=%s", visit.id)
    await sent.mark(visit.id, "confirmed")    # у себя отмечаем в любом случае
    await cb.message.edit_reply_markup(reply_markup=None)
    await cb.message.answer("Спасибо! Ждём вас.")
    await cb.answer()
```

Если API упал, пациент всё равно видит «Спасибо», а у себя мы отметку ставим — иначе он попадёт в обзвон, хотя подтвердил. Неудачные `confirm` стоит повторять той же периодической задачей: раз в 5 минут перебирать визиты с флагом `confirmed`, но без флага `confirmed_api`.

Если подходящего поля в API нет, остаётся `note` — спросите вендора, можно ли менять его у существующего визита.

**[S6.4] Пользователь:**

Поле есть, `confirmed`. Сделал, в СтомаОфисе появляется галочка. Повторы тоже.

**[S6.5] Ассистент:**

Отлично. Чтобы повторы не крутились вечно, ограничьте их: если визит уже прошёл, а `confirmed_api` так и не встал — перестаём пытаться и пишем одну строку в лог с id визита. Через API отмечать подтверждение прошедшего приёма смысла нет.

```python
async def retry_confirms(writer, reader):
    for visit_id in await sent.ids_with("confirmed", without="confirmed_api"):
        v = await reader.visit(visit_id)
        if v is None or v.start < datetime.now():
            await sent.mark(visit_id, "confirmed_api")   # больше не трогаем
            continue
        try:
            await writer.confirm(visit_id)
            await sent.mark(visit_id, "confirmed_api")
        except httpx.HTTPError:
            pass
```

**[S6.6] Пользователь:**

Теперь неприятное. Юрист клиники посмотрел логи на сервере, когда проверял, где хранятся данные пациентов, и устроил разнос: в логах ФИО и телефоны открытым текстом — из уведомлений регистратуре и из ошибок. Требование юриста клиники: в логах ФИО и телефоны маскируются. Как это сделать, не переписывая каждый вызов logging?

**[S6.7] Ассистент:**

Фильтр на уровне корневого логгера — он обрабатывает уже отформатированное сообщение, так что поймает и ваши вызовы, и логи aiogram, и трейсбеки:

```python
# clinic_bot/logging_setup.py
import logging
import re

PHONE_RE = re.compile(r"(?:\+7|8)[\s\-()]*\d(?:[\s\-()]*\d){9}")
FIO_RE = re.compile(r"\b[А-ЯЁ][а-яё]+(?:ов|ев|ин|ын|ский|цкий|ова|ева|ина|ына|ская|цкая)\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?")

def mask(text: str) -> str:
    text = PHONE_RE.sub(lambda m: "***" + re.sub(r"\D", "", m.group())[-2:], text)
    text = FIO_RE.sub(lambda m: m.group()[0] + "***", text)
    return text

class MaskFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = mask(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True

def setup():
    handler = logging.StreamHandler()
    handler.addFilter(MaskFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
```

Фильтр вешаем на handler, а не на логгер: фильтры логгера не срабатывают для сообщений дочерних логгеров (aiogram, httpx), а фильтр обработчика видит всё.

Регулярка по ФИО эвристическая — она не поймает «Ким Анна» и может замаскировать лишнее. Поэтому второе правило важнее: не передавайте в логи объекты пациента вообще. Логируйте `visit_id`, `card_id`, `tg_id` — по ним вы и так всё найдёте в учётке.

И отключите логирование апдейтов целиком на уровне DEBUG в проде: `logging.getLogger("aiogram.event").setLevel(logging.INFO)` — в DEBUG aiogram печатает содержимое сообщений, включая контакты.

**[S6.8] Пользователь:**

Сделал, в новых логах звёздочки. А что со старыми? Там за полтора месяца всё открытым текстом. И journald у меня хранит сколько хочет.

**[S6.9] Ассистент:**

Старые логи с персональными данными лучше удалить, а не маскировать задним числом — так проще объяснить юристу, что их нет.

```bash
# что сейчас занимает journald
journalctl --disk-usage

# удалить всё старше суток (после проверки, что новые логи уже с маской)
sudo journalctl --vacuum-time=1d

# если писали ещё и в файлы
sudo find /var/log/clinic-bot -name "*.log*" -mtime +0 -delete
```

И ограничьте срок хранения на будущее, чтобы через год не было той же истории:

```ini
# /etc/systemd/journald.conf
[Journal]
MaxRetentionSec=30day
SystemMaxUse=500M
```

`sudo systemctl restart systemd-journald`. Не забудьте про бэкапы: если VPS снапшотится в Selectel, в старых снапшотах логи остались. Их тоже стоит удалить или дождаться, пока они уйдут по ротации, — и сказать об этом юристу честно.

**[S6.10] Пользователь:**

Снапшоты есть, ротация 7 дней, подождём. Последнее на сегодня: сделай юнит systemd, чтобы бот сам поднимался, и пару команд, которые я потом оставлю Вадиму в инструкции.

**[S6.11] Ассистент:**

```ini
# /etc/systemd/system/clinic-bot.service
[Unit]
Description=Clinic Telegram bot
After=network-online.target wg-quick@wg0.service
Wants=network-online.target wg-quick@wg0.service

[Service]
User=clinicbot
WorkingDirectory=/opt/clinic-bot
EnvironmentFile=/etc/clinic-bot/.env
ExecStart=/opt/clinic-bot/.venv/bin/python -m clinic_bot
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`After=wg-quick@wg0` — чтобы бот не стартовал раньше туннеля до клиники и не сыпал ошибками подключения к Firebird.

Шпаргалка для инструкции — команды, которые можно копировать, ничего не понимая в Python:

| Что нужно | Команда |
|---|---|
| Работает ли бот | `sudo systemctl status clinic-bot` |
| Перезапустить | `sudo systemctl restart clinic-bot` |
| Последние 100 строк лога | `sudo journalctl -u clinic-bot -n 100` |
| Жив ли туннель до клиники | `sudo wg show` |
| Поменять настройки | `sudo nano /etc/clinic-bot/.env`, потом перезапустить |

Подключаться Вадиму удобнее всего через PuTTY или встроенный в Windows `ssh` — дайте ему отдельного пользователя с `sudo` только на эти команды, а не root.
