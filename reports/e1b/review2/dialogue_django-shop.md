# Диалог загрузки: Интернет-магазин велозапчастей «Спица» (Django)

## S1

**[S1.0] Пользователь:**

Привет. У меня магазин велозапчастей «Спица», Django 5, шаблоны на сервере плюс HTMX, никакого SPA. Делаю всё один, дизайнер иногда пару часов в неделю. Страница каталога стала грузиться секунды по три. Куда смотреть?

**[S1.1] Ассистент:**

На каталоге в Django три самые частые причины: N+1 запросы в шаблоне, отсутствие индексов под фильтры и тяжёлый `count()` в пагинации. Начните с замера:

```bash
pip install django-debug-toolbar
```

```python
# settings/dev.py
INSTALLED_APPS += ["debug_toolbar"]
MIDDLEWARE.insert(0, "debug_toolbar.middleware.DebugToolbarMiddleware")
INTERNAL_IPS = ["127.0.0.1"]
```

```python
# urls.py
if settings.DEBUG:
    urlpatterns += [path("__debug__/", include("debug_toolbar.urls"))]
```

Откройте каталог и посмотрите панель SQL: сколько запросов и есть ли одинаковые запросы, повторяющиеся десятки раз. Если запросов больше 20–30 на страницу — почти наверняка N+1.

**[S1.2] Пользователь:**

Поставил. 212 запросов на страницу, ага. В каталоге примерно 30 тысяч позиций, на странице 48 карточек. Повторяется запрос к бренду и к картинкам.

**[S1.3] Ассистент:**

Это N+1: для каждой карточки шаблон отдельно достаёт бренд (`product.brand.name`) и картинки (`product.images.first`). Лечится в queryset:

```python
def catalog(request):
    qs = (
        Product.objects.filter(is_active=True)
        .select_related("brand", "category")
        .prefetch_related(
            Prefetch("images", queryset=ProductImage.objects.order_by("position"), to_attr="ordered_images")
        )
        .only("id", "name", "slug", "price", "brand__name", "category__slug")
    )
    ...
```

В шаблоне берите `product.ordered_images.0`, а не `product.images.first` — `first()` делает новый запрос, игнорируя prefetch.

`select_related` — для ForeignKey (один JOIN), `prefetch_related` — для обратных связей и M2M (один дополнительный запрос на всю страницу). Должно стать 4–5 запросов.

**[S1.4] Пользователь:**

5 запросов, стало 400 мс. Остался тяжёлый count от пагинатора, на фильтрах до секунды.

**[S1.5] Ассистент:**

`Paginator` считает `COUNT(*)` по всему отфильтрованному набору. На 30 тысячах это не катастрофа, но с JOIN и фильтрами может тормозить. Варианты:

1. Индексы под частые фильтры:

```python
class Product(models.Model):
    ...
    class Meta:
        indexes = [
            models.Index(fields=["category", "is_active", "price"]),
            models.Index(fields=["brand", "is_active"]),
        ]
```

2. Считать по упрощённому запросу без JOIN: в пагинатор передайте queryset без `select_related`, а карточки грузите только для текущей страницы.

3. Отказаться от номера последней страницы и делать «Показать ещё» — тогда count вообще не нужен. С HTMX это делается очень просто.

**[S1.6] Пользователь:**

База у нас PostgreSQL, индексы добавлю. Давай «показать ещё» на HTMX, как раз хотел.

**[S1.7] Ассистент:**

Кнопка подгружает следующую порцию и сама себя заменяет:

```html
<!-- catalog/_page.html -->
{% for product in page %}
  {% include "catalog/_card.html" %}
{% endfor %}
{% if has_next %}
  <button hx-get="{% url 'catalog' %}?{{ query }}&after={{ last_id }}"
          hx-target="this" hx-swap="outerHTML"
          class="btn btn-outline w-full">Показать ещё</button>
{% endif %}
```

```python
def catalog(request):
    after = int(request.GET.get("after", 0))
    qs = filtered_products(request).filter(id__gt=after).order_by("id")[:49]
    items = list(qs)
    ctx = {"page": items[:48], "has_next": len(items) > 48,
           "last_id": items[47].id if len(items) > 48 else None,
           "query": request.GET.urlencode()}
    template = "catalog/_page.html" if request.htmx else "catalog/index.html"
    return render(request, template, ctx)
```

Пагинация «по ключу» (`id > after`) не делает ни count, ни offset — быстро на любой глубине. Для сортировки по цене ключ должен быть парой `(price, id)`. Не забудьте `django-htmx`, чтобы был `request.htmx`.

**[S1.8] Пользователь:**

Работает. Кстати, мобильного приложения у нас нет и не будет, только адаптивный сайт — так что на телефонах «показать ещё» как раз то, что надо.

**[S1.9] Ассистент:**

Тогда пара мелочей для мобильных:

- Кнопку «Показать ещё» можно заменить на автоподгрузку при прокрутке: `hx-trigger="revealed"` на последнем элементе. Но оставьте кнопку как запасной вариант — бесконечная лента мешает добраться до футера с контактами.
- Сохраняйте позицию: при возврате «назад» из карточки товара браузер должен вернуться к тому же месту. С HTMX помогает `hx-push-url` с параметром `after`, чтобы история знала, сколько было загружено.
- Картинки в карточках — `loading="lazy"` и правильный `sizes`, чтобы на телефоне не грузить десктопные размеры.

```html
<img src="{{ img.url_small }}" srcset="{{ img.url_small }} 320w, {{ img.url_medium }} 640w"
     sizes="(max-width: 640px) 50vw, 240px" loading="lazy" alt="{{ product.name }}">
```

**[S1.10] Пользователь:**

Сделал lazy. Теперь про оплату. Онлайн-оплата у нас через ЮKassa. Иногда заказ остаётся «не оплачен», хотя деньги списались. Как правильно ловить уведомления?

**[S1.11] Ассистент:**

Статус заказа нельзя менять по редиректу покупателя на «спасибо» — только по уведомлению (вебхуку) от платёжки или по проверке статуса платежа через API.

```python
@csrf_exempt
def yookassa_webhook(request):
    event = json.loads(request.body)
    payment_id = event["object"]["id"]
    # не доверяем телу: перепроверяем статус через API
    payment = Payment.find_one(payment_id)
    if payment.status == "succeeded":
        mark_order_paid.delay(payment.metadata["order_id"], payment_id)
    return HttpResponse(status=200)
```

Типичные причины «деньги списались, заказ не оплачен»:
1. Вебхук не дошёл (сервер отвечал 500 или долго) — платёжка повторит, но не бесконечно.
2. Обработчик упал после списания, но до обновления заказа — нужна идемпотентность: повторный вебхук с тем же `payment_id` не должен ломаться.
3. Нет фоновой сверки. Добавьте задачу раз в 10–15 минут: найти заказы «ожидают оплату» старше 15 минут и спросить статус платежа через API.

**[S1.12] Пользователь:**

Сверку сделаю. И ещё деталь: своего склада нет, товар лежит у поставщика. Остатки мы раз в час подтягиваем из его CSV. Импорт сейчас идёт минут 10 и в это время сайт подтупливает.

**[S1.13] Ассистент:**

Импорт на 30 тысяч строк не должен идти 10 минут. Скорее всего, он делает `save()` для каждой строки и тянет за собой сигналы. Ускорить:

```python
def import_stock(rows: list[dict]):
    by_sku = {p.sku: p for p in Product.objects.only("id", "sku", "stock")}
    changed = []
    for r in rows:
        p = by_sku.get(r["sku"])
        if p and p.stock != int(r["qty"]):
            p.stock = int(r["qty"])
            changed.append(p)
    Product.objects.bulk_update(changed, ["stock"], batch_size=1000)
```

Обновляются только изменившиеся позиции, одним пакетом. Обычно это секунды.

И уведите импорт с веб-процесса: пусть он идёт фоновой задачей, а не по запросу, тогда сайт не будет «подтупливать». Плюс защитите от двойного запуска (блокировка), чтобы два импорта не наложились.

**[S1.14] Пользователь:**

Да, импорт сейчас дёргается из cron через curl в вьюху, понимаю, что убого. На сегодня всё, спасибо.

## S2

**[S2.0] Пользователь:**

Привет. Перевёл импорт остатков на фоновые задачи: Celery с Redis, beat раз в час. Работает. Но иногда задача висит в «started» и больше не двигается. Что это может быть?

**[S2.1] Ассистент:**

«Повисшая» задача в Celery — обычно воркер умер посреди выполнения (OOM, перезапуск контейнера), а Redis-брокер не узнал об этом. Настройки, которые помогают:

```python
# celery.py
app.conf.update(
    task_acks_late=True,              # подтверждать после выполнения, а не при получении
    task_reject_on_worker_lost=True,  # вернуть задачу в очередь, если воркер умер
    worker_prefetch_multiplier=1,     # не забирать пачку задач заранее
    broker_transport_options={"visibility_timeout": 3600},
    task_time_limit=900,              # жёстко убить через 15 минут
    task_soft_time_limit=840,
)
```

С `acks_late` задача должна быть идемпотентной — импорт остатков у вас такой и так (повтор просто перезапишет те же числа). И проверьте память контейнера воркера: импорт 30 тысяч строк в одном списке может её съесть.

**[S2.2] Пользователь:**

Поставил, посмотрим. Знакомый сказал, что «нормальные люди» для очередей берут Kafka. Я подумал и отказался: я один, мне ещё один кластер поддерживать — перебор.

**[S2.3] Ассистент:**

Для вашей нагрузки это правильное решение. Kafka хороша, когда нужны потоки событий с хранением, много потребителей и миллионы сообщений, — и требует отдельной поддержки. У вас десятки фоновых задач в час: импорт, письма, SMS, сверка оплат. Celery с Redis с этим справляется, а Redis у вас уже есть для кэша.

Что стоит добавить в Celery, чтобы жить спокойно:
- Отдельные очереди для быстрых и медленных задач (`sms` и `import`), чтобы импорт не задерживал отправку кодов.
- Flower или хотя бы `celery inspect active` в мониторинге.
- Алерт, если beat не запускал импорт больше двух часов.

```python
app.conf.task_routes = {"shop.tasks.send_sms_code": {"queue": "fast"}, "shop.tasks.import_stock": {"queue": "slow"}}
```

**[S2.4] Пользователь:**

Очереди разделил. Про SMS как раз: у нас вход для покупателей по номеру телефона и коду из SMS, паролей нет вообще. Коды иногда приходят через минуту, люди злятся и жмут «отправить ещё раз».

**[S2.5] Ассистент:**

Задержка обычно на стороне SMS-шлюза или оператора, но вы можете уменьшить вред:

1. **Отдельная быстрая очередь** — уже сделали.
2. **Таймер повторной отправки** на кнопке (60 секунд), а повторная отправка — тем же кодом, если он ещё действителен. Иначе человек получит три разных кода и введёт первый пришедший — который уже не действует.

```python
def issue_code(phone: str) -> str:
    key = f"sms:{phone}"
    code = cache.get(key)
    if not code:
        code = f"{secrets.randbelow(10**4):04d}"
        cache.set(key, code, timeout=300)
    send_sms_code.delay(phone, code)
    return code
```

3. **Лимиты**: не больше 3 отправок на номер за 10 минут и 10 за сутки, лимит по IP — иначе SMS-бомбинг съест бюджет.
4. **Попытки ввода**: после 5 неверных — код сгорает.
5. На Android можно автоматически подставлять код (SMS Retriever / атрибут `autocomplete="one-time-code"` в поле ввода работает и в мобильных браузерах).

**[S2.6] Пользователь:**

Сделаю. Вопрос по инфраструктуре: хостимся в Яндекс Облаке, потому что персональные данные покупателей по 152-ФЗ должны лежать в России. Есть смысл переехать на их управляемый PostgreSQL вместо базы на виртуалке?

**[S2.7] Ассистент:**

Для одного разработчика управляемая база обычно окупается: бэкапы, обновления, восстановление на момент времени и реплика делаются кнопкой, а не вашими ночами.

Что взвесить:
- **Цена**: управляемый кластер дороже виртуалки той же мощности, но сравнивать надо с вашим временем на администрирование.
- **Восстановление**: PITR (на любой момент за последние дни) — главная причина переезда. На виртуалке вы, скорее всего, делаете дамп раз в сутки.
- **Переезд**: `pg_dump` → `pg_restore` в новый кластер, короткое окно техработ; или логическая репликация, если простой недопустим.
- **152-ФЗ**: управляемый сервис в том же облаке и регионе не меняет вашу позицию по локализации — данные остаются в российских дата-центрах.

Подключение из Django — обычный `DATABASES` с SSL-сертификатом облака:

```python
DATABASES["default"]["OPTIONS"] = {"sslmode": "verify-full", "sslrootcert": "/etc/ssl/yandex-ca.pem"}
```

**[S2.8] Пользователь:**

Подумаю. Доставка у нас только СДЭК. В корзине считаем стоимость доставки через их API, и иногда он отвечает секунд по 10, корзина висит.

**[S2.9] Ассистент:**

Внешний API не должен блокировать корзину. Три меры:

1. **Таймаут** на запрос: 2–3 секунды, после — показать «стоимость уточним» или последнее известное значение.

```python
try:
    r = requests.post(CDEK_CALC_URL, json=payload, headers=auth, timeout=(2, 3))
    r.raise_for_status()
except requests.RequestException:
    return cached_estimate(city_code, weight)  # или None → «рассчитаем при оформлении»
```

2. **Кэш**: стоимость между парой городов и весовой категорией меняется редко — кэшируйте на час по ключу `(city_from, city_to, weight_bucket)`.
3. **Асинхронно через HTMX**: корзина рендерится сразу, а блок доставки догружается отдельным запросом (`hx-get` с `hx-trigger="load"`), со спиннером. Тогда медленный СДЭК тормозит только этот блок.

Токен СДЭК живёт ограниченное время — кэшируйте и его, а не запрашивайте на каждый расчёт.

**[S2.10] Пользователь:**

Блок на HTMX — отлично. Меня ещё спрашивали, почему нет Почты России. Мы её пробовали подключать и отказались: трекинг плохой, посылки терялись, разборки с покупателями съедали время.

**[S2.11] Ассистент:**

Тогда в интерфейсе стоит заранее ответить на этот вопрос, чтобы он не приходил в поддержку:

- В корзине, если город не обслуживается СДЭК, — понятный текст: «В ваш город доставка СДЭК недоступна» и ближайший пункт выдачи, а не пустой список.
- В FAQ — «Почему только СДЭК»: надёжное отслеживание, страховка, пункты выдачи.
- Для отдалённых регионов можно предложить доставку до ближайшего ПВЗ СДЭК вместо курьера — это обычно дешевле.

```python
def delivery_options(city):
    points = cdek.pickup_points(city.code)
    if not points:
        nearest = cdek.nearest_city_with_points(city)
        return {"available": False, "nearest": nearest}
    return {"available": True, "points": points}
```

**[S2.12] Пользователь:**

Последнее. Нужно переименовать поле в модели заказа: `phone` → `contact_phone`. Боюсь потерять данные при миграции.

**[S2.13] Ассистент:**

Django при переименовании поля в модели спросит «Did you rename order.phone to order.contact_phone?» — ответьте «y», и миграция будет `RenameField`, без потери данных (под капотом `ALTER TABLE ... RENAME COLUMN`).

```python
operations = [
    migrations.RenameField(model_name="order", old_name="phone", new_name="contact_phone"),
]
```

Если Django не задал вопрос (например, вы одновременно поменяли тип поля), он сгенерирует удаление и добавление — тогда данные пропадут. Проверьте сгенерированную миграцию глазами перед применением и сначала прогоните её на копии базы.

И поищите старое имя по коду: шаблоны, админка, сериализаторы, `values("phone")` — в шаблонах ошибка не упадёт, просто покажется пустота.

**[S2.14] Пользователь:**

Ответил «y», посмотрел миграцию — RenameField. Нашёл ещё два места в шаблонах. Спасибо, до завтра.

## S3

**[S3.0] Пользователь:**

Привет. Покупают у нас в основном частники, но есть юрлица — веломастерские, — они платят по счёту. Сейчас счёт я делаю руками в Word. Хочу генерить PDF из заказа. Чем лучше?

**[S3.1] Ассистент:**

Для PDF из Django удобнее всего WeasyPrint: вы верстаете счёт обычным HTML-шаблоном с CSS, а он превращает его в PDF.

```python
from weasyprint import HTML

def invoice_pdf(request, order_id):
    order = get_object_or_404(Order.objects.select_related("company"), pk=order_id, company__isnull=False)
    html = render_to_string("invoices/invoice.html", {"order": order, "seller": settings.SELLER_REQUISITES})
    pdf = HTML(string=html, base_url=request.build_absolute_uri("/")).write_pdf()
    resp = HttpResponse(pdf, content_type="application/pdf")
    resp["Content-Disposition"] = f'inline; filename="schet-{order.number}.pdf"'
    return resp
```

В шаблоне счёта: номер и дата, реквизиты продавца и покупателя (ИНН, КПП, банк), таблица позиций, итог прописью. Сумму прописью делайте библиотекой `num2words` с `lang="ru", to="currency"`.

Нумерацию счетов ведите отдельной последовательностью в базе, а не по id заказа, — у бухгалтерии обычно свои требования к сквозной нумерации.

**[S3.2] Пользователь:**

WeasyPrint завёлся. По налогам: магазин на УСН без НДС, цены в рублях — в счёте пишу «Без НДС», так? И второе: в докере шрифты кракозябрами, шрифт PT Sans.

**[S3.3] Ассистент:**

Да, при УСН без НДС в счёте в графе налога пишут «Без НДС», а в итоге — сумму без выделения налога; можно добавить строку-основание (применение УСН). Держите это в шаблоне одной настройкой, а не текстом в разметке, — если налоговый режим поменяется, переделывать придётся в одном месте.

По шрифтам: в slim-образе нет шрифтов с кириллицей, а WeasyPrint берёт системные через fontconfig. Либо поставьте шрифты в образ, либо подключите их через `@font-face` в CSS счёта:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangoft2-1.0-0 fonts-paratype && rm -rf /var/lib/apt/lists/*
```

```css
@font-face { font-family: "PT Sans"; src: url("/static/fonts/PTSans-Regular.ttf"); }
body { font-family: "PT Sans", sans-serif; }
```

Через `@font-face` надёжнее: не зависите от пакетов дистрибутива, и счёт выглядит одинаково локально и на сервере. Убедитесь, что `base_url` указан, иначе относительный путь к шрифту не разрешится.

**[S3.4] Пользователь:**

Шрифт подключил через font-face, всё ок. Теперь поиск. Он у нас на полнотекстовом поиске PostgreSQL. Проблема: по артикулам с дефисами не находит, например «SM-BB52» ищут как «smbb52» или «sm bb52».

**[S3.5] Ассистент:**

Полнотекстовый парсер PostgreSQL режет «SM-BB52» на части по дефису и по-разному нормализует. Для артикулов лучше отдельное нормализованное поле и точное/префиксное совпадение по нему:

```python
class Product(models.Model):
    sku = models.CharField(max_length=64)
    sku_norm = models.CharField(max_length=64, db_index=True, editable=False)

    def save(self, *args, **kwargs):
        self.sku_norm = re.sub(r"[^0-9a-z]", "", self.sku.lower())
        super().save(*args, **kwargs)
```

```python
def search(q: str):
    norm = re.sub(r"[^0-9a-z]", "", q.lower())
    by_sku = Product.objects.filter(sku_norm__startswith=norm) if len(norm) >= 3 else Product.objects.none()
    vector = SearchVector("name", weight="A", config="russian") + SearchVector("description", weight="C", config="russian")
    by_text = Product.objects.annotate(rank=SearchRank(vector, SearchQuery(q, config="russian"))).filter(rank__gt=0.05)
    return list(by_sku[:10]) + list(by_text.order_by("-rank")[:40])
```

Для `bulk_update` из импорта `save()` не вызывается — заполните `sku_norm` и там, либо сделайте его генерируемой колонкой в PostgreSQL.

**[S3.6] Пользователь:**

Сделаю генерируемой колонкой. Мне советовали всё бросить и поставить Elasticsearch. Отказался: лишний сервис, а каталог у нас небольшой, PostgreSQL справляется.

**[S3.7] Ассистент:**

На 30 тысяч позиций PostgreSQL действительно хватает, а Elasticsearch добавил бы синхронизацию индекса, ещё один сервис в мониторинге и отдельную память.

Чтобы выжать из PostgreSQL больше:
- **GIN-индекс** по вектору, иначе поиск будет сканировать таблицу:

```sql
ALTER TABLE shop_product ADD COLUMN search_vec tsvector
  GENERATED ALWAYS AS (setweight(to_tsvector('russian', name), 'A') || setweight(to_tsvector('russian', coalesce(description, '')), 'C')) STORED;
CREATE INDEX product_search_idx ON shop_product USING GIN (search_vec);
```

- **Синонимы** для частых случаев («звезда» = «звёздочка», «каретка» = «BB»), через словарь синонимов или простую замену в запросе.
- **Подсветка** найденных слов через `SearchHeadline` — пользователям приятно.

**[S3.8] Пользователь:**

Индекс добавил, поиск стал мгновенный. Другая беда: фотки товаров. Они лежат в Яндекс Object Storage, миниатюры режет imgproxy на лету. На первой загрузке каталога миниатюры тормозят.

**[S3.9] Ассистент:**

imgproxy режет картинку при каждом запросе, если перед ним нет кэша. Первый просмотр каждой миниатюры всегда будет медленным, но повторные должны отдаваться из кэша.

1. **Кэш перед imgproxy**: CDN (у Яндекс Облака есть Cloud CDN) или nginx с `proxy_cache`:

```nginx
proxy_cache_path /var/cache/imgproxy levels=1:2 keys_zone=img:50m max_size=5g inactive=30d;
location /img/ {
    proxy_pass http://imgproxy:8080/;
    proxy_cache img;
    proxy_cache_valid 200 30d;
    add_header X-Cache $upstream_cache_status;
}
```

2. **Фиксированный набор размеров** (например, 240, 480, 960) — чтобы кэш переиспользовался, а не плодил варианты.
3. **Прогрев**: после загрузки нового товара фоновой задачей запросить его миниатюры, чтобы первый покупатель не ждал.
4. **Формат**: imgproxy умеет отдавать WebP/AVIF по заголовку `Accept` — включите `IMGPROXY_ENABLE_WEBP_DETECTION=true`.

**[S3.10] Пользователь:**

Поставлю nginx-кэш. И последнее на сегодня: заказами и ценами у нас занимается менеджер через Django admin. Он не технарь, ему неудобно: список заказов огромный, не видно главного. Можно его как-то очеловечить?

**[S3.11] Ассистент:**

Django admin хорошо настраивается под нетехнического человека:

```python
@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("number", "created_at", "customer_phone", "total_rub", "status_badge", "delivery_status")
    list_filter = ("status", ("created_at", admin.DateFieldListFilter), "payment_method")
    search_fields = ("number", "contact_phone", "company__inn")
    date_hierarchy = "created_at"
    list_per_page = 50
    readonly_fields = ("payment_id", "created_at")
    actions = ["mark_shipped"]

    @admin.display(description="Статус")
    def status_badge(self, obj):
        colors = {"new": "#2563eb", "paid": "#16a34a", "cancelled": "#9ca3af"}
        return format_html('<b style="color:{}">{}</b>', colors.get(obj.status, "#000"), obj.get_status_display())

    @admin.display(description="Сумма, ₽")
    def total_rub(self, obj):
        return f"{obj.total:,.0f}".replace(",", " ")
```

Плюс: по умолчанию фильтр «Новые и оплаченные», понятные русские названия полей (`verbose_name`), и убрать из формы технические поля. Если захочется современнее — тема `django-unfold`, но начните с настройки списков.

**[S3.12] Пользователь:**

Менеджер доволен, цветные статусы зашли. Спасибо.

## S4

**[S4.0] Пользователь:**

Привет. Большая новость: ушли с ЮKassa на CloudPayments, у них комиссия ниже. ЮKassa уже отключил полностью. Теперь надо принимать их уведомления — Pay, Fail. Как проверить, что запрос действительно от них?

**[S4.1] Ассистент:**

CloudPayments подписывает каждое уведомление: в заголовке `Content-HMAC` (или `X-Content-HMAC`) лежит HMAC-SHA256 тела запроса по вашему API Secret в base64. Проверяйте его до любой обработки:

```python
import base64, hashlib, hmac

def verify_cp(request) -> bool:
    sign = request.headers.get("Content-HMAC") or request.headers.get("X-Content-HMAC", "")
    digest = hmac.new(settings.CP_API_SECRET.encode(), request.body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), sign)

@csrf_exempt
def cp_pay(request):
    if not verify_cp(request):
        return HttpResponseForbidden()
    data = request.POST  # form-urlencoded
    mark_order_paid.delay(order_id=data["InvoiceId"], tx_id=data["TransactionId"], amount=data["Amount"])
    return JsonResponse({"code": 0})
```

Важно: ответ должен быть именно `{"code": 0}`, иначе CloudPayments будет повторять. Подписывайте по сырому `request.body`, а не по разобранным полям — порядок и кодировка важны. Сумму из уведомления сверяйте с суммой заказа.

**[S4.2] Пользователь:**

Подпись сходится. А старую фоновую сверку статусов, что делали для ЮKassa, куда девать?

**[S4.3] Ассистент:**

Логику оставьте, а источник поменяйте: у CloudPayments есть API поиска транзакции по `InvoiceId` (метод `payments/find`). Сверка делает то же самое — находит заказы «ожидают оплату» старше 15 минут и спрашивает статус:

```python
@app.task
def reconcile_pending():
    for order in Order.objects.filter(status="awaiting_payment", created_at__lt=now() - timedelta(minutes=15)):
        r = cp_client.post("payments/find", json={"InvoiceId": str(order.number)})
        tx = r.json().get("Model")
        if tx and tx["Status"] == "Completed":
            mark_order_paid(order.id, tx["TransactionId"], tx["Amount"])
```

Код работы с ЮKassa (вебхук, клиент, настройки) удалите, а не закомментируйте, — чтобы через полгода никто не перепутал, какая платёжка живая.

**[S4.4] Пользователь:**

Удалил. Ещё новость: поставщик наконец дал API с вебхуками. Остатки теперь обновляем сразу по вебхуку, а CSV раз в час убираем совсем. Боюсь, что вебхук придёт дважды или вперемешку.

**[S4.5] Ассистент:**

Обе проблемы реальные. Защиты:

1. **Идемпотентность**: у каждого события поставщика есть id — сохраняйте обработанные id и пропускайте повторы.
2. **Порядок**: используйте версию или время изменения остатка у поставщика и не применяйте событие, если оно старше уже записанного.

```python
@csrf_exempt
def supplier_stock_webhook(request):
    ev = json.loads(request.body)
    if not verify_supplier_signature(request):
        return HttpResponseForbidden()
    with transaction.atomic():
        _, created = ProcessedEvent.objects.get_or_create(event_id=ev["id"])
        if not created:
            return HttpResponse(status=200)  # повтор
        Product.objects.filter(sku=ev["sku"], stock_updated_at__lt=ev["updated_at"]).update(
            stock=ev["qty"], stock_updated_at=ev["updated_at"]
        )
    return HttpResponse(status=200)
```

3. **Пропуски**: вебхук может не дойти вообще. Раз в сутки делайте полную сверку через их API (не CSV) — лёгкую, только для расхождений.

**[S4.6] Пользователь:**

Полную сверку раз в сутки сделаю. ProcessedEvent будет пухнуть — чистить?

**[S4.7] Ассистент:**

Да, повторы приходят в пределах часов, а не месяцев. Храните id событий 7 дней и чистите задачей:

```python
@app.task
def purge_processed_events():
    ProcessedEvent.objects.filter(created_at__lt=now() - timedelta(days=7)).delete()
```

Индекс на `created_at` и уникальный индекс на `event_id`. При десятках тысяч событий в день таблица останется маленькой.

**[S4.8] Пользователь:**

Окей. Аналитика у нас — Яндекс Метрика. Хочу видеть, сколько людей дошло до оплаты и сколько реально оплатило. Сейчас цель «оплата» стоит на странице «спасибо», и цифры врут.

**[S4.9] Ассистент:**

Страница «спасибо» врёт в обе стороны: её открывают повторно (обновление, возврат по истории), и до неё не доходят те, кто оплатил, но закрыл вкладку банка. Разведите события:

- `checkout_start` — нажали «Оформить заказ»;
- `payment_redirect` — ушли на платёжную форму;
- `purchase` — подтверждённая оплата.

Первые два шлите из браузера через `ym(COUNTER_ID, 'reachGoal', 'checkout_start')`. А подтверждённую оплату отправляйте с сервера, когда пришло уведомление об оплате, — через Measurement Protocol Метрики или загрузку офлайн-конверсий по `ClientID`, сохранённому в заказе. Тогда «оплатил» в отчёте = действительно оплатил.

Для электронной коммерции в Метрике включите dataLayer и передавайте `purchase` с товарами — увидите выручку по товарам.

**[S4.10] Пользователь:**

ClientID в заказ уже пишу. Последнее — баг с промокодами. Правило у нас простое: один промокод на заказ, складывать нельзя. А кто-то умудрился применить два, получил минус 40%.

**[S4.11] Ассистент:**

Скорее всего, гонка или обход через несколько вкладок: две отправки формы почти одновременно, каждая видит «промокода ещё нет» и добавляет свой. Защита — на уровне данных, а не только формы:

```python
class Order(models.Model):
    promo = models.ForeignKey(PromoCode, null=True, blank=True, on_delete=models.PROTECT)  # одно поле, не M2M

def apply_promo(order_id: int, code: str):
    with transaction.atomic():
        order = Order.objects.select_for_update().get(pk=order_id)
        if order.promo_id:
            raise PromoError("К заказу уже применён промокод")
        promo = PromoCode.objects.get(code=code, is_active=True)
        order.promo = promo
        order.recalc_total()
        order.save(update_fields=["promo", "total"])
```

Если сейчас промокоды хранятся связью многие-ко-многим или списком в сессии — в этом и дыра. Одно поле в заказе плюс `select_for_update` делают второй промокод невозможным даже при параллельных запросах. Существующие заказы с двумя скидками найдите запросом и решите вручную.

**[S4.12] Пользователь:**

Так и было — M2M. Переделал на одно поле, двойных заказов нашлось три. До завтра.

## S5

**[S5.0] Пользователь:**

Привет. Бухгалтер обрадовала: с января магазин платит НДС. Значит, в ценах и в чеках НДС надо выделять. Онлайн-касса у нас через CloudPayments (их облачная касса). С чего начинать переделку?

**[S5.1] Ассистент:**

Переделка в трёх местах: данные товара, расчёт и фискальный чек.

1. **Ставка у товара.** Добавьте поле ставки НДС (по умолчанию 20%, но держите поле — для части товаров может быть иная ставка):

```python
class Product(models.Model):
    vat_rate = models.PositiveSmallIntegerField(default=20, choices=[(20, "20%"), (10, "10%"), (0, "0%")])
```

2. **Расчёт.** Розничные цены для покупателя обычно остаются «с НДС внутри»: сумма не меняется, просто из неё выделяется налог `price * rate / (100 + rate)`. Решите с бухгалтером, поднимаете ли цены или оставляете прежние.
3. **Чек.** В позициях чека для облачной кассы передавайте ставку НДС для каждой строки (`vat: 20`), а не «без НДС». И в счёте для юрлиц вместо «Без НДС» — сумма НДС отдельной строкой.

Сделайте переключение по дате: заказы, оформленные до 1 января, остаются со старыми правилами в истории и при возвратах.

**[S5.2] Пользователь:**

Цены оставляем прежние. Как правильно хранить ставку в заказе, чтобы при возврате старого заказа не пересчиталось по-новому?

**[S5.3] Ассистент:**

Фиксируйте всё, что нужно для чека, в строках заказа в момент оплаты — не ссылайтесь на текущие данные товара:

```python
class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name="items", on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    name = models.CharField(max_length=255)        # как назывался на момент покупки
    price = models.DecimalField(max_digits=10, decimal_places=2)
    qty = models.PositiveIntegerField()
    vat_rate = models.PositiveSmallIntegerField(null=True)  # None = без НДС (старые заказы)
```

Чек возврата строится из строк заказа: ставка та же, что была в чеке прихода. Для старых заказов `vat_rate=None` → в чеке возврата «без НДС». Так налоговая видит согласованные чеки прихода и возврата.

Миграция: для существующих строк заказа заполнить `vat_rate=None`, для новых — копировать ставку из товара при оформлении.

**[S5.4] Пользователь:**

Сделал. Про возвраты как раз: принимаем их 14 дней, обратная отправка тоже через СДЭК. Сейчас менеджер оформляет возврат руками в трёх местах: в заказе, в кассе, в СДЭК. Можно собрать в одну кнопку?

**[S5.5] Ассистент:**

Да, одно действие в админке, которое создаёт заявку на возврат и проводит шаги по очереди:

```python
@admin.action(description="Оформить возврат")
def make_return(modeladmin, request, queryset):
    for order in queryset:
        if order.delivered_at and now() - order.delivered_at > timedelta(days=14):
            modeladmin.message_user(request, f"Заказ {order.number}: прошло больше 14 дней", level=messages.WARNING)
            continue
        ret = Return.objects.create(order=order, status="requested")
        create_cdek_return.delay(ret.id)          # накладная на обратную отправку
```

Дальше по статусам:
1. СДЭК создал обратную накладную → покупателю SMS с номером.
2. Товар принят на склад поставщика → менеджер жмёт «Товар получен».
3. Возврат денег через API CloudPayments (`payments/refund`) → чек возврата формирует касса.

Проверку срока 14 дней делайте от даты получения, а не от даты заказа. Брак — отдельная ветка: там другие сроки, лучше пусть менеджер отмечает это явно.

**[S5.6] Пользователь:**

Сделаю статусами. Ещё хочу обновить Django — сейчас 5.0, есть 5.2 LTS. Чего бояться?

**[S5.7] Ассистент:**

Внутри 5.x переход обычно мягкий. План:

1. Включите предупреждения о депрекациях и прогоните тесты на текущей версии: `python -Wd -m pytest`. Всё, что выдаст `RemovedInDjango...Warning`, исправьте до обновления.
2. Проверьте сторонние пакеты: `django-htmx`, `django-debug-toolbar`, `django-celery-beat`, `weasyprint`-интеграцию — что они поддерживают 5.2.
3. Обновите в отдельной ветке, прогоните миграции на копии базы (`migrate --plan`, чтобы увидеть новые миграции из contrib-приложений).
4. Почитайте release notes 5.1 и 5.2 — раздел «Backwards incompatible changes».

Одна из частых мелочей — изменения в формах и виджетах админки, которые ломают кастомные шаблоны. У вас кастомизирован список заказов — посмотрите его глазами после обновления.

**[S5.8] Пользователь:**

Тесты у меня есть, но мало, и они на unittest. Хочу перейти на pytest-django и покрыть оплату и возвраты. С чего?

**[S5.9] Ассистент:**

pytest-django запускает и старые unittest-тесты, так что переход можно делать постепенно.

```ini
# pytest.ini
[pytest]
DJANGO_SETTINGS_MODULE = spica.settings.test
python_files = tests.py test_*.py
addopts = --reuse-db -q
```

Начните с самых дорогих ошибок — уведомления об оплате и возвраты:

```python
@pytest.mark.django_db
def test_cp_pay_marks_order_paid(client, order_factory, settings):
    order = order_factory(status="awaiting_payment", total=1500)
    body = urlencode({"InvoiceId": order.number, "TransactionId": "123", "Amount": "1500.00"})
    sign = base64.b64encode(hmac.new(settings.CP_API_SECRET.encode(), body.encode(), hashlib.sha256).digest()).decode()
    r = client.post("/pay/cp/pay/", body, content_type="application/x-www-form-urlencoded", HTTP_CONTENT_HMAC=sign)
    assert r.json() == {"code": 0}
    order.refresh_from_db()
    assert order.status == "paid"

@pytest.mark.django_db
def test_wrong_signature_rejected(client):
    r = client.post("/pay/cp/pay/", "InvoiceId=1", content_type="application/x-www-form-urlencoded", HTTP_CONTENT_HMAC="x")
    assert r.status_code == 403
```

Фабрики — `factory_boy`. Celery в тестах — `CELERY_TASK_ALWAYS_EAGER = True`, чтобы задачи выполнялись сразу. Внешние API (CloudPayments, СДЭК) мокайте через `responses`.

**[S5.10] Пользователь:**

Поехали так. Первые два теста зелёные. Дальше буду спрашивать по ходу.
