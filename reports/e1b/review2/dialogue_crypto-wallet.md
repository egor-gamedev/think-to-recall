# Диалог загрузки: Некастодиальный криптокошелёк «Кладезь» (React Native)

## S1

**[S1.0] Пользователь:**

Привет. Начинаем новый проект — кошелёк «Кладезь». Делаем мобилку на React Native, сразу под iOS и Android. Веб-версию не планируем, это точно, не хочу распыляться. Первый вопрос простой: какую навигацию брать? Хочется что-то скучное и надёжное.

**[S1.1] Ассистент:**

Для «скучно и надёжно» в React Native сейчас по сути один выбор — React Navigation. У него самая большая экосистема, нормальная документация и предсказуемые обновления.

Что конкретно поставить:

```bash
npm i @react-navigation/native @react-navigation/native-stack @react-navigation/bottom-tabs
npm i react-native-screens react-native-safe-area-context
cd ios && pod install
```

`native-stack` использует нативные контроллеры навигации (UINavigationController на iOS, Fragment на Android), поэтому переходы плавные и жесты «назад» работают как у системных приложений.

Типовой каркас:

```tsx
const Stack = createNativeStackNavigator<RootStackParamList>();
const Tabs = createBottomTabNavigator<TabsParamList>();

function MainTabs() {
  return (
    <Tabs.Navigator screenOptions={{ headerShown: false }}>
      <Tabs.Screen name="Wallet" component={WalletScreen} />
      <Tabs.Screen name="Activity" component={ActivityScreen} />
      <Tabs.Screen name="Settings" component={SettingsScreen} />
    </Tabs.Navigator>
  );
}

export function RootNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Main" component={MainTabs} options={{ headerShown: false }} />
        <Stack.Screen name="Send" component={SendScreen} />
        <Stack.Screen name="Receive" component={ReceiveScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

Альтернатива — Expo Router (файловая маршрутизация), но он тянет за собой Expo-подход ко всему проекту. Если вы на bare React Native, проще остаться на React Navigation.

**[S1.2] Пользователь:**

Мы на bare, без expo. Ок, берём React Navigation. Типизацию параметров экранов как лучше сделать, чтоб не было any везде?

**[S1.3] Ассистент:**

Заведите один файл с типами всех стеков и объявите глобальный тип для `useNavigation`, тогда автокомплит будет везде без ручных дженериков.

```ts
// src/navigation/types.ts
export type RootStackParamList = {
  Main: NavigatorScreenParams<TabsParamList>;
  Send: { asset: AssetId; prefillAddress?: string };
  Receive: { asset: AssetId };
};

export type TabsParamList = {
  Wallet: undefined;
  Activity: { filter?: 'in' | 'out' };
  Settings: undefined;
};

declare global {
  namespace ReactNavigation {
    interface RootParamList extends RootStackParamList {}
  }
}
```

В экране:

```ts
type Props = NativeStackScreenProps<RootStackParamList, 'Send'>;
export function SendScreen({ route }: Props) {
  const { asset, prefillAddress } = route.params; // уже типизировано
}
```

И ещё совет: держите названия экранов только в этом файле и не передавайте в параметрах большие объекты (баланс, список транзакций) — только идентификаторы. Параметры сериализуются при восстановлении состояния, большие объекты там ни к чему.

**[S1.4] Пользователь:**

Понял. Чтобы ты понимал масштаб: нас двое, я и джун. MVP надо показать через три месяца. Так что всё, что требует отдельного человека на поддержку, сразу мимо. Думаю сделать монорепу, чтобы мобилка и бэк жили вместе. Норм идея?

**[S1.5] Ассистент:**

Для двоих — да, монорепа хорошо подходит: один PR меняет и клиент, и сервер, общие типы не расходятся, один CI.

Минимальная структура:

```
kladez/
  apps/
    mobile/        # React Native
    api/           # бэкенд
  packages/
    shared/        # типы, константы сетей, утилиты форматирования
    config/        # общий eslint/tsconfig/prettier
  package.json
  pnpm-workspace.yaml
```

Инструменты: pnpm workspaces для зависимостей и, если захочется кешировать сборки, Turborepo. Но на старте можно и без Turborepo — это лишняя сущность для джуна.

Подводный камень с React Native в монорепе — Metro по умолчанию не видит пакеты вне `apps/mobile`. Нужно добавить `watchFolders` и запретить дубли React:

```js
// apps/mobile/metro.config.js
const path = require('path');
const root = path.resolve(__dirname, '../..');
module.exports = {
  watchFolders: [root],
  resolver: {
    nodeModulesPaths: [path.resolve(__dirname, 'node_modules'), path.resolve(root, 'node_modules')],
    disableHierarchicalLookup: true,
  },
};
```

И в `.npmrc` для pnpm: `node-linker=hoisted` — React Native до сих пор плохо дружит с симлинками pnpm.

**[S1.6] Пользователь:**

Сделал так, metro завёлся со второго раза. Теперь главное, с чего вообще надо было начать. Кошелёк некастодиальный: приватные ключи и сид-фраза живут только на устройстве и на сервер не уходят никогда. Ни в логи, ни в крэш-репорты, никуда.

**[S1.7] Ассистент:**

Это главное архитектурное ограничение, из него следует несколько практических правил.

1. **Где лежит секрет.** Никакого AsyncStorage и обычных файлов — только аппаратное хранилище платформы (Keychain на iOS, Keystore на Android). Детали подберём, когда дойдём до экрана создания кошелька.
2. **Крэш-репорты.** Если будете ставить Sentry или аналог, сразу включите `beforeSend`, который вырезает всё, похожее на мнемонику и приватный ключ. Лучше вообще не держать сид в state React — только в момент показа и подписи.
3. **Логи.** Запретите `console.log` в продовой сборке (babel-плагин `transform-remove-console`), чтобы джун случайно не залогировал ключ.
4. **Буфер обмена.** Если дадите копировать сид-фразу, очищайте буфер через 30–60 секунд.
5. **Скриншоты.** На экране показа сид-фразы — `FLAG_SECURE` на Android и затемнение превью в app switcher на iOS.

```ts
// правило для eslint, чтобы ловить подозрительное логирование
'no-restricted-syntax': ['error', {
  selector: "CallExpression[callee.object.name='console'][arguments.0.name=/mnemonic|seed|privateKey/i]",
  message: 'Не логируйте секреты',
}],
```

**[S1.8] Пользователь:**

Отлично, правило в линтер забираю. По сетям: поддерживаем Ethereum и Polygon. Solana обсуждали с ребятами и отказались — в команде никто под неё не писал, а со сроками экспериментировать не будем.

**[S1.9] Ассистент:**

Ethereum и Polygon — это один EVM-стек, вторую сеть вы получаете почти бесплатно: те же адреса, тот же формат транзакций, отличаются `chainId`, RPC и нативный токен.

Библиотека для работы с сетью — `viem`: легче ethers, лучше типизирована, нормально работает в React Native (нужны полифилы `crypto.getRandomValues` и `TextEncoder`).

```ts
// packages/shared/src/chains.ts
import { mainnet, polygon } from 'viem/chains';
export const SUPPORTED_CHAINS = { ethereum: mainnet, polygon } as const;
export type ChainKey = keyof typeof SUPPORTED_CHAINS;
```

Для полифилов в мобилке:

```ts
// apps/mobile/index.js — первой строкой
import 'react-native-get-random-values';
import 'text-encoding-polyfill';
```

Не забудьте проверить на настоящем устройстве генерацию ключа: в симуляторе `getRandomValues` иногда работает через JS-фолбэк, и вы не заметите, что нативный модуль не подключился.

**[S1.10] Пользователь:**

Теперь бэк. Пишем на NestJS, но он нужен только для двух вещей: курсы валют и пуш-уведомления. Аккаунтов у пользователей нет, клиента узнаём по device id. Никакой регистрации.

**[S1.11] Ассистент:**

Тогда бэкенд получается очень тонким — и это хорошо для команды из двух человек.

```ts
// apps/api/src/app.module.ts
@Module({
  imports: [ConfigModule.forRoot({ isGlobal: true }), RatesModule, DevicesModule, PushModule],
})
export class AppModule {}
```

Эндпоинты:

- `GET /rates?assets=eth,matic&fiat=eur` — отдаёт кэшированные курсы.
- `POST /devices` — регистрирует `deviceId` и push-токен, ничего больше.
- `PUT /devices/:id/watch` — список адресов, по которым слать пуши о входящих (адреса публичны, это не секрет, но всё равно не храните их дольше, чем нужно).

`deviceId` генерируйте на клиенте случайным UUID при первом запуске и храните в Keychain/Keystore, а не берите аппаратный идентификатор — так вы не собираете лишних персональных данных, а переустановка приложения честно даёт новое устройство.

Защитить эндпоинты от мусора можно лимитом запросов по IP и подписью запроса ключом устройства, но это уже на потом.

**[S1.12] Пользователь:**

Да, пока без подписи запросов. Давай линтер нормально настроим, чтоб конфиг был один на всю монорепу и джун не писал var и не забывал зависимости в хуках.

**[S1.13] Ассистент:**

Сделайте общий flat-config в `packages/config` и подключите его в каждом приложении.

```js
// packages/config/eslint/base.mjs
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';

export default [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    plugins: { 'react-hooks': reactHooks },
    rules: {
      'no-var': 'error',
      'prefer-const': 'error',
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  },
];
```

```js
// apps/mobile/eslint.config.mjs
import base from '@kladez/config/eslint/base.mjs';
export default [...base, { ignores: ['android/**', 'ios/**'] }];
```

Добавьте `lint-staged` + `husky`, чтобы линтер запускался перед коммитом, — для джуна это лучший учитель: ошибки видны сразу, а не на ревью.

**[S1.14] Пользователь:**

Сделал. По данным: историю цен храним в PostgreSQL, хочу графики за месяц-год строить и агрегаты считать нормально.

**[S1.15] Ассистент:**

Для истории цен в PostgreSQL достаточно одной таблицы с составным индексом:

```sql
CREATE TABLE price_points (
  asset      TEXT        NOT NULL,   -- 'eth' | 'matic'
  fiat       TEXT        NOT NULL,   -- 'eur' | 'usd' | 'rub'
  ts         TIMESTAMPTZ NOT NULL,
  price      NUMERIC(20, 8) NOT NULL,
  PRIMARY KEY (asset, fiat, ts)
);
```

Для графиков удобно хранить точки раз в 5 минут, а для длинных периодов строить агрегаты на лету:

```sql
SELECT date_trunc('day', ts) AS day, avg(price) AS price
FROM price_points
WHERE asset = 'eth' AND fiat = 'eur' AND ts > now() - interval '1 year'
GROUP BY 1 ORDER BY 1;
```

Объём маленький: две монеты × три валюты × 288 точек в день — около 630 тысяч строк в год, это для PostgreSQL ничто. В NestJS подключайте через Prisma или TypeORM — для двоих разработчиков Prisma проще: схема в одном файле, миграции генерируются.

**[S1.16] Пользователь:**

Возьмём Prisma. Слушай, а у нас баг на Android: иконки не показываются, вместо них квадратики. На iOS всё ок. Стоит react-native-vector-icons.

**[S1.17] Ассистент:**

Классическая проблема: на Android шрифты иконок нужно явно подключить в Gradle, а в монорепе ещё и путь к `node_modules` другой.

1. В `apps/mobile/android/app/build.gradle` в самом конце:

```gradle
apply from: file("../../node_modules/react-native-vector-icons/fonts.gradle")
```

Проверьте путь: если пакет поднят в корень монорепы, путь будет `../../../../node_modules/...`.

2. Если используете не все наборы, ограничьте их, чтобы не раздувать APK:

```gradle
project.ext.vectoricons = [ iconFontNames: ['MaterialCommunityIcons.ttf', 'Feather.ttf'] ]
```

3. Пересоберите с очисткой: `cd android && ./gradlew clean && cd .. && npx react-native run-android`.

Если квадратики остались — проверьте, что имя иконки существует именно в выбранном наборе: на Android несуществующее имя тоже рисуется квадратом.

**[S1.18] Пользователь:**

Да, путь был от старой структуры, поправил на четыре уровня вверх — иконки появились. Завтра садимся за онбординг, на сегодня всё.

## S2

**[S2.0] Пользователь:**

Привет, продолжаем. Сегодня экран создания кошелька. Генерим мнемонику из 12 слов, показываем пользователю, потом просим подтвердить пару слов. С чего начать?

**[S2.1] Ассистент:**

Порядок такой: генерация → показ → проверка → сохранение. Генерацию делайте через `@scure/bip39` — маленькая, аудированная библиотека, без лишних зависимостей:

```ts
import { generateMnemonic, validateMnemonic } from '@scure/bip39';
import { wordlist } from '@scure/bip39/wordlists/english';

export function createMnemonic(): string {
  return generateMnemonic(wordlist, 128); // 128 бит → 12 слов
}
```

Экран показа: слова сеткой 3×4 с номерами, кнопка «Я записал» активна не сразу, а через несколько секунд, чтобы человек не пролистал. Экран проверки: спросите 3 случайных слова по номерам и дайте выбрать каждое из 4 вариантов — так меньше опечаток, чем при ручном вводе.

Важно: держите мнемонику только в локальном state экрана и затирайте её при уходе со стека создания. Не кладите её в Redux/Zustand и не передавайте в параметрах навигации.

**[S2.2] Пользователь:**

Ок. Хранить будем в Secure Enclave на айфоне и Android Keystore — через react-native-keychain, доступ по биометрии. Покажи, как правильно сохранить, чтобы без отпечатка/лица было не достать.

**[S2.3] Ассистент:**

С `react-native-keychain` это делается флагами доступа:

```ts
import * as Keychain from 'react-native-keychain';

const SERVICE = 'kladez.wallet.v1';

export async function saveSecret(mnemonic: string) {
  await Keychain.setGenericPassword('wallet', mnemonic, {
    service: SERVICE,
    accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_CURRENT_SET,
    accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
    securityLevel: Keychain.SECURITY_LEVEL.SECURE_HARDWARE,
  });
}

export async function readSecret(): Promise<string | null> {
  const res = await Keychain.getGenericPassword({
    service: SERVICE,
    authenticationPrompt: { title: 'Подтвердите доступ к кошельку' },
  });
  return res ? res.password : null;
}
```

Пояснения:
- `BIOMETRY_CURRENT_SET` — если пользователь добавит новый отпечаток, доступ к секрету пропадёт. Это защищает от сценария «кто-то добавил свой палец», но значит, что без резервной копии сид-фразы пользователь может потерять доступ. Это надо явно сказать в онбординге.
- `THIS_DEVICE_ONLY` — секрет не попадёт в iCloud Keychain и не переедет на новое устройство.
- На Android `SECURE_HARDWARE` требует аппаратного Keystore; на старых устройствах без него будет ошибка — обработайте её понятным сообщением, а не падением.

**[S2.4] Пользователь:**

Про резервную копию. Обсуждали облачный бэкап сид-фразы в iCloud или Google Drive, зашифрованный паролем — отказались. Только бумажная копия, которую человек записывает сам. Меньше поверхность атаки и меньше вопросов к нам.

**[S2.5] Ассистент:**

Понятно, тогда онбординг должен очень серьёзно подводить к бумажной копии — это единственный путь восстановления.

Что помогает на практике:
- Отдельный экран перед показом слов: «Приготовьте бумагу и ручку. Не делайте скриншот. Без этих слов восстановить кошелёк невозможно».
- Блокировка скриншотов на экране слов (`FLAG_SECURE` на Android, на iOS — размытие при уходе в фон).
- Напоминание в настройках, если пользователь пропустил проверку: «Вы не подтвердили резервную копию».
- Восстановление: экран ввода 12 слов с автодополнением по словарю BIP39 и проверкой контрольной суммы.

```ts
export function isValidPhrase(input: string): boolean {
  const words = input.trim().toLowerCase().split(/\s+/);
  return words.length === 12 && validateMnemonic(words.join(' '), wordlist);
}
```

**[S2.6] Пользователь:**

Сделали. Джун прислал PR с экраном проверки слов, глянь на кусок, что-то мне не нравится, но не пойму что:

```tsx
const [words, setWords] = useState(mnemonic.split(' '));
useEffect(() => {
  setChallenge(pickRandom(words, 3));
});
```

**[S2.7] Ассистент:**

Тут две проблемы.

1. `useEffect` без массива зависимостей запускается после каждого рендера. `setChallenge` вызывает новый рендер, тот снова запускает эффект — получится бесконечный цикл (или, если `pickRandom` возвращает тот же массив, — постоянная смена вопросов). Нужен `[]` или `[words]`.

2. `useState(mnemonic.split(' '))` делит строку на каждом рендере, хотя результат нужен только при первом. Используйте ленивую инициализацию: `useState(() => mnemonic.split(' '))`. Мелочь, но хорошая привычка.

Исправленный вариант:

```tsx
const words = useMemo(() => mnemonic.split(' '), [mnemonic]);
const [challenge] = useState(() => pickRandom(words, 3));
```

Здесь вопросы выбираются один раз при открытии экрана и не меняются, пока пользователь отвечает. Линтер `react-hooks/exhaustive-deps`, который мы включили, кстати, должен был подсветить первый случай — проверьте, что у джуна в редакторе ESLint вообще работает.

**[S2.8] Пользователь:**

Ха, у него eslint в vscode был выключен. Ладно. Ещё вопрос про рынок: пользователи у нас в ЕС. Покупку крипты за фиат, on-ramp, в MVP не делаем, отложили на потом. Что нам из регуляторики надо держать в голове уже сейчас?

**[S2.9] Ассистент:**

Для некастодиального кошелька без on-ramp нагрузка небольшая, но несколько вещей стоит заложить сразу.

- **MiCA.** Регламент ЕС о криптоактивах в основном касается поставщиков услуг, которые хранят средства клиентов или проводят обмен за них. Некастодиальный кошелёк, где ключи только у пользователя, обычно не считается таким поставщиком. Но как только появится покупка за фиат или обмен, где вы выступаете стороной, картина меняется — это уже вопрос к юристу.
- **GDPR.** Пользователи в ЕС — значит, к любым персональным данным, которые вы собираете (даже device id и push-токен), применим GDPR: нужна политика конфиденциальности, минимизация данных и способ удалить данные по запросу.
- **Правила сторов.** Apple и Google требуют отдельно описывать криптофункции; для некоторых стран у них свои ограничения.

Это не юридическая консультация, но архитектурно вы в хорошей позиции: чем меньше вы знаете о пользователе, тем меньше обязательств.

**[S2.10] Пользователь:**

Да, в эту сторону и думаем. Поэтому никаких сторонних SDK аналитики — ни Firebase Analytics, ни Amplitude. Только своя обезличенная статистика, и то если пользователь согласился, opt-in.

**[S2.11] Ассистент:**

Тогда можно сделать простой собственный сборщик событий:

```ts
type Event = { name: 'onboarding_step' | 'wallet_created' | 'send_started' | 'send_done'; step?: number };

export async function track(e: Event) {
  if (!(await settings.analyticsOptIn())) return;
  queue.push({ ...e, day: new Date().toISOString().slice(0, 10) }); // только дата, без времени
}
```

Принципы:
- Без идентификатора пользователя и устройства — только счётчики по дням.
- Отправка пачкой раз в день, а не по событию, чтобы по времени запросов нельзя было восстановить поведение.
- На бэкенде — таблица агрегатов `(day, event, count)`, сырые события не храните.
- Экран согласия с понятным текстом и выключателем в настройках.

Воронку онбординга так вполне можно увидеть: сколько дошло до каждого шага в день.

**[S2.12] Пользователь:**

Норм. Слушай, ещё мелочь: даты в истории транзакций показываются как 10/21/2026, а у нас половина пользователей русскоязычные. Как сделать нормально?

**[S2.13] Ассистент:**

Используйте `Intl.DateTimeFormat` с локалью устройства, а не ручное форматирование:

```ts
import { getLocales } from 'react-native-localize';

const locale = getLocales()[0]?.languageTag ?? 'en-US';
const fmt = new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short', year: 'numeric' });
fmt.format(new Date(tx.timestamp * 1000)); // «21 окт. 2026 г.» или «Oct 21, 2026»
```

В Hermes `Intl` поддерживается, но на Android нужно, чтобы в сборке был полный ICU — проверьте, что в `android/app/build.gradle` не стоит урезанный JSC/ICU. Для относительного времени («5 минут назад») есть `Intl.RelativeTimeFormat`.

Суммы тоже форматируйте через `Intl.NumberFormat`, но аккуратно: для крипты нельзя полагаться на `number` — держите значения в `bigint` (wei) и переводите в строку своей функцией, иначе потеряете точность на больших суммах.

**[S2.14] Пользователь:**

Ага, на бигинтах и так. Последнее на сегодня — пуши о входящих транзакциях. Шлём через Firebase Cloud Messaging на Android и APNs на iOS. Как это связать с нашим бэком на Nest?

**[S2.15] Ассистент:**

Схема:

1. Клиент получает push-токен (FCM на Android, APNs на iOS — через `@react-native-firebase/messaging` можно получить оба) и отправляет его на `POST /devices` вместе с `deviceId`.
2. Клиент сообщает список публичных адресов для наблюдения: `PUT /devices/:id/watch`.
3. Бэкенд подписывается на входящие переводы по этим адресам (вебхуки провайдера узлов или опрос новых блоков) и шлёт пуш.

```ts
@Injectable()
export class PushService {
  constructor(private readonly fcm: FcmClient, private readonly apns: ApnsClient) {}

  async notifyIncoming(device: Device, tx: IncomingTx) {
    const body = `Получено ${formatAmount(tx)} ${tx.symbol}`;
    if (device.platform === 'android') await this.fcm.send(device.pushToken, { title: 'Кладезь', body });
    else await this.apns.send(device.pushToken, { alert: { title: 'Кладезь', body } });
  }
}
```

Не кладите в текст пуша адрес отправителя целиком и не пишите сумму, если пользователь включил «скрывать балансы» — уведомление видно на экране блокировки.

**[S2.16] Пользователь:**

Про скрытие балансов хорошая мысль, добавлю в бэклог. Всё, до завтра.

## S3

**[S3.0] Пользователь:**

Привет. Неделя прошла, онбординг собрали. Сейчас про бэк. Курсы валют тянем с CoinGecko каждые 5 минут, и при каждом запросе от клиента лезем в базу. Можно как-то проще?

**[S3.1] Ассистент:**

Курсы обновляются раз в 5 минут, а читаются постоянно — значит, держите последний срез в памяти процесса и отдавайте из неё, а в базу пишите только историю.

```ts
@Injectable()
export class RatesService implements OnModuleInit {
  private latest = new Map<string, number>(); // 'eth:eur' -> 2345.67
  private updatedAt = 0;

  constructor(private readonly gecko: CoinGeckoClient, private readonly repo: PriceRepo) {}

  onModuleInit() { void this.refresh(); setInterval(() => void this.refresh(), 5 * 60_000); }

  async refresh() {
    const rows = await this.gecko.simplePrice(['ethereum', 'matic-network'], ['eur', 'usd', 'rub']);
    for (const r of rows) this.latest.set(`${r.asset}:${r.fiat}`, r.price);
    this.updatedAt = Date.now();
    await this.repo.insertMany(rows);
  }

  get(asset: string, fiat: string) {
    return { price: this.latest.get(`${asset}:${fiat}`), updatedAt: this.updatedAt };
  }
}
```

Плюс HTTP-кэш на ответе: `Cache-Control: public, max-age=60`. Клиенты будут реже ходить, а если перед бэком появится CDN или обратный прокси, он тоже сможет кэшировать.

**[S3.2] Пользователь:**

Сделал. И вот ещё что решили, пока считали деньги: от PostgreSQL отказываемся. Историю цен держим в SQLite прямо на том же сервере — данных копейки, отдельный постгрес ради этого дорого и лишний сервис поддерживать. Prisma с SQLite же дружит?

**[S3.3] Ассистент:**

Дружит, и для такого объёма SQLite — разумный выбор. Что поменять:

```prisma
datasource db {
  provider = "sqlite"
  url      = "file:/data/kladez.db"
}

model PricePoint {
  asset String
  fiat  String
  ts    DateTime
  price Decimal
  @@id([asset, fiat, ts])
}
```

Нюансы перехода:
- **Типы.** В SQLite нет `TIMESTAMPTZ` и настоящего `NUMERIC` — Prisma хранит `DateTime` как текст/число, `Decimal` как текст. Для графиков это не важно, но агрегаты (`avg`) считайте в коде или через `CAST`.
- **`date_trunc`** из PostgreSQL нет — группировку по дням делайте через `strftime('%Y-%m-%d', ts)`.
- **Режим WAL.** Включите `PRAGMA journal_mode=WAL;`, чтобы чтение не блокировалось записью раз в 5 минут.
- **Миграции.** `prisma migrate` работает и с SQLite; старые миграции под PostgreSQL удалите и сгенерируйте заново, они несовместимы.
- **Перенос данных.** Если в PostgreSQL уже есть история — выгрузите в CSV и загрузите скриптом; если там пара недель тестовых данных, проще начать с нуля.

**[S3.4] Пользователь:**

Тестовые там, начнём с нуля. Слушай, в dev-сборке мобилки при запросе к локальному API ловлю CORS ошибку. Хотя это же мобилка, откуда там CORS?

**[S3.5] Ассистент:**

В нативном приложении CORS действительно нет — браузерного механизма там не существует. Значит, запрос идёт не из нативного кода. Варианты:

1. Вы открываете экран через отладчик в Chrome (старый remote JS debugging) — тогда JS выполняется в браузере, и CORS появляется. С Hermes и новым дебаггером такого быть не должно.
2. Запрос делает WebView (например, страница, встроенная в приложение) — там CORS настоящий.
3. Ошибка не CORS, а `Network request failed`, которую кто-то в коде оборачивает в сообщение про CORS.

Чаще всего на Android проблема в другом: эмулятор не видит `localhost` хоста. Используйте `10.0.2.2` вместо `localhost`, а для HTTP без TLS добавьте в debug-манифест `android:usesCleartextTraffic="true"`.

Если всё-таки хотите разрешить CORS на бэке для отладки:

```ts
app.enableCors({ origin: process.env.NODE_ENV === 'development' ? true : false });
```

**[S3.6] Пользователь:**

Был remote debugging, выключил — прошло. Теперь деплой. Хостимся на одном VPS в Hetzner, всё в Docker Compose. Kubernetes не берём, нам двоим он не нужен. Как сделать, чтобы обновление было без простоя хотя бы минимального?

**[S3.7] Ассистент:**

На одном сервере с Compose проще всего так: обратный прокси (Caddy или Traefik) перед API и выкатка новой версии рядом со старой.

```yaml
services:
  caddy:
    image: caddy:2
    ports: ["80:80", "443:443"]
    volumes: ["./Caddyfile:/etc/caddy/Caddyfile", "caddy_data:/data"]
  api:
    image: ghcr.io/kladez/api:${API_TAG}
    env_file: .env
    volumes: ["db:/data"]
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:3000/health"]
      interval: 10s
volumes: { caddy_data: {}, db: {} }
```

Выкатка: `docker compose pull api && docker compose up -d --no-deps api`. Простой будет несколько секунд, пока стартует контейнер; для MVP это нормально. Если нужно совсем без простоя — поднимайте `api_blue` и `api_green` и переключайте upstream в Caddy, но это уже усложнение.

Важно для SQLite: том `db` должен быть один и тот же, и одновременно с базой должен работать только один экземпляр API, иначе получите блокировки записи.

**[S3.8] Пользователь:**

Понял, сине-зелёный не надо. Бэкапы как сделать? Сервер один, если умрёт — всё.

**[S3.9] Ассистент:**

С SQLite бэкап — это копия одного файла, но копировать его нужно правильно, не `cp` во время записи.

```bash
# /opt/kladez/backup.sh
set -euo pipefail
ts=$(date +%F-%H%M)
docker compose exec -T api sqlite3 /data/kladez.db ".backup '/data/backup-$ts.db'"
docker compose cp api:/data/backup-$ts.db /opt/kladez/backups/
docker compose exec -T api rm /data/backup-$ts.db
rclone copy /opt/kladez/backups/ hetzner-storage:kladez-backups --max-age 2d
find /opt/kladez/backups -mtime +7 -delete
```

Крон раз в сутки. Хранилище — Hetzner Storage Box или любой S3-совместимый бакет в другом дата-центре. Раз в месяц проверяйте восстановление: скачайте бэкап и откройте его `sqlite3 backup.db 'select count(*) from PricePoint'`.

И помните, что на сервере нет ничего критичного для пользователей: ключи у них на устройствах. Потерять историю цен неприятно, но не катастрофа.

**[S3.10] Пользователь:**

Да, это и успокаивает. Дальше ноды. Берём узлы у Alchemy, бесплатный тариф. Там жёсткий лимит запросов в секунду, мы его уже пару раз ловили на тестах, когда грузили балансы всех токенов.

**[S3.11] Ассистент:**

На бесплатном тарифе лимит считается в «compute units» в секунду, и разные методы стоят по-разному: `eth_getBalance` дешёвый, а `eth_getLogs` по большому диапазону — дорогой. Что помогает:

1. **Батчинг.** Балансы токенов одного адреса — одним вызовом `alchemy_getTokenBalances` вместо N вызовов `balanceOf`, или через Multicall3:

```ts
const results = await client.multicall({
  contracts: tokens.map((t) => ({ address: t.address, abi: erc20Abi, functionName: 'balanceOf', args: [owner] })),
});
```

2. **Ретраи с паузой** при 429: экспоненциальная задержка с джиттером, не больше 3 попыток.
3. **Обновление по требованию.** Не опрашивайте балансы по таймеру каждые 10 секунд; обновляйте при открытии экрана и по пушу о входящей транзакции.
4. **Разные ключи** для бэкенда и для мобилки, чтобы мобилки одного пользователя-энтузиаста не съели лимит бэкенда.

**[S3.12] Пользователь:**

Про обновление по требованию — как раз в тему. Требование от меня: приложение должно показывать балансы и без сети, из локального кэша, и писать, когда последний раз обновлялось. В метро у нас половина пользователей.

**[S3.13] Ассистент:**

Тогда балансы — это локальное состояние, которое иногда синхронизируется, а не данные, которые всегда тянутся из сети.

```ts
type CachedBalances = { address: string; chain: ChainKey; items: { token: string; raw: string }[]; updatedAt: number };

export async function loadBalances(address: string, chain: ChainKey) {
  const cached = await storage.get<CachedBalances>(`bal:${chain}:${address}`);
  if (cached) store.setBalances(cached); // сразу показываем
  if (!(await NetInfo.fetch()).isConnected) return;
  try {
    const fresh = await fetchBalances(address, chain);
    await storage.set(`bal:${chain}:${address}`, fresh);
    store.setBalances(fresh);
  } catch { /* остаёмся на кэше */ }
}
```

В интерфейсе: под суммой мелкая подпись «обновлено 12 мин назад», а без сети — «нет сети, данные от 14:32». Кнопку «Отправить» без сети блокируйте с понятным текстом: подписать транзакцию можно и офлайн, но отправить нельзя.

Храните кэш в MMKV — он быстрый и синхронный. Балансы не секрет, шифровать их отдельно не обязательно, но если включено «скрывать балансы», не показывайте их и из кэша.

**[S3.14] Пользователь:**

MMKV у нас уже есть для настроек, туда и положим. Последнее — логи на бэке. Сейчас NestJS пишет в консоль, docker это собирает, но читать невозможно.

**[S3.15] Ассистент:**

Переведите на структурные JSON-логи и ограничьте размер логов Docker.

```ts
// main.ts
import { Logger } from 'nestjs-pino';
const app = await NestFactory.create(AppModule, { bufferLogs: true });
app.useLogger(app.get(Logger));
```

```ts
LoggerModule.forRoot({
  pinoHttp: {
    level: process.env.LOG_LEVEL ?? 'info',
    redact: ['req.headers.authorization', 'req.body.pushToken'],
    customProps: () => ({ service: 'api' }),
  },
}),
```

В Compose:

```yaml
api:
  logging:
    driver: json-file
    options: { max-size: "20m", max-file: "5" }
```

Читать удобно через `docker compose logs -f api | npx pino-pretty`. Обратите внимание на `redact`: push-токены и заголовки авторизации в логах не нужны.

**[S3.16] Пользователь:**

Сделал, стало читаемо. Спасибо, на этой неделе всё.

## S4

**[S4.0] Пользователь:**

Привет. Новости: добавляем обмен токенов прямо в приложении. Через 0x API, берём комиссию 0,3% в свою пользу. Начнём с экрана обмена — как его устроить, чтобы было понятно, сколько человек получит?

**[S4.1] Ассистент:**

Экран обмена держится на котировке: пользователь вводит сумму, вы запрашиваете цену, показываете итог и только потом строите транзакцию.

У 0x для этого два шага: `price` (индикативная цена, дешёвый запрос — можно дёргать при вводе) и `quote` (твёрдая котировка с данными транзакции — только по нажатию «Обменять»).

```ts
const params = new URLSearchParams({
  chainId: String(chainId),
  sellToken, buyToken,
  sellAmount: amountWei.toString(),
  taker: address,
  swapFeeRecipient: FEE_WALLET,
  swapFeeBps: '30',            // 0,3%
  swapFeeToken: buyToken,
});
const res = await fetch(`https://api.0x.org/swap/permit2/price?${params}`, {
  headers: { '0x-api-key': ZEROX_KEY, '0x-version': 'v2' },
});
```

На экране показывайте: «Вы отдаёте», «Вы получаете (примерно)», «Минимум с учётом проскальзывания», «Комиссия сервиса 0,3%», «Комиссия сети ≈ … €». Комиссию сервиса показывайте явно отдельной строкой — и сторам, и пользователям это важно.

Ключ 0x не кладите в приложение: проксируйте запросы через ваш бэкенд, иначе его вытащат из сборки за пять минут.

**[S4.2] Пользователь:**

Окей, через бэк проксируем. Комиссию сети как посчитать заранее, чтобы показать в евро?

**[S4.3] Ассистент:**

Комиссия сети = `gas × (baseFee + priorityFee)`. 0x в ответе `quote` возвращает оценку газа; цену газа берите у узла:

```ts
const fees = await client.estimateFeesPerGas(); // { maxFeePerGas, maxPriorityFeePerGas }
const feeWei = BigInt(quote.transaction.gas) * fees.maxFeePerGas;
const feeEth = Number(feeWei) / 1e18;          // для показа, не для расчётов
const feeEur = feeEth * rates.get('eth', 'eur').price;
```

Нюансы:
- Показывайте верхнюю оценку (`maxFeePerGas`), а фактически спишется меньше — лучше приятно удивить.
- В Polygon нативная монета другая, курс берите для неё.
- Если у пользователя не хватает нативной монеты на комиссию, ошибку надо показать до подписи: «Недостаточно ETH для оплаты сети», а не дать транзакции упасть.
- При обмене ERC-20 первым шагом может понадобиться одобрение (approve) или подпись Permit2 — это отдельная комиссия или отдельная подпись, предупредите об этом.

**[S4.4] Пользователь:**

Так, и вторая новость, побольше. Помнишь, говорили, что веба не будет? Передумали: делаем веб-дашборд на Next.js. Только просмотр портфеля — балансы, история, графики. Никаких отправок средств с сайта.

**[S4.5] Ассистент:**

Понял, это меняет картину: появляется второй клиент, и ему нужно как-то понять, чей портфель показывать. Раз дашборд только для просмотра, базовая версия может вообще ничего не хранить: пользователь вводит или подключает адрес, вы показываете публичные данные блокчейна.

Структура в монорепе:

```
apps/
  mobile/
  api/
  web/          # Next.js, App Router
packages/
  shared/       # типы, сети, форматирование сумм — теперь пригодится вебу
```

Графики — из того же `GET /rates/history`, балансы — через тот же код, что в мобилке, вынесенный в `packages/shared`. Хороший момент проверить, что в `shared` нет ничего React Native-специфичного.

**[S4.6] Пользователь:**

Только сразу: логинов и паролей у нас нет ни в приложении, ни на сайте и не будет. Личность — это адрес кошелька и подпись. Хочу, чтобы на сайт входили, подключив кошелёк.

**[S4.7] Ассистент:**

Тогда стандартный путь — Sign-In with Ethereum (EIP-4361): сайт показывает сообщение со случайным nonce, пользователь подписывает его в кошельке (через WalletConnect с телефона или расширением), сервер проверяет подпись и узнаёт адрес.

```ts
// web: формирование сообщения
import { createSiweMessage } from 'viem/siwe';
const message = createSiweMessage({
  domain: 'app.kladez.io', address, chainId: 1,
  uri: 'https://app.kladez.io', version: '1',
  nonce, statement: 'Вход в дашборд Кладезь',
});
const signature = await walletClient.signMessage({ account: address, message });
```

```ts
// api: проверка
import { verifySiweMessage } from 'viem/siwe';
const ok = await verifySiweMessage(publicClient, { message, signature, nonce: storedNonce });
```

Nonce одноразовый и живёт пару минут — храните его на сервере и удаляйте после проверки.

**[S4.8] Пользователь:**

Ага. И чтобы не подписывать на каждый клик: после проверки подписи бэк выдаёт короткую JWT-сессию для дашборда. Значит, бэк теперь не только курсы и пуши. Как это аккуратно сделать в Nest?

**[S4.9] Ассистент:**

Отдельный модуль `AuthModule` только для веба: проверка SIWE → выдача JWT → guard на эндпоинтах дашборда.

```ts
@Controller('auth')
export class AuthController {
  constructor(private readonly siwe: SiweService, private readonly jwt: JwtService) {}

  @Get('nonce')
  nonce() { return { nonce: this.siwe.issueNonce() }; }

  @Post('verify')
  async verify(@Body() dto: VerifyDto, @Res({ passthrough: true }) res: Response) {
    const address = await this.siwe.verify(dto.message, dto.signature);
    const token = await this.jwt.signAsync({ sub: address.toLowerCase() }, { expiresIn: '15m' });
    res.cookie('session', token, { httpOnly: true, secure: true, sameSite: 'lax' });
    return { address };
  }
}
```

Решения, которые стоит зафиксировать:
- JWT в `httpOnly`-cookie, а не в localStorage — меньше риск кражи через XSS.
- Короткое время жизни; продление — повторной подписью или отдельным refresh-токеном (обсудим, когда понадобится).
- Мобильному приложению эта сессия не нужна: у него свои ключи на устройстве, и эндпоинты курсов/пушей остаются без авторизации, как были.

**[S4.10] Пользователь:**

15 минут пока ок. Слушай, а что делать, если человек в кошельке нажал «отклонить» на подписи? Сейчас у нас на сайте просто белый экран.

**[S4.11] Ассистент:**

Отклонение подписи — это нормальный сценарий, а не ошибка. Кошельки возвращают его как `UserRejectedRequestError` (код 4001), его надо ловить отдельно:

```ts
import { UserRejectedRequestError } from 'viem';

try {
  const signature = await walletClient.signMessage({ account, message });
  await api.verify(message, signature);
} catch (e) {
  if (e instanceof UserRejectedRequestError || (e as any)?.code === 4001) {
    setState({ kind: 'idle', notice: 'Вход отменён — подпись не была подтверждена' });
    return;
  }
  setState({ kind: 'error', message: 'Не удалось войти. Попробуйте ещё раз' });
  reportError(e);
}
```

Белый экран, скорее всего, — необработанное исключение в компоненте. Добавьте в Next.js `error.tsx` на уровне сегмента, чтобы любая ошибка показывала аккуратную заглушку, а не пустую страницу.

**[S4.12] Пользователь:**

Поймал, было исключение в рендере. Последнее на сегодня — вёрстка дашборда. Хочу слева список активов, справа график, на мобильном вебе — одно под другим. Tailwind у нас.

**[S4.13] Ассистент:**

На Tailwind это сетка с переключением колонок на брейкпоинте:

```tsx
export default function Dashboard() {
  return (
    <main className="mx-auto max-w-6xl p-4 grid gap-4 grid-cols-1 lg:grid-cols-[320px_1fr]">
      <aside className="rounded-xl border p-3">
        <AssetList />
      </aside>
      <section className="rounded-xl border p-3 min-h-[360px]">
        <PortfolioChart />
      </section>
    </main>
  );
}
```

График — `recharts` или `lightweight-charts` (последний легче и лучше для финансовых рядов). Числа в списке выравнивайте `tabular-nums`, чтобы суммы не «прыгали» при обновлении. И не забудьте режим скрытия балансов из мобилки — на вебе он тоже нужен, хотя бы как переключатель в шапке.

**[S4.14] Пользователь:**

lightweight-charts возьмём. Всё, до завтра.

## S5

**[S5.0] Пользователь:**

Привет. Готовимся к релизу. E2E-тесты у нас на Detox, CI в GitHub Actions. На Android прогон зелёный, а на iOS один тест стабильно падает — «экран отправки»: Detox не находит кнопку «Отправить». Локально у меня проходит.

**[S5.1] Ассистент:**

«Локально проходит, в CI нет» на iOS почти всегда про одно из трёх:

1. **Другой симулятор.** В CI по умолчанию может быть iPhone SE с маленьким экраном, и кнопка внизу формы просто вне видимой области. Detox `tap()` по невидимому элементу падает. Решение — прокрутить перед нажатием:

```ts
await waitFor(element(by.id('send-submit'))).toBeVisible().whileElement(by.id('send-scroll')).scroll(200, 'down');
await element(by.id('send-submit')).tap();
```

2. **Клавиатура перекрыла кнопку** после ввода суммы. Перед нажатием закройте клавиатуру: `await element(by.id('amount-input')).tapReturnKey();`.

3. **Таймаут синхронизации**: на медленной машине CI анимация или сетевой запрос не успевает. Включите логи `--loglevel trace` и посмотрите, что Detox ждёт.

Зафиксируйте симулятор в `.detoxrc.js` явно (`device: { type: 'iPhone 15' }`), чтобы локально и в CI было одно и то же.

**[S5.2] Пользователь:**

Был iPhone SE, точно. Прокрутку добавил, зелёное. Как настроить в Actions, чтобы iOS-прогон не занимал по 40 минут?

**[S5.3] Ассистент:**

Основное время съедают сборка и `pod install`. Кэшируйте всё, что можно:

```yaml
jobs:
  e2e-ios:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: pnpm }
      - run: pnpm install --frozen-lockfile
      - uses: actions/cache@v4
        with:
          path: apps/mobile/ios/Pods
          key: pods-${{ hashFiles('apps/mobile/ios/Podfile.lock') }}
      - run: cd apps/mobile/ios && pod install
      - uses: actions/cache@v4
        with:
          path: apps/mobile/ios/build
          key: ios-build-${{ hashFiles('apps/mobile/ios/**', 'apps/mobile/package.json') }}
      - run: cd apps/mobile && npx detox build -c ios.sim.release
      - run: cd apps/mobile && npx detox test -c ios.sim.release --retries 1
```

И запускайте iOS-E2E не на каждый пуш, а на PR в `main` и по ночам — macOS-раннеры дорогие и медленные. На каждый пуш хватит юнит-тестов и линтера.

**[S5.4] Пользователь:**

Так и сделаю. Теперь сторы. Интерфейс у нас на русском и английском — описание и скриншоты тоже на двух языках делать? И что с иконкой, Apple придирается?

**[S5.5] Ассистент:**

Да, раз интерфейс двуязычный, делайте две локализации карточки: русскую и английскую. Скриншоты снимайте в соответствующем языке интерфейса — сторы показывают их по языку устройства.

Скриншоты удобно генерировать тем же Detox: прогоняете сценарий с `--device-locale ru-RU` и `en-US` и снимаете `device.takeScreenshot('wallet')`. Нужны размеры для 6.7" и 6.5" iPhone и для Android — телефон и 7" планшет, если поддерживаете.

Иконка:
- iOS: один файл 1024×1024 без прозрачности и без скруглений (скругляет система). Прозрачный фон — частая причина отказа.
- Android: адаптивная иконка — отдельные foreground и background слои, иначе в некоторых лаунчерах будет белая рамка.

Для крипто-кошельков Apple обычно спрашивает в ревью, кто владеет ключами. Сразу напишите в заметках для ревьюера, что кошелёк некастодиальный, ключи хранятся только на устройстве, а обмен идёт через сторонний сервис.

**[S5.6] Пользователь:**

Про ревьюера хорошая мысль. Кстати, нас спросили на созвоне с потенциальным партнёром про аудит. Своих смарт-контрактов у нас нет, мы только вызываем чужие — 0x и токены. Что тогда вообще аудировать, если спросят ещё раз?

**[S5.7] Ассистент:**

Если своих контрактов нет, аудит контрактов в классическом смысле не к чему применять. Имеет смысл говорить об аудите **приложения**:

- **Работа с ключами:** генерация (источник энтропии), хранение в Keychain/Keystore, что сид не попадает в логи, крэш-репорты, буфер обмена, скриншоты.
- **Подписание:** что пользователь видит перед подписью (адрес получателя, сумма, контракт, на который даётся approve), защита от подмены данных транзакции.
- **Интеграция обмена:** что котировка от 0x проверяется (сеть, токены, минимальная сумма) и что approve не даётся на бесконечную сумму без явного согласия.
- **Бэкенд и веб:** SIWE (nonce, домен, срок), cookie-сессия, прокси к 0x без утечки ключа.
- **Цепочка поставки:** зависимости npm, сборка в CI, подпись релизов.

Для небольшой команды подойдёт пентест мобильного приложения от студии, которая делала это для кошельков, плюс самопроверка по OWASP MASVS.

**[S5.8] Пользователь:**

Ок, запишу MASVS. Перед релизом хочу обновить зависимости, там dependabot насыпал 30 PR. Как это разгрести без боли?

**[S5.9] Ассистент:**

30 отдельных PR — это много ручной работы. Настройте группировку в `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: npm
    directory: "/"
    schedule: { interval: weekly }
    groups:
      dev-tooling:
        patterns: ["eslint*", "@typescript-eslint/*", "prettier", "jest*"]
      react-native:
        patterns: ["react-native*", "@react-native*", "react"]
      minor-and-patch:
        update-types: ["minor", "patch"]
    ignore:
      - dependency-name: "react-native"
        update-types: ["version-update:semver-major"]
```

Порядок перед релизом:
1. Сливайте `minor-and-patch` пачкой — после прогона CI.
2. Мажорные обновления React Native — не перед релизом, а сразу после, отдельной веткой, с React Native Upgrade Helper.
3. Библиотеки, которые трогают ключи и подписи (`@scure/bip39`, `viem`, `react-native-keychain`), обновляйте только читая changelog и только по одной.

**[S5.10] Пользователь:**

Логично, крипто-либы по одной. Ещё вопрос про версию: какую ставить в сторы, 1.0.0 или 0.9?

**[S5.11] Ассистент:**

Для сторов это в основном маркетинг. Технически важнее другое:

- `versionName`/`CFBundleShortVersionString` — то, что видит пользователь (1.0.0).
- `versionCode`/`CFBundleVersion` — должен расти с каждой загрузкой в стор. Генерируйте его в CI из номера сборки, чтобы не забывать.

Если продукт уже работает и вы готовы его поддерживать — ставьте 1.0.0. «0.9» в сторе выглядит как бета и снижает доверие, а для кошелька доверие важнее всего. Для бета-тестеров есть TestFlight и закрытое тестирование в Google Play — там можно хоть 0.1.

Автоматизировать выкладку можно через fastlane: `fastlane ios beta` и `fastlane android internal` из того же GitHub Actions.

**[S5.12] Пользователь:**

Тогда 1.0.0. Отправляем в TestFlight на этой неделе. Спасибо, дальше буду спрашивать по ходу.
