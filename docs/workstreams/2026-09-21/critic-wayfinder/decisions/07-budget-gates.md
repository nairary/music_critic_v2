# D7 — Цена, минимальные опыты и условия продолжения

Статус: **план, не запуск и не подтверждённый доступный бюджет**.
Зависимости: D1–D5; [карта](../README.md). Владелец планирования: агент.
Перед каждым существенным выполнением: новый stable ID и planned record по
[experiment protocol](../../../../../.agents/skills/experiment-protocol/SKILL.md),
потом terminal closure, ledger/status. В этой карте коды E1…E7 — этапы плана,
**не зарегистрированные экспериментальные IDs**. Новые records сейчас не нужны:
нет нового результата/import/diagnostic execution, только обзор retained evidence.

## Уточнение пользователя и рекомендуемая очередь — 22 сентября

Пользователь сообщает остаток **около 7300₽** в Yandex Cloud и хочет сначала
увидеть критика на искажённых треках, а также обучить changes. Это новый
ориентир остатка со слов пользователя; Billing API, текущие начисления и квоты
не проверены. Порядок ниже предложен для выбора, платные запуски не начаты.
Он сужает прежнюю карту: probability/PLL и новое обучение backbone с нуля
отложены; исторические варианты ниже сохранены как альтернативы.

Последующее уточнение меняет **содержание первого critic**, сохраняя critic-first
порядок: пользователь хочет многомасштабную последовательную модель и четыре
аспекта. [Предложенная конструкция](../../../../research/MULTISCALE_ASPECT_CRITIC_DESIGN_20260922_RU.md)
заменяет tiny beat head как основной вариант; тот остаётся контролем.
Прежний1000₽ потолок не является оценкой стоимости этой более богатой модели.
До исполнения нужны размер TRAIN expansion, стоимость feature extraction и
native memory/throughput gate; при несовместимости с потолком пересмотреть
размер пилота/конверт, не расширять расходы автоматически.

| Очередь | Конкретный результат | Предлагаемый потолок |
|---|---|---:|
| 1. Первый critic pilot | Проверить загрузку checkpoint и multilevel export; bounded recurrent/aspect readout по D4, с raw/beat/pooling controls. Existing suite — демонстрация/feasibility; достаточный TRAIN expansion определить до запуска. Плеер показывает A/B, четыре относительные оценки, взвешенный итог и scope. Human A/B — последующая проверка. Локальные вклады не объявлять проверенной локализацией | прежние1000₽; вместимость нового дизайна не установлена |
| 2. Changes pilot | На подходящем frozen temporal backbone обучить только99-output masked changes head; короткий screen128 TRAIN/до32 VAL, затем расширять доступный TRAIN только в пределах лимита. Сравнить с TRAIN prevalence, отдельно сообщить поддержку классов и положительные наборы | 1000₽ |
| 3. Условное совместное дообучение | После успешных первых шагов и подготовки pcset v2 сравнить короткие ветви с/без changes от одного checkpoint, с одинаковыми данными, v2 targets, optimizer/draw budget и старым objective weighting. Измерять и theory retention, и сохранённый human endpoint | 3000₽ |
| Резерв | Storage, перенос, подготовка, непредвиденные расходы и следующий выбранный шаг; не автоматические retries | 2300₽ |

Сумма предложенных конвертов7300₽. Первое предложение ограничивается этапами1–2,
суммарно до2000₽ при подтверждённой вместимости нового дизайна;
этап3 не запускается автоматически. Потолки не являются прогнозами стоимости
или обещанием завершить обучение. Перед GPU execution нужны live account/source
admission, новый record и остановка по первому пределу денег/wall/updates;
число шагов этапа3 выбирается после короткого throughput benchmark.

Для первого результата предпочтителен доступный completed temporal checkpoint.
Не задерживать просмотр ради нового80k run: C2 model-only локально сохранился
и byte-verified в [D1](01-backbone.md), поэтому является резервной исходной
точкой первого demo. Его использование не означает автоматическую совместимость
с temporal changes runtime; для этапа2 отдельно связать backbone/source/targets.
Инференс сначала оценить локально, cloud использовать только при необходимости.

Реакция старых18 classifiers не является готовым score качества. Первая
диагностическая страница и weak synthetic ranker могут появиться до разметки;
согласие направления «лучше/хуже» с людьми требует listener labels. Existing20works/149variants —
development; не считать original автоматически лучшим, не переносить его
теоретические targets на изменённый MIDI. Делить пары по произведениям,
оставить целые unseen works/families для дальнейшей независимой оценки. Рендер
для MIDI-only критика не должен добавлять неслышимый моделью drum stem.

Исправленный pcset v2 уже реализован, но frozen backbone остаётся исторически
обученным. Отдельный changes-only probe не использует pcset-loss; совместный
old18+changes run требует нового v2 data/support/weights/augmentation/runtime
протокола. Обе сравниваемые ветви получают одинаковое исправление, чтобы его
эффект не приписывать changes. Старый3090 scratch80k план не переносится в Cloud.

Публичный прайс проверен заново22сентября: T4+16vCPU+64GiB стоит
`70.27 + 16*1.15 + 64*0.3074 = 108.3436₽/h` compute с НДС по
[официальной таблице](https://github.com/yandex-cloud/docs/blob/master/md-docs/compute/pricing.md).
7300₽ соответствуют примерно67.4compute h до других расходов, а не готовому
бюджету67.4часов обучения. По [правилам](https://yandex.cloud/ru/docs/compute/pricing)
диски/хранение, сеть и публичный IP оплачиваются отдельно. Рекомендация использует
обычную T4; spot/resume — отдельный выбор.

## Что проверено о стоимости

На 21 сентября официальный published source
[Yandex Cloud pricing](https://github.com/yandex-cloud/docs/blob/master/md-docs/compute/pricing.md)
указывает обычную T4:70.27₽/GPU·h,1.15₽/vCPU·h,0.3074₽/GiB·h с НДС.
Расчёт для 1 T4+16vCPU+64GiB: **108.3436₽/h**. Это оценка compute по публичному
прайсу, не quote аккаунта и не сохранённый расход. Перед заказом сверить SKU/zone
и актуальные цены. [Правила тарификации](https://yandex.cloud/ru/docs/compute/pricing):
диски, storage, egress и IP учитываются отдельно; остановленные диски тарифицируются.

Формула лимита: `cost_cap = hours_cap × current_compute_rate + storage/network reserve`.
`usable_balance = verified remaining funds - committed other work - reserve`.
18000₽/месяц — пользовательский ориентир, **не измеренный остаток**. Даже при
полном бюджете это около 166compute h до хранения/прочих расходов; не план
заполнить их обучением. Рекомендуемый первый конверт: до 24 T4h (~2600₽compute)
+до 500₽резерва, только если подтверждён доступный остаток и отдельное согласование.

Callable account toolkit не найден; `yc` не найден в PATH и обычных локальных
roots. Доступна документация Cloud, но она не показывает account balance/quotas.
Текущий остаток, GPU quota/free capacity, VM state, сохранность bucket/disk и
права скачивания остаются unknown. Старые STOPPED snapshots не являются live
проверкой. Для планирования этого достаточно; native work до закрытия gate нельзя.
Не запрашивались новые credentials, не запускались VM, не менялись роли.

## Исторические измерения — только ориентир

[Pipeline8](../../../../experiments/CLOUD_CPU_PIPELINE_RESULT_RU.md):1.379→0.819 с/update
на коротком temporal B2 участке; [B4](../../../../experiments/CLOUD_BATCH4_RESULT_RU.md):
+26.63%draw throughput; [prepared_v1](../../../../experiments/CLOUD_T4_PREPARED_BENCHMARK_RU.md):
1.807×slowdown. Новый masked decoder означает несколько masked forwards и
новые memory/IPC costs. Не умножать прежнюю скорость на 80k и выдавать это за
гарантию. Измерять cold/warm wall, valid blocks/s, draws, encoder/decoder forwards,
mask closure size, p95 latency, GPU/RAM peaks, transfer/storage overhead.

RTX3090 недоступна по текущему указанию пользователя. Исторические 3090
авторизации и protocols не разрешают запуск в Cloud. Spot/preemptible вариант
сначала требует resume/admission и явного выбора; дешёвая цена не равна
дешёвому завершённому результату.

## Историческая последовательность с минимальным бюджетом

Таблица E0–E7 ниже описывает прежний small-probe/probability дизайн. Она остаётся
оценкой альтернатив, **не протоколом нового multiscale critic**. Его актуальный
critic-first порядок и неопределённая вместимость отражены в начале документа;
в частности, E4 human labels не prerequisite его weak-supervision пилота.

Размеры — нижняя граница **feasibility** и твёрдый stop cap, не обещание
статистической силы. Cap применяется к первому достигнутому пределу
(updates, wall, деньги). Если data admission не уложился, завершить failed/aborted,
сохранить причину и не расширять бюджет автоматически. TEST всегда закрыт.

| Этап | Зависимости | Минимальный полезный дизайн / cap | Продолжать, если | STOP / результат при провале |
|---|---|---|---|---|
| E0: contract | D3; D2 для probability | Synthetic pair-swap/group/mask fixtures; normalization и hidden-content tests только для probability; 0 cloud ₽; ≤1 local CPU ч | Проверены контракты выбранного пути | Ошибка блокирует затронутый путь |
| E1: native admission | E0, D1, live cloud gate | E1A frozen readout; E1B masked decoder отдельно. До 16 TRAIN works, суммарно ≤32 updates + resume/all-shift fixture, общий ≤0.5 T4 h ≈54 ₽ | Memory reserve, replay/finite gradients, source/schema bindings выбранного пути | OOM, drift, source mismatch блокируют соответствующий путь; другой не наследует допуск |
| E2: probability feasibility | E1B | ≤128 TRAIN/32 VALIDATION records с group closure; prior, frozen Bernoulli и small AR; ≤1000 updates/learned arm, общий ≤4 T4 h ≈433 ₽ | Held-out NLL/calibration лучше proper prior; AR добавляет пользу при приемлемой цене | Нет gain → закрыть текущий decoder; это не блокирует frozen preference |
| E3: context/fragility diagnostics | E1A для embeddings; E2 только для PLL | ≤32 VAL, fixed ≤32 blocks/work, 4 context views для probability; subset ≤48 dev pairs; общий ≤1.5 T4 h ≈163 ₽ | Контекст помогает выбранному signal; neutral/changed cases различимы без density shortcut | Нет conditional evidence → ограничить claim; support mismatch → unavailable |
| E4: human rubric | D3, D5; параллельно E0–E3 | 48 pairs/12dev works,≥2 raters,6repeats;2–4 ч/слушатель;0 GPU | Aspect понятен, доступны useful ties/decisive labels | Не учить preference из неподтверждённого ordering |
| E5: frozen critic controls | E1A, E4; независимо от E2 | 8 train/4 dev-val groups; C2, C3, raw summaries, same small head; ≤1000 updates/arm, ≤3 T4 h ≈325 ₽ всего incl embeddings | Signal лучше cheap control на dev, не только operator/genre | Нет gain → сохранить дешёвый baseline, final comparison всё равно frozen |
| E6: один optional escalation | E2/E5+review | Выбрать **либо** partial-finetune vs frozen (≤2000 updates/arm), **либо** regularized signal combination; общий≤4 T4h≈433₽ | Заранее выбранный dev criterion/retention выполнен | Не запускать обе ветви без нового решения; нет gain → отказаться от усложнения |
| E7: bounded human evaluation | A: E4+E5; B: E4+E2+E3; combination: готовность выбранных signals + E6 | Frozen ≥24 new works×4 pairs; ≥2 raters; fixed score schedule; ≤3 T4 h ≈325 ₽ scoring, 3–5 ч/слушатель | Предзаданный human gate из D5 + localization/benign controls | CI включает 0 → неопределённо; не выбирать новые weights по holdout |

E1A/E1B оформляются разными admission records, если выбраны обе ветки; общий
0.5 h cap не гарантирует, что обе поместятся. Сначала оценить размер и выбрать
одну или согласовать иной конверт. Ни E0 probability tests, ни E2 success не
являются обязательным blocker для самостоятельного frozen preference пути.

Основной compute cap E1–E7 с optional E6 —16 T4h (~1733.50₽), до 24h envelope
оставляет запас 8h только для **заранее выделенной** подготовки/экспорта и
scoring overhead. Это не бюджет на автоматические retries. Если estimated
cost с overhead не помещается, уменьшить число masks/decoder или выбрать
одну ветку до старта. Human labor не входит в Cloud grant.

Одинаковые updates сравнивать только вместе с draws/LR/optimizer/forward budget.
E2 сравнивает декодеры при одинаковом frozen encoder и масках; E5 сравнивает
готовые representations, не matched pretraining. После дешёвого screen второй
seed допускается только новым протоколом/конвертом; один seed — pilot.

## Что должно остаться после каждого опыта

Перед запуском: bound source/config/masks/split/checkpoint, sample schedule,
seed/RNG, target semantic ID, objectives, final stopping/selection rule,
целевой cloud cost и observed-access logging. После: immutable artifacts,
hashes/size/portable locator, actual applied/skipped updates/draws/forwards,
wall/cost, VALIDATION/TEST observations отдельно от policy, terminal result.

Если дешёвые сигналы не согласуются с людьми — это полезный отрицательный
результат. Не переходить автоматически к большему GNN/JEPA/PDMX. Сначала
выяснить, различает ли измерение нужный аспект и достаточно ли независимых works.
Дополнительные направления и отдельные caps находятся в [D6](06-expansion.md).
