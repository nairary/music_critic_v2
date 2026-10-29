# Доразметка HIERGRAPH — 28 октября 2026

## Цель и границы

По запросу пользователя текущий ресурс направлен на доразметку. Работа идёт
только в isolated harmonic-event-pilot checkout. Исходный release, старые
frozen попытки и terminal records не перезаписываются. Используются существующие
TRAIN/fit произведения: current67 и split44fit/10selection/13dev сохранены;
TEST закрыт. Растёт глубина разметки, число независимых произведений не выросло.

## Принятые результаты

| Произведение | Принятая работа | Независимая приёмка |
|---|---|---|
| WORK047 Scarlatti K.1 | 32 исправления прежних полей, 36 известных полей 12 новых объектов, 42 связи | REVIEW001 accepted |
| WORK052 Festive Minor | 154 unknown→known прежних полей, 44 известных поля 13 объектов, 34 связи; затем два уточнения области действия | REVIEW001 changes_requested сохранён; REVIEW002 accepted |
| WORK062 ELO Don’t Bring Me Down | 127 прежних полей, 18 известных полей 6 новых объектов, 41 связь | REVIEW001 accepted, ограниченный объём |
| WORK051 丁当 — 手掌心 | Принят старый author002: 162 изменённых field envelopes, из них 114 semantic changes; 12 связей | Новый REVIEW001 accepted; авторство Sol/high сохранено |
| WORK049 T.R.Y — 不是因为寂寞才想你 | Исправлены 6 объяснений интервала; ранее выполненные 5 604 baseline changes доведены до приёмки | REVIEW001 accepted; новых музыкальных gains в текущем исправлении 0 |
| WORK056 Gala — 追梦赤子心 | Исправлены 20 ошибок dimension/scale/direction/accent; прежние 26 gains, 6 value corrections и 8 связей сохранены | REVIEW001 changes_requested и старый отрицательный отзыв сохранены; REVIEW002 accepted |
| WORK058 Mendelssohn Op.12:I | Проверен и принят импорт существующего независимого bounded weak review; 14 дельт воспроизведены в обе стороны | AUDIT001 accepted; историческая неизменность ранее unsealed R2 не подтверждена |
| WORK063 SHE — 他还是不懂 | Сняты 4 неподтверждённых maintains в masked unresolved; остальные 909 исторических изменений сохранены | REVIEW001 accepted; свежие gains 0 |
| WORK037 乌兰托娅 — 高原蓝 | Закрыты три boundary-scope замечания; 27 043 historical gains и 415 withdrawals относительно прежней базы приняты с сохранением R1 carry | REVIEW001 accepted, fresh bounded R2 |
| WORK054 Fats Waller — Honeysuckle Rose | Принят старый author004 после семи исправлений; исторические 113 known gains, 12 refinements и 26 edges сохранены | REVIEW001 accepted; историческое авторство сохранено |

| WORK064 POP909556 | Удалены901 ошибочные not_applicable receipts, музыкальные значения сохранены; принята историческая дельта9 gains/11 withdrawals/10 changes | REVIEW002 accepted; failed author001 сохранён |
| WORK061 Beyond the Sea | Два принятых авторских участка:133+226 новых доступных полей,10 процессов,52 связи; первый негативный отзыв исправлен | REVIEW002 и REVIEW003 accepted; q0..96 с адресными boundary witnesses |
| WORK043 Schulhoff Stomp | 183 новых доступных поля,7 изменений прежних значений,4 объекта,14 связей; сортировка transforms и описание четырёх лиг исправлены | REVIEW001 changes_requested, REVIEW002 accepted |
| WORK032 Steely Dan Peg | 56+22 новых поля,16+4 связи; четыре E7→Am и отдельный вступительный участок | REVIEW001 и REVIEW002 accepted |
| WORK050 Clara Schumann Op23No2 | Исторический accepted import и четыре новые метки/одна связь: A4→G4 в кадансовом обороте | AUDIT001 и REVIEW001 accepted; фактический backend root неизвестен |

Это разные единицы учёта: изменение field envelope не обязательно добавляет
новый таргет или новое музыкальное суждение. Исправления и снятие ошибочных
меток сохраняют ценность, хотя не увеличивают число доступных targets.

WORK047 и WORK052 имеют full_work_review_complete=true в рамках протокола.
Это не плотная разметка, human gold или доказательство исчерпания возможностей.
У остальных указанных работ полнота не заявляется; strict_complete=false у всех.

## Собранные версии native датасета

`EXP-HIERGRAPH-ANNOTATION-RELEASE-001` completed, immutable evidence:
`outputs/hiergraph_annotation_20261028/release001`.
В неё включены первые пять принятых работ: WORK047/052/062/051/049.
Release001 сохранена неизменной. Более полная release002 описана ниже.

- Exact current67 baseline и canonical identities подтверждены; 62 работы не изменены.
- Доступные native поля этих пяти работ: 53 289 → 58 124, net +4 835.
- 4 577 unknown→known, 305 известных полей новых объектов, 47 снятых меток.
- 31 новый объект, 244 новые связи; включает прежнюю выполненную, но не принятую работу.
- 132 799 свежих numeric assertions прошли; OOV 0; 74 570 полей остались masked.
- Native per-work event/object vocabularies и числовые targets сохранены.
- Общая 97-полевая проекция HIERGRAPH V3 пока не пересоздавалась.

Net +4 835 — механическая разница с точным родительским native baseline,
а не число новых независимых наблюдений или меток, написанных Astra сегодня.

`EXP-HIERGRAPH-ANNOTATION-RELEASE-002` completed:
`outputs/hiergraph_annotation_20261028/release002` включает все десять принятых
работ из таблицы. 57 current67 works остаются без изменений; split сохранён.

| Показатель release002 | Значение |
|---|---:|
| Native available fields, baseline → current | 161 783 → 193 561 |
| Net available difference | +31 778 |
| unknown→known / inserted known / withdrawn | 32 086 / 305 / 613 |
| Изменённые доступные JSON values | 1 191 |
| Новые объекты / связи | 31 / 407 |
| Masked fields | 282 238 |
| Native numeric assertions / OOV | 476 237 / 0 |
| Canonical notes в выбранных works | 23 496 |
| Full-work review complete / strict complete | 2 / 0 |
| Wall, native integration | 27.124 s |

Эти разницы включают поздно интегрированные исторические решения. Метаданные
и evidence-path изменения посчитаны отдельно; это не дополнительная музыкальная
супервизия. Полная 97-полевая HIERGRAPH V3 проекция пока не пересоздана.

## Итоговая native release003

`EXP-HIERGRAPH-ANNOTATION-RELEASE-003` completed, immutable:
`outputs/hiergraph_annotation_20261028/release003`.
Включены последние принятые версии15 произведений;52 из current67 без изменений.
Новые независимые произведения не добавлялись, split44/10/13 сохранён.

| Показатель release003 | Значение |
|---|---:|
| Available native fields, exact parent → accepted | 309 503 → 323 062 |
| Net available difference | +13 559 |
| unknown→known / inserted known / withdrawn | 32 656 / 345 / 19 442 |
| Changed available values | 1 413 |
| Added objects / relations | 45 / 547 |
| Masked fields | 410 499 |
| Native numeric assertions / OOV | 734 145 / 0 |
| Canonical notes | 36 157 |
| Full-work review complete / strict complete | 2 / 0 |
| Native integration wall | 39.755s |

Это механическая разница с точным current67 parent, включающая исторические
аннотации, теперь доведённые до приёмки. Она не равна числу сегодняшних новых
музыкальных решений. В последних четырёх потоках WORK061/043/032/050 отдельно
приняты624 новых доступных поля и87 связей относительно их принятых входных
версий; повторяющиеся фигуры и несколько полей одного события не становятся
независимыми наблюдениями.

Важное различие баз: у Peg current67 parent содержал52 726 доступных полей,
а принятая старая author005 версия, с которой началась новая разметка,33 904.
Свежие два прохода добавили78 и довели её до33 982; относительно старого
current67 это всё ещё−18 744 доступных поля. Поэтому общий net release003
меньше net release002, хотя новые решения добавлены.18 744 из19 442 переводов
known→masked приходятся на эту замену исторической версии Peg. Это не ошибка
суммирования и не измерение качества модели; точные старые и новые envelopes
сохранены в сравнениях, неизвестное не превращено в отрицательные targets.

Native per-work event/object vocabulary и targets сохранены с точными хешами.
Общая97-полевая HIERGRAPH V3 проекция ещё не пересоздана: старые training tensors
автоматически не получают эти метки. Следующий инженерный шаг — применить
accepted overlay при новом зарегистрированном экспорте с прежним split и masks.

## Закрытие и оставшаяся очередь

Текущие author/review попытки завершены; отрицательные отзывы, ошибки проверок
и исправления сохранены. WORK061 прежние mixed/initial-versus-maintains findings
закрыты в R2, следующий участок принят в R3. WORK064 accounting исправлен в
новой версии без изменения музыкальных labels. WORK043 R1 исправлен в R2.
WORK050 check001 отклонил null requested-model provenance; check002 явно
разделяет желаемый пользователем Astra/ultra профиль и неизвестный root backend.
Это не подтверждение, что root фактически выполнялся на Astra.

Metadata-only FITQUEUE-001 проверил32 оставшихся на момент инвентаризации fit
works:16 возможных импортов существующих отзывов,14 работ для новой разметки,
две unresolved metadata routes WORK027/042. Три из кандидатов этой очереди
WORK043/032/050 затем отдельно допущены, проверены и расширены. Приёмка не
выводилась из одного metadata status. Ранее unsealed файлы имеют observed
snapshot pins; историческая неизменность не выдумывается.

## Проверки и ограничения

Reader gate: 8/8 synthetic tests. Root проверяет frozen bytes, basis3 и native
review receipts перед закрытием каждого record. Независимые source/numeric
проверки: WORK047 77 736 + 5 648; WORK052 R2 20 846 + 557 slots;
WORK062 61 173 event + 409 object + 894 endpoint checks;
WORK051 67 784 numeric + 16 637 note checks;
WORK049 50 283 event + 948 object + 9 915 participant + 446 relation checks;
WORK056 116 225 свежих assertions;
WORK058 667 401 свежих assertions;
WORK063 295 599; WORK037 1 295 672; WORK054 346 024 свежих assertions. WORK064 R2 —635 534; WORK061 R3 —334 540; WORK043 R2 —72 479; WORK032 R1 —279 595,R2 —273 707; WORK050 R1 —88 569. Перенесённые исторические проверки
учитываются отдельно. Числовая корректность не доказывает музыкальное качество.

Сохранены выявленные ошибки: WORK052 source-span findings и их исправление;
WORK056 metadata findings; WORK063 неподтверждённые maintains; технические
handoff/checker failures и новые успешные попытки. WORK052 содержит неблокирующую
унаследованную неточность фразы вне исправляемого target scope, отмеченную R2.

Метки остаются weak source/model labels. Для ряда работ source score и
materialized MIDI подтверждены, original MIDI unbound. Для POP909 точность
нотной геометрии не означает акустическую эквивалентность: CC64/pedal и конец
файла могут отличаться. Raw pretraining overlap неизвестен.

GPU/training/LLM API/new paid resources:0. Legacy не использовался. Commit/push
не выполнялись. Объём наблюдений quota относится ко всему аккаунту, не только
к этой задаче: в06:35UTC было78%used, около06:45UTC уже4%used и более поздний
reset epoch. Точный момент rollover и расход этой задачи неизвестны. После
наблюдения rollover новые авторские партии не открывались; завершались только
текущие независимые проверки и сборка. Источник: ignored
`resource-rollover-observation.json`; обновление в07:00UTC заранее не предполагалось
подтверждённым. Новые лимиты не используются для продолжения бесконечной очереди.
