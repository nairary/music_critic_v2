# Harmonic event graph: данные, задачи и оценка

25 октября 2026. T08, proposed research specification.
Статус: согласуется постановка четырёх модулей; implementation-ready интерфейс
NBF/refiner↔Query2box, ontology mapping и compute budget ещё не определены.
Никакое обучение, data migration или новый доступ к TEST этим документом
не разрешаются. Default architecture и terminal records не меняются.

Последующий [review перед реализацией](review.md) уточняет условия старта:
same-source negative supervision, отдельные NBF/box пути, bounded H proposals
и контролируемое различение conditioning/propagation. Исходная таблица этапов
ниже остаётся постановкой; исполняемый protocol должен включить эти уточнения.

Ближайшая предложенная кампания конкретизирована в [pilot plan](pilot-plan.md):
oracle H→note membership, frozen C3, четыре readout/refiner условия и максимум
три paired seeds. Это выбор первого диагностического вопроса; raw H discovery,
box queries, encoder adaptation и группы остаются отдельными следующими опытами.

## Основания и проверенные данные

Проблема: существующий relation pilot читает статические note/slice states
через MLP/TransE. Новое направление сохраняет C3, добавляет HarmonyEvent
proposals, query-conditioned reasoning и box representation составных запросов.
Связанные [domain/design note](../../../research/HARMONIC_EVENT_GRAPH_DESIGN_20261025_RU.md)
и [research](../../../research/QUERY_CONDITIONED_HARMONIC_GRAPHS_20261025_RU.md).
Исходный source revision: `287bf856643f9e555352e1661a7bf0563772178a`, dirty checkout.

Источник — [RELEASE003](../../../experiments/records/EXP-CORPUS400-COMPLETION-015-RELEASE003.json):
67current-v2 works с исходной и LLM-обогащённой разметкой; historical-v1 не
добавляется. Сохранённый current67-field-coverage.json проверен по binding
в terminal record:6096bytes,
SHA256 `6e474d46c7fcf9157acaf144e7ec7a4ef75297e87e03d0477e156eacdee4c95c`.
Новый corpus census не выполнялся. Ниже опубликованные known-field slots по67,
а не новые метрики, число независимых событий или человеческая accuracy.

| Поле/состояние | Опубликованная доступность | Для чего применимо |
|---|---:|---|
| harmony/analysis.sonority known | 29339 | Структура/тип после TRAIN-only нормализации |
| harmony/analysis.key known | 43647 | Отдельный тональный target, не raw input |
| harmony/analysis.function known | 11809 из45263slots | Контекстная функция, классы/совместимость ещё проверить |
| harmony/analysis.roman known | 16745 | Отдельный структурированный target; не автоматически T/S/D |
| notes/analysis.chord_membership known | 146785 | Core membership note-context→H |
| notes/analysis.chord_degree known | 142038 | Роль/ступень там, где определена |
| resolution linked/none/masked | 10671/21497/242768 contexts | Отдельные directed targets и source-scoped known absence |
| preparation linked/none/masked | 7730/23115/244091 contexts | Дополнительное семейство после семантического gate |

Known counts включают разные аналитические views; исходные labels, источник,
confidence и masks сохраняются. Они не доказывают достаточность каждого класса
по независимым произведениям или наличие joint-query/hierarchy-depth targets.
Для функции всего26.09%слотов имеют known value; unreviewed/unresolved не negatives.

Текущий сопоставимый pilot scope —64runtime works,52fit/12dev, после исключений
геометрии/track limit. Данные и split закреплены
[PREPARE006](../../../experiments/records/EXP-LLM-RELATION-PREPARE-006.json) и
[REVIEW001](../../../experiments/records/EXP-LLM-RELATION-REVIEW-001.json).
Это benchmark предыдущего decoder, не intrinsic ограничение нового датасета.
Для первого controlled comparison сохраняется его admission/split; возвращение
исключённых works или расширение — отдельная версия и паспорт.

Предыдущий prepared006 export содержит sanitized raw и узкие resolution
targets. Для нового пути требуется отдельный authenticated export H/fields/
memberships из исходных bundles; прежний файл не является готовым датасетом
всех четырёх модулей. До payload reads нужен новый planned preparation record
и exact flat admission по действующему протоколу.

## Один обучающий пример

Единица верхнего уровня — произведение/его версия и analytical view;
батч может содержать ограниченный фрагмент. Рабочий интерфейс loader должен
разделить:

```text
raw: notes, exact time, tracks, raw temporal/containment graph
targets.H: instance IDs, intervals, existence/coverage, field values+masks
targets.membership: note-context IDs, H IDs, context/core role, mask
targets.relations: typed source/target participants, time/view, mask
targets.queries: anchors, operators, typed answer IDs, known pos/neg/unknown
provenance: source and release bindings, origin, confidence, analytic view
```

Это candidate interface, не новая схема, уже реализованная в коде.
Scope уточнён в [domain note](../../../research/HARMONIC_EVENT_GRAPH_DESIGN_20261025_RU.md):
participant_scope (одна наблюдаемая дорожка, группа, ансамбль) отделяется от
конкретных members и от context_scope, доступного для интерпретации. Группы
могут пересекаться и меняться во времени; raw track не равен voice/part.
Первый H путь предлагается на ensemble scope, если он соответствует source
targets; local/group H supervision требует собственной поддержки. Ensemble
labels не распространяются автоматически на все уровни. Learned group discovery
и отдельная музыкальная редукция остаются следующими гипотезами.
Кандидатные H и их features строятся из raw evidence на inference. Gold H,
function/key/voice labels и target edges не становятся encoder inputs.
Числовой vocabulary/mapping строится по fit-данным; OOV в dev учитывается
отдельно, не заменяется известным отрицанием или новым классом из dev.

## Задачи и метрики модулей

| Задача | Target и ответ | Основная оценка |
|---|---|---|
| H discovery | Существование HarmonyEvent и его интервал | Event precision/recall/F1 по1:1interval matching; boundary F1; unmatched/coverage |
| Context membership | Какая note-context звучит в контексте H | Masked incidence P/R/F1 и coverage |
| Core membership/role | Chord-member и доступная роль относительно H | P/R/F1 по классам и works; обратный H→notes set score |
| H attributes | Sonority/root/quality/inversion, key и function в их scope | MacroF1 и per-class P/R, duration-weighted known-label accuracy; OOV/coverage |
| Atomic relations | `resolves_to`, `chord_member`; затем supported families | Query-macro Recall@1/5, Hits@1/5; existence P/R/F1 и Brier |
| Composite queries | Один/несколько/нет ответов на typed AND/path query | Multi-answer Recall@k; set P/R/F1/exact match на полностью reviewed queries; empty-answer FPR |
| Full predicted graph | Узлы и typed edges после raw-only pipeline | Matched typed-edge F1 на reviewed scopes, end-to-end answer recall |

### H proposals и attributes

Для event matching предлагается label-agnostic temporal IoU с1:1assignment;
primary threshold0.5 и stricter0.75 — предварительные значения, которые
фиксируются до training/evaluation. Границы остаются exact rational;
boundary tolerance в quarter-note units нужно также заранее связать с protocol.
Нельзя подбирать matching rule после просмотра результатов.

Event precision/F1 и penalty за лишний slot допустимы на scopes с полным
набором целевых H для выбранного view. На частичной разметке unmatched
prediction не становится background. Там показываются supported recall и
coverage; полного precision ground truth нет.

`sounds_in_context` не равно `chord_member`. Роли удержанных нот меняются по
интервалам. Обратный H→notes — retrieval существующих участников, не генерация
всей фактуры. Exact membership-set score допустим лишь при полностью reviewed
наборе; иначе считаются masked positive/negative решения.

Структура аккорда и функция раздельны. T/S/D не назначаются автоматически по
root, roman или pitch template. Нужен совместимый TRAIN-only словарь и явно
одобренный mapping, если данные требуют преобразования. Unknown function
не объединяется с классом «не доминанта» или silence.

### Atomic reasoner

Стартовые семейства: прежний note-context `resolves_to` и core-membership
note-context→H. `prepares` добавляется после проверки scope/mechanism.
Harmony→harmony resolution не выводится автоматически из D→T или одной
нотной стрелки; ему нужны отдельные поддержанные targets.

Ranking — средняя доля известных gold answers в первых k на запрос, допускает
несколько правильных целей. Source-query существование/пустота оценивается
отдельно, чтобы always-linked стратегия не скрывалась ranking-метрикой.
Кандидатная recall до predictor обязательна; отсутствующие правильные H/notes
не исключаются молча из end-to-end знаменателя.

Вместе публикуются global metrics, per-class precision/recall, распределение
по works и constant-linked/constant-none controls. Work macroF1 на поддержанных
классах не является единственным критерием. Threshold выбирается без доступа
к финальному оценочному holdout; Brier применяется на известных binary targets.

### Query2box и составные запросы

Первый тип запроса:

```text
q(H) = chord_member(n1,H) AND chord_member(n2,H)
```

При поддержанном контекстном target функции:

```text
q(H) = chord_member(n1,H) AND function(H, tonal_context_K)=D
```

Примеры условные. Query compilation строится из совместимых known facts
одного view/interval; оно не превращает labels в observed inference graph.
Anchor IDs и запрошенный класс допустимы как часть вопроса, target edges — нет.
Запросы генерируются только после family split, с ограничением числа на work:
комбинаторное размножение внутри пьесы не создаёт независимых произведений.

Для каждого кандидата query target имеет состояние true/false/unknown.
AND-positive требует обоих известных positive; один reviewed-false конъюнкт
даёт false в том же scope. Остальные случаи unknown. Для отрицания existential
или empty query нужно исключить все допустимые witnesses: отсутствие записей
или низкая score этого не доказывают.

На полностью проверенном query subset доступны precision/set-F1/exact-set
accuracy; на остальных — known-answer recall, known-negative FPR и coverage.
Нельзя считать все неуказанные ответы ложными. Простое пересечение сравнивается
с min/product atomic scores; learned box не получает победу только за наличие
оператора intersection. Logic-consistency checks необходимы, но не доказывают
музыкальную правильность. Shared-witness DAG queries — отдельная следующая
задача с явной идентичностью промежуточного участника.

## Раздельная и сквозная оценка

Две обязательные дорожки:

1. **Oracle-object diagnostic.** Истинные H intervals/IDs явно предоставлены
   reasoner для локализации ошибки. Не выдаётся за raw-MIDI inference.
2. **End-to-end.** Предсказаны H, memberships, labels и затем relation/query
   answers. H matching выполняется по заранее определённому правилу, missed
   object/answer считается пропуском; oracle node/label не подставляется.

Нужен временной и object-level coverage отчёт: сколько source contexts,
H и true endpoints вообще мог представить candidate generator. Отдельно
считать ошибку boundaries, type/function, membership и reasoning; один общий
score не позволяет выбрать исправление.

Основной вопрос первого этапа — качество конкретных узлов/отношений/ответов.
Общее гармоническое понимание, человеческая accuracy, preference и эстетика
не измеряются этими proxies. Небольшой независимо проверенный query subset
предлагается для оценки precision и пустых ответов, но его ещё нет как
подтверждённого human-gold benchmark.

## План обучения и сравнения

| Этап | Что обучается/проверяется | Gate следующего этапа |
|---|---|---|
| 0. Task passport | Только data admission, label normalization, masks/support по works | Поддержка targets, candidate coverage, certified negative scopes |
| 1. H/object path | Frozen C3 + H proposals/memberships/attribute heads | Конечные gradients, detector и явные known-negative controls |
| 2. Atomic reasoning | Статический decoder против query-conditioned refiner | Условные и end-to-end метрики, параметрический/marker controls |
| 3. Query operators | Fixed atomic min/product против box query head | Joint labels, multi-answer/empty query metrics, held-out combinations |
| 4. Adaptation | Отдельный full-finetune C3 + новые модули | Matched protocol, paired seeds и учёт compute |

C3 не получает отдельной новой задачи в первом frozen этапе. Следующие losses
передаются в общий encoder только в явно выбранном joint-finetune условии.
Веса losses, class/work sampling и matching targets фиксируются до запуска;
их нельзя считать решёнными простым суммированием всех имеющихся labels.

Для первого сравнения один C3, одинаковые H proposals/candidates, fit/dev,
trainable-budget controls, seeds и mask policy. Изолируются изменения readout,
query operator и C3 adaptation, а не всё одновременно. Добавляются raw/static
heads, constant decisions и простые score-combination baselines. Наличие
модуля Query2box/NBF само по себе не является критерием успеха.

Разбиение по work/source-family, без смешивания фрагментов одного произведения.
Для сопоставимости можно начать со старых52/12, но12dev уже использовались
в прежнем пилоте и не являются новым untouched test. Selection/tuning должны
иметь отдельный внутренний fit split. Дополнительные family folds/held-out
query combinations требуют predeclared protocol. Corpus TEST остаётся закрыт;
raw C3 pretraining overlap неизвестен, held-out labels не доказывают unseen raw.

## Производительность, иерархия и приёмка

Для3090 и любого другого runtime отдельно считать end-to-end и cached-C3
query latency, peak VRAM, queries/batch, nodes/edges/candidates, preprocessing
и wall time. Ни ETA, ни достаточная память нового пути пока не измерены.

Настоящая редукционная иерархия и parent-child targets пока не доказаны
известными function/member counts. При их появлении отдельные задачи:
parent/ancestor edges, сохранение событий по depth и согласованность уровней;
оценка typed edge F1 и нарушений конкретного заданного порядка. Существующая
encoder hierarchy не считается уже решённой задачей музыкальной редукции.

Наблюдаемая приёмка реализации: raw-only inference; exact identity/time/view;
unknown не даёт negative gradient; unmatched unknown H не штрафуется как фон;
несколько/нет ответов; oracle и predicted-object режимы не смешиваются;
candidate failure отражён в метриках. Synthetic tests проверяют механику этих
контрактов, native admission — работоспособность/ресурсы, scientific evaluation —
качество. Одно не заменяет другое.

Изменяемые на этой стадии файлы — данный spec и STATUS. Code/data/checkpoints,
legacy и terminal records не менялись. Новых experiment IDs для execution,
compute budget и запусков нет. Перед следующим существенным admission/run
создаётся новый planned record по
[EXPERIMENT_LOGGING](../../../EXPERIMENT_LOGGING.md).
