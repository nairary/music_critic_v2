# Ближайший опыт: H → аккордовые участники, frozen C3

25 октября 2026. T08. Предлагаемый ограниченный protocol для реализации;
на этой стадии выполнено только планирование. Data admission, числовой config
после engineering gate и новый бюджет исполнения ещё не зарегистрированы.
Основания: [spec](spec.md), [review](review.md),
[scope H](../../../research/HARMONIC_EVENT_GRAPH_DESIGN_20261025_RU.md).

## Вопрос и выбор первой задачи

При одинаковых C3 features, известных границах H и общем наборе кандидатов
помогает ли query-conditioned propagation по графу отличать аккордовых
участников H от явно размеченных неаккордовых нот того же события?

Выбран конкретный H/member опыт после обсуждения oracle H и scope ансамбля.
Предложенный в review note-resolution опыт остаётся отдельной возможностью,
но первым не запускается: его текущая negative policy не обеспечивает
различение правильных и неверных целей одного linked source.

В [существующем validator](../../../../src/music_critic/tasks/whole_work_annotation.py)
`analysis.chord_membership` имеет boolean value, а note-context может ссылаться
на `harmony_id`. Это подтверждает форму нужной supervision, но не число
поддержанных true/false examples внутри одного H. Counts по 67 работам из
spec не являются готовой H/member выборкой. Primary corpus payloads здесь
не читались, наличие пригодного subset ещё требуется проверить.

## Данные и обязательный паспорт

Планируемый источник — published current-v2 67; начальная сравнимая область —
прежние 64 runtime works / 52 fit / 12 dev. Новые exclusions публикуются;
возврат исключённых работ получает отдельную версию данных. Все методы
обучаются заново на новом membership export. Старые resolves_to метрики
не являются baseline для этой новой задачи.

До authenticated H/member export создать новый planned preparation record.
Паспорт связывает source hashes, exact intervals, view, annotation origin,
scope, confidence/masks, known membership true/false и unresolved состояния.
Отдельно считать на каждом split: works/families, H, positive/negative contexts,
H с обоими классами, полностью reviewed H, конфликты views, scope/time exclusions.
H/member target принимается только при явной связи note-context→H в том же
интервале/view; guessed labels и автоматические pitch-template negatives запрещены.

Для primary metric нужны H с ≥1 known member и ≥1 known nonmember. Нельзя
обучать или проверять сложную модель только на нескольких H одной семьи и
выдавать это за перенос: passport должен обосновать independent-family support.
При недостаточной поддержке итог подготовки — blocked scientific question
и конкретный план дополнительной review-разметки, без автоматического LLM API.
Не заменять отсутствующую поддержку переименованием unknown в false.

Family split сохраняется. Для selection/threshold выделить детерминированные
20% fit families, не подбирая split по качеству модели; принадлежность запечатать.
12 dev уже использовались ранее и остаются exploratory holdout. Corpus TEST
закрыт; C3 raw-pretraining overlap остаётся unknown. Native и LLM origins
показываются отдельно; основной результат означает agreement с weak targets.

## Первый режим: oracle-object diagnostic

Вход: raw notes/tracks/timing и C3 graph features всех поддержанных дорожек
в одном фиксированном окне; дополнительно явно даны интервалы и допустимая
область H. Это oracle geometry, не raw-only discovery готового анализатора.
Source scope должен соответствовать разметке; ensemble не назначается всем
source H автоматически. Число H и границы пока не предсказываются.

Для каждого H строится raw geometric incidence со всеми подходящими нотами
его области, включая unknown/nonmember. Этот edge не означает chord membership.
H feature агрегируется по всем этим нотам, не по gold chord members. Форма
агрегации фиксирована: mean C3 note/slice features с raw geometry и общей
по архитектуре type projection. C3 frozen и кешируется одинаково для всех.

Кандидат ответа — note-context в H. Identity/time grid строится из raw timing
и разрешённых oracle H boundaries; выбор размеченных строк не задаёт candidate
universe. Labels сопоставляются отдельно с проверкой точного интервала и view;
неоднозначные или меняющиеся внутри candidate targets masked. Границы tile,
held notes и unsupported H считаются в coverage, не обрезаются молча.

Gold members, sonority/key/function, source voice, textual rationale и derived
label features не поступают в модель. Work/entity IDs — маршрутизация, не
learnable lookup embeddings. C3 raw cache не зависит от target sidecar.
Oracle H geometry cache имеет отдельный явный binding.

## Что реализовать

Общий пакет предлагается под `src/music_critic/experiments/harmonic_event_reasoner/`:

1. Admission/export и dataset contracts: raw / geometry / targets / provenance.
2. H candidate/graph builder: oracle intervals сейчас; raw candidates отдельно.
3. Общий `score_query(graph, features, query, candidate_ids)` interface,
   query = source H + relation `chord_member_of_H`; scores возвращаются по IDs.
4. Четыре readout/refiner варианта ниже, общий masked trainer и evaluator.
5. Registry-bound runner, cache manifests, final/step0 metrics, resource caps.

Это proposal путей и интерфейса, файлов model package ещё нет. Существующие
MLP/geometry primitives можно адаптировать из relation_probe; его frozen source,
terminal artifacts и протокол сохраняются. Legacy код не нужен.

| Код | Модель | Что проверяет |
|---|---|---|
| M | Frozen C3 + H/note pair MLP | Сильный простой статический decoder |
| T | Frozen C3 + typed TransE-like score | Пользу translation geometry в той же задаче |
| Q | Frozen C3 + query-conditioned self-only refiner | Дополнительную обработку запроса без сообщений между узлами |
| G | Frozen C3 + тот же refiner с typed propagation | Добавочный эффект доступа к путям графа |

Предлагаемый начальный размер: 64d shared latent, 3 refinement rounds для Q/G,
без поиска depth/width. H и note имеют одинаковые по интерфейсу type projections
во всех условиях. Q и G получают source H features и relation conditioning
**на каждом кандидате**; source marker не должен оставаться недоступным Q.
У Q есть self updates, у G дополнительно сообщения по raw temporal/containment
и geometric H incidence. Это NBF-inspired адаптация, не буквальная реплика paper.
Графа gold semantic edges нет ни в одном условии.

Loss первого опыта — masked binary membership BCE, с одинаковым усреднением
known candidates внутри H и H внутри work, общим fit sampling schedule.
Unknown не даёт gradient; все H с известными labels могут участвовать в loss,
в том числе одноклассовые. Primary AUC имеет отдельно объявленный mixed-class
support. Class weights/threshold получают значения только из inner-fit policy.
Дополнительные function/sonority losses пока не смешиваются с основной задачей.

Controls без отдельной сетки архитектур: all-member, all-nonmember, одна
source-only оценка на H для всех его кандидатов, geometry-only ranking и
step0 каждой модели. Source-only создаёт ties внутри H и primary AUC=0.5.
Правило geometry ranking и tie handling фиксируется до dev; программа не
выбирает удобный heuristic после сравнения.

## Метрики и решение

Primary: внутри каждого H с обоими известными классами считать долю пар
member/nonmember, где member выше, с 0.5 за tie (binary AUROC). Затем среднее
H внутри work, затем среднее works. Публиковать eligible H/works и покрытие:
primary metric не характеризует неразмеченные или одноклассовые H.

Дополнительно: known-target global macroF1, P/R каждого класса, FPR по known
nonmembers, Brier и per-work distribution. Threshold выбирается на inner-fit.
Recall@1/5 по известным members — вторичная метрика, с числом candidates/answers;
фиксированный k не считается полноценной оценкой множества большой мощности.
Set precision/exact set и empty-answer accuracy доступны лишь для scopes,
где соответствующая полнота действительно подтверждена. Полное H discovery
или предсказание существования resolves_to в этой кампании не измеряются.

Контроль ресурсов: trainable parameters, actual H queries/known-pair exposure,
applied/attempted/skipped updates, train/eval wall, preprocessing/cache time,
peak VRAM, cached-query и end-to-end latency. Одинаковые updates не приравнивать
к одинаковому compute. Paired seeds имеют общий примерный schedule по H/works.
Неопределённость оценивать по work/family, не независимым bootstrap всех нот;
seed variability показывать отдельно.

Предлагаемое правило продолжения: G улучшает primary AUC относительно и M,
и Q в среднем минимум на 0.03, paired difference положительна на всех трёх
seeds, а known-target macroF1 не падает более чем на 0.01. Это заранее выбранный
порог полезности пилота, не доказанная статистическая мощность. Work-level
интервалы и концентрация эффекта публикуются; неубедительная неопределённость
означает inconclusive. T — дополнительный comparator; если он лучше G,
преимущество G над всеми простыми readouts не заявляется.

Если Q≈G и оба лучше M, поддержана польза дополнительной query processing,
а не propagation. Если G не даёт полезного улучшения, следующий шаг — диагноз
данных/readout, а не автоматическое увеличение числа модулей.

## Матрица запусков и предлагаемый предел

Nearest campaign = только oracle H/member comparison:

- preparation/admission и краткий native engineering gate;
- M/T/Q/G на seed17; screen проверяет корректность, finite learning и ресурсы;
- те же четыре условия на seeds42/73 при достаточном бюджете;
- максимум 12 training conditions, без гиперпараметрической сетки.

Seed17 не используется для произвольного отбора понравившихся условий по dev.
Пропуск/ошибка отдельного условия остаётся видимым; incomplete matrix не
объявляется завершённым трёхseed сравнением.

Предлагаемый новый campaign wall cap — 4 часа на одной существующей RTX3090,
включая gate, cache, training и evaluation; это потолок, не ETA и не перенос
старого шестичасового разрешения. Native gate до 15 минут; каждый training
condition до 10 минут. Финальное сопоставимое число updates U одинаково для
всех условий, до 2000; выбрать его до dev по throughput самой медленной модели
с резервом времени. Условия cap/memory/cleanup фиксируются в execution config.
Если какой-либо run не достигает U, сохраняется partial evidence, но primary
matched-U сравнение не подменяется результатами разной длительности.
Контрольные checkpoints сохраняют fallback common exposure для описательной
диагностики, не для post-hoc выбора лучшей точки.

Доступность сервера/GPU здесь не проверялась. Перед исполнением нужны актуальный
native admission, согласованный новый budget и planned records на свежие attempts.
Нет фонового мониторинга, SSH, платных ресурсов или новых LLM-вызовов на этапе
планирования. Документ не запускает кампанию.

## Полная программа и очередность следующих опытов

A выше — первый контролируемый вопрос, а не сокращение всей программы до
membership. В текущем task passport следует заранее описать доступность всех
следующих targets: H properties/function/key, directed relations, совместные
query answers, part/group/process и hierarchy labels. Это план учёта поддержки,
не разрешение дополнительных payload reads или обещание обучить все поля.
Interface сразу сохраняет exact time/view, typed entities и relations,
participant/context scope; многочисленные головы и losses одновременно не включаются.

| Очередь / опыт | Условия | Что необходимо для старта |
|---|---|---|
| Сейчас: A, H/member | M/T/Q/G, frozen C3, oracle H | Known member/nonmember внутри H и family support |
| С начала программы: V, независимая проверка | Небольшой reviewed набор atomic/composite queries, неверных и пустых ответов | Явная scope/completeness review; новые LLM calls не предполагаются автоматически |
| Следующая волна: S, semantic heads | Отдельные heads структуры, тональности, функции; atomic relation families по одному | Совместимый finite vocabulary, source scope, masks; within-source negatives для адресных relations |
| Следующая волна: C, Query2box | Explicit min/product против inductive Query2box на oracle H и том же candidate universe | Надёжный atomic scoring и различающие AND queries с joint supervision |
| Следующая волна: D, encoder adaptation | Frozen / pretrained-finetune / scratch C3 с одним фиксированным readout | Работоспособная задача, общий trainer/selection и сопоставимый exposure |
| Следующая волна: B, raw H candidates | Простой и graph readouts с общим bounded raw generation; новые train/eval runs | Поддержанный exact transfer targets и измеренная candidate coverage |
| После базового event path: E, музыкальные группы | Ensemble-only против добавленных part/group candidates | Собственная scope-bound supervision либо явно исследовательская latent постановка |
| После поддержанных events/relations: F, временные процессы | Последовательности H, phrases/process instances и их участники | Проверенные длительные relations/intervals; временное соседство не является semantic label |
| Отдельная ветка: J, сложная логика/иерархия | Shared-witness query operators; parent/ancestor/reduction heads при наличии targets | Identity witnesses для DAG queries; отдельные targets реального частичного порядка |
| Отдельная ветка: L, скрытые факторы | Semantic representation + residual latent; probes/stability против baseline | Собственный проверяемый auxiliary objective и критерий, без автоматического музыкального имени фактора |
| Интеграция: K, Music Critic/HCM | Existing critic против добавленных event/relation features или auxiliary learning | Закреплённый encoder/interface и отдельные downstream data/metrics |

S/C/D/B имеют частичные зависимости, но не требуют строгой цепочки друг за
другом: подготовка может идти независимо, GPU runs выполняются по очереди
в отдельных bounded campaigns. Query2box не ждёт raw H detector, потому что
oracle H достаточно для его собственного диагностического сравнения.
Function-ограниченные queries ждут соответствующих atomic heads/targets;
membership-only queries могут исследоваться раньше. Даже без победы NBF в A
можно использовать MLP/TransE scores для следующей проверки query operators.
Порог успеха G в A решает вопрос о дополнительной propagation, а не судьбу
всего event-graph направления. Единый NBF→box bridge не предполагается готовым.

В S функция относительно явно заданного в запросе K — отдельный conditional
режим, который можно изучать на oracle H раньше собственного key inference.
Raw-only function analysis использует предсказанный context и имеет отдельную
сквозную оценку. V готовится рано и служит общей оценкой музыкального смысла
для веток; до фактической независимой проверки существующие LLM labels не
переименовываются в gold. Local track events можно включать раньше E при
собственных targets; E обозначает проверку добавленной ценности group discovery,
а не задержку самого scope-aware интерфейса.

Иерархию relation/type vocabulary можно фиксировать при нормализации сейчас;
она не заменяет обучение вложению групп, временным процессам или музыкальной
редукции. Эти структуры имеют разные targets. В E сначала нужны явные scope
и ограниченные группы; свободное sparse group discovery — отдельная абляция.
Полноценный learned H detector с boundaries/count/objectness следует за B
только при достаточном supervision полноты или заявленной partial-label модели.

B не обучается только на gold H с последующим молчаливым переключением на raw.
Нужны общий train/inference candidate regime и отдельные oracle/common-support/
end-to-end показатели. Перенос membership нельзя делать только по interval IoU:
кандидат может пересечь две гармонии. Нужны однозначный reference H, совпадение
note-context/time/view/scope; остальные targets unknown. Count/objectness H
не объявляется решённым, пока нет supervision полноты объектов.

C запускается при совместной разметке и запросах, где оба ограничения нужны,
а простой time-overlap не определяет ответ. D/E и остальные ветки не
перемножаются с первыми условиями в общий grid. Все последующие опыты имеют
отдельные records/бюджеты и не входят в предложенные 12 runs / 4 часа.

Обратный H→notes retrieval уже является первой задачей A; направление
note→H можно проверить на том же incidence representation с собственной
candidate/coverage оценкой. Function→новые ноты — conditional generation,
не inverse pooling; её можно связывать с HCM только отдельным опытом после K.
PLL/likelihood, agreement с theory labels, human preference и музыкальная
эстетика в K остаются разными оценками. Свободным факторам L не приписываются
музыкальные значения по красивой визуализации; проверяются перенос, устойчивость
и связь с отдельными наблюдаемыми признаками. Hyperbolic/cone geometry возможна
в J при подходящем отношении порядка, но не обязательна для всего графа.

Очередность конкретна по зависимостям; календарные сроки реализации и GPU ETA
не измерены. Первый следующий артефакт — паспорт всей предполагаемой supervision
с ближайшим исполняемым subset A, затем общий event/query interface. Такой
порядок сохраняет большой замысел и ограничивает число одновременно меняемых
компонентов, не делает следующие направления бессрочно зависимыми от одного scorer.

## Приёмка реализации и текущие изменения

До native данных synthetic tests должны проверить: target-sidecar invariance,
oracle geometry как явный режим, отсутствие negative gradient от unknown,
membership scopes/time/view, сохранение всех eligible candidates, query доступ
в Q/G, несколько/нет известных ответов, AUC ties и порядок H/work aggregation,
runtime caps и сохранение partial evidence. Engineering tests не доказывают
качество модели. При изменении implementation каждое поведение тестируется.

В этой задаче изменены только этот plan, ссылка в spec и STATUS. Code/data/
legacy/weights/terminal records не изменялись, raw corpus payloads не читались.
Перед substantial preparation/training действует
[EXPERIMENT_LOGGING](../../../EXPERIMENT_LOGGING.md); records создаются до работы,
а не задним числом. Следующий непосредственный результат — authenticated task
passport, после него реализуемый config и конечный набор условий.
