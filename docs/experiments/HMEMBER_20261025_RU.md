# Первый oracle-H/member пилот

## Итог кампании и проверка доказательств — 25 октября 2026

**Все12условий завершены** по2 000updates, failed=0, aborted=0. Конец:13:07:53UTC (16:07:53МСК).
Wall supervisor775.79s (12min56s), включая244.85s gate/cache; сумма wall обучающих
условий519.27s. Новое обучение или inference при проверке результатов не запускалось.
Heartbeat приостановлен после первой проверки; процессов этой кампании на сервере не найдено.

Все значения ниже относятся к exploratory dev внутри global TRAIN. Primary AUROC —
среднее внутри mixed-class H, затем H→work→works. Общая опора:596mixed H
в7независимых works/families. MacroF1/FPR считаются по всем известным dev-кандидатам.
Пороги выбраны на inner selection отдельно для каждого run; dev не выбирал checkpoint.

| Условие | Seed | Updates | Wall,s | AUROC | MacroF1 | FPR | Recall@1 | Recall@5 | Known coverage | Статус |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MLP | 17 | 2000 | 37.16 | 0.8975 | 0.6877 | 0.5329 | 0.1294 | 0.5012 | 62.68% | completed |
| TransE-like | 17 | 2000 | 37.65 | 0.8468 | 0.6246 | 0.6239 | 0.1290 | 0.4996 | 62.68% | completed |
| Self-only | 17 | 2000 | 46.41 | 0.9063 | 0.6846 | 0.5444 | 0.1302 | 0.5014 | 62.68% | completed |
| Graph | 17 | 2000 | 52.94 | 0.9204 | 0.6900 | 0.5568 | 0.1296 | 0.5013 | 62.68% | completed |
| MLP | 42 | 2000 | 36.72 | 0.8644 | 0.7201 | 0.5521 | 0.1293 | 0.5001 | 62.68% | completed |
| TransE-like | 42 | 2000 | 37.21 | 0.8368 | 0.6228 | 0.7366 | 0.1290 | 0.4988 | 62.68% | completed |
| Self-only | 42 | 2000 | 45.88 | 0.7916 | 0.6458 | 0.7184 | 0.1287 | 0.5002 | 62.68% | completed |
| Graph | 42 | 2000 | 53.33 | 0.8217 | 0.7016 | 0.4209 | 0.1285 | 0.5003 | 62.68% | completed |
| MLP | 73 | 2000 | 36.73 | 0.9195 | 0.7396 | 0.5147 | 0.1296 | 0.5009 | 62.68% | completed |
| TransE-like | 73 | 2000 | 37.48 | 0.8551 | 0.6051 | 0.8179 | 0.1278 | 0.4988 | 62.68% | completed |
| Self-only | 73 | 2000 | 45.85 | 0.9018 | 0.6992 | 0.5936 | 0.1294 | 0.5006 | 62.68% | completed |
| Graph | 73 | 2000 | 51.91 | 0.9307 | 0.7095 | 0.5470 | 0.1294 | 0.5016 | 62.68% | completed |

Среднее±sample SD по3seeds (не доверительный интервал по произведениям):

| Условие | AUROC | MacroF1 | FPR mean | Wall mean,s |
|---|---:|---:|---:|---:|
| MLP | 0.8938±0.0277 | 0.7158±0.0262 | 0.5332 | 36.87 |
| TransE-like | 0.8462±0.0091 | 0.6175±0.0108 | 0.7261 | 37.45 |
| Self-only | 0.8666±0.0650 | 0.6765±0.0276 | 0.6188 | 46.04 |
| Graph | 0.8909±0.0602 | 0.7004±0.0098 | 0.5083 | 52.73 |

Фиксированные controls одинаковы во всех12runs:

| Control | AUROC | MacroF1 | FPR | Recall@1 | Recall@5 |
|---|---:|---:|---:|---:|---:|
| Все member | 0.5000 | 0.4870 | 1.0000 | 0.1274 | 0.4945 |
| Все nonmember | 0.5000 | 0.0481 | 0.0000 | 0.1274 | 0.4945 |
| Только raw duration | 0.7829 | 0.5416 | 0.6567 | 0.1288 | 0.4966 |

Control raw duration использует duration/(1+duration), threshold0.3 выбран на selection.
Для постоянного score retrieval ties разрешаются стабильным порядком candidates;
это не случайный ranking baseline. Source-only score постоянный внутри H имеет AUROC0.5.

**Сопоставимость.** Для каждого seed совпадают SHA256 последовательности work/H draws
у всех4условий,2000attempted=2000applied,0skipped; known-pair exposure:
seed17=52 453,seed42=52 371,seed73=52 805. Cache SHA и dev coverage совпадают.
Проверены согласованность result/progress/records и восстановление primary AUROC
из per-work values. Self/graph имеют по344 321параметру; MLP187 009,TransE153 986.
Одинаковые updates/exposure не означают одинаковые compute/parameter budgets.

**Покрытие.** Dev содержит10 475H и93 530candidates:58 621known (55 656positive,
2 965negative),34 909unknown. Primary поддержан596H/7works; retrieval —3 418H/12works.
Unknown не считаются negative и остаются конкурентами в retrieval. Known negatives
сосредоточены:work005=1052,work014=972; вместе68.26%всех negatives. У7mixed works
очень разная поддержка:work003 имеет11negative candidates,work016 —15.

**Выводы.** Graph−self AUROC по seeds17/42/73: +0.0141/+0.0301/+0.0289
(среднее+0.0244). Это согласованный описательный сигнал пользы распространения
сообщений в данной постановке. Средняя разница положительна в6из7mixed works,
но sparse support и один split ограничивают интерпретацию. Превосходства над MLP
не показано:Graph0.8909 противMLP0.8938 AUROC;MacroF1 0.7004 против0.7158,
Graph требует52.73s против36.87s. TransE ниже MLP по AUROC и MacroF1 во всех3seeds.
Graph FPR≈0.508:около51%известных nonmembers всё ещё предсказаны member при
selection-порогах. Recall@5 близок между моделями и control, поэтому он не является
решающей метрикой этого пилота. Duration-only AUROC0.7829 показывает сильный
raw признак; learned heads превосходят его здесь, но это не доказательство понимания гармонии.

| Утверждение | Evidence / integrity | Вердикт |
|---|---|---|
| Все12bounded runs завершены |13canonical runtime records; SHA/size13compact artifacts + cache manifest; matched2000updates | supported |
| Propagation помогает относительно self-only здесь |+AUROC на3paired seeds, одинаковые exposure/cache/support;7mixed families | descriptive |
| Graph лучше простого MLP |Mean AUROC/F1 не выше; compute больше | unsupported |
| Raw H discovery, human harmonic accuracy, generalization к неизвестному raw pretraining |Oracle H, weak labels, overlap unknown; TEST закрыт | unsupported |

**Сохранность.**41compact JSON получен в ignored `outputs/hmember_20261025/monitor001`.
Проверены fingerprints13runtime records, SHA/size12result.json+gate-result.json,
cache manifest binding.60training checkpoint files и4gate timing weights остаются
на сервере; в этой проверке их bytes не скачивались и локально не проверялись.
12prediction streams также остались на сервере с записанными bindings.
Terminal records не переписывались; planned/running продвинуты вперёд.
Closure: `EXP-HMEMBER-REVIEW-001`; детали в compact comparison/integrity artifacts.

**Следующий шаг, без запуска:** отдельный подтверждающий протокол MLP/self/graph
с проверкой устойчивости по независимым families и заранее заданным operating point
для nonmember false positives; сохранять raw-duration control. Данная проверка
не меняет default architecture, не запускает Query2box и не расширяет доступ к TEST.
Original release checkout не изменялся; code/model tests повторно не запускались
для этой документальной сверки; выполнены integrity/parity/registry checks.

## История подготовки и запуска

25 октября 2026. Изолированная реализация T08/A по разрешению пользователя.
Этот worktree не входит в параллельно готовящийся GitHub release для научного
руководителя. Ни commit, ни push не выполняются.

## Вопрос и модели

Помогает ли распространение сообщений, зависящее от запроса H→notes, отличать
известные chord members от известных nonmembers **внутри одного H**?
Frozen C3 сохраняет отдельные состояния нот, временных срезов и других узлов.
Поверх них сравниваются MLP, TransE-like, query self-only и query graph.
У self/graph одинаковы параметры и доступ к source/relation query. Graph
добавляет типизированные сообщения между узлами. Это NBF-inspired refiner,
не воспроизведение оригинального NBFNet.

Кандидаты строятся по raw notes/time grid и oracle H boundaries. Поля chord,
key, function, chord template, membership и confidence не входят в признаки.
H получает среднее **всех** геометрических кандидатов, включая unknown.
Теоретические membership edges являются целью, а не входными рёбрами.
Unknown/conflicting/partially covered интервалы маскируются. Geometric
incidence обозначает пересечение во времени и не означает chord membership.

В source нет общего надёжного track-scope контракта H. Поэтому все наблюдаемые
pitched tracks образуют candidate superset, но labels остаются строго в своём
H/view. Это не утверждение, что все H являются гармонией ансамбля. Альтернативные
views сохранены отдельно; конфликтующие known targets при одинаковом H interval
и наборе candidates маскируются во всех эквивалентных views.

## Данные и ограничения

Сохранены прежние64runtime works; исключённые ранее работы не возвращались.
Новый exact admission фиксирует128файлов canonical/bundle с SHA/size до decode.
Подготовка `EXP-HMEMBER-PREPARE-001` завершена:43 590H;121 926positive,
11 242negative,119 543unknown note-context fields;442нулевых контекста исключены,
94контекста без H не присоединены по догадке. Это counts исходных аннотаций,
а не будущих raw-context candidates или независимых наблюдений.

| Раздел внутри TRAIN | Works | H | Mixed H | Works с mixed H |
|---|---:|---:|---:|---:|
| Train |42|24 968|3 667|24|
| Inner selection |10|7 855|812|5|
| Exploratory dev |12|10 767|615|7|

Inner selection выделена по family hash до metrics; прежний dev сохранён.
Global VALIDATION/TEST не используются. Все labels слабые LLM/source labels,
human accuracy и отсутствие raw-pretraining overlap не установлены.
После проверки ingestion доступные поля имеют state=known (133 225 включая
контексты, исключённые геометрией); ужесточение state guard не меняет export.

## Измерение и бюджет

Primary: AUROC внутри каждого mixed-class H, затем среднее H внутри work,
затем среднее works. Source-only constant внутри H даёт0.5. Дополнительно:
fixed-two-class macroF1, P/R/FPR, Brier/BCE, Recall@1/5 по известным positive
при конкуренции всех candidates, masks/coverage, excluded H, runtime и exposure.
Порог выбирается только на inner selection; финальная оценка при одном общем
числе updates, а не на best dev checkpoint.

[Config](../../configs/hmember_20261025.json):4conditions × seeds17/42/73,
не более4h суммарно;15min gate;10min каждое условие включая evaluation.
Общий update count50..2000 выбирается по TRAIN-only timing до dev quality.
При сбое/таймауте кампания прекращается, оставшиеся условия явно закрываются
как skipped/aborted. Дубликат запуска запрещён. Процессы удалённые и не зависят
от включённого ноутбука.

## Проверки и состояние

Локальные pure tests проходят; native Torch tests запускаются на существующем
RTX3090 окружении без установки зависимостей. Legacy не читался/не копировался.
Повторно использованы текущие V2 raw sanitizer, PLL graph/encoder и registry.
Независимый review выявил и исправил parent-cache authentication, lifecycle
при mkdir failure и fail-closed annotation states. Native run status, точные
итоги тестов и artifact bindings добавляются после gate/launch.

Из source001 на сервере прошли73tests без skips:12model,8cache,53data/metrics/
training/evidence/supervisor. Native log:430bytes,
SHA256 `61a99fe8548bb71c2021ea42556522ebce15e82cfd8d17fd1531ee7259c9c45f`.
Source archive:1 448 258bytes,
SHA256 `ec99e7e832e0ab212b7add31628f85fdf55bb1fae220eec4d84c3496d6cb02bb`.
После двух независимых review axes открытых блокеров не осталось. Первичный
тестовый upload пропустил fixture; пакет исправлен, итоговая проверка полного
замороженного source001 прошла. Это не обучение и не scientific result.

Supervisor запущен25.10.2026 12:54:57UTC; первый этап — C3 cache/native timing
gate. Runtime evidence root на сервере:
`/home/humtech/music_critic_hmember_20261025/campaign001`.
Относительные locators runtime records разрешаются от этого evidence root;
checkpoint bytes остаются на сервере. Local compact copies размещаются в
ignored `outputs/hmember_20261025/monitorNNN`, terminal records не переписываются.
Кампания имеет13planned records (gate+12conditions); запуск условия требует
успешного gate. Native gate не использует dev quality при выборе updates.

Gate завершён за244.85s: RTX3090, CUDA12.6, Torch2.8.0+cu126, PyG2.7.0.
Обработаны все64works. Сохранены41 633из43 590H;1 957H исключены по
tile/raw geometry и лимитам без усечения. После candidate masking mixed H:
3 330train/765selection/596dev, соответственно24/5/7независимых families.
Конфликтов одинаковых H между views в этом cache не найдено (0masked).
Candidate counts отличаются от source annotation counts из-за пересечения
notes с raw time slices; отсутствующие labels остаются masked.

Timing policy допустила2 000общих updates на условие до просмотра dev quality.
В снимке `monitor000-progress` завершены3/12условий (MLP, TransE и self-only
seed17), выполняется graph seed17. MLP:2 000applied updates,37.16s включая evaluation; веса
сохранены. Сравнение моделей пока не завершено и quality выводов не делаем.
Heartbeat `h-member-rtx3090`:1check/hour, max4, pause on terminal/error/noSSH;
без новых launches. Чекпоинты остаются на сервере, секреты не сохранялись.

Query2box, discovered H, новые semantic heads, C3 finetune/scratch и группы
остаются следующими отдельными опытами; baseline architecture не заменена.
