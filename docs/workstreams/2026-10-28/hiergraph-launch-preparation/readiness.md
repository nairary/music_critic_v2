# HIERGRAPH E1–E5: подготовка к первому сравнению

Статус: **готово к запуску пяти ограниченных пилотов; обучение не запущено**.
Подготовлен H/F slice с frozen C3 и пятью разными H constructors. Все 67 работ
представлены в raw cache; native E1–E5 forward/backward и CUDA replay прошли.
Пять training records остаются `planned`. [Спецификация](spec.md),
[независимые Standards/Spec reviews](review.md),
[финальная evidence review](../../../experiments/records/EXP-HIERGRAPH-READY-REVIEW-001.json).

## Подтверждённые данные и их идентичность

`EXP-HIERGRAPH-READY-EXPORT-002` завершён: повторно спроецированы те же 67 работ
исходного глобального TRAIN, с 15 принятыми native заменами из annotation
release003. Точные canonical и baseline SHA/size каждой замены совпали с прежним
admission; сохранены индивидуальные review records и трёхчастный musical-review
basis. Это принятые слабые source/model labels, не human gold.

| Проверка | Подтверждённый результат |
|---|---:|
| Работы / downstream split | 67 / 44 fit, 10 selection, 13 dev |
| H / note-context h / objects | 45 263 / 274 936 / 10 611 |
| Новые families | 0 |
| Проверенные target bindings | 134: projection + numeric для каждой работы |
| Геометрия H/h относительно предыдущего V3 | Все 45 263 H и 274 936 h совпали |
| 52 работы без принятой замены | Source-projection evidence совпало; mask gains/losses 0/0 |
| Ontology / eligible поля / eligible H | 97 / 47 / 42 |
| Source payload reads | 134 distinct files, 3 208 612 839 bytes; 201 read operations |
| CPU export / verification | 892.224 s / 83.606 s |
| Peak worker RSS / verification RSS | 4 588 716 032 / 2 683 240 448 bytes |

Vocabulary fitted только на 44 fit families. Selection/dev не расширяли классы;
OOV остаются masked и учитываются отдельно. В новом export остаются 81 event OOV
и 4 object OOV; это суммы field occurrences, не число независимых событий.
Глобальные VALIDATION/TEST закрыты. Downstream dev расположен внутри исходного
TRAIN, является exploratory; отсутствие пересечения с raw C3 pretraining неизвестно.

Первый `EXP-HIERGRAPH-READY-EXPORT-001` сохранён как **failed**: исходные payload
paths были разрешены относительно isolated checkout вместо main read-only.
Рабочий процесс завершился до source payload read (`reads=[]`). Retry002 получил
новую запись и новый output, явно разрешив исходные пути через read-only fallback;
предыдущие попытки и artifacts не заменялись.

- [Record002](../../../experiments/records/EXP-HIERGRAPH-READY-EXPORT-002.json),
  [retained failed001](../../../experiments/records/EXP-HIERGRAPH-READY-EXPORT-001.json).
- Полные локальные artifacts: `outputs/hiergraph_ready_20261028/export002/` и
  `outputs/hiergraph_ready_20261028/metadata002/`.
- Dataset manifest SHA-256:
  `2d808660243ad68a24b29474140eebd9b9abb7e0818b0ca60e7af7fe0e0bd415`;
  bytes `2415466`; dataset fingerprint
  `052c28d75ec06337bfbc083d7d439330948b14f2004c9c5b559d72d8382bb22f`.
- Split fingerprint:
  `88641a45a4cd57081f7c0afdda899d25d08fefce7f7761772552201a8e3e68a1`.
- Vocabulary fingerprint:
  `3bd39448a1a0289ee4b9f94cc563b16db43bc832721c1fb655b001feec5f2e36`;
  vocabulary file SHA-256
  `4f55954ebad5d75b5055ac023149f7417330f427a18f3c7691f38d27a91d4d10`.
- Projection: `HiergraphProjection@3.0.0`; contract fingerprint
  `89e9f8affc6805d4af3a2ca8ffde9613035232f26fe4a77a19a89eeedb20ded8`.

## Какие именно H/F свойства входят в 42 heads

42 — число отдельных eligible output fields. Оно включает независимые PC bins;
это не 42 вида гармонических событий. Общий минимальный gate: минимум три fit
families и два известных fit класса; он не доказывает достаточность supervision.
Таблица получена только из authenticated heads/vocabulary metadata, без нового
чтения corpus payloads.

| Группа | Eligible / exported | Активные поля; fit support |
|---|---:|---|
| Функция H, то есть текущая F | 1 / 1 | `H.function`: 7 классов, 38 fit families |
| Тональность | 2 / 2 | `H.key.tonic_pc`: 12 классов/43 families; `H.key.mode`: 3/41 |
| Root, quality, core, seventh | 4 / 4 | `H.sonority.root_pc` 12/41; `.quality` 9/22; `.core` 6/40; `.seventh` 3/24 |
| Bass и inversion | 3 / 3 | `H.bass_pc` 5/5; `H.bass_degree` 3/5; `H.inversion` 4/36 |
| Интервальный состав интерпретированного аккорда | 12 / 12 | `H.chord.interval_pc.0..11`; по два класса, 21–29 families |
| Именованные additions | 0 / 7 | `H.sonority.addition.*`: недостаточно классов и/или families |
| Именованные alterations | 0 / 6 | `H.sonority.alteration.*`: недостаточно классов и/или families |
| Именованные extensions | 1 / 7 | Только `H.sonority.extension.9`: 2 класса, 4 families |
| Explicit omissions | 0 / 7 | `H.sonority.omission.*`: нет двух известных fit классов |
| Прежний positive-only PC set | 0 / 12 | `H.sonority.pitch_class.0..11`: PU negative-loss protocol не принят; implicit zeros запрещены |
| Полный наблюдаемый абсолютный состав | 12 / 12 | `H.observed.pitch_class.0..11`: по два класса, 6 families |
| Root наблюдаемого созвучия | 1 / 1 | `H.observed.root_pc`: 12 классов, 6 families |
| Наблюдаемый root-relative состав | 6 / 12 | `H.observed.interval_pc.{3,4,6,7,10,11}`: по два класса, 6 families |
| **H всего** | **42 / 86** | Одна и та же head manifest для E1–E5 |
| Note-context h, сохранены для последующих slices | 5 / 9 | `h.chord_membership`, `.preparation.linked`, `.resolution.linked`, `.temporal_role`, `.voice_role` |
| Процессы P, evidence only | 0 / 2 | `P.function`, `P.process` отложены по scope первого сравнения |
| **Полный export** | **47 / 97** | Первый runtime выбирает только 42 H поля |

Наблюдаемые sets и их zeros проверяют состав при source H geometry; это частично
детерминированная задача по MIDI и не доказательство понимания гармонической
функции. `H.observed.*` и `H.chord.*` имеют разные корни/смысл; их маски независимы.
Неразмеченный root не заменяется tonic или bass. В первых PC bins теряется spelling
и различие compound degree; TODO `HIERGRAPH-ENHARMONIC-001` остаётся открытым.

Полные fitted typed-value maps, пустые fitted maps неподдержанных полей и отдельные
ontology maps сохранены в loader. ID берутся из соответствующего fitted field map;
`false` и integer `0` не становятся одним значением, ontology order не используется
вместо fitted class order.

## Target-side contract и учёт исключений

Model forward получает только raw tensors: frozen C3 features, наблюдаемую
геометрию, schema node/edge types и raw time/bank routing. Target IDs, view IDs,
availability, границы H и теоретические поля в forward не передаются. Exact
canonical timing хранится как rational/Fraction; floating coordinates создаются
только для нормализованного model/loss пространства.

- Каждый source H принадлежит ровно одному core по условию
  `core_start <= H.start < core_end`. Target set допускает H только при полном
  containment в halo; границы не обрезаются.
- Owned H вне halo остаются в `coverage.outside_halo` и primary known-H denominator.
  `outside_halo_known_fields` сохраняет соответствующий H/F denominator. Их
  исключение из localization loss не превращается в улучшение reported recovery.
- Point H не становятся positive-duration целями; `point_events_owned` учитывает
  их отдельно. Held physical notes сохраняют identity и несколько raw fragments.
- Разные analytical views образуют отдельные target sets при одинаковом raw tile.
  Matching совместим по view/scope. `scope_id='harmonies'` именует коллекцию;
  подробный source interpretation scope остаётся в field namespace/evidence.
- Неизвестные labels имеют mask=false и numeric sentinel `-1`. Неявных negatives
  из отсутствия поля, unmatched event или пустой части annotation нет.
- `known_negative_spans` по умолчанию пуст. Negative objectness допустим только
  внутри явно complete совместимой области без positive/unknown/ambiguous overlap;
  matched positive query не получает одновременно negative BCE.
- Training assignment допускает low-IoU пары для обучения localization. Metrics
  используют отдельно заданный overlap threshold и one-to-one matching. Общий
  prediction budget определяется raw duration, а не числом gold H.
- Precision доступна только в доказанно complete scopes. Основная метрика —
  family-macro known-H recovery; joint H+F, conditional macro-F1, boundary error,
  coverage и capacity сообщаются отдельно. Oracle-H — отдельная диагностика.
- Raw-cache capacity exclusions должны сохраняться с owned target misses; полные
  native coverage/capacity числа пока pending.

## Пять механизмов и граница первого сравнения

Все ветви используют общие raw hierarchy, typed raw updater, H refiner и H heads.
Raw residual channels и nonchord notes сохраняются. Общий output budget и общие
head masks не делают разное число candidates и вычисления автоматически равными;
фактические ресурсы и exposure нужно измерять отдельно.

| Ветка | Механизм создания H |
|---|---|
| E1 | Raw-grid span bank; scoring/selection сырых кандидатов |
| E2 | Boundary segmentation с differentiable boundary learning |
| E3 | Anchor-free endpoints от raw time anchors |
| E4 | Learned event queries с cross-attention к raw states |
| E5 | Learned sigmoid graph assignments, нормировка по входам каждой группы и независимый interval head |

E5 assignments — latent pooling weights, не semantic membership и не
воспроизведение AutoSchA или проверенная Schenkerian analysis. Note, h context и
H сохраняют разные смыслы. Gold count не устанавливает K, gold segmentation не
требуется при raw inference.

Первое сравнение использует **frozen C3**, H discovery и перечисленные H/F heads.
**Augmentation=`none` одинаково для всех пяти ветвей.** Transposition/full orbit
будет отдельной будущей ablation; нельзя транспонировать cached C3 vectors как
готовые признаки. Для такой ablation нужны новые raw notes, согласованно
преобразованные absolute targets и заново вычисленные C3 features.

P construction, многоуровневый P feedback, predicted F/f/P feedback, NBF-like
addressed decoding, Query2box composition, live C3 adaptation, spelling restoration
и дополнительные note-context heads не объявляются реализованным результатом
этого первого H/F сравнения. Текущая подготовка также не подтверждает theory
accuracy, human preference, audio aesthetics или превосходство какой-либо ветви.

## Подтверждённая инженерная готовность

Portable `inputs001`: 67 canonical pieces, 67 numeric shards и три metadata files,
137 файлов / 222 675 127 bytes. Target manifest перенесён byte-for-byte, bundles
и full projections не нужны runtime. Receipt SHA:
`7bb5e261d8bdd65f6957dd72ef47dba95989fdfd8a65ac99fda6b6d8647a54e5`.

| Проверка | Итог |
|---|---|
| Локальные synthetic tests | 125 запущено: **124 passed**, один CUDA-only skip; 1.591 s |
| Отдельный тест на RTX3090 | **1 passed**, все пять конструкторов, 1.182 s; точное tensor replay Adam |
| CUDA unit accounting | 30 synthetic optimizer steps; это не обучение на корпусе |
| Независимые Standards / Spec reviews | Исправлены все найденные blockers; scope H/F slice |
| Финальный C3 cache004 | **67 работ, 3256 окон**, 254.904 s, 1 810 429 344 bytes tile tensors |
| Native gate001 | **5/5 моделей × 3 raw окна**, конечные losses/gradients, 36.542 s с подготовкой inventory |
| Real-data optimizer updates | **0** |
| Training command admission | Пять точных planned команд проверены на сервере; не исполнены |
| Registry / ledger | **531 valid records**; render и render --check прошли |

Native gate измерил один forward+loss+backward на каждом из трёх размеров
локального raw графа с добавленным whole-work bar context. Это одиночные
измерения без warmup, не сравнение скорости и не MIDI-to-analysis latency.
C3 уже закэширован; веса frozen C3 не входят в GPU memory этих измерений.

| Ветка | Обучаемые параметры | F/B на трёх примерах, s | Peak allocated, MiB |
|---|---:|---:|---:|
| E1 | 1,563,019 | 0.0177–0.4130 | 343.3 |
| E2 | 1,628,812 | 0.0181–0.0654 | 194.3 |
| E3 | 1,563,277 | 0.0166–0.0506 | 222.8 |
| E4 | 1,604,621 | 0.0230–0.0321 | 193.9 |
| E5 | 1,588,045 | 0.0267–0.0335 | 194.3 |

Данные gate: RTX3090, PyTorch2.13.0+cu130, CUDA13.0. CPU mechanics проверены в
отдельном local PyTorch2.15.0/PyG2.8.1 environment с существующими declared deps.
CUDA replay использовал model/runtime из source004; проверено, что соответствующие
model/runtime/loss/matching/metric bytes идентичны финальному source005.

## Покрытие и сохранённые неудачные попытки

Raw-only профиль допускает **64 дорожки**, сохраняя все исходные дорожки. Он
отделён от PLL генератора: его ContextProfile/Codec/Event лимит16 не изменён.
Синтетические 17- и 64-дорожечные pieces прошли настоящий C3 encoder. Это проверка
формы/исполнения, не доказательство качества C3 на непривычных инструментовках.

Осталось одно raw исключение: `work044`, core `[160,173)` qn, halo `[156,177)` qn,
ошибка canonical bar validator при удалении source-only provenance. Геометрия
источника не исправлялась. Семь принадлежащих окну H остаются пропусками в метрике.

Fit coverage проверена против compact export counts: **26 196 исходных H =
25 591 admitted + 598 outside-halo + 7 raw-excluded**. То есть 97.69% fit H
входят в обучаемые окна; остальные сохраняются в primary denominator. Это
ограничение текущего horizon/coverage protocol, не оценка музыкальной точности.
Selection содержит 7 855 H, exploratory dev — 11 212 H; всего45 263.

| Попытка | Статус и причина |
|---|---|
| EXPORT-001 | failed до source payload read: paths относились к main checkout |
| EXPORT-002 | completed; явный read-only fallback и accepted15 overlay |
| CACHE-001 | failed, 0 updates: canonical binding уже имел `work_id` |
| CACHE-002 | failed после1961 окон: необработанная validator rejection irregular measure |
| CACHE-003 | completed,3206 окон; исторический профиль16 исключал две работы целиком |
| CACHE-004 | completed,3256 окон; raw64 возвращает обе работы, одно окно исключено явно |
| GATE-001 | completed, все пять условий без real optimizer steps |
| REVIEW-001 | completed, инженерная готовность подтверждена |

Все эти имена имеют prefix `EXP-HIERGRAPH-READY-`; terminal records и outputs
не переписывались. Никакие checkpoint/cache tensors локально не скачивались.
Compact JSON/receipts проверены локально; C3 checkpoint SHA и58 805 104 bytes
проверены на сервере. Raw tensors перепроверяются SHA/size ленивым loader при чтении.

## Точный первый запуск — подготовлен, не исполнен

- [Общая конфигурация](../../../../configs/hiergraph_e1_e5_first_20261028.json).
- [Пять list-form команд и environment](../../../../configs/hiergraph_e1_e5_launch_20261028.json).
- Records `EXP-HIERGRAPH-HFIRST-E1-S17-001` … `EXP-HIERGRAPH-HFIRST-E5-S17-001`
  находятся в `docs/experiments/records/` со статусом **planned**.
- Frozen source: `/home/humtech/music_critic_hiergraph_ready_20261028/source005`.
  Из этого cwd использовать `../training-config001.json`, соответствующую запись
  `../EXP-HIERGRAPH-HFIRST-E*-S17-001.json` и новый `../runs/E*-s17-001`.
- Пять условий последовательно, **seed17, 2000 attempted draws**, одинаковый sampler,
  Adam3e-4, clip1, hidden128, rounds3, decode≤64 и≤4H/qn, IoU0.5.
- Checkpoints250/500/1000/2000. Выбор — earliest maximum inner-selection
  family-macro known-H recovery. Dev один выбранный checkpoint + отдельный oracle-H
  diagnostic. Different applied/selected updates не выдаются за matched-update сравнение.
- Hard cap1800 s на условие, outer timeout1820 s с cleanup10 s; до9100 s на все
  пять при последовательном исполнении. Это cap, не оценка длительности.
- Frozen C3, augmentation=none. Следующий этап после пилота — разбор coverage,
  losses и ошибок, затем отдельно спланированные paired17/42/73 replications.

| Binding | SHA-256 |
|---|---|
| Source manifest005 | `05ac3a3482c51eac0ff543138e49206805eb2ec0c0cd11dbe87b88d87e3efffa` |
| Target manifest | `2d808660243ad68a24b29474140eebd9b9abb7e0818b0ca60e7af7fe0e0bd415` |
| Cache004 manifest | `5d4e2634bff7a9cffeea22c83e0be52325be7b4bcb89eb7321771f0bc0d45a32` |
| Training config001 | `bcf00d51ff367950cc26dba4d29397c49a3bda729f68825b71f48ee259e60ee2` |
| C3 checkpoint | `d651511441907d4d559d0d3009c925107b9e371ca462db953f189ad4528e4a35` |

## Scientific-evidence-review: граница выводов

| Утверждение | Evidence | Verdict |
|---|---|---|
| Пять bounded H/F pilots готовы к запуску | source/config/data/cache hashes, native gate, planned admission | supported |
| Loss/backward и resume механически работают | CPU suite, реальные fit gradients, synthetic CUDA exact replay | supported в проверенном scope |
| Приведённые GPU time/memory описывают эти три примера | primary gate JSON,15 измерений без warmup | descriptive |
| Какая архитектура лучше понимает гармонию | Обучение/сравнение ещё не проводилось | unsupported |
| Есть generalization на TEST или преимущество перед LLM | TEST закрыт, end-to-end/LLM benchmark отсутствует | unsupported |
| Реализована полная H/h/F/f/P/p иерархия | Первый slice содержит только shared raw graph и H/F | unsupported |

Новые модули opt-in. Legacy checkout не читался и не использовался в runtime.
Переиспользованы текущие V2 canonical loading/sanitization, raw tiling/graph,
проверенный C3 loader и `_encode_graph` seam. Основной checkout для GitHub release
не редактировался; commit/push не выполнялись.

## Файлы текущей data-подготовки

Созданы в isolated worktree; основной checkout не редактировался:

- `src/music_critic/experiments/hierarchical_graph_analyzer/data.py` — authenticated
  target-only loader, typed maps, per-view exact core/halo targets и coverage.
- `src/music_critic/experiments/hierarchical_graph_analyzer/overlay.py` — exact
  accepted-native admission и immutable metadata для существующего V3 exporter.
- `scripts/prepare_hiergraph_overlay.py` — зарегистрированный CPU export entrypoint.
- `scripts/package_hiergraph_inputs.py` — authenticated canonical/numeric subset copy.
- `tests/experiments/test_hiergraph_data.py` (6 tests),
  `test_hiergraph_overlay.py` (3), `test_hiergraph_packaging.py` (1).
- `docs/experiments/records/EXP-HIERGRAPH-READY-EXPORT-001.json` и `002.json` —
  отдельные failed/completed lifecycle records.
- Этот `readiness.md` — интеграционный draft с явно открытыми native gates.

Общие model/cache/matching/loss/metrics/runtime/pipeline/CLI files и их tests
принадлежат root и соседним implementation slices; их окончательный inventory
и последнее подтверждение root добавляет после integration. Legacy logic не
инспектировалась, не импортировалась и не копировалась. Dataset/checkpoint/cache
payloads не добавлялись в Git; staging/commit/push/publication не выполнялись.
