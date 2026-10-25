# Эксперименты: карта исследовательских кампаний

Автоматическая навигация по существующим записям; не новый анализ результатов.
Статусы взяты из реестра: это не проверка живых процессов.
Терминальные записи, старые конфиги и отрицательные результаты сохраняются неизменными.

[Главная](../../README.md) · [Модели](../MODELS.md) · [Данные](../DATASETS.md) · [Полный ledger](../EXPERIMENT_LEDGER.md)

Всего записей: **373**. aborted: 21; completed: 288; failed: 33; invalid: 8; planned: 10; running: 13.

| Кампания | Вопрос | Записей |
|---|---|---:|
| [PLL / HCM: условная вероятность музыкальных событий](#hcm) | Как использовать локальный и полный контекст? | 45 |
| [Парный и аспектный критик](#critic) | Можно ли обучить сравнение музыкальных вариантов на frozen C3? | 18 |
| [Корпус и теоретическая разметка](#annotation) | Какие источники, события и теоретические поля подтверждены? | 124 |
| [Авторские произведения и музыкальные вмешательства](#authoring) | Какие контролируемые примеры и изменения пригодны для исследования? | 18 |
| [C4: POP909 и перенос между корпусами](#c4) | Как работает обучение на POP909 и обратный перенос? | 8 |
| [Temporal / C3 / attention / relation gate](#c3) | Что меняют точные временные события и варианты обмена сообщениями? | 43 |
| [Fusion / task attention / MMoE](#fusion) | Помогает ли обмен между теоретическими задачами? | 30 |
| [C0 / C1 / C2 и замороженный перенос](#baselines) | Какие контроли и преобразования данных поддержаны сохранёнными сравнениями? | 13 |
| [Самообучение и контекст](#ssl) | Работают ли маскирование и перенос представлений без утечки целей? | 10 |
| [CPU / GPU / облако и производительность](#runtime) | Какие среды, ресурсы и способы исполнения были допущены? | 60 |
| [Обзоры и сверка сохранённых результатов](#reporting) | Какие выводы можно восстановить из уже сохранённых свидетельств? | 4 |
| [Требуют классификации](#unclassified) | Какие новые семейства ещё не включены в ручную карту? | 0 |

<a id="hcm"></a>

## PLL / HCM: условная вероятность музыкальных событий

Как использовать локальный и полный контекст?

[HCM-1](../HCM1.md), [полный контекст](../HCM_WHOLE.md). NLL/PLL, свободная генерация и человеческие предпочтения — разные проверки.

**Сохранённый итог и решение:** Сохранены запуски HCM-1/whole и development-проверки HCM-2; [отчёт сравнения](HCM2_RANDOM_3090_VALIDATION_20261001_RU.md) описывает разницу NLL на фиксированных масках, с различиями в истории обучения. Новый baseline этим обзором не выбирается.

**Открытый вопрос:** Польза полного контекста/памяти и качество свободной генерации требуют отдельных сопоставимых проверок.

<details>
<summary>Попытки и свидетельства: 45</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-HCM-WHOLE-3090-GATE-001](records/EXP-HCM-WHOLE-3090-GATE-001.json) | failed | engineering_gate | RTX3090 whole-song CPU CUDA parity, memory and resume gate |
| [EXP-HCM-WHOLE-3090-GATE-002](records/EXP-HCM-WHOLE-3090-GATE-002.json) | completed | engineering_gate | RTX3090 whole-song CPU CUDA parity, memory and resume gate |
| [EXP-HCM-WHOLE-3090-TRAIN-001](records/EXP-HCM-WHOLE-3090-TRAIN-001.json) | completed | training | Whole-song HCM pilot1000 updates on RTX3090 |
| [EXP-HCM-WHOLE-COVERAGE-001](records/EXP-HCM-WHOLE-COVERAGE-001.json) | completed | data_preparation | Screen complete TRAIN source geometry before whole native gate |
| [EXP-HCM-WHOLE-PREPARE-001](records/EXP-HCM-WHOLE-PREPARE-001.json) | completed | data_preparation | Prepare complete POP909-CL TRAIN songs for whole-context launch |
| [EXP-HCM1-3090-EVAL-001](records/EXP-HCM1-3090-EVAL-001.json) | completed | evaluation | Fixed small HCM10000 restoration and common-context authored evaluation |
| [EXP-HCM1-3090-EVAL-002](records/EXP-HCM1-3090-EVAL-002.json) | completed | evaluation | Expanded fixed development evaluation of HCM at10000updates |
| [EXP-HCM1-3090-EVAL-003](records/EXP-HCM1-3090-EVAL-003.json) | completed | evaluation | Expanded fixed development evaluation of HCM at20000updates |
| [EXP-HCM1-3090-GATE-001](records/EXP-HCM1-3090-GATE-001.json) | completed | engineering_gate | Full-state native migration and next-draw replay on3090 |
| [EXP-HCM1-3090-TRAIN-001](records/EXP-HCM1-3090-TRAIN-001.json) | completed | training | HCM second10000-update schedule segment on3090 |
| [EXP-HCM1-DIAGNOSTIC-PREPARE-001](records/EXP-HCM1-DIAGNOSTIC-PREPARE-001.json) | failed | data_preparation | Fixed12 validation works,12 TRAIN comparison windows and40 authored development pairs |
| [EXP-HCM1-DIAGNOSTIC-PREPARE-002](records/EXP-HCM1-DIAGNOSTIC-PREPARE-002.json) | completed | data_preparation | Fixed12 validation works,12 TRAIN comparison windows and40 authored development pairs |
| [EXP-HCM1-DIAGNOSTIC-PREPARE-003](records/EXP-HCM1-DIAGNOSTIC-PREPARE-003.json) | completed | data_preparation | Fixed12 validation works,12 TRAIN comparison windows and40 authored development pairs |
| [EXP-HCM1-DIAGNOSTIC-PREPARE-004](records/EXP-HCM1-DIAGNOSTIC-PREPARE-004.json) | completed | preparation | Expand head-dev to120 pairs and24 corpus validation works |
| [EXP-HCM1-HUMAN-RATINGS-IMPORT-001](records/EXP-HCM1-HUMAN-RATINGS-IMPORT-001.json) | completed | result_import | First user listening ratings bound to HCM10000 audio and scores |
| [EXP-HCM1-HUMAN-RATINGS-IMPORT-002](records/EXP-HCM1-HUMAN-RATINGS-IMPORT-002.json) | failed | result_import | Second human listening export: deletion bias and fixed 10000/20000 score comparison |
| [EXP-HCM1-HUMAN-RATINGS-IMPORT-003](records/EXP-HCM1-HUMAN-RATINGS-IMPORT-003.json) | completed | result_import | Second human listening export: deletion bias and fixed 10000/20000 score comparison |
| [EXP-HCM1-LISTENING-PAGE-001](records/EXP-HCM1-LISTENING-PAGE-001.json) | completed | preparation | Interactive listening page bound to HCM checkpoint10000 diagnostic |
| [EXP-HCM1-LISTENING-PAGE-002](records/EXP-HCM1-LISTENING-PAGE-002.json) | completed | preparation | Listening page with readable example labels and generated examples first |
| [EXP-HCM1-LISTENING-PAGE-003](records/EXP-HCM1-LISTENING-PAGE-003.json) | completed | preparation | Expanded listening page with preserved human ratings and paired10000/20000 scores |
| [EXP-HCM1-LISTENING-PREPARE-001](records/EXP-HCM1-LISTENING-PREPARE-001.json) | completed | preparation | Fixed original/edit audio playback admission before HCM evaluation |
| [EXP-HCM1-SOURCES-PREPARE-001](records/EXP-HCM1-SOURCES-PREPARE-001.json) | failed | data_preparation | Prepare and admit TRAIN source windows for HCM-1 T4 pilot |
| [EXP-HCM1-SOURCES-PREPARE-002](records/EXP-HCM1-SOURCES-PREPARE-002.json) | invalid | data_preparation | Prepare HCM TRAIN windows after full-visible graph sanitization fix |
| [EXP-HCM1-SOURCES-PREPARE-003](records/EXP-HCM1-SOURCES-PREPARE-003.json) | completed | data_preparation | HCM TRAIN source preparation with native and project split intersection |
| [EXP-HCM1-T4-GATE-001](records/EXP-HCM1-T4-GATE-001.json) | completed | engineering_gate | Native HCM-1 frozen-C3 CPU/CUDA, memory and checkpoint-resume gate on preemptible T4 |
| [EXP-HCM1-T4-TRAIN-001](records/EXP-HCM1-T4-TRAIN-001.json) | completed | training | HCM-1 initial frozen-C3 conditional reconstruction pilot on preemptible T4 |
| [EXP-HCM2-3090-CONTINUE-001](records/EXP-HCM2-3090-CONTINUE-001.json) | completed | training | Continue full-state HCM2 from1000 to10000 on unchanged POP songs |
| [EXP-HCM2-3090-CONTINUE-002](records/EXP-HCM2-3090-CONTINUE-002.json) | completed | result_import | Import of the primary terminal record for the HCM-2 continuation, restoring the verified data-access observations |
| [EXP-HCM2-3090-CONTINUE-003](records/EXP-HCM2-3090-CONTINUE-003.json) | completed | training | Full-state HCM2 continuation from 10000 to 20000 updates on RTX 3090 |
| [EXP-HCM2-3090-HUMAN-DEV-EVAL-001](records/EXP-HCM2-3090-HUMAN-DEV-EVAL-001.json) | completed | evaluation | HCM2 at 20000 versus retained HCM1 scores on all 62 human-rated pairs |
| [EXP-HCM2-3090-VAL20K-001](records/EXP-HCM2-3090-VAL20K-001.json) | completed | evaluation | Paired 10000 versus 20000 HCM2 development validation on RTX 3090 |
| [EXP-HCM2-3090-VAL20K-CORRECTION-001](records/EXP-HCM2-3090-VAL20K-CORRECTION-001.json) | completed | saved_metrics_analysis | Exact invocation provenance correction for 3090 validation |
| [EXP-HCM2-3090-VALCURVE-001](records/EXP-HCM2-3090-VALCURVE-001.json) | completed | evaluation | Development validation curve over all whole-song HCM-2 checkpoints |
| [EXP-HCM2-3090-VALCURVE-002](records/EXP-HCM2-3090-VALCURVE-002.json) | completed | evaluation | Equivalence and speed verification of the cached-encoder sharded curve harness |
| [EXP-HCM2-3090-VALCURVE-003](records/EXP-HCM2-3090-VALCURVE-003.json) | completed | saved_metrics_analysis | Correction of the VALCURVE-002 control mislabeling and restatement of its equivalence evidence |
| [EXP-HCM2-PILOT-DIAGNOSTIC-001](records/EXP-HCM2-PILOT-DIAGNOSTIC-001.json) | failed | evaluation | Matched full-song development evaluation and motif null diagnosis |
| [EXP-HCM2-PILOT-DIAGNOSTIC-002](records/EXP-HCM2-PILOT-DIAGNOSTIC-002.json) | completed | evaluation | Matched full-song development evaluation and motif null diagnosis |
| [EXP-HCM2-RANDOM-3090-VAL10K-001](records/EXP-HCM2-RANDOM-3090-VAL10K-001.json) | failed | evaluation | Cloud random C3 HCM2 10000 fixed-mask development validation on RTX 3090 |
| [EXP-HCM2-RANDOM-3090-VAL10K-002](records/EXP-HCM2-RANDOM-3090-VAL10K-002.json) | completed | evaluation | Retry cloud random HCM2 10000 fixed-mask validation on RTX 3090 |
| [EXP-HCM2-RANDOM-T4-GATE-001](records/EXP-HCM2-RANDOM-T4-GATE-001.json) | completed | engineering_gate | Random trainable C3 and fresh whole-song HCM2 native T4 admission |
| [EXP-HCM2-RANDOM-T4-TRAIN-001](records/EXP-HCM2-RANDOM-T4-TRAIN-001.json) | failed | training | Full random C3 and HCM2 trained together from scratch for10000updates |
| [EXP-HCM2-RANDOM-T4-TRAIN-002](records/EXP-HCM2-RANDOM-T4-TRAIN-002.json) | aborted | training | Full random C3 and HCM2 trained together from scratch for10000updates |
| [EXP-HCM2-RANDOM-T4-TRAIN-003](records/EXP-HCM2-RANDOM-T4-TRAIN-003.json) | aborted | training | Resume random C3 HCM2 from1750 to10000 after provider preemption |
| [EXP-HCM2-RANDOM-T4-TRAIN-004](records/EXP-HCM2-RANDOM-T4-TRAIN-004.json) | aborted | training | Resume random C3 HCM2 from 5750 to 10000 after second provider preemption |
| [EXP-HCM2-RANDOM-T4-TRAIN-005](records/EXP-HCM2-RANDOM-T4-TRAIN-005.json) | completed | training | Resume random C3 HCM2 from 9750 to 10000 after provider preemption |

</details>

<a id="critic"></a>

## Парный и аспектный критик

Можно ли обучить сравнение музыкальных вариантов на frozen C3?

[Контракт критика](../CRITIC_V1.md). Искусственные правки и авторские примеры не заменяют независимую проверку человеческих предпочтений.

**Сохранённый итог и решение:** [Отчёт перехода TRAIN002](CRITIC_TRAIN002_TRANSITION_20260927_RU.md) сохраняет родительский checkpoint и отдельное продолжение с оптимизацией. Прежние попытки не заменены новой записью.

**Открытый вопрос:** Обобщение сравнений на независимые человеческие предпочтения остаётся отдельной проверкой; статус живого обучения здесь неизвестен.

<details>
<summary>Попытки и свидетельства: 18</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-CRITIC-AUTHORED-BACKBONE-IMPORT-001](records/EXP-CRITIC-AUTHORED-BACKBONE-IMPORT-001.json) | completed | artifact_import | Retrieve selected C3 checkpoint from retained Object Storage without starting a VM |
| [EXP-CRITIC-AUTHORED-CONSTANTS-GATE-001](records/EXP-CRITIC-AUTHORED-CONSTANTS-GATE-001.json) | failed | engineering_gate | Strict native check of packed geometry constants against source004 |
| [EXP-CRITIC-AUTHORED-CONSTANTS-GATE-002](records/EXP-CRITIC-AUTHORED-CONSTANTS-GATE-002.json) | completed | engineering_gate | Strict native check of packed geometry constants against source004 |
| [EXP-CRITIC-AUTHORED-CURRICULUM-GATE-001](records/EXP-CRITIC-AUTHORED-CURRICULUM-GATE-001.json) | failed | engineering_gate | Native RTX3090 gate for full-piece authored curriculum and frozen C3 critic |
| [EXP-CRITIC-AUTHORED-CURRICULUM-GATE-002](records/EXP-CRITIC-AUTHORED-CURRICULUM-GATE-002.json) | aborted | engineering_gate | Native RTX3090 gate with deterministic runtime and separate encoder numeric tolerance |
| [EXP-CRITIC-AUTHORED-CURRICULUM-GATE-003](records/EXP-CRITIC-AUTHORED-CURRICULUM-GATE-003.json) | completed | engineering_gate | RTX3090 gate for exact geometry preparation and evaluation reuse |
| [EXP-CRITIC-AUTHORED-CURRICULUM-GATE-004](records/EXP-CRITIC-AUTHORED-CURRICULUM-GATE-004.json) | failed | engineering_gate | RTX3090 gate for row batching against pinned serial source004 reference |
| [EXP-CRITIC-AUTHORED-CURRICULUM-PREPARE-001](records/EXP-CRITIC-AUTHORED-CURRICULUM-PREPARE-001.json) | failed | data_preparation | New authored45 campaign curriculum admission and pair preparation |
| [EXP-CRITIC-AUTHORED-CURRICULUM-PREPARE-002](records/EXP-CRITIC-AUTHORED-CURRICULUM-PREPARE-002.json) | invalid | data_preparation | New authored45 campaign curriculum admission and pair preparation |
| [EXP-CRITIC-AUTHORED-CURRICULUM-PREPARE-003](records/EXP-CRITIC-AUTHORED-CURRICULUM-PREPARE-003.json) | completed | data_preparation | New authored45 campaign curriculum admission and pair preparation |
| [EXP-CRITIC-AUTHORED-CURRICULUM-PREPARE-004](records/EXP-CRITIC-AUTHORED-CURRICULUM-PREPARE-004.json) | completed | data_preparation | Prepare bounded source-bound authored curriculum pair bundle and joint-review dossiers |
| [EXP-CRITIC-AUTHORED-CURRICULUM-TRAIN-001](records/EXP-CRITIC-AUTHORED-CURRICULUM-TRAIN-001.json) | aborted | training | Fresh source004 critic seed17 with explicit seven-day cap amendment on RTX3090 |
| [EXP-CRITIC-AUTHORED-CURRICULUM-TRAIN-002](records/EXP-CRITIC-AUTHORED-CURRICULUM-TRAIN-002.json) | aborted | training | Authored curriculum continuation from update605 using exact constant-packing runtime |
| [EXP-CRITIC-AUTHORED-PARITY-DIAG-001](records/EXP-CRITIC-AUTHORED-PARITY-DIAG-001.json) | completed | engineering_diagnostic | Localize FP32 single-versus-batch discrepancy before authored critic training |
| [EXP-CRITIC-C3G-GATE-001](records/EXP-CRITIC-C3G-GATE-001.json) | completed | engineering_gate | Native frozen C3 critic: full-bank feature extraction and CUDA continuation/throughput admission |
| [EXP-CRITIC-C3G-PREPARE-001](records/EXP-CRITIC-C3G-PREPARE-001.json) | failed | data_preparation | Frozen C3 x3 relationgate critic: eligible TRAIN intervention bank and checkpoint admission |
| [EXP-CRITIC-C3G-PREPARE-002](records/EXP-CRITIC-C3G-PREPARE-002.json) | completed | data_preparation | Frozen C3 critic TRAIN bank: manifest-bound missing embedded split admission |
| [EXP-CRITIC-C3G-TRAIN-001](records/EXP-CRITIC-C3G-TRAIN-001.json) | aborted | critic_training | Frozen C3 x3 relationgate multiscale critic: bounded four-aspect TRAIN run |

</details>

<a id="annotation"></a>

## Корпус и теоретическая разметка

Какие источники, события и теоретические поля подтверждены?

[Данные](../DATASETS.md), [контракт](../DATA_CONTRACT.md). Полнота проверки, полнота меток и семантическое качество учитываются отдельно; неизвестное сохраняет маску.

**Сохранённый итог и решение:** Сохранены source-bound native/weak слои и отдельные reviews/releases. [Контракт](../DATA_CONTRACT.md) различает полную проверку произведения, известные targets и corpus-quota admission; частичное принятие не становится полной разметкой.

**Открытый вопрос:** Остаются семантические пробелы и непринятые попытки; рабочие статусы и маски должны сопровождать каждый release.

<details>
<summary>Попытки и свидетельства: 124</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-CHANGES-ALIGNMENT-001](records/EXP-CHANGES-ALIGNMENT-001.json) | completed | target_alignment_audit | DLC changes raw-candidate alignment and transposition gate |
| [EXP-CHANGES-ALIGNMENT-002](records/EXP-CHANGES-ALIGNMENT-002.json) | completed | target_alignment_audit | DLC changes alignment with conservative notation provenance gate |
| [EXP-CHANGES-ALIGNMENT-FULLTRAIN-001](records/EXP-CHANGES-ALIGNMENT-FULLTRAIN-001.json) | completed | target_alignment_audit | Full manifest-bound TRAIN changes alignment and cleaned class support |
| [EXP-CHANGES-TEMPORAL-PREPARE-001](records/EXP-CHANGES-TEMPORAL-PREPARE-001.json) | completed | target_alignment_audit | Authenticate TRAIN and prepare frozen-vocabulary VALIDATION temporal changes |
| [EXP-CHANGES-TEMPORAL-ULTRA-AUDIT-001](records/EXP-CHANGES-TEMPORAL-ULTRA-AUDIT-001.json) | completed | target_alignment_audit | Independent ultra repeat audit and readiness remediation of DLC changes |
| [EXP-CHANGES-TRAIN-CENSUS-001](records/EXP-CHANGES-TRAIN-CENSUS-001.json) | completed | source_audit | DLC TRAIN changes vocabulary and source-event census |
| [EXP-CHANGES-TRAIN-TARGETS-001](records/EXP-CHANGES-TRAIN-TARGETS-001.json) | failed | target_preparation | DLC TRAIN source-native changes target sidecars |
| [EXP-CHANGES-TRAIN-TARGETS-002](records/EXP-CHANGES-TRAIN-TARGETS-002.json) | completed | target_preparation | DLC TRAIN source-native changes target sidecars |
| [EXP-CORPUS400-ANNOTATION-001](records/EXP-CORPUS400-ANNOTATION-001.json) | completed | whole_work_astra_annotation | Author and independently review the first8 complete-event musical analyses |
| [EXP-CORPUS400-ANNOTATION-002](records/EXP-CORPUS400-ANNOTATION-002.json) | aborted | whole_work_astra_annotation | Author32 further full-event multigenre analyses under explicit scope accounting |
| [EXP-CORPUS400-ANNOTATION-003](records/EXP-CORPUS400-ANNOTATION-003.json) | completed | whole_work_astra_annotation | Continue interrupted32-work annotation and assemble verified prior reviews |
| [EXP-CORPUS400-ANNOTATION-004](records/EXP-CORPUS400-ANNOTATION-004.json) | completed | whole_work_astra_annotation | Author and independently review64new work families from authenticated TRAIN versions |
| [EXP-CORPUS400-ANNOTATION-005](records/EXP-CORPUS400-ANNOTATION-005.json) | completed | data_preparation | Bounded continuation: four new TRAIN families with individual seven-layer analysis and review |
| [EXP-CORPUS400-CANONICAL-REPAIR-001](records/EXP-CORPUS400-CANONICAL-REPAIR-001.json) | completed | data_preparation | Source-grounded canonical repair and complete-basis review of Clara Schumann Op23No2 |
| [EXP-CORPUS400-CATALOG-001](records/EXP-CORPUS400-CATALOG-001.json) | completed | metadata_candidate_selection | Select 400 full-work candidates across four genres |
| [EXP-CORPUS400-COMPLETION-001](records/EXP-CORPUS400-COMPLETION-001.json) | completed | data_preparation | Substantive completion of four previously accepted complete-source musical works |
| [EXP-CORPUS400-COMPLETION-002](records/EXP-CORPUS400-COMPLETION-002.json) | completed | data_preparation | Preparation and resolution enrichment of three existing works before expanding the 75-work corpus |
| [EXP-CORPUS400-COMPLETION-003](records/EXP-CORPUS400-COMPLETION-003.json) | completed | data_preparation | Deepen Bach Suite6 Prelude, Festive Minor and POP909-294 before corpus expansion |
| [EXP-CORPUS400-COMPLETION-004](records/EXP-CORPUS400-COMPLETION-004.json) | invalid | data_preparation | Deepen Mendelssohn Op12 I, ELO Dont Bring Me Down and POP909-057 before corpus expansion |
| [EXP-CORPUS400-COMPLETION-005](records/EXP-CORPUS400-COMPLETION-005.json) | completed | data_preparation | Remediate source-read scope; review retained ELO and resume Mendelssohn; recreate POP057 |
| [EXP-CORPUS400-COMPLETION-006](records/EXP-CORPUS400-COMPLETION-006.json) | aborted | data_preparation | Deepen existing Clara Op23No2, Beyond the Sea and original POP810 |
| [EXP-CORPUS400-COMPLETION-007](records/EXP-CORPUS400-COMPLETION-007.json) | completed | data_preparation | Deepen existing Clara Op23No2, Beyond the Sea and original POP810 |
| [EXP-CORPUS400-COMPLETION-008](records/EXP-CORPUS400-COMPLETION-008.json) | completed | data_preparation | Deepen existing Schulhoff Stomp, Miss Anns Tempo and original POP466 |
| [EXP-CORPUS400-COMPLETION-009](records/EXP-CORPUS400-COMPLETION-009.json) | completed | data_preparation | Deepen existing Hunnybee, exact Liszt Suisse family and original POP090 |
| [EXP-CORPUS400-COMPLETION-010](records/EXP-CORPUS400-COMPLETION-010.json) | completed | data_preparation | Deepen three explicitly selected existing accepted works after COMPLETION-009 |
| [EXP-CORPUS400-COMPLETION-011](records/EXP-CORPUS400-COMPLETION-011.json) | completed | data_preparation | Deepen three explicitly selected existing accepted works after COMPLETION-010 |
| [EXP-CORPUS400-COMPLETION-012](records/EXP-CORPUS400-COMPLETION-012.json) | completed | data_preparation | Deepen three explicitly selected existing accepted works after COMPLETION-011 |
| [EXP-CORPUS400-COMPLETION-013](records/EXP-CORPUS400-COMPLETION-013.json) | failed | data_preparation | Deepen three explicitly selected existing accepted works after COMPLETION-012 |
| [EXP-CORPUS400-COMPLETION-014](records/EXP-CORPUS400-COMPLETION-014.json) | aborted | data_preparation | Sequential whole-version deep annotation: Chasing Cars after sparse accepted pass011 |
| [EXP-CORPUS400-COMPLETION-015](records/EXP-CORPUS400-COMPLETION-015.json) | running | data_preparation | Ten-work detailed enrichment on GPT-6.1 Sol high with three-worker rolling concurrency |
| [EXP-CORPUS400-COMPLETION-015-RELEASE001](records/EXP-CORPUS400-COMPLETION-015-RELEASE001.json) | completed | data_preparation | COMPLETION015 incremental local release001: eight independently accepted weak annotations |
| [EXP-CORPUS400-COMPLETION-015-RELEASE002](records/EXP-CORPUS400-COMPLETION-015-RELEASE002.json) | completed | data_preparation | COMPLETION015 incremental local release002: four independently accepted weak additions |
| [EXP-CORPUS400-COMPLETION-015-RELEASE003](records/EXP-CORPUS400-COMPLETION-015-RELEASE003.json) | completed | data_preparation | COMPLETION015 incremental local release003: eight independently accepted weak additions |
| [EXP-CORPUS400-COMPLETION-015-RELEASE004](records/EXP-CORPUS400-COMPLETION-015-RELEASE004.json) | running | data_preparation | COMPLETION015 incremental local release004: four independently accepted bounded weak views |
| [EXP-CORPUS400-COMPLETION-015-WORK007](records/EXP-CORPUS400-COMPLETION-015-WORK007.json) | completed | data_preparation | COMPLETION015 work007: Johann Sebastian Bach — 2. Fuga / Sonata No. 3 in C major |
| [EXP-CORPUS400-COMPLETION-015-WORK008](records/EXP-CORPUS400-COMPLETION-015-WORK008.json) | completed | data_preparation | COMPLETION015 work008: Edvard Grieg — Lyric Pieces / Lyric Pieces, Op. 65 |
| [EXP-CORPUS400-COMPLETION-015-WORK009](records/EXP-CORPUS400-COMPLETION-015-WORK009.json) | completed | data_preparation | COMPLETION015 work009: The Coral — Dreaming of You |
| [EXP-CORPUS400-COMPLETION-015-WORK010](records/EXP-CORPUS400-COMPLETION-015-WORK010.json) | completed | data_preparation | COMPLETION015 work010: Ai Mini — 遇 |
| [EXP-CORPUS400-COMPLETION-015-WORK011](records/EXP-CORPUS400-COMPLETION-015-WORK011.json) | completed | data_preparation | COMPLETION015 continuous work011: Nirvana — Smells Like Teen Spirit (The Gallows arrangement) |
| [EXP-CORPUS400-COMPLETION-015-WORK012](records/EXP-CORPUS400-COMPLETION-015-WORK012.json) | completed | data_preparation | COMPLETION015 continuous work012: Vince Guaraldi — Linus & Lucy |
| [EXP-CORPUS400-COMPLETION-015-WORK013](records/EXP-CORPUS400-COMPLETION-015-WORK013.json) | completed | data_preparation | COMPLETION015 continuous work013: Johann Sebastian Bach — 7. Gigue / French Suite No. 4 |
| [EXP-CORPUS400-COMPLETION-015-WORK014](records/EXP-CORPUS400-COMPLETION-015-WORK014.json) | completed | data_preparation | COMPLETION015 continuous work014: DJ费玉清 — 一剪梅 DJ费玉清 |
| [EXP-CORPUS400-COMPLETION-015-WORK015](records/EXP-CORPUS400-COMPLETION-015-WORK015.json) | completed | data_preparation | COMPLETION015 continuous work015: The Beatles — Please Please Me |
| [EXP-CORPUS400-COMPLETION-015-WORK016](records/EXP-CORPUS400-COMPLETION-015-WORK016.json) | completed | data_preparation | COMPLETION015 continuous work016: François Couperin — <font size="13.5"/><font face="SnellRoundhand BlackScript"/>Concerts Royaux (1722) / Concert Royal no. 4 |
| [EXP-CORPUS400-COMPLETION-015-WORK017](records/EXP-CORPUS400-COMPLETION-015-WORK017.json) | completed | data_preparation | COMPLETION015 continuous work017: 万芳 — 新不了情 (POP909 485) — original MIDI version |
| [EXP-CORPUS400-COMPLETION-015-WORK018](records/EXP-CORPUS400-COMPLETION-015-WORK018.json) | completed | data_preparation | COMPLETION015 continuous work018: 孙燕姿 — 开始懂了 (POP909 354) — original MIDI version |
| [EXP-CORPUS400-COMPLETION-015-WORK019](records/EXP-CORPUS400-COMPLETION-015-WORK019.json) | completed | data_preparation | COMPLETION015 continuous work019: 刘德华 — 练习 (POP909 718) — original MIDI version |
| [EXP-CORPUS400-COMPLETION-015-WORK020](records/EXP-CORPUS400-COMPLETION-015-WORK020.json) | completed | data_preparation | COMPLETION015 continuous work020: Girolamo Frescobaldi — Bergamasca |
| [EXP-CORPUS400-COMPLETION-015-WORK020-METADATA001](records/EXP-CORPUS400-COMPLETION-015-WORK020-METADATA001.json) | completed | metadata_remediation | Bergamasca current author receipt consistency, metadata-only continuation |
| [EXP-CORPUS400-COMPLETION-015-WORK021](records/EXP-CORPUS400-COMPLETION-015-WORK021.json) | completed | data_preparation | COMPLETION015 continuous work021: 吴克群 — 为你写诗 (POP909 080) — original MIDI version |
| [EXP-CORPUS400-COMPLETION-015-WORK022](records/EXP-CORPUS400-COMPLETION-015-WORK022.json) | completed | data_preparation | COMPLETION015 continuous work022: 周杰伦 — 说好的幸福呢 (POP909 779) — original MIDI version |
| [EXP-CORPUS400-COMPLETION-015-WORK023](records/EXP-CORPUS400-COMPLETION-015-WORK023.json) | completed | data_preparation | COMPLETION015 continuous work023: Sara — 我的心好冷 |
| [EXP-CORPUS400-COMPLETION-015-WORK024](records/EXP-CORPUS400-COMPLETION-015-WORK024.json) | completed | data_preparation | COMPLETION015 continuous work024: 周传雄 — 青花 (POP909 862) — original MIDI version |
| [EXP-CORPUS400-COMPLETION-015-WORK025](records/EXP-CORPUS400-COMPLETION-015-WORK025.json) | completed | data_preparation | COMPLETION015 continuous work025: 任然 — 疑心病 |
| [EXP-CORPUS400-COMPLETION-015-WORK026](records/EXP-CORPUS400-COMPLETION-015-WORK026.json) | completed | data_preparation | COMPLETION015 continuous work026: Portugal. The Man; Czar Vergantinos — feel it still / Feel It Still |
| [EXP-CORPUS400-COMPLETION-015-WORK027](records/EXP-CORPUS400-COMPLETION-015-WORK027.json) | completed | data_preparation | COMPLETION015 continuous work027: S.H.E — 你曾是少年 (POP909 135) — original MIDI version |
| [EXP-CORPUS400-COMPLETION-015-WORK028](records/EXP-CORPUS400-COMPLETION-015-WORK028.json) | completed | data_preparation | COMPLETION015 continuous work028: The White Stripes — seven nation army / Seven Nation Army |
| [EXP-CORPUS400-COMPLETION-015-WORK029](records/EXP-CORPUS400-COMPLETION-015-WORK029.json) | completed | data_preparation | COMPLETION015 continuous work029: 刀郎 — 冲动的惩罚 (POP909 196) — original MIDI version |
| [EXP-CORPUS400-COMPLETION-015-WORK030](records/EXP-CORPUS400-COMPLETION-015-WORK030.json) | completed | data_preparation | COMPLETION015 continuous work030: Johann Christian Bach — Keyboard Sonata in A major, W.A 11 / 1 |
| [EXP-CORPUS400-COMPLETION-015-WORK031](records/EXP-CORPUS400-COMPLETION-015-WORK031.json) | completed | data_preparation | COMPLETION015 continuous work031: Carl Philipp Emanuel Bach — Sonata in A Major / Sonata in A Major, Wq.55/4 |
| [EXP-CORPUS400-COMPLETION-015-WORK032](records/EXP-CORPUS400-COMPLETION-015-WORK032.json) | completed | data_preparation | COMPLETION015 continuous work032: Steely Dan — Peg |
| [EXP-CORPUS400-COMPLETION-015-WORK033](records/EXP-CORPUS400-COMPLETION-015-WORK033.json) | completed | data_preparation | COMPLETION015 continuous work033: Juan Tizol — Perdido (full big-band arrangement) |
| [EXP-CORPUS400-COMPLETION-015-WORK034](records/EXP-CORPUS400-COMPLETION-015-WORK034.json) | completed | data_preparation | COMPLETION015 continuous work034: 任贤齐 — 伤心太平洋 |
| [EXP-CORPUS400-COMPLETION-015-WORK035](records/EXP-CORPUS400-COMPLETION-015-WORK035.json) | completed | data_preparation | COMPLETION015 continuous work035: 亦心Elissa — 走江南 |
| [EXP-CORPUS400-COMPLETION-015-WORK036](records/EXP-CORPUS400-COMPLETION-015-WORK036.json) | completed | data_preparation | COMPLETION015 continuous work036: 五月天 — 我不愿让你一个人 |
| [EXP-CORPUS400-COMPLETION-015-WORK037](records/EXP-CORPUS400-COMPLETION-015-WORK037.json) | running | data_preparation | COMPLETION015 continuous work037: 乌兰托娅 — 高原蓝 |
| [EXP-CORPUS400-COMPLETION-015-WORK038](records/EXP-CORPUS400-COMPLETION-015-WORK038.json) | completed | data_preparation | COMPLETION015 continuous work038: Tfboys — 青春修炼手册 |
| [EXP-CORPUS400-COMPLETION-015-WORK039](records/EXP-CORPUS400-COMPLETION-015-WORK039.json) | completed | data_preparation | COMPLETION015 continuous work039: Panic! At the Disco; Panic! at the Disco — the ballad of mona lisa / The Ballad of Mona Lisa |
| [EXP-CORPUS400-COMPLETION-015-WORK040](records/EXP-CORPUS400-COMPLETION-015-WORK040.json) | completed | data_preparation | COMPLETION015 continuous work040: Grant Green; Grant Green — Miss Ann's Tempo / Miss Ann's Tempo |
| [EXP-CORPUS400-COMPLETION-015-WORK041](records/EXP-CORPUS400-COMPLETION-015-WORK041.json) | completed | data_preparation | COMPLETION015 continuous work041: Francois Couperin; François Couperin — Fuguéte / Nouveau Concert, ou Les Goûts-Réunis no. 14 |
| [EXP-CORPUS400-COMPLETION-015-WORK042](records/EXP-CORPUS400-COMPLETION-015-WORK042.json) | completed | data_preparation | COMPLETION015 continuous work042: Radiohead — creep / Creep |
| [EXP-CORPUS400-COMPLETION-015-WORK043](records/EXP-CORPUS400-COMPLETION-015-WORK043.json) | completed | data_preparation | COMPLETION015 continuous work043: Erwin Schulhoff — Suite dansante en jazz / Suite dansante en jazz, WV 98 |
| [EXP-CORPUS400-COMPLETION-015-WORK044](records/EXP-CORPUS400-COMPLETION-015-WORK044.json) | completed | data_preparation | COMPLETION015 continuous work044: Claudio Monteverdi — O primavera gioventù del anno, SV68 |
| [EXP-CORPUS400-COMPLETION-015-WORK045](records/EXP-CORPUS400-COMPLETION-015-WORK045.json) | completed | data_preparation | COMPLETION015 continuous work045: 吴静 — 女儿情 |
| [EXP-CORPUS400-COMPLETION-015-WORK046](records/EXP-CORPUS400-COMPLETION-015-WORK046.json) | completed | data_preparation | COMPLETION015 continuous work046: Johann Sebastian Bach — 1. Prelude / Suite No. 6 in D major |
| [EXP-CORPUS400-COMPLETION-015-WORK047](records/EXP-CORPUS400-COMPLETION-015-WORK047.json) | running | data_preparation | COMPLETION015 continuous work047: Domenico Scarlatti — Sonata K.1 |
| [EXP-CORPUS400-COMPLETION-015-WORK048](records/EXP-CORPUS400-COMPLETION-015-WORK048.json) | completed | data_preparation | COMPLETION015 continuous work048: 云飞 — 云在飞 |
| [EXP-CORPUS400-COMPLETION-015-WORK049](records/EXP-CORPUS400-COMPLETION-015-WORK049.json) | running | data_preparation | COMPLETION015 continuous work049: T.R.Y — 不是因为寂寞才想你 |
| [EXP-CORPUS400-COMPLETION-015-WORK050](records/EXP-CORPUS400-COMPLETION-015-WORK050.json) | completed | data_preparation | COMPLETION015 continuous work050: Clara Schumann — An einem lichten Morgen, Op.23 No.2 |
| [EXP-CORPUS400-COMPLETION-015-WORK051](records/EXP-CORPUS400-COMPLETION-015-WORK051.json) | running | data_preparation | COMPLETION015 continuous work051: 丁当 — 手掌心 |
| [EXP-CORPUS400-COMPLETION-015-WORK052](records/EXP-CORPUS400-COMPLETION-015-WORK052.json) | planned | data_preparation | COMPLETION015 continuous work052: Gerry Mulligan; Gerry MulliganArr. R. Morris — Festive Minor / Festive Minor Quartet |
| [EXP-CORPUS400-COMPLETION-015-WORK053](records/EXP-CORPUS400-COMPLETION-015-WORK053.json) | running | data_preparation | COMPLETION015 continuous work053: Frédéric Chopin — Mazurka, Op.50 No.2 |
| [EXP-CORPUS400-COMPLETION-015-WORK054](records/EXP-CORPUS400-COMPLETION-015-WORK054.json) | running | data_preparation | COMPLETION015 continuous work054: Fats Waller — Honeysuckle Rose |
| [EXP-CORPUS400-COMPLETION-015-WORK055](records/EXP-CORPUS400-COMPLETION-015-WORK055.json) | completed | data_preparation | COMPLETION015 continuous work055: Fool's Garden; arr. Teddy — lemon tree / Lemon Tree |
| [EXP-CORPUS400-COMPLETION-015-WORK056](records/EXP-CORPUS400-COMPLETION-015-WORK056.json) | running | data_preparation | COMPLETION015 continuous work056: Gala — 追梦赤子心 |
| [EXP-CORPUS400-COMPLETION-015-WORK057](records/EXP-CORPUS400-COMPLETION-015-WORK057.json) | planned | data_preparation | COMPLETION015 continuous work057: Unknown Mortal Orchestra; Unknown Mortal Orchestra — hunnybee / Hunnybee Final |
| [EXP-CORPUS400-COMPLETION-015-WORK058](records/EXP-CORPUS400-COMPLETION-015-WORK058.json) | running | data_preparation | COMPLETION015 continuous work058: Felix Mendelssohn — String Quartet Op.12: I |
| [EXP-CORPUS400-COMPLETION-015-WORK059](records/EXP-CORPUS400-COMPLETION-015-WORK059.json) | completed | data_preparation | COMPLETION015 continuous work059: Franz Liszt — Années de Pèlerinage, Première année: Suisse / Années de Pèlerinage, Première année: Suisse, S.160 |
| [EXP-CORPUS400-COMPLETION-015-WORK060](records/EXP-CORPUS400-COMPLETION-015-WORK060.json) | completed | data_preparation | COMPLETION015 continuous work060: Imagine Dragons — Radioactive (instrumental arrangement) |
| [EXP-CORPUS400-COMPLETION-015-WORK061](records/EXP-CORPUS400-COMPLETION-015-WORK061.json) | running | data_preparation | COMPLETION015 continuous work061: Robbie Williams; English: Robbie WilliamsFrançais: Charles Tenet — beyond the sea / Beyond the Sea :: La Mer |
| [EXP-CORPUS400-COMPLETION-015-WORK062](records/EXP-CORPUS400-COMPLETION-015-WORK062.json) | planned | data_preparation | COMPLETION015 continuous work062: Electric Light Orchestra — Don't Bring Me Down / Don't Bring Me Down |
| [EXP-CORPUS400-COMPLETION-015-WORK063](records/EXP-CORPUS400-COMPLETION-015-WORK063.json) | running | data_preparation | COMPLETION015 continuous work063: SHE — 他还是不懂 |
| [EXP-CORPUS400-COMPLETION-015-WORK064](records/EXP-CORPUS400-COMPLETION-015-WORK064.json) | running | data_preparation | COMPLETION015 continuous work064: 云菲菲 — 殇雪 |
| [EXP-CORPUS400-COMPLETION-015-WORK065](records/EXP-CORPUS400-COMPLETION-015-WORK065.json) | completed | data_preparation | COMPLETION015 continuous work065: Archie Bleyer; Archie Bleyer Arr. Dan Wattis — Hernando's Hideaway / Hernando's Hideaway - Archie Bleyer |
| [EXP-CORPUS400-COMPLETION-015-WORK066](records/EXP-CORPUS400-COMPLETION-015-WORK066.json) | planned | data_preparation | COMPLETION015 continuous work066: Ignaz Pleyel — String Quartet in A major / String Quartet in A major, Op. 2, No. 1, B.307 |
| [EXP-CORPUS400-COMPLETION-015-WORK067](records/EXP-CORPUS400-COMPLETION-015-WORK067.json) | planned | data_preparation | COMPLETION015 continuous work067: Aki阿杰 — 东风志 |
| [EXP-CORPUS400-COMPLETION-SOURCE-001](records/EXP-CORPUS400-COMPLETION-SOURCE-001.json) | completed | source_import | Acquire pinned Mendelssohn Op.12 first-movement score for existing TRAIN annotation |
| [EXP-CORPUS400-MATERIALIZE-001](records/EXP-CORPUS400-MATERIALIZE-001.json) | completed | corpus400_materialization | Acquire and materialize 400 full-file candidates with exact event mapping |
| [EXP-CORPUS400-MATERIALIZE-002](records/EXP-CORPUS400-MATERIALIZE-002.json) | completed | corpus400_materialization | Recover exact PDMX numeric representations and materialize distinct reserve works |
| [EXP-CORPUS400-MATERIALIZE-003](records/EXP-CORPUS400-MATERIALIZE-003.json) | completed | corpus400_materialization | Materialize screened reserves and additional creator-linked pop and rock works |
| [EXP-CORPUS400-MATERIALIZE-004](records/EXP-CORPUS400-MATERIALIZE-004.json) | completed | corpus400_materialization | Preserve100 original multitrack POP909 alternative versions |
| [EXP-CORPUS400-METADATA009-INCIDENT-001](records/EXP-CORPUS400-METADATA009-INCIDENT-001.json) | invalid | data_preparation | Quarantined unplanned009 metadata-only source-selection preparation |
| [EXP-CORPUS400-PACKETS-001](records/EXP-CORPUS400-PACKETS-001.json) | completed | corpus400_analysis_packetization | Build complete exact-event analysis packets for443 materialized candidates |
| [EXP-CORPUS400-PACKETS-002](records/EXP-CORPUS400-PACKETS-002.json) | completed | corpus400_analysis_packetization | Prepare exact-event packets for200 additional and original source versions |
| [EXP-CORPUS400-PREPLAN010-001](records/EXP-CORPUS400-PREPLAN010-001.json) | invalid | data_preparation | COMPLETION010 metadata preplan rejected absolute temporary provenance locators |
| [EXP-DENSE-HARMONIC-TARGETS-001](records/EXP-DENSE-HARMONIC-TARGETS-001.json) | completed | dense_source_harmonic_target_materialization | Complete-note harmonic interval targets on three admitted TRAIN works |
| [EXP-HARMONIC-DATA-AUDIT-001](records/EXP-HARMONIC-DATA-AUDIT-001.json) | completed | source_audit | Source format feasibility for a harmonic rule analyzer |
| [EXP-HT-POP-PDMX-OVERLAP-001](records/EXP-HT-POP-PDMX-OVERLAP-001.json) | completed | metadata_overlap_audit | HookTheory POP909 PDMX metadata overlap candidates |
| [EXP-JAZZ-CORPUS-OVERLAP-001](records/EXP-JAZZ-CORPUS-OVERLAP-001.json) | invalid | metadata_overlap_audit | Jazz analysis datasets versus four local music catalogs |
| [EXP-JAZZ-CORPUS-OVERLAP-002](records/EXP-JAZZ-CORPUS-OVERLAP-002.json) | completed | metadata_overlap_audit | Jazz catalog overlap from verified identity projections after acquisition incident |
| [EXP-JHT-HT-PDMX-DEMO-002](records/EXP-JHT-HT-PDMX-DEMO-002.json) | completed | retained_evidence_registration | Validated registration of Saints enrichment evidence after malformed terminal serialization |
| [EXP-MULTIGENRE-THEORY-ENRICHMENT-001](records/EXP-MULTIGENRE-THEORY-ENRICHMENT-001.json) | completed | multigenre_auxiliary_target_materialization | Autonomous event-bound harmonic enrichment across admitted TRAIN corpora |
| [EXP-RAW-TIME-COVERAGE-001](records/EXP-RAW-TIME-COVERAGE-001.json) | completed | diagnostic | TRAIN-only raw-time readout candidate coverage |
| [EXP-RAW-TIME-COVERAGE-002](records/EXP-RAW-TIME-COVERAGE-002.json) | completed | diagnostic | Raw-time plus legacy beat fallback and fixed-subdivision coverage |
| [EXP-RESEARCH-DILEMMA-BINDINGS-001](records/EXP-RESEARCH-DILEMMA-BINDINGS-001.json) | completed | engineering_gate | Dilemma frozen split and existing TRAIN VALIDATION cache integrity |
| [EXP-RESEARCH-DILEMMA-METADATA-002](records/EXP-RESEARCH-DILEMMA-METADATA-002.json) | completed | engineering_gate | Supplemental target-free B3 candidate piece mapping |
| [EXP-SAINTS-HARMONIC-PROCESS-001](records/EXP-SAINTS-HARMONIC-PROCESS-001.json) | completed | bounded_harmonic_interpretation_demo | Expose harmonic process and hierarchical relations in admitted Saints JHT example |
| [EXP-TARGET-SLOT-CENSUS-001](records/EXP-TARGET-SLOT-CENSUS-001.json) | failed | diagnostic | TRAIN-only exact target-to-output slot conflict census |
| [EXP-TARGET-SLOT-CENSUS-002](records/EXP-TARGET-SLOT-CENSUS-002.json) | completed | diagnostic | TRAIN-only exact target-to-output slot conflict census |
| [EXP-TONAL-RELATIONS-ALIGNMENT-001](records/EXP-TONAL-RELATIONS-ALIGNMENT-001.json) | invalid | source_alignment_feasibility | Four-genre published analysis to symbolic events pilot |
| [EXP-TONAL-RELATIONS-ALIGNMENT-002](records/EXP-TONAL-RELATIONS-ALIGNMENT-002.json) | completed | source_alignment_feasibility_remediation | Revalidate four admitted musical analysis cases after shared-page incident |
| [EXP-TONAL-RELATIONS-ASTRA-PILOT-001](records/EXP-TONAL-RELATIONS-ASTRA-PILOT-001.json) | completed | bounded_astra_assisted_relations_annotation | Source-version relation episodes with Astra annotations and simple-rule diagnostics |
| [EXP-TONAL-RELATIONS-MAP-001](records/EXP-TONAL-RELATIONS-MAP-001.json) | completed | research_relation_map_engineering | Build inspectable multilevel relation maps from admitted source events |
| [EXP-TONAL-RELATIONS-TARGETS-001](records/EXP-TONAL-RELATIONS-TARGETS-001.json) | completed | canonical_weak_target_materialization | Astra participant roles bound to canonical TRAIN notes and loadable target bundles |

</details>

<a id="authoring"></a>

## Авторские произведения и музыкальные вмешательства

Какие контролируемые примеры и изменения пригодны для исследования?

[Итог авторской кампании](LLM_GENRE_CAMPAIGN_FINAL_40_RU.md). Успех построения/рендера не равен музыкальному принятию.

**Сохранённый итог и решение:** [Итог кампании](LLM_GENRE_CAMPAIGN_FINAL_40_RU.md) сохраняет авторские произведения и их проверки. Контролируемые [вмешательства](MUSIC_INTERVENTION_PILOT_RU.md) исследуются отдельно.

**Открытый вопрос:** Готовый файл, renderer acceptance и человеческая оценка не взаимозаменяемы; перенос на реальные произведения требует проверки.

<details>
<summary>Попытки и свидетельства: 18</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-GENRE-INTERVENTION-SUITE-001](records/EXP-GENRE-INTERVENTION-SUITE-001.json) | completed | controlled_intervention_pilot | Twenty real compositions across four genres with multi-aspect listening interventions |
| [EXP-HARMONIC-REPLACEMENT-PILOT-001](records/EXP-HARMONIC-REPLACEMENT-PILOT-001.json) | completed | controlled_intervention_pilot | Contextual chord and resolution replacements in three TRAIN excerpts |
| [EXP-LLM-GENRE-AUTHORING-001](records/EXP-LLM-GENRE-AUTHORING-001.json) | completed | controlled_intervention_pilot | Five genres: composition-specific LLM analysis and deliberate single/combined edits |
| [EXP-LLM-GENRE-AUTHORING-002](records/EXP-LLM-GENRE-AUTHORING-002.json) | completed | controlled_intervention_pilot | Expand five-work contextual edits and reviewed compositions |
| [EXP-LLM-GENRE-AUTHORING-003](records/EXP-LLM-GENRE-AUTHORING-003.json) | completed | controlled_intervention_pilot | Five existing works: five harmony melody rhythm candidates each |
| [EXP-LLM-GENRE-AUTHORING-004](records/EXP-LLM-GENRE-AUTHORING-004.json) | completed | controlled_intervention_pilot | Five new composition candidates: source admission then contextual authoring |
| [EXP-LLM-GENRE-AUTHORING-005](records/EXP-LLM-GENRE-AUTHORING-005.json) | completed | controlled_intervention_pilot | Five further distinct works: contextual authoring after source admission |
| [EXP-LLM-GENRE-AUTHORING-006](records/EXP-LLM-GENRE-AUTHORING-006.json) | completed | controlled_intervention_pilot | Five new contextual works toward20pergenre |
| [EXP-LLM-GENRE-AUTHORING-007](records/EXP-LLM-GENRE-AUTHORING-007.json) | completed | controlled_intervention_pilot | Five new contextual works toward20pergenre |
| [EXP-LLM-GENRE-AUTHORING-008](records/EXP-LLM-GENRE-AUTHORING-008.json) | completed | controlled_intervention_pilot | Sixth five-work batch toward20pergenre |
| [EXP-LLM-GENRE-AUTHORING-009](records/EXP-LLM-GENRE-AUTHORING-009.json) | completed | controlled_intervention_pilot | Seventh five-work batch toward20pergenre |
| [EXP-LLM-GENRE-AUTHORING-009-RENDER-001](records/EXP-LLM-GENRE-AUTHORING-009-RENDER-001.json) | failed | controlled_intervention_pilot | 009 diagnostic whole-work rendering attempt 001 |
| [EXP-LLM-GENRE-AUTHORING-009-RENDER-002](records/EXP-LLM-GENRE-AUTHORING-009-RENDER-002.json) | completed | controlled_intervention_pilot | 009 diagnostic whole-work rendering attempt 002 |
| [EXP-LLM-GENRE-AUTHORING-010](records/EXP-LLM-GENRE-AUTHORING-010.json) | completed | controlled_intervention_pilot | Eighth five-work batch toward20pergenre |
| [EXP-LLM-GENRE-AUTHORING-011](records/EXP-LLM-GENRE-AUTHORING-011.json) | completed | controlled_intervention_pilot | Ninth five-work batch toward20pergenre |
| [EXP-LLM-GENRE-AUTHORING-012](records/EXP-LLM-GENRE-AUTHORING-012.json) | aborted | controlled_intervention_pilot | Tenth five-work batch toward20pergenre |
| [EXP-LLM-GENRE-CAMPAIGN-001](records/EXP-LLM-GENRE-CAMPAIGN-001.json) | aborted | controlled_intervention_pilot | 500-work contextual intervention campaign |
| [EXP-MUSIC-INTERVENTION-PILOT-001](records/EXP-MUSIC-INTERVENTION-PILOT-001.json) | completed | controlled_intervention_pilot | Three TRAIN works: composable local musical interventions |

</details>

<a id="c4"></a>

## C4: POP909 и перенос между корпусами

Как работает обучение на POP909 и обратный перенос?

[Обучение C4](C4_POP909_3090_20261005_RU.md), [перенос](C4_DILEMMA_TRANSFER_20261005_RU.md). Сравнения сохраняют источник, split и конкретные задачи.

**Сохранённый итог и решение:** Сохранены [обучение на POP909](C4_POP909_3090_20261005_RU.md) и [обратный перенос](C4_DILEMMA_TRANSFER_20261005_RU.md). Разница между корпусами не служит выбором универсальной лучшей модели.

**Открытый вопрос:** Причины разницы переноса и качество на других источниках требуют отдельных контролей.

<details>
<summary>Попытки и свидетельства: 8</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-C4-DILEMMA-TRANSFER-001](records/EXP-C4-DILEMMA-TRANSFER-001.json) | completed | evaluation | C4 POP-trained encoder: Dilemmadata VALIDATION reverse transfer |
| [EXP-C4-POP909-3090-80K-001](records/EXP-C4-POP909-3090-80K-001.json) | completed | training | C4: fresh POP909-CL training, RTX3090 B4 W8 80k |
| [EXP-C4-POP909-3090-GATE-001](records/EXP-C4-POP909-3090-GATE-001.json) | completed | engineering_gate | C4 RTX3090 FP32 pipeline8 all-shift and resume admission |
| [EXP-C4-POP909-PREPARE-001](records/EXP-C4-POP909-PREPARE-001.json) | failed | engineering_gate | C4 POP909-CL TRAIN/VALIDATION preparation for RTX3090 |
| [EXP-C4-POP909-PREPARE-002](records/EXP-C4-POP909-PREPARE-002.json) | completed | engineering_gate | C4 POP909-CL verified TRAIN/VALIDATION source packaging |
| [EXP-C4-POP909-PREPARE-003](records/EXP-C4-POP909-PREPARE-003.json) | completed | engineering_gate | C4 native TRAIN target support and orbit preparation |
| [EXP-C4-POP909-PREPARE-RECEIPT-001](records/EXP-C4-POP909-PREPARE-RECEIPT-001.json) | completed | engineering_gate | C4 preparation receipt integrity correction |
| [EXP-C4-TRANSFER-DIAGNOSIS-001](records/EXP-C4-TRANSFER-DIAGNOSIS-001.json) | planned | diagnostic | Why C4 POP to Dilemmadata transfer degrades: retained-prediction audit |

</details>

<a id="c3"></a>

## Temporal / C3 / attention / relation gate

Что меняют точные временные события и варианты обмена сообщениями?

[Temporal](../TEMPORAL_SLICE_GRAPH.md), [обзор C3](C3_THREEPASS_FINAL_RU.md), [POP909](C3_POP909_FULL_20261001_RU.md). Исторические бюджеты и batch size различаются; это не единый рейтинг архитектур.

**Сохранённый итог и решение:** [Итог C3](C3_THREEPASS_FINAL_RU.md) описывает неодинаковые изменения по головам и разные batch/draw budgets; primary selection в этом отчёте сохраняется. [POP909-проверка](C3_POP909_FULL_20261001_RU.md) оформлена отдельно.

**Открытый вопрос:** Причинный эффект внимания/gate при равном бюджете и перенос между источниками не выводятся из общего списка запусков.

<details>
<summary>Попытки и свидетельства: 43</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-3090-C3-RELATIONGATE-CHANGES-80K-001](records/EXP-3090-C3-RELATIONGATE-CHANGES-80K-001.json) | planned | training | C3 relationgate plus changes from scratch: 80k, RTX3090 B2 W4 |
| [EXP-3090-C3-RELATIONGATE-CHANGES-GATE-001](records/EXP-3090-C3-RELATIONGATE-CHANGES-GATE-001.json) | planned | engineering_gate | C3 relationgate plus changes from scratch: gate, RTX3090 B2 W4 |
| [EXP-3090-C3-RELATIONGATE-CHANGES-MEMORY-001](records/EXP-3090-C3-RELATIONGATE-CHANGES-MEMORY-001.json) | planned | engineering_gate | C3 relationgate plus changes from scratch: memory, RTX3090 B2 W4 |
| [EXP-3090-C3-RELATIONGATE-CHANGES-PREPARE-001](records/EXP-3090-C3-RELATIONGATE-CHANGES-PREPARE-001.json) | completed | engineering_gate | C3 relationgate plus changes from scratch: prepare, RTX3090 B2 W4 |
| [EXP-C3-THREEPASS-FINAL-REVIEW-001](records/EXP-C3-THREEPASS-FINAL-REVIEW-001.json) | completed | result_import | Final threepass and relation-gate comparison plus supervisor figures |
| [EXP-C3-THREEPASS-FINAL-REVIEW-002](records/EXP-C3-THREEPASS-FINAL-REVIEW-002.json) | completed | result_import | Artifact-binding correction superseding the C3 threepass and relation-gate final review record |
| [EXP-C3-THREEPASS-HEADS-REVIEW-001](records/EXP-C3-THREEPASS-HEADS-REVIEW-001.json) | completed | saved_metrics_analysis | Per-task interpretation of threepass versus relation gate at60k |
| [EXP-C3-THREEPASS-PROGRESS-REVIEW-003](records/EXP-C3-THREEPASS-PROGRESS-REVIEW-003.json) | completed | progress_review | Read-only current threepass/gate progress, quality and elapsed time |
| [EXP-CLOUD-T4-C3-ATTENTION-80K-001](records/EXP-CLOUD-T4-C3-ATTENTION-80K-001.json) | aborted | training | C3 attention: train, batch4, eight workers |
| [EXP-CLOUD-T4-C3-ATTENTION-80K-EVIDENCE-001](records/EXP-CLOUD-T4-C3-ATTENTION-80K-EVIDENCE-001.json) | completed | result_import | Artifact binding for the aborted C3 attention 80k record, supplying the hash-bound primary evidence its closure relied on |
| [EXP-CLOUD-T4-C3-ATTENTION-GATE-001](records/EXP-CLOUD-T4-C3-ATTENTION-GATE-001.json) | completed | engineering_gate | C3 attention: gate, batch4, eight workers |
| [EXP-CLOUD-T4-C3-ATTENTION-MEMORY-001](records/EXP-CLOUD-T4-C3-ATTENTION-MEMORY-001.json) | completed | engineering_gate | C3 attention: memory, batch4, eight workers |
| [EXP-CLOUD-T4-C3-CONTROL-80K-001](records/EXP-CLOUD-T4-C3-CONTROL-80K-001.json) | aborted | training | C3 control: train, batch4, eight workers |
| [EXP-CLOUD-T4-C3-CONTROL-80K-EVIDENCE-001](records/EXP-CLOUD-T4-C3-CONTROL-80K-EVIDENCE-001.json) | completed | result_import | Artifact binding for the aborted C3 control 80k record, supplying the hash-bound primary evidence its closure relied on |
| [EXP-CLOUD-T4-C3-CONTROL-GATE-001](records/EXP-CLOUD-T4-C3-CONTROL-GATE-001.json) | completed | engineering_gate | C3 control: gate, batch4, eight workers |
| [EXP-CLOUD-T4-C3-CONTROL-MEMORY-001](records/EXP-CLOUD-T4-C3-CONTROL-MEMORY-001.json) | completed | engineering_gate | C3 control: memory, batch4, eight workers |
| [EXP-CLOUD-T4-C3-RELATIONGATE-80K-001](records/EXP-CLOUD-T4-C3-RELATIONGATE-80K-001.json) | completed | training | C3 relationgate: train, batch4, eight workers |
| [EXP-CLOUD-T4-C3-RELATIONGATE-GATE-001](records/EXP-CLOUD-T4-C3-RELATIONGATE-GATE-001.json) | completed | engineering_gate | C3 relationgate: gate, batch4, eight workers |
| [EXP-CLOUD-T4-C3-RELATIONGATE-MEMORY-001](records/EXP-CLOUD-T4-C3-RELATIONGATE-MEMORY-001.json) | failed | engineering_gate | C3 relationgate: memory, batch4, eight workers |
| [EXP-CLOUD-T4-C3-RELATIONGATE-MEMORY-002](records/EXP-CLOUD-T4-C3-RELATIONGATE-MEMORY-002.json) | completed | engineering_gate | C3 relationgate: memory, batch4, eight workers; retry after host IPC lifetime correction |
| [EXP-CLOUD-T4-C3-THREEPASS-80K-001](records/EXP-CLOUD-T4-C3-THREEPASS-80K-001.json) | aborted | training | C3 threepass: train, batch4, eight workers |
| [EXP-CLOUD-T4-C3-THREEPASS-80K-EVIDENCE-001](records/EXP-CLOUD-T4-C3-THREEPASS-80K-EVIDENCE-001.json) | completed | result_import | Artifact binding for the aborted C3 threepass 80k record, supplying the hash-bound primary evidence its closure relied on |
| [EXP-CLOUD-T4-C3-THREEPASS-GATE-001](records/EXP-CLOUD-T4-C3-THREEPASS-GATE-001.json) | completed | engineering_gate | C3 threepass: gate, batch4, eight workers |
| [EXP-CLOUD-T4-C3-THREEPASS-MEMORY-001](records/EXP-CLOUD-T4-C3-THREEPASS-MEMORY-001.json) | failed | engineering_gate | C3 threepass: memory, batch4, eight workers |
| [EXP-CLOUD-T4-C3-THREEPASS-MEMORY-002](records/EXP-CLOUD-T4-C3-THREEPASS-MEMORY-002.json) | completed | engineering_gate | C3 threepass: memory, batch4, eight workers; retry after host IPC lifetime correction |
| [EXP-POP-C3-FULL-EVAL-001](records/EXP-POP-C3-FULL-EVAL-001.json) | invalid | evaluation | Rejected C3 full evaluation startup after planning serialization failure |
| [EXP-POP-C3-FULL-EVAL-002](records/EXP-POP-C3-FULL-EVAL-002.json) | completed | evaluation | Frozen C3 relationgate 80000 on all 908 accepted POP909-CL songs |
| [EXP-POP-C3-FULL-INPUTS-001-ENCODING-001](records/EXP-POP-C3-FULL-INPUTS-001-ENCODING-001.json) | failed | data_preparation | Acquire all accepted pinned POP909-CL source MIDI for C3 frozen evaluation |
| [EXP-POP-C3-FULL-INPUTS-003](records/EXP-POP-C3-FULL-INPUTS-003.json) | completed | data_preparation | Acquire all accepted pinned POP909-CL source MIDI for C3 frozen evaluation |
| [EXP-TEMPORAL-C2-80K-001](records/EXP-TEMPORAL-C2-80K-001.json) | completed | training | Fresh transposed C2 with exact temporal slices to80k |
| [EXP-TEMPORAL-FINAL-001](records/EXP-TEMPORAL-FINAL-001.json) | failed | result_import | Temporal80k final producer review and next experiment assessment |
| [EXP-TEMPORAL-FINAL-002](records/EXP-TEMPORAL-FINAL-002.json) | completed | result_import | Temporal80k final review after SSH access recovery |
| [EXP-TEMPORAL-GPU-GATE-001](records/EXP-TEMPORAL-GPU-GATE-001.json) | failed | engineering_gate | Native temporal C2 acceleration and full-resume equivalence gate |
| [EXP-TEMPORAL-GPU-GATE-002](records/EXP-TEMPORAL-GPU-GATE-002.json) | completed | engineering_gate | Retry native temporal C2 equivalence/resume with primitive environment binding |
| [EXP-TEMPORAL-GPU-MEMORY-001](records/EXP-TEMPORAL-GPU-MEMORY-001.json) | failed | engineering_gate | Largest TRAIN temporal graphs native CUDA admission |
| [EXP-TEMPORAL-GPU-MEMORY-002](records/EXP-TEMPORAL-GPU-MEMORY-002.json) | completed | engineering_gate | Largest TRAIN temporal CUDA admission with verified desktop process exceptions |
| [EXP-TEMPORAL-LATENT-RESEARCH-001](records/EXP-TEMPORAL-LATENT-RESEARCH-001.json) | completed | result_import | Temporal latent refinement: attention, multiscale context and residual controls |
| [EXP-TEMPORAL-PROGRESS-001](records/EXP-TEMPORAL-PROGRESS-001.json) | completed | result_import | Temporal C2 early5k/10k comparison with historical C2 |
| [EXP-TEMPORAL-PROGRESS-002](records/EXP-TEMPORAL-PROGRESS-002.json) | completed | result_import | Current temporal C2 validation progress versus historical C2 |
| [EXP-TEMPORAL-PROGRESS-003](records/EXP-TEMPORAL-PROGRESS-003.json) | completed | result_import | Temporal65k progress versus historical C2 matched step,80k and120k |
| [EXP-TEMPORAL-SIZING-001](records/EXP-TEMPORAL-SIZING-001.json) | failed | engineering_gate | Full TRAIN temporal graph sizing before GPU admission |
| [EXP-TEMPORAL-SIZING-002](records/EXP-TEMPORAL-SIZING-002.json) | completed | engineering_gate | Retry full TRAIN temporal graph sizing with canonical planned-record filename |
| [EXP-TEMPORAL-SLICE-CPU-001](records/EXP-TEMPORAL-SLICE-CPU-001.json) | completed | diagnostic | Exact temporal graph C2: bounded TRAIN CPU forward/backward admission |

</details>

<a id="fusion"></a>

## Fusion / task attention / MMoE

Помогает ли обмен между теоретическими задачами?

[Карта исторических веток](../BRANCHES_AND_RELEASES.md). Незавершённые, остановленные и отрицательные попытки сохранены рядом с контролями.

**Сохранённый итог и решение:** Попытки fusion, task attention и MMoE сохранены вместе с остановками/контролями. [Решение интеграции](../BRANCHES_AND_RELEASES.md) исторически оставило temporal основой main, fusion — отдельной веткой.

**Открытый вопрос:** Совместимость и эффективность новых сочетаний с temporal требуют отдельной реализации и сравнения.

<details>
<summary>Попытки и свидетельства: 30</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-FUSION-ACCEL-C2-GATE-001](records/EXP-FUSION-ACCEL-C2-GATE-001.json) | failed | engineering_gate | Accelerated C2 fusion equivalence |
| [EXP-FUSION-ACCEL-C2-GATE-002](records/EXP-FUSION-ACCEL-C2-GATE-002.json) | completed | engineering_gate | Accelerated C2 fusion equivalence |
| [EXP-FUSION-C2-15K-COMPARISON-001](records/EXP-FUSION-C2-15K-COMPARISON-001.json) | completed | result_import | Fusion15k versus historicalC2 at15k: metrics and elapsed time |
| [EXP-FUSION-C2-80K-FINAL-001](records/EXP-FUSION-C2-80K-FINAL-001.json) | completed | result_import | Completed fusion80k versus historical C2 |
| [EXP-FUSION-C2-80K-PROGRESS-001](records/EXP-FUSION-C2-80K-PROGRESS-001.json) | completed | result_import | Fusion continuation progress versus historical C2 |
| [EXP-FUSION-C2-80K-PROGRESS-002](records/EXP-FUSION-C2-80K-PROGRESS-002.json) | completed | result_import | Text erratum for fusion interim report: four improved heads |
| [EXP-FUSION-C2-CONTINUE-80K-001](records/EXP-FUSION-C2-CONTINUE-80K-001.json) | completed | training | Continue accelerated C2 fusion from15000 to80000 updates |
| [EXP-FUSION-C2-SCRATCH-15K-001](records/EXP-FUSION-C2-SCRATCH-15K-001.json) | completed | training | From-scratch C2 with harmonic fusion, 15000 updates |
| [EXP-FUSION-CUDA-SYNTHETIC-001](records/EXP-FUSION-CUDA-SYNTHETIC-001.json) | failed | engineering_gate | Planned bounded synthetic CUDA fusion continuation and strict resume gate |
| [EXP-FUSION-CUDA-SYNTHETIC-002](records/EXP-FUSION-CUDA-SYNTHETIC-002.json) | completed | engineering_gate | Planned bounded synthetic CUDA fusion continuation and strict resume gate |
| [EXP-FUSION-P512-CONTROL-001](records/EXP-FUSION-P512-CONTROL-001.json) | aborted | paired_validation_comparison | Bound registered fusion P512 arm |
| [EXP-FUSION-P512-ENABLED-001](records/EXP-FUSION-P512-ENABLED-001.json) | aborted | paired_validation_comparison | Bound registered fusion P512 arm |
| [EXP-FUSION-P512-POSTHOC-001](records/EXP-FUSION-P512-POSTHOC-001.json) | completed | posthoc_evidence_verification | Bounded artifact-only verification of retained outcomes after fusion report abort |
| [EXP-FUSION-TRAIN-REFERENCE-001](records/EXP-FUSION-TRAIN-REFERENCE-001.json) | completed | engineering_gate | Bound registered fusion TRAIN reference smoke |
| [EXP-MMOE-C2-80K-001](records/EXP-MMOE-C2-80K-001.json) | aborted | training | Two-group MMoE plus attention from scratch80k |
| [EXP-MMOE-GATE-001](records/EXP-MMOE-GATE-001.json) | completed | engineering_gate | Two-group MMoE native equivalence gate |
| [EXP-MMOE-PROGRESS-001](records/EXP-MMOE-PROGRESS-001.json) | completed | result_import | MMoE progress from user supplied 16k stdout excerpt |
| [EXP-MMOE-PROGRESS-002](records/EXP-MMOE-PROGRESS-002.json) | completed | result_import | MMoE supplied-log progress to21k |
| [EXP-MMOE-PROGRESS-003](records/EXP-MMOE-PROGRESS-003.json) | completed | result_import | Live MMoE80k progress and continuation assessment |
| [EXP-MMOE-STOP-001](records/EXP-MMOE-STOP-001.json) | completed | engineering_gate | User-authorized early termination of MMoE with retained checkpoints |
| [EXP-TASKATTN-80K-REVIEW-001](records/EXP-TASKATTN-80K-REVIEW-001.json) | failed | result_import | Attention80k endpoint, same-runtime control status and failure-hypothesis review |
| [EXP-TASKATTN-80K-REVIEW-002](records/EXP-TASKATTN-80K-REVIEW-002.json) | completed | result_import | Attention80k endpoint, same-runtime control status and failure-hypothesis review |
| [EXP-TASKATTN-C2-80K-001](records/EXP-TASKATTN-C2-80K-001.json) | completed | training | train-attention official-style task attention controlled80k campaign |
| [EXP-TASKATTN-CONTROL-80K-001](records/EXP-TASKATTN-CONTROL-80K-001.json) | aborted | training | train-control official-style task attention controlled80k campaign |
| [EXP-TASKATTN-CONTROL-GATE-001](records/EXP-TASKATTN-CONTROL-GATE-001.json) | completed | engineering_gate | gate-control official-style task attention controlled80k campaign |
| [EXP-TASKATTN-CONTROL-STOP-001](records/EXP-TASKATTN-CONTROL-STOP-001.json) | completed | result_import | User-requested cancellation of redundant control80k |
| [EXP-TASKATTN-GATE-001](records/EXP-TASKATTN-GATE-001.json) | completed | engineering_gate | gate-attention official-style task attention controlled80k campaign |
| [EXP-TASKATTN-PROGRESS-001](records/EXP-TASKATTN-PROGRESS-001.json) | completed | result_import | Attention fusion progress versus historical C2 and linear fusion |
| [EXP-TASKATTN-PROGRESS-002](records/EXP-TASKATTN-PROGRESS-002.json) | completed | result_import | Attention fusion progress versus historical C2 and linear fusion |
| [EXP-TASKATTN-PROGRESS-003](records/EXP-TASKATTN-PROGRESS-003.json) | completed | result_import | Attention fusion progress versus historical C2 and linear fusion |

</details>

<a id="baselines"></a>

## C0 / C1 / C2 и замороженный перенос

Какие контроли и преобразования данных поддержаны сохранёнными сравнениями?

[Решения](../DECISIONS.md), [модели](../MODELS.md). Сопоставлять только совместимые данные, головы, seeds и бюджеты; baseline не выбирается этим каталогом.

**Сохранённый итог и решение:** [ADR-111](../DECISIONS.md#2026-09-02--adr-111-select-c0-and-defer-c1-after-the-paired-full-screen) фиксирует исторический выбор C0 и отсрочку C1 после paired screen; дальнейшая [интеграция temporal](../BRANCHES_AND_RELEASES.md) — отдельное решение.

**Открытый вопрос:** Нельзя переносить ранний выбор на другие данные, бюджет, seed или семейство HCM/critic без новой проверки.

<details>
<summary>Попытки и свидетельства: 13</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-9EB5D-C0-001](records/EXP-9EB5D-C0-001.json) | completed | scientific_training | C0 seed-17 no-transposition 10k screen |
| [EXP-9EB5D-C1-001](records/EXP-9EB5D-C1-001.json) | completed | scientific_training | C1 seed-17 stochastic-transposition 10k screen |
| [EXP-9EB5H-C2-001](records/EXP-9EB5H-C2-001.json) | completed | scientific_training | C2 seed-17 full-orbit 120k run |
| [EXP-9EB5K-C0-120K-001](records/EXP-9EB5K-C0-120K-001.json) | completed | scientific_training_control | C0 seed-17 compute-matched 120k control attempt 001 |
| [EXP-9EB5K-C0-C2-AUDIT-001](records/EXP-9EB5K-C0-C2-AUDIT-001.json) | completed | validation_analysis_import | Independent C0/C2 seed-17 120k evidence audit |
| [EXP-POP-TRANSFER-C0-001](records/EXP-POP-TRANSFER-C0-001.json) | completed | evaluation | Frozen C0 transfer to 101 POP909-CL VALIDATION pieces |
| [EXP-POP-TRANSFER-C2-001](records/EXP-POP-TRANSFER-C2-001.json) | completed | evaluation | Frozen C2 transfer to 101 POP909-CL VALIDATION pieces |
| [EXP-POP-TRANSFER-CLASS-DISTRIBUTION-001](records/EXP-POP-TRANSFER-CLASS-DISTRIBUTION-001.json) | completed | posthoc_analysis | Per-class recognition from saved frozen POP C0/C2 predictions |
| [EXP-POP-TRANSFER-EXPORT-001](records/EXP-POP-TRANSFER-EXPORT-001.json) | completed | engineering_gate | Extract unchanged best C0/C2 model-only weights through a restricted prefix |
| [EXP-POP-TRANSFER-INPUTS-001](records/EXP-POP-TRANSFER-INPUTS-001.json) | failed | engineering_gate | Verify best C0/C2 artifacts and allowlisted POP VALIDATION sources |
| [EXP-ROMAN-COMPOSITION-AUDIT-001](records/EXP-ROMAN-COMPOSITION-AUDIT-001.json) | completed | target_composition_audit | TRAIN-frozen Roman component reconstruction audit |
| [EXP-ROMAN-COMPOSITION-C2-EVAL-001](records/EXP-ROMAN-COMPOSITION-C2-EVAL-001.json) | completed | validation_inference | Frozen C2 paired simple numeral decoder evaluation |
| [PLAN-9EB5K-C0-120K](records/PLAN-9EB5K-C0-120K.json) | planned | scientific_training_control | Planned C0 seed-17 compute-matched 120k control |

</details>

<a id="ssl"></a>

## Самообучение и контекст

Работают ли маскирование и перенос представлений без утечки целей?

[SSL](../PHASE7A_SSL_BASELINE.md), [протокол сравнения](../PHASE8B2_COMPARISON_PROTOCOL.md). Реконструкция представлений не является PLL или оценкой эстетики.

**Сохранённый итог и решение:** [Контракт SSL](../PHASE7A_SSL_BASELINE.md) фиксирует механические границы маскирования, resume и encoder transfer. [Протокол сравнения](../PHASE8B2_COMPARISON_PROTOCOL.md) сохраняется отдельно от этих проверок.

**Открытый вопрос:** Польза представлений для downstream и музыкального качества не устанавливается прохождением механических tests/gates.

<details>
<summary>Попытки и свидетельства: 10</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-CTX-C2-ARTIFACT-001](records/EXP-CTX-C2-ARTIFACT-001.json) | failed | posthoc_analysis | Saved false-D7 and Dilemma regression context diagnostics |
| [EXP-CTX-C2-ARTIFACT-002](records/EXP-CTX-C2-ARTIFACT-002.json) | completed | posthoc_analysis | Saved false-D7 and Dilemma regression context diagnostics |
| [EXP-CTX-C2-LINUX-001](records/EXP-CTX-C2-LINUX-001.json) | completed | engineering_gate | Native Linux supervisor and frozen context source preflight |
| [EXP-CTX-C2-LOCAL-001](records/EXP-CTX-C2-LOCAL-001.json) | completed | diagnostic | Bounded Dilemma TRAIN-only nearest-centroid versus preselected frozen C2 |
| [EXP-CTX-C2-SUITE-GPU-001](records/EXP-CTX-C2-SUITE-GPU-001.json) | completed | diagnostic | Bounded frozen C2 synthetic context sensitivity GPU suite |
| [EXP-SSL-GROUP-CUDA-001](records/EXP-SSL-GROUP-CUDA-001.json) | failed | engineering_gate | Planned C2 to WS2 random-vs-beat masking CUDA fixture and resume gate |
| [EXP-SSL-GROUP-CUDA-002](records/EXP-SSL-GROUP-CUDA-002.json) | completed | engineering_gate | Planned retry of sealed review002 CUDA fixture with autocast API compatibility |
| [EXP-SSL-GROUP-SCIENTIFIC-PILOT-001](records/EXP-SSL-GROUP-SCIENTIFIC-PILOT-001.json) | completed | scientific_comparison | C2 paired independent-note versus beat-pitch pilot and theory recovery |
| [EXP-SSL-GROUP-SCIENTIFIC-PREPARE-001](records/EXP-SSL-GROUP-SCIENTIFIC-PREPARE-001.json) | failed | engineering_gate | Compile review-003 Dilemma paired masking inputs |
| [EXP-SSL-GROUP-SCIENTIFIC-PREPARE-002](records/EXP-SSL-GROUP-SCIENTIFIC-PREPARE-002.json) | completed | engineering_gate | Compile review-003 Dilemma paired masking inputs |

</details>

<a id="runtime"></a>

## CPU / GPU / облако и производительность

Какие среды, ресурсы и способы исполнения были допущены?

[Облачная история](../YANDEX_CLOUD_DEPLOYMENT.md). Инженерные gates и throughput не доказывают качество модели.

**Сохранённый итог и решение:** В [облачной истории](../YANDEX_CLOUD_DEPLOYMENT.md) сохранены загрузка/восстановление комплектов и проверки исполнения. CPU/GPU attempts и неудачные gates остаются в реестре.

**Открытый вопрос:** Для другой машины/версии необходим собственный admission; исторический gate не подтверждает текущее состояние устройства.

<details>
<summary>Попытки и свидетельства: 60</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-9EP1-B2-MEMORY-REPLAY-001](records/EXP-9EP1-B2-MEMORY-REPLAY-001.json) | completed | engineering_gate | Fresh-process B2 memory and update equality gate |
| [EXP-9EP1-C2-CUDA-EQUIVALENCE-001](records/EXP-9EP1-C2-CUDA-EQUIVALENCE-001.json) | completed | engineering_gate | C0 isolated performance replay (fixture) |
| [EXP-9EP1-CPU-FIXTURE-001](records/EXP-9EP1-CPU-FIXTURE-001.json) | completed | engineering_gate | C0 isolated performance replay (fixture) |
| [EXP-9EP1-DESKTOP-PREFLIGHT-001](records/EXP-9EP1-DESKTOP-PREFLIGHT-001.json) | completed | engineering_gate | Bounded idle desktop GPU observation for versioned screen launch policy |
| [EXP-9EP1-GPU-DIAGNOSTIC-001](records/EXP-9EP1-GPU-DIAGNOSTIC-001.json) | failed | engineering_gate | C0 isolated performance replay (production) |
| [EXP-9EP1-GPU-DIAGNOSTIC-002](records/EXP-9EP1-GPU-DIAGNOSTIC-002.json) | completed | engineering_gate | C0 isolated performance replay (production) |
| [EXP-9EP1-GPU-EVIDENCE-001](records/EXP-9EP1-GPU-EVIDENCE-001.json) | completed | engineering_gate | Import sealed first GPU queue evidence |
| [EXP-9EP1-GPU-EVIDENCE-002](records/EXP-9EP1-GPU-EVIDENCE-002.json) | completed | engineering_gate | Import sealed second GPU queue evidence |
| [EXP-9EP1-GPU-EVIDENCE-003](records/EXP-9EP1-GPU-EVIDENCE-003.json) | completed | engineering_gate | Import sealed update-only isolation evidence |
| [EXP-9EP1-GPU-EVIDENCE-004](records/EXP-9EP1-GPU-EVIDENCE-004.json) | completed | engineering_gate | Import sealed compact-cache B2 throughput evidence |
| [EXP-9EP1-GPU-EVIDENCE-005](records/EXP-9EP1-GPU-EVIDENCE-005.json) | completed | engineering_gate | Import sealed prefetch CUDA fixture and full TRAIN size index |
| [EXP-9EP1-GPU-EVIDENCE-006](records/EXP-9EP1-GPU-EVIDENCE-006.json) | completed | engineering_gate | Import exact screen-job serialization reservation evidence |
| [EXP-9EP1-GPU-EVIDENCE-007](records/EXP-9EP1-GPU-EVIDENCE-007.json) | completed | engineering_gate | Import historical C2 synthetic CUDA and fresh-runtime restoration gate |
| [EXP-9EP1-GPU-EVIDENCE-008](records/EXP-9EP1-GPU-EVIDENCE-008.json) | completed | engineering_gate | Import finite fresh-process TRAIN B2 memory and update equality gate |
| [EXP-9EP1-GPU-EVIDENCE-009](records/EXP-9EP1-GPU-EVIDENCE-009.json) | completed | engineering_gate | Import exact reservation remeasurement for worker runtime observation variant |
| [EXP-9EP1-GPU-EVIDENCE-010](records/EXP-9EP1-GPU-EVIDENCE-010.json) | aborted | engineering_gate | Import the fixed one-repeat 0/1/2/4 worker screen |
| [EXP-9EP1-GPU-EVIDENCE-011](records/EXP-9EP1-GPU-EVIDENCE-011.json) | completed | engineering_gate | Independent import of ordered worker screen with versioned desktop launch policy |
| [EXP-9EP1-GPU-EVIDENCE-012](records/EXP-9EP1-GPU-EVIDENCE-012.json) | completed | engineering_gate | Independent import of longer C0 prefetch reservation sizing |
| [EXP-9EP1-GPU-EVIDENCE-013](records/EXP-9EP1-GPU-EVIDENCE-013.json) | completed | engineering_gate | Independent import of integrated compact-cache sizing |
| [EXP-9EP1-GPU-EVIDENCE-014](records/EXP-9EP1-GPU-EVIDENCE-014.json) | completed | engineering_gate | Independent import of one integrated three-arm GPU comparison |
| [EXP-9EP1-GPU-EVIDENCE-015](records/EXP-9EP1-GPU-EVIDENCE-015.json) | completed | engineering_gate | Post-hoc engineering review of completed integrated replay after a false cuDNN assertion |
| [EXP-9EP1-GPU-FIXTURE-001](records/EXP-9EP1-GPU-FIXTURE-001.json) | completed | engineering_gate | C0 isolated performance replay (fixture) |
| [EXP-9EP1-GPU-THROUGHPUT-001](records/EXP-9EP1-GPU-THROUGHPUT-001.json) | completed | engineering_gate | C0 isolated performance replay (production) |
| [EXP-9EP1-GPU-THROUGHPUT-002](records/EXP-9EP1-GPU-THROUGHPUT-002.json) | completed | engineering_gate | C0 isolated performance replay (production) |
| [EXP-9EP1-GPU-UPDATE-ONLY-001](records/EXP-9EP1-GPU-UPDATE-ONLY-001.json) | completed | engineering_gate | C0 isolated performance replay (production) |
| [EXP-9EP1-PREFETCH-CUDA-FIXTURE-001](records/EXP-9EP1-PREFETCH-CUDA-FIXTURE-001.json) | completed | engineering_gate | C0 isolated performance replay (fixture) |
| [EXP-9EP1-PREFETCH-INTEGRATED-001](records/EXP-9EP1-PREFETCH-INTEGRATED-001.json) | completed | engineering_gate | C0 isolated performance replay (production) |
| [EXP-9EP1-PREFETCH-RESERVATIONS-001](records/EXP-9EP1-PREFETCH-RESERVATIONS-001.json) | completed | data_audit | Warm TRAIN serialized job reservation gate for B2 prefetch |
| [EXP-9EP1-PREFETCH-RESERVATIONS-002](records/EXP-9EP1-PREFETCH-RESERVATIONS-002.json) | completed | data_audit | Warm TRAIN serialized job reservation gate for B2 prefetch |
| [EXP-9EP1-PREFETCH-RESERVATIONS-003](records/EXP-9EP1-PREFETCH-RESERVATIONS-003.json) | completed | data_audit | Warm TRAIN serialized job reservation gate for B2 prefetch |
| [EXP-9EP1-PREFETCH-RESERVATIONS-004](records/EXP-9EP1-PREFETCH-RESERVATIONS-004.json) | completed | data_audit | Warm TRAIN serialized job reservation gate for B2 prefetch |
| [EXP-9EP1-PREFETCH-SCREEN-002](records/EXP-9EP1-PREFETCH-SCREEN-002.json) | completed | engineering_gate | C0 isolated performance replay (production) |
| [EXP-9EP1-PREFLIGHT-001](records/EXP-9EP1-PREFLIGHT-001.json) | completed | engineering_gate | Remote C0 directory and runtime preflight |
| [EXP-9EP1-RECOVERY-001](records/EXP-9EP1-RECOVERY-001.json) | completed | engineering_gate | Read-only host recovery and resource preflight after user resumption |
| [EXP-9EP1-TRAIN-MEMORY-INDEX-001](records/EXP-9EP1-TRAIN-MEMORY-INDEX-001.json) | completed | data_audit | Frozen TRAIN graph size index for a future B2 memory gate |
| [EXP-CLOUD-T4-15K-RESULT-IMPORT-001](records/EXP-CLOUD-T4-15K-RESULT-IMPORT-001.json) | completed | result_import | Cloud T4 temporal 15k: terminal evidence import and historical timing comparison |
| [EXP-CLOUD-T4-BATCH4-GATE-001](records/EXP-CLOUD-T4-BATCH4-GATE-001.json) | completed | engineering_gate | B4 selected runtime exact update and full resume gate |
| [EXP-CLOUD-T4-BATCH4-MEMORY-001](records/EXP-CLOUD-T4-BATCH4-MEMORY-001.json) | completed | engineering_gate | Largest selected B4 graph groups: native memory screen |
| [EXP-CLOUD-T4-BATCHSIZE-BENCH-001](records/EXP-CLOUD-T4-BATCHSIZE-BENCH-001.json) | completed | engineering_gate | Matched-draw B2 versus B4 throughput with eight workers |
| [EXP-CLOUD-T4-BATCHSIZE-RESULT-001](records/EXP-CLOUD-T4-BATCHSIZE-RESULT-001.json) | completed | result_import | B2/B4 matched-draw speed and admission verification |
| [EXP-CLOUD-T4-CPU-PIPELINE-BENCH-001](records/EXP-CLOUD-T4-CPU-PIPELINE-BENCH-001.json) | completed | engineering_gate | Exact CPU pipeline and four/eight worker comparison on T4 |
| [EXP-CLOUD-T4-CPU-PIPELINE-GATE-001](records/EXP-CLOUD-T4-CPU-PIPELINE-GATE-001.json) | completed | engineering_gate | Native coalesced temporal training with packed 8-worker pipeline: exact update and resume gate |
| [EXP-CLOUD-T4-CPU-PIPELINE-GATE-RESULT-001](records/EXP-CLOUD-T4-CPU-PIPELINE-GATE-RESULT-001.json) | completed | result_import | Independent import of native TRAIN pipeline admission gate |
| [EXP-CLOUD-T4-CPU-PIPELINE-RESULT-001](records/EXP-CLOUD-T4-CPU-PIPELINE-RESULT-001.json) | completed | result_import | CPU pipeline benchmark verification and speed comparison |
| [EXP-CLOUD-T4-CPU-PREP-DIAG-001](records/EXP-CLOUD-T4-CPU-PREP-DIAG-001.json) | completed | engineering_gate | Bounded CPU worker preparation profile on T4 host |
| [EXP-CLOUD-T4-FP32-PAIRED-BENCH-001](records/EXP-CLOUD-T4-FP32-PAIRED-BENCH-001.json) | completed | engineering_gate | T4 reference versus prepared FP32 paired speed benchmark |
| [EXP-CLOUD-T4-PREPARED-GATE-001](records/EXP-CLOUD-T4-PREPARED-GATE-001.json) | completed | engineering_gate | Prepared T4 exact equivalence and resume gate |
| [EXP-CLOUD-T4-PREPARED-MEMORY-001](records/EXP-CLOUD-T4-PREPARED-MEMORY-001.json) | completed | engineering_gate | Prepared T4 worst-size memory admission |
| [EXP-CLOUD-T4-PREPARED-RESULT-IMPORT-001](records/EXP-CLOUD-T4-PREPARED-RESULT-IMPORT-001.json) | completed | result_import | T4 prepared runtime: paired speed and exactness terminal import |
| [EXP-CLOUD-T4-SPEED-REVIEW-001](records/EXP-CLOUD-T4-SPEED-REVIEW-001.json) | completed | result_import | Cloud T4 versus historical RTX3090 temporal runtime review |
| [EXP-CLOUD-T4-TEMPORAL-15K-001](records/EXP-CLOUD-T4-TEMPORAL-15K-001.json) | completed | training | Cloud T4 temporal C2 from scratch to15k with four workers |
| [EXP-CLOUD-T4-TEMPORAL-GATE-001](records/EXP-CLOUD-T4-TEMPORAL-GATE-001.json) | aborted | diagnostic | T4 temporal acceleration and full-resume equivalence gate |
| [EXP-CLOUD-T4-TEMPORAL-GATE-002](records/EXP-CLOUD-T4-TEMPORAL-GATE-002.json) | completed | diagnostic | T4 temporal acceleration and full-resume equivalence gate |
| [EXP-CLOUD-T4-TEMPORAL-MEMORY-001](records/EXP-CLOUD-T4-TEMPORAL-MEMORY-001.json) | failed | diagnostic | T4 worst-size temporal FP32 capacity admission |
| [EXP-CLOUD-T4-TEMPORAL-MEMORY-002](records/EXP-CLOUD-T4-TEMPORAL-MEMORY-002.json) | completed | diagnostic | T4 worst-size temporal FP32 capacity admission |
| [EXP-HARDWARE-PROFILE-3090-001](records/EXP-HARDWARE-PROFILE-3090-001.json) | completed | engineering_gate | Reference temporal deployed hardware profile: 3090 |
| [EXP-HARDWARE-PROFILE-COMPARISON-001](records/EXP-HARDWARE-PROFILE-COMPARISON-001.json) | completed | result_import | Reference310 hardware profiles: terminal verification and stage comparison |
| [EXP-HARDWARE-PROFILE-T4-001](records/EXP-HARDWARE-PROFILE-T4-001.json) | completed | engineering_gate | Reference temporal deployed hardware profile: T4 |
| [EXP-PILOT-RAW-SIZING-001](records/EXP-PILOT-RAW-SIZING-001.json) | completed | engineering_gate | Raw-only pilot sizing with selected sidecar container counts |
| [EXP-PILOT-SUPERVISOR-LINUX-001](records/EXP-PILOT-SUPERVISOR-LINUX-001.json) | completed | engineering_gate | Native Linux acceptance of pilot time and memory wrapper |

</details>

<a id="reporting"></a>

## Обзоры и сверка сохранённых результатов

Какие выводы можно восстановить из уже сохранённых свидетельств?

[Полный ledger](../EXPERIMENT_LEDGER.md). Анализ сохранённых результатов не считается новым обучением.

**Сохранённый итог и решение:** Отдельные обзоры связывают сохранённые результаты с источниками. Их записи доступны ниже; они не заменяют исходные training/evaluation records.

**Открытый вопрос:** Недостающие первичные артефакты и несопоставимые условия сохраняются как ограничения; новых запусков этим индексом нет.

<details>
<summary>Попытки и свидетельства: 4</summary>

| Запись | Статус в реестре | Вид | Название |
|---|---|---|---|
| [EXP-ARCH-GRAPH-RESEARCH-001](records/EXP-ARCH-GRAPH-RESEARCH-001.json) | completed | result_import | Graph architecture information-flow research and source-free bottleneck diagnostic |
| [EXP-ARTICLE-PRECISION-RECALL-001](records/EXP-ARTICLE-PRECISION-RECALL-001.json) | completed | posthoc_analysis | Standalone precision and recall PDF from retained validation metrics |
| [EXP-ARTICLE-TASK-METRICS-001](records/EXP-ARTICLE-TASK-METRICS-001.json) | completed | posthoc_analysis | Per-task precision recall F1 and support from retained article experiments |
| [EXP-POSTBREAK-AUDIT-001](records/EXP-POSTBREAK-AUDIT-001.json) | completed | result_import | Independent post-break implementation and scientific-contract audit |

</details>

<a id="unclassified"></a>

## Требуют классификации

Какие новые семейства ещё не включены в ручную карту?

Записи сохраняются здесь автоматически; категория — навигация, а не научное заключение.

**Сохранённый итог и решение:** Автоматическая страховочная группа: ни одна неизвестная запись не исчезает из каталога.

**Открытый вопрос:** Добавить ручную карту нового семейства; научный итог до её составления неизвестен.

Записей нет.

## Все текстовые отчёты

<details>
<summary>Отчёты в этом каталоге: 103</summary>

- [C0_C2_120K_REVIEW.md](C0_C2_120K_REVIEW.md)
- [C3_ATTENTION_ARCHITECTURE_AUDIT_RU.md](C3_ATTENTION_ARCHITECTURE_AUDIT_RU.md)
- [C3_POP909_FULL_20261001_RU.md](C3_POP909_FULL_20261001_RU.md)
- [C3_RELATIONGATE_CHANGES_3090_PROTOCOL_RU.md](C3_RELATIONGATE_CHANGES_3090_PROTOCOL_RU.md)
- [C3_THREEPASS_FINAL_RU.md](C3_THREEPASS_FINAL_RU.md)
- [C3_THREEPASS_HEADS_60K_RU.md](C3_THREEPASS_HEADS_60K_RU.md)
- [C3_THREEPASS_PROGRESS_60K_RU.md](C3_THREEPASS_PROGRESS_60K_RU.md)
- [C4_DILEMMA_TRANSFER_20261005_RU.md](C4_DILEMMA_TRANSFER_20261005_RU.md)
- [C4_POP909_3090_20261005_RU.md](C4_POP909_3090_20261005_RU.md)
- [CLOUD_ACCELERATION_AND_ABLATIONS_RU.md](CLOUD_ACCELERATION_AND_ABLATIONS_RU.md)
- [CLOUD_BATCH4_RESULT_RU.md](CLOUD_BATCH4_RESULT_RU.md)
- [CLOUD_C3_PAIR_80K_PROTOCOL_RU.md](CLOUD_C3_PAIR_80K_PROTOCOL_RU.md)
- [CLOUD_C3_THREEPASS_PROTOCOL_RU.md](CLOUD_C3_THREEPASS_PROTOCOL_RU.md)
- [CLOUD_CPU_PIPELINE_RESULT_RU.md](CLOUD_CPU_PIPELINE_RESULT_RU.md)
- [CLOUD_T4_15K_FINAL_RU.md](CLOUD_T4_15K_FINAL_RU.md)
- [CLOUD_T4_PREPARED_BENCHMARK_RU.md](CLOUD_T4_PREPARED_BENCHMARK_RU.md)
- [CLOUD_T4_VS_3090_SPEED_RU.md](CLOUD_T4_VS_3090_SPEED_RU.md)
- [CRITIC_AUTHORED_CURRICULUM_20260927_RU.md](CRITIC_AUTHORED_CURRICULUM_20260927_RU.md)
- [CRITIC_C3G_CLOUD_PROTOCOL_RU.md](CRITIC_C3G_CLOUD_PROTOCOL_RU.md)
- [CRITIC_CONSTANT_PACKING_20260927_RU.md](CRITIC_CONSTANT_PACKING_20260927_RU.md)
- [CRITIC_TRAIN002_TRANSITION_20260927_RU.md](CRITIC_TRAIN002_TRANSITION_20260927_RU.md)
- [DENSE_HARMONIC_TARGETS_20261006_RU.md](DENSE_HARMONIC_TARGETS_20261006_RU.md)
- [DLC_CHANGES_ALIGNMENT_RU.md](DLC_CHANGES_ALIGNMENT_RU.md)
- [DLC_CHANGES_FULLTRAIN_RU.md](DLC_CHANGES_FULLTRAIN_RU.md)
- [DLC_CHANGES_TEMPORAL_READY_RU.md](DLC_CHANGES_TEMPORAL_READY_RU.md)
- [DLC_CHANGES_ULTRA_AUDIT_RU.md](DLC_CHANGES_ULTRA_AUDIT_RU.md)
- [FUSION_C2_15K_COMPARISON_RU.md](FUSION_C2_15K_COMPARISON_RU.md)
- [FUSION_C2_80K_FINAL_RU.md](FUSION_C2_80K_FINAL_RU.md)
- [FUSION_C2_80K_PROGRESS_RU.md](FUSION_C2_80K_PROGRESS_RU.md)
- [FUSION_C2_80K_PROGRESS_RU_CORRECTED.md](FUSION_C2_80K_PROGRESS_RU_CORRECTED.md)
- [GENRE_INTERVENTION_SUITE_RU.md](GENRE_INTERVENTION_SUITE_RU.md)
- [HARDWARE_PROFILE_T4_3090_RU.md](HARDWARE_PROFILE_T4_3090_RU.md)
- [HARMONIC_REPLACEMENT_PILOT_RU.md](HARMONIC_REPLACEMENT_PILOT_RU.md)
- [HCM1_10000_GENERALIZATION_20260928_RU.md](HCM1_10000_GENERALIZATION_20260928_RU.md)
- [HCM1_3090_HANDOFF_20260928_RU.md](HCM1_3090_HANDOFF_20260928_RU.md)
- [HCM1_EXPANDED_LISTENING_20260928_RU.md](HCM1_EXPANDED_LISTENING_20260928_RU.md)
- [HCM1_EXPANDED_RESULTS_20260928_RU.md](HCM1_EXPANDED_RESULTS_20260928_RU.md)
- [HCM1_T4_PILOT_20260928_RU.md](HCM1_T4_PILOT_20260928_RU.md)
- [HCM2_3090_20K_VALIDATION_20260930_RU.md](HCM2_3090_20K_VALIDATION_20260930_RU.md)
- [HCM2_HUMAN_COMPARISON_20261001_RU.md](HCM2_HUMAN_COMPARISON_20261001_RU.md)
- [HCM2_PILOT_DIAGNOSTIC_20260928_RU.md](HCM2_PILOT_DIAGNOSTIC_20260928_RU.md)
- [HCM2_RANDOM_3090_VALIDATION_20261001_RU.md](HCM2_RANDOM_3090_VALIDATION_20261001_RU.md)
- [HCM2_RANDOM_CLOUD_20260928_RU.md](HCM2_RANDOM_CLOUD_20260928_RU.md)
- [HCM2_RANDOM_RESUME_20260929_RU.md](HCM2_RANDOM_RESUME_20260929_RU.md)
- [HCM2_TO_10000_20260928_RU.md](HCM2_TO_10000_20260928_RU.md)
- [HCM2_VALIDATION_CURVE_20260929_RU.md](HCM2_VALIDATION_CURVE_20260929_RU.md)
- [HCM_WHOLE_3090_LAUNCH_20260928_RU.md](HCM_WHOLE_3090_LAUNCH_20260928_RU.md)
- [JAZZ_CORPUS_OVERLAP_20261007_RU.md](JAZZ_CORPUS_OVERLAP_20261007_RU.md)
- [JHT_HT_PDMX_DEMO_20261007_RU.md](JHT_HT_PDMX_DEMO_20261007_RU.md)
- [LLM_GENRE_AUTHORING_002_RU.md](LLM_GENRE_AUTHORING_002_RU.md)
- [LLM_GENRE_AUTHORING_003_RU.md](LLM_GENRE_AUTHORING_003_RU.md)
- [LLM_GENRE_AUTHORING_004_RU.md](LLM_GENRE_AUTHORING_004_RU.md)
- [LLM_GENRE_AUTHORING_005_RU.md](LLM_GENRE_AUTHORING_005_RU.md)
- [LLM_GENRE_AUTHORING_006_RU.md](LLM_GENRE_AUTHORING_006_RU.md)
- [LLM_GENRE_AUTHORING_007_RU.md](LLM_GENRE_AUTHORING_007_RU.md)
- [LLM_GENRE_AUTHORING_008_RU.md](LLM_GENRE_AUTHORING_008_RU.md)
- [LLM_GENRE_AUTHORING_009_RU.md](LLM_GENRE_AUTHORING_009_RU.md)
- [LLM_GENRE_AUTHORING_010_RU.md](LLM_GENRE_AUTHORING_010_RU.md)
- [LLM_GENRE_AUTHORING_011_RU.md](LLM_GENRE_AUTHORING_011_RU.md)
- [LLM_GENRE_AUTHORING_20260923_RU.md](LLM_GENRE_AUTHORING_20260923_RU.md)
- [LLM_GENRE_CAMPAIGN_FINAL_40_RU.md](LLM_GENRE_CAMPAIGN_FINAL_40_RU.md)
- [MMOE_PROGRESS_001_RU.md](MMOE_PROGRESS_001_RU.md)
- [MMOE_PROGRESS_002_RU.md](MMOE_PROGRESS_002_RU.md)
- [MMOE_PROGRESS_003_RU.md](MMOE_PROGRESS_003_RU.md)
- [MULTIGENRE_THEORY_ENRICHMENT_20261007_RU.md](MULTIGENRE_THEORY_ENRICHMENT_20261007_RU.md)
- [MUSIC_INTERVENTION_PILOT_RU.md](MUSIC_INTERVENTION_PILOT_RU.md)
- [PERFORMANCE_REPLAY_C2_RESTORATION_REVIEW.md](PERFORMANCE_REPLAY_C2_RESTORATION_REVIEW.md)
- [PERFORMANCE_REPLAY_COMPACT_CACHE_REVIEW.md](PERFORMANCE_REPLAY_COMPACT_CACHE_REVIEW.md)
- [PERFORMANCE_REPLAY_COMPACT_RESERVATION_REVIEW.md](PERFORMANCE_REPLAY_COMPACT_RESERVATION_REVIEW.md)
- [PERFORMANCE_REPLAY_GPU_REVIEW.md](PERFORMANCE_REPLAY_GPU_REVIEW.md)
- [PERFORMANCE_REPLAY_INTEGRATED_GATE_REJECTION.md](PERFORMANCE_REPLAY_INTEGRATED_GATE_REJECTION.md)
- [PERFORMANCE_REPLAY_INTEGRATED_POSTHOC_REVIEW.md](PERFORMANCE_REPLAY_INTEGRATED_POSTHOC_REVIEW.md)
- [PERFORMANCE_REPLAY_LONGER_RESERVATION_REVIEW.md](PERFORMANCE_REPLAY_LONGER_RESERVATION_REVIEW.md)
- [PERFORMANCE_REPLAY_NATIVE_MEMORY_REVIEW.md](PERFORMANCE_REPLAY_NATIVE_MEMORY_REVIEW.md)
- [PERFORMANCE_REPLAY_PREFETCH_FIXTURE_INDEX_REVIEW.md](PERFORMANCE_REPLAY_PREFETCH_FIXTURE_INDEX_REVIEW.md)
- [PERFORMANCE_REPLAY_RECOVERY_REVIEW.md](PERFORMANCE_REPLAY_RECOVERY_REVIEW.md)
- [PERFORMANCE_REPLAY_RESERVATION_REFRESH_REVIEW.md](PERFORMANCE_REPLAY_RESERVATION_REFRESH_REVIEW.md)
- [PERFORMANCE_REPLAY_RESERVATION_REVIEW.md](PERFORMANCE_REPLAY_RESERVATION_REVIEW.md)
- [PERFORMANCE_REPLAY_WORKER_SCREEN_REVIEW.md](PERFORMANCE_REPLAY_WORKER_SCREEN_REVIEW.md)
- [POP909_CL_C0_C2_FROZEN_REVIEW.md](POP909_CL_C0_C2_FROZEN_REVIEW.md)
- [POSTBREAK_AUDIT_20260915_RU.md](POSTBREAK_AUDIT_20260915_RU.md)
- [RAW_TIME_READOUT_PREPARATION_RU.md](RAW_TIME_READOUT_PREPARATION_RU.md)
- [RESEARCH_PILOTS_20260913_REPORT_RU.md](RESEARCH_PILOTS_20260913_REPORT_RU.md)
- [RESEARCH_PILOTS_20260913_REVIEW.md](RESEARCH_PILOTS_20260913_REVIEW.md)
- [ROMAN_COMPOSITION_AUDIT_RU.md](ROMAN_COMPOSITION_AUDIT_RU.md)
- [SAINTS_HARMONIC_PROCESS_20261007_RU.md](SAINTS_HARMONIC_PROCESS_20261007_RU.md)
- [TARGET_SLOT_CENSUS_RU.md](TARGET_SLOT_CENSUS_RU.md)
- [TASK_ATTENTION_80K_FINAL_RU.md](TASK_ATTENTION_80K_FINAL_RU.md)
- [TASK_ATTENTION_PROGRESS_001_RU.md](TASK_ATTENTION_PROGRESS_001_RU.md)
- [TASK_ATTENTION_PROGRESS_002_RU.md](TASK_ATTENTION_PROGRESS_002_RU.md)
- [TASK_ATTENTION_PROGRESS_003_RU.md](TASK_ATTENTION_PROGRESS_003_RU.md)
- [TEMPORAL_80K_FINAL_RU.md](TEMPORAL_80K_FINAL_RU.md)
- [TEMPORAL_C2_LAUNCH_RU.md](TEMPORAL_C2_LAUNCH_RU.md)
- [TEMPORAL_FINAL_001_ACCESS_STATUS_RU.md](TEMPORAL_FINAL_001_ACCESS_STATUS_RU.md)
- [TEMPORAL_PROGRESS_001_RU.md](TEMPORAL_PROGRESS_001_RU.md)
- [TEMPORAL_PROGRESS_002_RU.md](TEMPORAL_PROGRESS_002_RU.md)
- [TEMPORAL_PROGRESS_003_RU.md](TEMPORAL_PROGRESS_003_RU.md)
- [TEMPORAL_SLICE_CPU_RU.md](TEMPORAL_SLICE_CPU_RU.md)
- [TONAL_RELATIONS_ALIGNMENT_PILOT_20261005_RU.md](TONAL_RELATIONS_ALIGNMENT_PILOT_20261005_RU.md)
- [TONAL_RELATIONS_ALIGNMENT_REMEDIATION_20261005_RU.md](TONAL_RELATIONS_ALIGNMENT_REMEDIATION_20261005_RU.md)
- [TONAL_RELATIONS_ASTRA_PILOT_20261006_RU.md](TONAL_RELATIONS_ASTRA_PILOT_20261006_RU.md)
- [TONAL_RELATIONS_MAP_BUILD_20261005_RU.md](TONAL_RELATIONS_MAP_BUILD_20261005_RU.md)
- [TONAL_RELATIONS_TARGETS_20261006_RU.md](TONAL_RELATIONS_TARGETS_20261006_RU.md)

</details>

## Обновление

Из корня репозитория:

```bash
python3 scripts/build_project_catalog.py
python3 scripts/build_project_catalog.py --check
```

Новый запуск сначала регистрируется по [протоколу](../EXPERIMENT_LOGGING.md).
Этот генератор только читает записи и не меняет их статусы, метрики или хеши.
