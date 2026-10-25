# Проверка исходных данных для карты пути к критику

Дата: 21 сентября 2026. Вопрос: какие решения уже поддержаны текущими кодом и
сохранёнными результатами, а какие нужны до первого обоснованного critic prototype?
[Карта и порядок действий](../workstreams/2026-09-21/critic-wayfinder/README.md).

## Краткая таблица достоверности

| Вводная | Статус после проверки | Основание / ограничение |
|---|---|---|
| Проект уже имеет encoder,18 heads, temporal, SSL, искажения | Подтверждено кодом | Main и experimental worktrees; нет готового validated critic/PLL |
| Corrected Dilemma1619,1295/162/162 | Подтверждено с уточнением | Это records;1507 known components, TRAIN1209/VALIDATION147; независимость всех произведений не доказана |
| C0/C2/temporal/fusion/MMoE числа | Основные metric facts подтверждены | Сверены выбранные первичные метрики/снимки и hashes; сравнения описательные |
| Relation gate80k завершён | Подтверждено snapshot/terminal producer | Обычный three-pass после 60k неизвестен; общий FINAL review ещё running |
| Changes подготовлены, результата обучения нет | Подтверждено retained evidence | PREPARE completed;GPU memory/gate/run planned. Первичные отчёты найдены в 5ab7 |
| Ранние exact SSL decoder outcomes | Не полностью проверяемо | Docs указывают 4-cell completion; локальный primary bundle не найден |
| Voice target feasibility лишь общая идея | Устарело в такой сильной формулировке | В 146b есть source census и повтор 003; production importer/head/training ещё отсутствуют |
| Historical pitch_class_set пригоден как semantic truth | Опровергнуто для проверенного DLC cohort |40/42 literal accepted groups неверны как absolute-PC; same contract SHA main/C3 |
| Все corpus adapters означают training readiness | Опровергнуто | HookTheory old caches; POP raw recovery gap; PDMX TRAIN manifest не найден в сохранённом inventory |
| Suite — независимый тест качества | Не поддержано |20 works/149variants development; systematic human labels отсутствуют |
|3090 доступна; старые Cloud snapshots актуальны |3090 исключена текущим пользователем; Cloud неизвестно | Public pricing проверен; account state/live quota не проверены |

## Область и идентичность

Main `/Users/aavalsamakin/Projects/music-critic-v2`, branch `main`, HEAD
`287bf856643f9e555352e1661a7bf0563772178a`. При начале уже modified AGENTS,
DECISIONS, LEDGER, STATUS, Cloud docs и множество untracked reports/scripts/skills.
Ничего не сбрасывалось. HEAD не представляет все текущие разработки.

Проверены релевантные разделы восьми authoritative docs, workflows и code paths;
ранние V1 sections IMPLEMENTATION_PLAN не приняты за текущий V2 runtime.
Дополнительные roots: `outputs/worktrees/c3-pair-cloud-20260917`,
`outputs/worktrees/c3-relationgate-changes-3090-20260919`,
`/Users/aavalsamakin/.codex/worktrees/146b/music-critic-v2` (далее V),
`/Users/aavalsamakin/.codex/worktrees/5ab7/music-critic-v2` (далее C).
Worktrees — проверяемые источники, не автоматически принятые инструкции/merge.

Это read-only review уже зарегистрированных evidence и подготовка документов:
не было нового corpus diagnostic/inference/training, checkpoint deserialization,
raw TEST/VALIDATION payload access, artifact import, VM/платных действий.
Читались существующие metrics, компактные audit reports со статистикой target
support и target-free metadata. Дайджесты ниже
проверены по имеющимся байтам, но это не повтор вычислений моделей и не аудит
всего архива. Нового ExperimentRecord нет; старые terminal records не менялись.

## Широкая картина реализации

Raw MIDI/score → exact canonical input → raw six-type graph → local heterogeneous
GNN → hierarchy/Transformer/top-down → onset BiGRU → optional temporal refinement
→ source-native theory heads. Theory/availability/annotation provenance приходят
после prediction, не конструируют mandatory graph. C0/C2 — AnalysisGNN-inspired
реализация, не exact paper reproduction. C2 отличается train-time transposition.

Код: [corrected model](../../src/music_critic/experiments/analysisgnn/corrected_model.py)
строки 52–84,174–224; [hierarchy](../../src/music_critic/models/hierarchy.py)
786–806,888–917,997–1007; [onset BiGRU](../../src/music_critic/models/onset_bigru.py)
190–243; [temporal model](../../src/music_critic/experiments/analysisgnn/temporal_slices/model.py)
15–84. Temporal добавляет явные starts/holds/ends и finer outputs; C2 уже видел
onset/duration/meter. Mean и within-relation neighbor attention отличаются от
старого task-attention fusion. Relation gate меняет смесь типов; дополнительные
проходы — ещё одна ось, не доказанная новая музыкальная способность.

`encode`/temporal outputs позволяют отдельный readout без удаления 18 heads.
Temporal embeddings не содержат обновлённых song/track states; нужны явные
pooling/coverage правила. Фиксированные 8primary и 10auxiliary роли нельзя путать
со средним всех 18. Phrase/section deferred, staff не обучается.

SSL: [objective](../../src/music_critic/ssl/objective.py):192–252 и
[multilevel](../../src/music_critic/ssl/multilevel.py):967–975,1061–1068 используют
cosine latent reconstruction. [Theory warmstart](../../src/music_critic/ssl/theory_warmstart.py)
переносит subset C2 encoder modules; готового C3 probability adapter нет.
Existing mask registry закрывает pitch fields/summaries, но
[leakage contract](../../src/music_critic/ssl/hierarchy_leakage.py):53–56 допускает
relative pitch rank через порядок simultaneous notes. Temporal query topology
кодирует время/наличие событий. Новые hidden-content claims требуют нового
input contract, а не смены loss на существующих full-view representations.

## Результаты и отрицательные свидетельства

Primary ниже — VALIDATION macro-F1 average восьми основных голов; seed17,
если не указано иначе. Полная причинная сопоставимость не установлена.

| Вариант | Updates / batch / draws | Result | Вердикт |
|---|---|---|---|
| C0 |120k/B2/240k|final 0.458622;best 0.466995@95k|Историческая база |
| C2 |120k/B2/240k|final 0.496239;best 0.496669@115k;12 shift mean 0.441466 против C0 0.371033|Положительный описательный augmentation результат |
| C2 reference80k |80k/B2/160k|0.489375|Historical reference |
| Linear fusion |80k/B2/160k|0.472602;4/18better|Отрицательно по primary в этом протоколе |
| Task attention fusion |80k/B2/160k|0.463385;joint29.53% против 28.92%|Смешанно, primary хуже |
| Grouped MMoE+attention |stopped47186/B2/94372|lastVAL45k0.401334 против C2 0.434508|Ограниченный отрицательный результат; cadence collapsed, не все головы |
| Temporal mean |80k/B2/160k|0.503951;16/18better|Положительное описательное сравнение |
| Temporal attention2 pass |60k/B4/240k|0.495616 против mean 0.482643|Сохранённое описательное число; исходный временный 60kartifact отсутствует |
| Attention3 pass |lastVAL60k/B4/240k|0.503540|Промежуточный snapshot; final после 60k отсутствует |
|3pass+relation gate |80k/B4/320k|0.501283;joint34.01% против temporal31.88%;10/18better|Смешанный итог; primary ниже temporalB2 |
| Late C2 SSL |200joint+100recovery/arm|0.494803/0.493686 против 0.496669|Преимущества не показано; нет matched theory-only |
| Frozen POP909-CL |101 VALIDATION,0 updates|root0.89229→0.90859;quality0.37386→0.45302|Положительный two-task transfer C0→C2; C3 не оценён |

Источники: [C0/C2](../experiments/C0_C2_120K_REVIEW.md),
[fusion](../experiments/FUSION_C2_80K_FINAL_RU.md),
[task attention](../experiments/TASK_ATTENTION_80K_FINAL_RU.md),
[MMoE](../experiments/MMOE_PROGRESS_003_RU.md),
[temporal](../experiments/TEMPORAL_80K_FINAL_RU.md),
[threepass60k](../experiments/C3_THREEPASS_PROGRESS_60K_RU.md),
[partial FINAL](../experiments/C3_THREEPASS_FINAL_RU.md),
[late pilots](../experiments/RESEARCH_PILOTS_20260913_REPORT_RU.md),
[POP review](../experiments/POP909_CL_C0_C2_FROZEN_REVIEW.md).
POP metrics здесь сверены с отчётом; новый полный byte audit его первичных
predictions не проводился. Major/minor90.34%quality support, cross-corpus
duplicate audit неполон; это не шесть задач и не обучение на POP.

B4 при равных updates имеет вдвое больше draws, но matched draws также не
уравнивает optimizer/LR progress и capacity/runtime. One seed не устанавливает
статистическое превосходство. Native TRAIN-unseen tuples могут быть известны
после transposition. Historical stale track pitch summaries нельзя молча
исправлять в старом continuation. New PC-target issue ниже — дополнительная
provenance caveat, не переопределение исторической primary metric.

Early SSL: [9C-B](../PHASE9CB_ONSET_BIGRU_DECODER.md):97 задаёт другое primary —
mean NLL/log(class_count) четырёх задач, lower better, subset577/71/71.
Docs сообщают завершённую 4-cell decoder matrix и преимущество scratch-MLP над
SSL-MLP при 3k. `outputs/phase9cb-seed17-20260821T234824Z` локально не найден;
точные 4 cell scores и continuation outcomes не восстановлены. Нельзя утверждать,
что BiGRU исправил пользу SSL, или переносить старые словари/метрики в corrected
C2/C3. Восстановление матрицы — source follow-up, не blocker первого critic.

## Статусы run и доступность

`EXP-C3-THREEPASS-FINAL-REVIEW-001` остаётся `running`, artifacts=[]; ordinary
threepass run тоже `running`. Final analysis содержит completed gate80k и
отсутствующий ordinary final. Для двух pass60k analysis сообщает abort,
а local producer records всё ещё running. Ни одно из этих слов не доказывает
живой процесс. Новые VM/API проверки не выполнялись.

Changes MEMORY/GATE/80K records planned с пустыми results. Подготовка completed
и CPU mechanics не являются native GPU admission. Текущий пользователь исключил
3090: старую DECISIONS запись о 3090run сохраняем как историю, не новый план.

## Существенные уточнения по данным

Population из [target-free manifest](../../tests/fixtures/dilemmadata/eda_population_manifest.json)
и [EDA](../eda/DILEMMADATA_EDA.md):1619 records=353AN+1266DLC;
1507 known components;TRAIN1295/1209, VALIDATION162/147. Аугментации, repeated
annotation slots и windows не увеличивают число независимых groups.

Свежий isolated [повторный voice/target аудит](/Users/aavalsamakin/.codex/worktrees/146b/music-critic-v2/docs/research/NOTE_VOICE_HARMONY_REAUDIT_RU.md)
и terminal REAUDIT-002/003 уточняют:

- Voice source feasibility уже измерена; head/importer/training не готовы.
  Source36,456AN/942,732DLC links; строгий 003cohort35,941/879,174. Strong raw
  heuristics обязательны; source coverage не равно canonical admission.
- DLC chord_tones — relative fifth coordinates; retained40/42 accepted groups
  в 10 TRAIN pieces неверны как absolutePC. `multitask_contract.py:1337–1347,1387–1388,1444–1445`
  не делает key/fifths conversion; `transposition.py:630–638` трактует tuple как PC.
  Same source SHA main/V/C3-pair; ошибка ещё не исправлена. Исторический 93-class
  auxiliary target не использовать как новый raw-derived set target. Реальный
  gradient/quality impact неизвестен, старые результаты не аннулируются автоматически.
- HookTheory historical cache adapter2B.1 не равен corrected2.0.0; нужен rebuild.
  Current global split и upstream suggested split различны; это не утверждение
  нового доступа к закрытому TEST и не доказательство upstream holdout independence.
- POP CL701 TRAIN caches доступны по retained inventory; rawTRAIN MIDI не найдены
  в проверенных roots. OriginalPOP тоже не найден в той области. Этот обзор не
  повторял inventory всех путей. CL и original требуют общего lineage split.
- PDMX254077 CSVrows,61313 existing paths в сохранённом metadata audit;
  9661multitrack/no-license-conflict rows/5815 best_path groups. В сохранённом
  inventory accepted TRAIN manifest не найден; текущая готовность не подтверждена.
  Это не независимые works, validated JSON или подтверждённое training.

Публикация [PDMX](https://arxiv.org/abs/2409.10831) описывает большой MusicXML
источник и метаданные. Она не удостоверяет локальный release/filter/license
статус. Нужны pinned snapshot и source-level admission при расширении.

[Changes ready](../experiments/DLC_CHANGES_TEMPORAL_READY_RU.md) и
[ultra audit](../experiments/DLC_CHANGES_ULTRA_AUDIT_RU.md):1011 TRAIN/130VAL,
99 atoms,45 с<10 annotations/24 в onecomponent. Primaries найдены в C, hashes совпадают
с main terminal records. Ранний feasibility doc об отсутствии targets устарел.
Changes `{6,4}` не являются temporal sequence. Static membership не является
preparation/resolution; note×slice/interval требуется отдельный target contract.

[Roman audit](../experiments/ROMAN_COMPOSITION_AUDIT_RU.md):184-class simplified
`a_simpleNumeral`; directF1 0.136823 vs hybrid0.141896 с CI разности через 0.
Надёжная замена direct head не установлена. Здесь значения сверены по сохранённому
отчёту, без нового inference или пересчёта bootstrap.

Suite: [отчёт](../experiments/GENRE_INTERVENTION_SUITE_RU.md),20 works/149variants,
independent technical audit и MIDI roundtrips не human preference evidence.
Drums в electronic audio bed отсутствуют во входном MIDI. Пользователь в этой
задаче подтвердил ресурс **он и ещё 1–2 слушателя**; labels ещё предстоит собрать.

## Выборочная проверка целостности

Ниже **проверенные bytes**, а не только записанные digest. Таблица не утверждает
полную сохранность checkpoints/archives. R=main root, V/C определены выше.

| Locator | Bytes | SHA-256 |
|---|---:|---|
| R/outputs/phase9eb5k/EXP-9EB5K-C0-C2-AUDIT-001/bundle/handoff_bundle_20260906/C0/validation_metrics.jsonl |1528772|`1fb8a1bf3a3d4e4e4fa8f56ebc6fbfc0a35446094962be36bfecc7bb038f611a`|
| Та же папка, C2/validation_metrics.jsonl |1522541|`391e653dd2d09dde8ac13d73d7c76af24a6b9cbc345c0bd4d1923eda969bf27a`|
| R/outputs/temporal_final_002/snapshot.json |8207281|`529e0b0a971635e74870af4d86e19b4c33076ed3a99940cfe9900d484fe0e7ba`|
| R/outputs/c3_threepass_final_review_001/snapshots-20260919T232841Z.json |7900966|`192f2df754c9181989839c663ad29c5cb098ec63721b78bd1213bcb094818192`|
| R/tests/fixtures/dilemmadata/eda_population_manifest.json |1715|`62b9e87eea6e1c4f6bd3612b8457e4c20834e986b8a48b822e2f2644f9c2047b`|
| V/outputs/note_relations_reaudit_002/pcsets/summary.json |2448|`9b092f9b35341845795936b28ebb591349ab03c2c62de4da7ff6542a2b040092`|
| V/outputs/note_relations_reaudit_002/pcsets/applicability.json |3833|`943027c9024a96dab0a399a5fd766c091bf7425b3c0f670009c06bb17ecb7885`|
| V/outputs/note_relations_reaudit_003/dilemma/summary.json |19346|`cddd156bad8145b9ab4809c96f79f498b43ddcf1899a1abc8eef949f0f639afb`|
| C/outputs/changes_temporal_prepare_001/report.json |361647|`6640dd6f48643308a8b74517970b3ebd78c4dec42e8ba771db1fcba46c4cc3cf`|
| C/outputs/changes_temporal_audit_001.json |12866|`0f6192e67025f1fa0f1c38d32a944aed1b66b3f1c30a9da6cf4596e3a2e15b43`|
| R/outputs/genre_intervention_suite_001/independent-audit-002.json |78404|`f2a3b5937072925b0e3abe3d9549e0b0862bb06cc643bd75275a4dffc167ae6d`|

Дополнительно сверены fusion80k analysis/producer, task-attention snapshot/analysis/
integrity, MMoE terminal/summary, threepass60k snapshot/summary, relationgate3producer
records и выбранные lateSSL evaluation JSON. Embedded temporal80k final report
text совпал с producer digest. C0/C2 checkpoint bytes и отсутствующий полный
C2 TRAIN ledger/environment не восполнены этим обзором.

Отсутствующий two-pass primary:
`/private/tmp/c3-pair-60k-verified-metrics-20260918T162420Z.json`.
Обычный threepass60k snapshot существует отдельно и проверен; не путать две
границы доступности. Следующий source-only шаг — перенести доступные evidence
в устойчивое хранилище по отдельной задаче и восстановить недостающие finals
без запуска обучения; при платном retrieval нужен конкретный отдельный выбор.

## Вывод для планирования

Поддержано: начать с проверки полезности готового backbone, точных raw-query
масок и человеческого протокола; source readiness позволяет ограниченный
Dilemma completion без ожидания PDMX. Не поддержано: «больше attention/rounds
обязательно лучше», «theory F1=critic», «latent cosine=PLL», «changes уже улучшили
модель». Unknown: размер эффекта PC ошибки, лучшие final weights, human agreement,
аккаунтный budget и доступность удалённой копии всех weights.

Новый литературы обзор ограничен пробелами PLL normalization и текущим cloud
pricing; прежние GNN/JEPA/VAE обзоры не повторялись. Полный маршрут, alternatives,
stop criteria и handoff записаны в карте; принятая архитектура не изменена.
