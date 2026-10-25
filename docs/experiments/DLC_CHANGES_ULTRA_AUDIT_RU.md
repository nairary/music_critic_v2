# DLC changes: независимый повторный аудит

18 сентября 2026. Эксперимент: `EXP-CHANGES-TEMPORAL-ULTRA-AUDIT-001`.
Повторная проверка подтверждает сохранённые TRAIN/VALIDATION targets и их
привязку к raw attack onsets. Найденные недостатки проверки входов, objective
и resume исправлены отдельно от неизменяемых исторических артефактов.
Обученное качество и готовность native/GPU campaign этим не установлены.

## Проверка данных

Новый `scripts/audit_changes_temporal_ultra.py` использует собственный расчёт,
не вызывает production parser/alignment/validator и не копирует итоговые счётчики
старого отчёта. До payload-доступа он связывает метаданные с terminal records,
проверяет frozen campaign/split membership и полный список допустимых файлов.
Затем сверяет сохранённые source groups, исходные строки, точные рациональные
anchors и note durations, notation gate, masks, dense targets и веса.

Для каждой доступной аннотации сумма её вкладов по пригодным onsets равна одному.
Проверяется и вес каждой позиции, а не только общая сумма: перераспределение
весов между аннотациями больше не может пройти проверку сохранением total.

| Показатель | TRAIN | VALIDATION |
|---|---:|---:|
| Произведения | 1 011 | 130 |
| Исходные аннотации | 257 771 | 35 021 |
| Пригодные аннотации | 251 038 | 34 532 |
| Непустые пригодные аннотации | 34 055 | 4 070 |
| Подтверждённые пустые множества | 216 983 | 30 462 |
| Исходные нотные строки | 1 661 584 | 213 225 |
| Строки продолжения лиг | 41 419 | 5 593 |
| Raw temporal slices | 910 112 | 119 704 |
| Onsets с supervision | 840 925 | 112 528 |
| Onsets с непустым target | 93 032 | 10 135 |
| Onsets под маской | 28 069 | 1 933 |
| Source anchor times без raw attack | 1 932 | 196 |
| Классы с положительной поддержкой | 99 | 53 |

Все 1 141 записи прошли повторную проверку. Выполнено 2 551 сравнение хеша
корневых/полных артефактов и ещё четыре сравнения хеша provenance metadata.
Canonical hashes в обеих сохранённых привязках совпадают для всех 1 141 записей;
это сверка закреплённых идентичностей, а не новая декомпрессия campaign pieces.

Семь OOV-аннотаций VALIDATION остаются под маской: `1` один раз и `b14` шесть раз.
45 TRAIN-классов имеют менее десяти пригодных аннотаций; 46 классов не имеют
положительных примеров в VALIDATION. Полные распределения по аннотациям,
позициям и независимым source components сохранены в JSON-результате.

Продолжение лиги сохраняет собственный source anchor и lineage canonical note.
Оно не переписывает target исходной атаки. Source anchor без raw attack не
создаёт prediction candidate. Длительно звучащая нота проверяется относительно
её точного сохранённого onset/duration; ближайший chord и gold interval не
используются как fallback. `64` остаётся множеством `{6,4}`. Пустой валидный
набор, отсутствующий target, конфликт и OOV остаются разными состояниями.

## Исправленные границы допуска

- Ранее заново sealed sidecar мог перенести target на существующий offset/beat
  slice. Новый общий validator требует полный точный inventory raw attacks,
  согласованные index/id/time, sparse/dense labels, masks и source identities.
  Повтор одной физической source row в разных onset targets отвергается.
- Веса проверяются как точная сумма `1 / usable_onsets_for_annotation` для каждой
  позиции. Re-seal или новый SHA сами по себе не подтверждают семантическую
  корректность targets.
- Batch targets связываются с порядком candidate keys, vocabulary и raw graph
  fingerprint. Одинаковой формы tensor для допуска к loss/metrics недостаточно.
- Старый `scripts/audit_changes_temporal.py` проверял произвольный список
  artifacts до проверки split и брал часть counts из отчёта. Теперь он отвергает
  TEST/unknown splits, дубликаты, недопустимые пути, лишние/отсутствующие payloads
  до их открытия или хеширования; затем валидирует sidecars и пересчитывает counts.
  Его прежний CLI сохранён. Он проверяет объявленное membership; независимое
  закрепление frozen membership выполняет новый ultra audit.

Исторические `EXP-CHANGES-TEMPORAL-PREPARE-001`, отчёты, TRAIN/VALIDATION targets
и checkpoint artifacts не исправлялись задним числом. Ошибок значений в старых
1 141 sidecars повторный проход не выявил. Найденные недостатки касались того,
какие некорректные входы прежний код мог принять.

## Objective, metrics и resume

Objective теперь проверяет masks, значения и идентичность batch перед расчётом,
приводит targets/weights к FP32 и использует устойчивую нормировку весов.
Недоступные позиции и позиции с нулевой confidence исключаются до arithmetic.
Treatment и control используют одну функцию коэффициентов доступных групп,
включая нестандартный `old_weight`. Исходная нормировка old18 остаётся внутри
старого objective. Неизвестная source confidence сохраняется как `null`;
baseline reliability 1 — правило loss, а не оценка эксперта.

В метриках отдельно выводятся обычные position PRF и annotation-weighted PRF.
Source components объединяются по всему потоку; множество gold positives
и exact-set accuracy для непустых наборов не заменяются accuracy по нулевым
элементам. Это устраняет смешение поддержки аннотаций с повторением по onsets.

Новый формат `resume@1.1.0` связывает фактический model/backbone contract,
зависимости LR-функции — globals, defaults, keyword defaults, closure и
используемые math functions — и расширенный CPU runtime. Проверяются типы
cursor/envelope, CPU GradScaler и отсутствие незавершённой scaler attempt.
Сохранение публикует полный файл атомарно без overwrite. Восстановление проверяет
post-load fingerprint независимо от возможной мутации аргументов loader и при
ошибке откатывает все принадлежащие ему domains. Старые checkpoints не
переинтерпретируются как новый resume format.

Итоговые объединённые тесты: **281 passed, 1 skipped, 1 warning, 17,87 с**.
Проверка реальных frozen C2/C3/relationgate классов на CPU fixtures:
**45 passed, 37,13 с, без warnings**.
Пропущенный модуль обычного прогона требует более нового temporal runtime и
полностью проверен отдельным native-source прогоном. Предупреждение обычного
прогона — Torch JIT deprecation. Native source manifests до/после совпали;
source files не изменились во время выполнения.
Она охватывает B4, все 12 shifts, следующий update/resume, обе парные ветви и
отклонение alias mutations до переноса batch, но не допускает
campaign `coalesced_reference` queue, real-data training или GPU execution.

## Идентичности манифестов и воспроизведение

Расхождение названий полей в старых метаданных не означает изменения bytes:

| Поле/объект | SHA-256 |
|---|---|
| Original package manifest; `inputs.campaign_manifest_sha256` | `58085b4e58e03f3ad362a58d2bfef693b5946e9f676d8e5c3dfe20b2af1406ae` |
| Relocated manifest; terminal `protocol.campaign_manifest_sha256` | `48a5301dcf83c12eb6ba5f5a00e658e3d0972221bc9db5eea5fa1e62f384dd38` |

Оба хеша совпадают с соответствующими полями закреплённого package receipt.
Новый отчёт сохраняет оба имени явно.

Точная list-form команда выполненного прохода; environment: `PYTHONPATH=src:.`:

```json
[
  "outputs/changes-venv/bin/python",
  "scripts/audit_changes_temporal_ultra.py",
  "--prepared", "outputs/changes_temporal_prepare_001",
  "--parent-record", "docs/experiments/records/EXP-CHANGES-TEMPORAL-PREPARE-001.json",
  "--campaign-manifest", "/Users/aavalsamakin/Projects/music-critic-v2/outputs/cloud_migration_20260916/prepared/original-manifest.json",
  "--split-assignments", "/Users/aavalsamakin/Projects/music-critic-v2/outputs/research_resume_20260913/context-local-inputs/metadata/split_assignments.jsonl",
  "--evidence-root", "/Users/aavalsamakin/Projects/music-critic-v2",
  "--output", "outputs/changes_ultra_audit_001/independent_alignment_audit.json"
]
```

Exit status: `0`. Измеренный wall time этого отдельного прохода не сохранён и
остаётся неизвестным. stdout/stderr сохранены без overwrite в
`outputs/changes_ultra_audit_001/independent_alignment_audit.log`.
JSON: 18 667 bytes, SHA-256
`da4cb18ca73e3e03073b67b6d65e6f05b776661e476ec846e56cd8b536d8e63e`.

Отдельный pre-launch snapshot нового ultra script не был сохранён.
`before-manifest.json` фиксирует прежний код, но не этот вновь созданный script.
После выполнения зафиксирован его SHA-256
`d7f4b2e66b550c63a835380ed94503be32886f0a0ff09ceeb073a930710c9a83`;
это final snapshot, а не доказательство сохранённого pre-launch snapshot.
Новый реальный повтор требует нового experiment/attempt ID и output path.

## Границы вывода

Исходные TSV не перечитывались и corpus conversion не повторялся. Проверялись
hash-bound сохранённые source events, alignment rows, raw identities и frozen
membership metadata; сохранённые source-row aggregates не подтверждались новым
чтением оригинального корпуса. Campaign tar не декомпрессировался заново.
Нет real-data forward/training, TEST payloads, GPU, legacy или cloud mutation.

Нотационный gate проверяет документированное соответствие literal changes и
It/Ger/Fr expansion; он не является новой экспертной разметкой подготовки,
задержания, голосов или разрешения. Улучшение музыкального распознавания и
критики не измерялось. Следующий самостоятельный этап — native prepared-runtime
admission с одним закреплённым parent и затем зарегистрированное парное обучение.


Итоговый код: `src/music_critic/experiments/changes_baseline/`; проверки:
`tests/tasks/test_changes_*`. Изменены также audit CLI, парный config,
DATA_CONTRACT, DECISIONS и STATUS. Legacy не использовался. Старые source
snapshots и12 artifacts, перечисленные в PREPARE-001, повторно сверены: хеши
сохранились вместе с terminal record. Полные логи первоначальных ошибок и
промежуточных тестов сохранены в outputs и перечислены в новом manifest.

Мой прежний вывод о готовности уточнён: числовое покрытие и semantic scope
подтвердились, но прежней глубины unit tests было недостаточно для строгого
input admission и full-resume. Теперь эти дефекты закрыты, а совместимость
CPU adapter с настоящими C3 исходниками проверена исполнением. Утверждение
о GPU/campaign readiness по-прежнему не поддерживается. Новый исследовательский
вывод о качестве модели не сделан.

Registry: **58 passed, 0,91 с**. Первоначальный registry gate обнаружил только
неперегенерированный ledger после нового planned record; ledger обновлён,
повторная проверка прошла. Финальное закрытие проверяет все119 records.
