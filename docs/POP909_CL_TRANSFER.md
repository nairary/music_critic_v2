# POP909-CL: frozen C0/C2 transfer

2026-10-01: пользователь запросил полный POP909 и подтвердил C3 three-pass
relation gate, шаг 80 000. Отдельный `scripts/evaluate_c3_pop909.py` требует
matching planned record, явного scoped TEST разрешения и точного pinned
индекса из 908 принятых записей. Исторические C0/C2 и synthetic APIs сохраняют
прежний запрет TEST. Новый wrapper наследует metric policy и объявляет
собственную full-population execution policy.
Полный C3-прогон завершён: native root macro-F1 0.897446,
quality macro-F1 0.361193, joint accuracy 0.726104; все 908 принятых записей.
На сопоставимом прежнем VALIDATION: 0.920000 / 0.416032 / 0.748803.
[Полные метрики, протокол и проверка](experiments/C3_POP909_FULL_20261001_RU.md).

2026-09-08. Реальная frozen-оценка C0/C2 завершена на одинаковых 101
произведении VALIDATION, без дообучения и TEST. C2 лучше по обеим основным
метрикам: root macro-F1 0.908590 против 0.892292, quality macro-F1 0.453021
против 0.373857. [Результаты, coverage и ограничения](experiments/POP909_CL_C0_C2_FROZEN_REVIEW.md).
TRAIN/VALIDATION entrypoint реализован отдельным increment после WS4;
старый [synthetic-only протокол](POP909_CL_FROZEN_EVALUATION.md) сохраняется.

## Что проверяем

Может ли corrected модель, обученная на Dilemmadata, распознавать гармонию
POP909-CL без дообучения? Это проверка переноса двух гармонических задач,
а не музыкального качества, PLL/fragility или всех 18 задач.

Для первого сравнения предлагается весь существующий POP909-CL VALIDATION:
101 произведение. C0 и C2 получают одинаковый список. Число определяется
замороженным split, искусственного ограничения 1000 здесь нет. Primary
checkpoints — выбранные по Dilemmadata VALIDATION best: C0 95k и C2 115k;
фактические SHA/size, parent run, selection/update provenance проверяются до
регистрации. Final 120k, если понадобится, оценивается отдельной заранее
заданной парой. Нельзя выбирать checkpoint по его POP-метрике.

Сохраняются все правила WS4: root37 softmax суммируется в 12 pitch classes;
quality предсказывается среди всех 17 классов, а допустимые gold — subset9.
Outside-subset prediction остаётся ошибкой. Native — primary,
metadata_removed — secondary. Каждое исходное chord event сопоставляется с
raw beat по точному началу; коллизии нескольких аккордов в одном beat
считаются и не скрываются. Root/quality имеют независимые masks.

Отчёт сохраняет per-event observations, class confusion/support, coverage и
число конфликтующих аккордов на beat. Group ID — component fingerprint
исходного общего split, а не произвольно переопределённая группа.

Сравнение не доказывает отсутствие межкорпусных дубликатов: полный поиск
пересечений с Dilemmadata здесь не реализован. Чужой корпус также отличается
нотацией и распределением классов; низкую метрику сначала разбираем по
coverage, классам, разрешению beat и входным views. По результатам POP нельзя
настраивать fusion и затем называть тот же набор независимой проверкой.

## Входы и границы исполнения

Новые файлы:

- `src/music_critic/adapters/pop909_cl_frozen_source.py` — читает один
  SHA-bound MIDI, строит только channel-0 projection, проверяет её SHA и
  source-group identity. Channel0 уже содержит melody и accompaniment.
  Никакие ноты из channel1 не добавляются. Source bytes читаются один раз для
  hash и raw parse; нормализация chord targets здесь не вызывается.
- `src/music_critic/experiments/pop909_cl_frozen/transfer.py` — явный request,
  preflight, проверка experiment record и отдельный real-data report.
- `src/music_critic/experiments/pop909_cl_frozen/transfer_run.py` — CLI.

Preflight читает только все constituent index JSON и исходный split JSON,
проверенные по SHA и размеру, затем проверяет полное source/lineage component
closure через прежний `validate_split_manifest`. Только после этого отбирает
явный список произведений из запрошенного TRAIN или VALIDATION. Данные,
canonical caches и веса preflight не открывает. Наличие путей и пригодность
checkpoint bytes этим этапом не подтверждаются.

Execution требует действительного **planned** ExperimentRecord, где
`protocol.pop909_transfer_binding_fingerprint` совпадает с preflight.
Declared policy должна запрещать весь TEST и разрешать VALIDATION при его
оценке. Нельзя исполнить terminal record повторно. Каждый запуск требует
нового output path; ошибки оставляют отдельный failure receipt. Caller
финализирует запись и ledger по [EXPERIMENT_LOGGING](EXPERIMENT_LOGGING.md),
включая неуспешные попытки; runner сам не редактирует записи или ledger.

После строгой model-only загрузки обрабатывается одно произведение за раз
на CPU, в eval/inference mode. Новый raw reader проверяет source/projection
до forward. Только после предсказаний обеих views вызывается прежний
`convert_pop909_cl_file(include_targets=False)` для chord evidence с повторной
проверкой source SHA; raw projection должна совпасть. `include_targets=False`
сам по себе не отключает извлечение evidence, поэтому этот adapter нельзя
вызывать на раннем raw этапе. Missing targets у 367/658 остаются unavailable;
quarantine 172 не включается. TEST запрет действует до чтения request при
явном `--split test`, а неверно выбранная TEST-запись отклоняется до source
или checkpoint reads. Метаданные общего split могут перечислять TEST IDs;
это не чтение нот или значений разметки TEST.

Старые `evaluate_frozen`, `evaluate_from_readers`, smoke CLI и их fingerprints
остаются synthetic-only. Общая арифметика вынесена в private core, старый
report побайтово проверяется тестом. Production report имеет отдельные schema,
protocol, membership/checkpoint bindings и access observations. Variable wall
time исключён только из `scientific_fingerprint`; весь отчёт имеет отдельный
`report_fingerprint`. Обновлений параметров нет.

## Формат request и запуск

`Pop909ClTransferRequest@1.0.0` — JSON с ровно такими полями:

| Поле | Значение |
| --- | --- |
| `schema_version` | `Pop909ClTransferRequest@1.0.0` |
| `split` | `train` или `validation` |
| `indices` | Непустой список всех index bindings общего split, включая HookTheory |
| `split_manifest` | Binding исходного общего split |
| `checkpoint` | Binding конкретного corrected checkpoint |
| `source_root` | Абсолютный корень для `source_relative_path` из POP index |
| `piece_ids` | Явный непустой список уникальных `piece:pop909-cl-NNN` |

Каждый file binding содержит ровно `path` (абсолютный), `sha256` и
`size_bytes`. Все файлы проверяются по исходным bytes, paths не входят в
portable identity. Для полного VALIDATION список берётся из существующих
assignment rows; новый split не строится. Binding включает checkpoint, так
что для C0 и C2 нужны разные planned records.

```sh
PYTHONPATH=src python -m music_critic.experiments.pop909_cl_frozen.transfer_run \
  --preflight --split validation --request /absolute/request.json \
  --output /absolute/new-preflight.json

PYTHONPATH=src python -m music_critic.experiments.pop909_cl_frozen.transfer_run \
  --evaluate --split validation --request /absolute/request.json \
  --experiment-record docs/experiments/records/EXP-ID.json \
  --output /absolute/new-result.json
```

Это примеры интерфейса, не команды для имеющихся реальных артефактов.
Default checkpoint loader использует restricted `weights_only=True` и
не восстанавливает optimizer или checkpoint RNG. Исторические C0/C2 envelopes
были отклонены отдельным input gate ниже; unsafe fallback отсутствует.
Последующий `checkpoint_export.py` читает ограниченный доказанный model-only
prefix и создаёт артефакт, который проходит прежний строгий loader. Исходный
checkpoint SHA, внешний update/selection provenance и совпадение model-state
fingerprint сохранены отдельно. Loader не восстанавливает отсутствующие
сведения из имени файла.

## История input gate — 2026-09-07

На Mac найдены metadata files `data/cache/pop909_cl.index.json` (908),
`data/cache/hooktheory.index.json` (26175) и `data/cache/global.split.json`.
В последнем POP имеет TRAIN 701, VALIDATION 101, TEST 106. Это существующий
split POP/HookTheory, он не равен Dilemmadata 1295/162/162. Проверена только
доступность нескольких разрешённых cached paths; их содержимое не читалось.
При подготовке реализации локальный исходный MIDI-root и checkpoint bytes
ещё не были подтверждены. Последующий `EXP-POP-TRANSFER-INPUTS-001` скопировал
оба best checkpoint и ровно 101 исходный VALIDATION MIDI: их SHA/size совпали
с сохранёнными identities и полным проверенным allowlist. TEST payloads не
копировались; разбор MIDI и извлечение targets ещё не запускались.

Input gate завершён как **failed**: прежний restricted loader отклонил C0
на `numpy.core.multiarray._reconstruct`, C2 на
`numpy._core.multiarray._reconstruct`. Исторический training envelope хранит
NumPy RNG вместе с весами. Проверка выполнена на Python 3.12.14 / Torch 2.13.0;
unsafe fallback и allowlisting не применялись. Совпадение SHA подтверждает
целостность копирования, но проверка tensors/model-state ещё не пройдена.
Этот отказ не свидетельствует о повреждении весов и не даёт метрики переноса.

План после этого отказа — отдельно зарегистрированное получение model-only артефактов
с сохранением source SHA, selection/update provenance и model-state fingerprint,
затем проверка обоих прежним строгим loader. После этого заполнить два request
и planned records, выполнить одинаковую оценку и сопоставить классы/coverage.
Полные артефакты input gate находятся в ignored `outputs/pop909_transfer/inputs-001/`;
компактная запись связывает десять файлов по SHA и размеру. Legacy не использовался; новые
dependencies, GPU запуск, обучение fusion/SSL, изменение historical baseline,
TEST, commit/push не входят в объём.

План архитектурной абляции отдельно:
[HARMONIC_LOGIT_FUSION_ABLATION](HARMONIC_LOGIT_FUSION_ABLATION.md).

## Завершение экспорта и оценки — 2026-09-08

`EXP-POP-TRANSFER-EXPORT-001` успешно экспортировал обе модели. Проверены
исходные SHA/размеры, структура ограниченного prefix, формы и значения тензоров,
model contract и одинаковый с исходным model-state fingerprint. Для GRU
поддержаны ограниченные contiguous views общего storage. NumPy RNG/optimizer
tail не исполняется; `weights_only=False` и дополнительный allowlisting
не используются. Исходные checkpoints не изменены.

Отдельные `EXP-POP-TRANSFER-C0-001` и `EXP-POP-TRANSFER-C2-001` прошли
preflight и оценку по прежнему subset9-протоколу. Обе predictions выполнялись
до извлечения chord targets каждого произведения. Ноты melody/accompaniment
берутся из channel0; channel1 chord evidence не входит в raw graph. Обе
модели получили одинаковые raw inputs, события и маски. Полные отчёты,
исходные snapshots, progress и provenance сохранены под
`outputs/pop909_transfer/20260908/`; SHA/размеры — в terminal records.

Развёрнутые результаты и следующий диагностический шаг представлены в
[paired review](experiments/POP909_CL_C0_C2_FROZEN_REVIEW.md).
Будущее обучение надстройкам описано отдельно в
[HARMONIC_MODIFIERS_FUTURE](HARMONIC_MODIFIERS_FUTURE.md) и сейчас не запущено.

## Проверка реализации

В increment 2026-09-08: **56 passed** (9.17s) — transfer/core/raw reader;
**59 passed** (4.52s) — новый exporter и прежний strict checkpoint loader;
**6 passed** (1.00s) — repository contract. Всего **121 passed**, без failures
и skips, четыре upstream Torch JIT deprecation warnings в двух вызовах.
Зависимости проекта восстановлены в отдельной временной среде; новые
runtime dependencies в проект не добавлены. Legacy не использован.

Предыдущая проверка increment 2026-09-07:

Root повторил два synthetic CPU gate: **142 passed, 1 CUDA-only test
deselected** (5.55s) для старого WS4, нового raw reader, repository contract
и corrected model; **24 passed** для нового transfer/CLI. В каждом прогоне
два прежних upstream Torch JIT deprecation warnings; `git diff --check` чист.
Независимое review не нашло оставшихся блокеров в границах этих контрактов.

Проверены TRAIN и VALIDATION, полная component closure с дополнительным
корпусом, TEST до payload readers, metadata-only preflight, неверные SHA и
membership, planned-record binding/policy, source-path escape, обе predictions
до targets, изменение source после predictions, отсутствие optimizer,
неизменность model state, сохранение failure/KeyboardInterrupt receipts и
запрет перезаписи результатов. Старый synthetic report сохраняет все 36 333
байта контрольной fixture. Это механика исполнения, не GPU/runtime
совместимость исторических checkpoints и не качество на настоящем POP909.
