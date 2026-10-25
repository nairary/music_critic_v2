# DLC changes: фактическая готовность temporal baseline

18 сентября 2026. `EXP-CHANGES-TEMPORAL-PREPARE-001`.
Локальная CPU-подготовка и аудит; новых real-data forward/training, GPU,
исторических checkpoint loads, SSH и TEST payload/targets не было.

| Показатель | TRAIN | VALIDATION |
|---|---:|---:|
| Независимые произведения / components | 1 011 | 130 |
| Исходные аннотации | 257 771 | 35 021 |
| Пригодные аннотации | 251 038 | 34 532 |
| Пригодные непустые наборы | 34 055 | 4 070 |
| Пригодные пустые наборы | 216 983 | 30 462 |
| Raw temporal slices | 910 112 | 119 704 |
| Slice-позиции с supervision | 840 925 | 112 528 |
| Все masked slice-позиции | 69 187 | 7 176 |
| Классы с положительной поддержкой | 99 | 53 |

TRAIN числа аннотаций и positive sets независимо совпали с прошлым экспортом.
45 TRAIN-классов имеют менее10 аннотаций,24 — поддержку в одном произведении.
VALIDATION не расширял vocabulary:46 классов не имеют positives,16 классов
встречаются в одном произведении. Семь аннотаций содержат OOV: `1` —1,
`b14` —6. Они не стали отрицательными примерами. Причины масок по VALIDATION
onsets: source-unavailable/OOV1772, notation150, raw repair17, conflicting
sets2. Счётчики причин могут пересекаться. Дополнительные offset/beat slices
без подтверждённой onset-аннотации также masked; переноса gold spans нет.

## Проверенная цепочка происхождения

- TRAIN manifest, vocabulary/events и полный alignment manifest проверены по
  прежним terminal records. TRAIN-membership повторно сверена с фактическими
  сохранёнными draws и frozen split; VALIDATION — с тем же frozen split.
- Архив campaign прочитан потоково. Читались только выбранные DLC raw pieces;
  старые model target-sidecars не открывались, корпус не распаковывался.
- Все23 части пакета проверены по SHA/size. Хеш receipt совпал с tracked cloud
  release. Manifest SHA `48a5301dcf83c12eb6ba5f5a00e658e3d0972221bc9db5eea5fa1e62f384dd38`
  совпал с launch sequence обеих C3 arms. Успешно связаны все1 141 DLC raw pieces.
- SHA-256 каждого canonical piece совпал с реальным piece artifact campaign.
  Дополнительно проверены полный raw graph, точные note IDs/times и raw slots.
  Изначальная неопределённость historical-cache equality теперь закрыта для
  **этого конкретного manifest**, не для произвольного другого cache.
- VALIDATION TSV hashes зафиксированы при чтении. Их raw projections совпали
  с frozen B3 metadata. Это newly observed full source hashes; отдельного
  historical full-source hash для этих VALIDATION changes не было заявлено.
- Независимый post-pass проверил1 533 emitted artifact hashes, полные annotation
  weights и поддержку по независимым components. Время основного прохода:
  673,9065с CPU, один worker,0 model updates.

Frozen temporal graph source отличается от этой ветки лишь дополнительным
loader fast path, пропускающим повторную canonical validation после authenticated
loads. Raw graph builder, slot builder и transposition source совпали byte-for-byte.
Проверка C3 ABI проведена чтением исходников; C3 native execution не утверждается.

## Код, проверки и остающиеся условия

Реализованы opt-in Linear(128,99), authenticated loader, exact post-forward
alignment, annotation-weighted multilabel BCE, availability-normalized groups,
streaming per-class PRF/component support, all/positive-only sets, CPU update
со штатным old18 objective и отдельный полный CPU resume. При отключении
wrapper не создаётся; прежние18 outputs и checkpoints не меняются.

Тесты покрывают known-empty/missing, unknown atoms, source conflicts, точное
время, все12 транспозиций, input/target separation, batch isolation, gradient
head/backbone, old18 retention, совместный update, full next-update resume,
ошибочные optimizer states, rollback и TEST-before-open. Итоговые результаты
зафиксированы в STATUS и compact experiment record; полный лог —
`outputs/changes-final-tests.log`.

Полные sidecars: `outputs/changes_temporal_prepare_001/`; независимый аудит:
`outputs/changes_temporal_audit_001.json`. Raw pieces остаются в исходном
read-only evidence root. Generated outputs не предназначены для Git.

Ограничения: редкие TRAIN-классы и46 unsupported VALIDATION classes; source
confidence unknown; нет доказательства полезности новой головы. Exact snapshot
всех импортированных Python modules до начала data-прохода не сохранён. Финальный
source snapshot отражает итоговую реализацию; последующие изменения затрагивали
метрики/resume и усиление preflight, а raw binding algorithm не менялся. Primary
input/output hashes и логи сохранены; это не заявляется как полный historical
execution environment archive.

Следующий этап — выбрать один authenticated parent, заполнить его bindings и
optimizer/LR values в `configs/changes/paired_continuation.json`, зарегистрировать
парное continuation и native admission. CPU reference join заново проверяет
raw graph и не заменяет оптимизированный prepared loader текущей GPU campaign.
Интеграция с ним, native full-resume/memory gate и реальный launch остаются
условиями будущего запуска. Обучение здесь не запущено.

Подробности/команды: [baseline](../DLC_CHANGES_TEMPORAL_BASELINE.md).
Композиционная модель: [отдельное предложение](../DLC_CHANGES_COMPOSITIONAL_ABLATION.md).
Legacy не читался и не использовался.
