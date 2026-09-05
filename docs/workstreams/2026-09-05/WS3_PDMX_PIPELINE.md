# WS3 — PDMX pipeline и cache

## Цель и фаза
Инженерная основа Phase 10: source manifest → canonical raw score → source
groups/splits → bar windows → versioned cache → lazy loader и отчет.
Разработка и проверки только на synthetic fixtures. Это не завершение Phase 10,
не подготовленный реальный корпус и не запуск SSL. Общие правила: [README](README.md).

## Состояние и источники
adapters/pdmx_eda.py воспроизводит исторические compact facts и не является
production adapter. docs/eda/PDMX_EDA.md содержит ограничения provenance.
Прочитать IMPLEMENTATION_PLAN разделы 18, windowing и Phase 10; DATA_CONTRACT,
MULTISOURCE_DATASET и текущие corpus/loading APIs.
Предпочтительный source format — supplied MusicRender/JSON; original MXL
отложен. Не добавлять runtime dependency MusicRender, не исполнять serialized
objects. Schema profile должен опираться на закрепленные первичные определения
формата/исходники, доступные read-only. Synthetic fixture не доказывает реальный
source schema. Неизвестная release identity остается unknown, не угадывается.

Число 1000 не является лимитом, обязательной sample policy или встроенной
константой. Historical 1k acceptance — возможная последующая инженерная проверка.
Объем, filters, workers, window geometry и budgets конфигурируемы.

## Реализация
1. Versioned source/selection manifest: release/profile identity, source IDs,
   relative paths/hashes, license/filter facts, доказанные duplicate/lineage
   связи, parser/projection versions. Stable reasons для exclusions. Rating
   можно использовать как declared filter, нельзя объявлять gold quality.
   Разделить explicit fixture mode и production readiness: неизвестные
   обязательные bindings не заменять fixture identity.
2. Raw JSON adapter с exact ticks/resolution→RationalTime, raw notes/tracks,
   tempo/meter и bars/beats согласно принятому schema. Optional notation,
   voices/roles/spelling не становятся обязательными encoder inputs.
   Не выдумывать voices; next_in_track когда voice evidence нет. Key signature
   не local-key target. Никаких псевдоаккордов/Roman labels.
3. Grouping и split до windowing: supplied lineage + exact hash components,
   затем explicit split manifest. Все windows одной component наследуют split.
   Filename не доказанный work ID. Неизвестные cross-corpus связи обозначить.
   TEST guard выполняется до любого source/cache reader; в этой волне только
   явно разрешенные synthetic TRAIN/VALIDATION examples.
4. Configurable bar-count/stride, global piece/group identity, source bar range,
   original note IDs и exact global-to-local offset. Для crossing notes принять
   clipping с provenance/boundary semantics, не молчаливую потерю.
   Переносить действующий на start tempo/meter context. Объяснить grace notes,
   pickup/incomplete bars, empty window и very long score.
   Overlap не увеличивает число независимых произведений; evaluation overlap
   не разрешать молча.
5. Streaming preprocessing с bounded worker queue; не держать corpus целиком
   в RAM. Deterministic order/results независимо от workers. Versioned cache
   shards/index, atomic publication и hashes; resumable journal связан с source,
   filter, split, window и projection fingerprints. Partial/corrupt artifact
   не cache hit; измененный contract не переиспользует старые данные.
   Изолировать bad input; programming exceptions не скрывать под invalid source.
6. Lazy window loader, совместимый с текущим raw graph/SSL collator boundary.
   Counts отдельно records/groups/windows, filtered/quarantined/cache hits;
   reasons, clipping, resource rejects. Graph-size percentiles только по реально
   построенным fixture graphs с denominator, иначе not_computed.
   Не заявлять throughput или corpus distribution по fixtures.
7. CLI принимает explicit source/config/output, без автоматического доступа к
   локальному data/cache или необязательного corpus scan при import.
   Реальный запуск не выполнять в этой задаче.

## Повторное использование
- tasks/corpus.py: CanonicalCorpusInput, CorpusCacheConfig, cache_canonical_corpus,
  load_cached_piece.
- tasks/loading.py: SplitManifest, create_split_manifest, plan_group_hash_split,
  validate_split_manifest, IndexedMultiSourceDataset, make_multisource_dataloader.
- data canonical serialization/validation и graph/builder.py build_raw_graph.
Не переписывать общий cache/graph стек. Generic MIDI compatibility path может
быть явным дополнительным режимом, но не подменяет заявленный MusicRender reader.
Если реальный JSON schema недоступен, реализовать независимую pipeline часть,
отразить конкретный blocker формата и не заявлять production adapter готовым.

## Владение файлами
- src/music_critic/adapters/pdmx.py;
- src/music_critic/pdmx_pipeline/**;
- scripts/preprocess_pdmx.py; configs/pdmx/**;
- tests/adapters/test_pdmx.py; tests/pdmx_pipeline/**;
- tests/fixtures/pdmx_pipeline/** (только tiny synthetic);
- docs/PDMX_PIPELINE_CONTRACT.md;
- docs/workstreams/2026-09-05/WS3_RESULT.md;
- собственные STATUS и при необходимости append-only WS3 ADR.
Общие corpus/loading/schema/graph APIs, pdmx_eda replay и другие workstreams
не менять без конкретного согласования с координатором.

## Приемка
Exact timing/meter changes; optional metadata absent; invalid/empty/oversize
input; grouping before windows/no split crossing; clipping/context; identical
workers output; uninterrupted vs interrupted/resumed artifacts; partial writes,
corruption/stale fingerprint; lazy reader evidence; TEST rejected before
callbacks; raw graphs without theory annotations. Проверить memory bounding
архитектурно и на ограниченной generated sequence, не corpus stress run.

Запустить новые tests и относящиеся tests/tasks/test_multisource_dataset.py,
tests/training/test_split_and_corpus.py, tests/adapters/test_pdmx_eda.py,
tests/adapters/test_midi.py. Не включать real-data integration tests.
Handoff: точные tests, schema support vs unresolved evidence, fixture-only
statistics и условия будущей реальной preprocessing acceptance.
