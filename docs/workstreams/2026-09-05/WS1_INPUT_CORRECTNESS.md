# WS1 — корректность raw inputs

## Цель и фаза
Ограниченная корректировка Phase 9E transposition input contract и Phase 2B
HookTheory raw timeline. Результат: новые корректные пути с регрессионными
проверками и явной совместимостью исторических экспериментов.
Общие правила: [README](README.md). Выполнить реализацию и CPU synthetic tests.

## Основания
- `src/music_critic/experiments/analysisgnn/transposition.py`:
  `transpose_raw_graph_view_directed` меняет note pitch, pitch_class, octave,
  track_relative_pitch, но не track mean/min/max pitch. Проверить весь
  фактический raw feature registry; не считать перечисление исчерпывающим.
- `src/music_critic/adapters/hooktheory.py`:
  `convert_hooktheory_record` включает chord_end в content_end перед
  `_raw_duration`, поэтому изменение target annotation extent способно менять
  raw duration/bars/beats. Это проверяемое нарушение границы, не утверждение о
  величине ущерба уже обученной модели.
- Прочитать PHASE9EB5F_TRANSPOSITION_CORRECTNESS,
  PHASE9EB5G_TRITONE_ROUND_TRIP_REMEDIATION_TZ,
  PHASE9EB5H_FULL_ORBIT_TRANSPOSITION_PROFILE, HOOKTHEORY_MIGRATION,
  HARMONIC_SUPERVISION и действующие feature builder/registry.

## Реализация
1. Создать synthetic regression, воспроизводящую stale track statistics.
   Oracle: raw graph, построенный из корректно транспонированных canonical
   raw notes, с тем же layout/ownership. Проверить все pitch-derived признаки,
   availability и statistics, не только числовой сдвиг note.pitch.
2. Реализовать отдельный versioned complete transposition contract/API.
   Старые frozen forward APIs/profiles и их tensor behavior сохранить:
   текущие C0/C2 должны воспроизводиться с прежним protocol identity.
   Новый API должен быть явно доступен будущим потребителям и документирован;
   не подменять старые функции или старые config fingerprints молча.
3. Пересчитывать производные признаки из фактических note/track ownership и
   действующего feature contract. Сохранить topology, exact timing, IDs,
   source graph immutability и non-pitch fields; исключить percussion из
   pitch shifts согласно текущему контракту. Range failures остаются явными.
   Новые availability правила не изобретать: свериться с raw builder.
4. Убрать chord annotation extent из обязательного raw timing HookTheory.
   Source duration и raw melody/observable timeline задают raw geometry.
   Зафиксировать clipping/masking/reason для annotation за пределами raw
   duration и empty melody случая; не расширять timeline ради labels.
   Target-bearing canonical artifacts могут отличаться; обязательная raw
   проекция, raw serialization/cache identity и graph должны быть независимы.
5. Изменение HookTheory adapter semantics получает новую явную version identity
   и cache invalidation boundary. Старые cached pieces/records не переписывать.
   Проверить include_targets true/false, изменение/удаление/удлинение chords.
   Теория, нужная для интерпретации относительной melody encoding, не является
   предметом этого исправления; не переписывать принятую pitch reconstruction.
6. Описать старый/новый путь, причины, поддержанные гарантии и нерешенные
   случаи. Исправленный directed inverse для tritone уже существует —
   не объявлять его новой найденной ошибкой.

## Владение файлами
- transposition.py и при необходимости новый соседний transposition_inputs.py;
- adapters/hooktheory.py и HookTheory-specific tests/config identity glue;
- tests/experiments/test_phase9eb5*_transposition*.py и новый
  tests/experiments/test_complete_transposition_inputs.py;
- tests/adapters/test_hooktheory_adapter.py и отдельный
  tests/adapters/test_hooktheory_raw_timeline.py;
- docs/INPUT_CORRECTNESS_CONTRACT.md;
- docs/workstreams/2026-09-05/WS1_RESULT.md, собственные STATUS/DECISIONS записи.
Graph builder, canonical schema, trainer, vocabularies и frozen profile files
read-only. Если compatibility требует shared edit, сначала описать координатору.

Согласованное уточнение координатора при старте: WS1 владеет минимальным
HookTheory adapter-version glue в src/music_critic/tasks/corpus.py и synthetic
regression для version/config identity нового index. Общие cache APIs/schema
остаются прежними; WS3 уведомлен и использует corpus.py read-only.

Второе согласованное уточнение: внутри build_hooktheory_corpus_cache разрешены
cache_input_sha256=hooktheory_raw_input_fingerprint(piece) и raw config projection
без include_targets, с новой version identity. Существующий общий cache API
уже поддерживает отдельную raw-input identity и artifact identity по полному
canonical SHA. Сохранить полный source digest, canonical SHA и target/config
provenance. Тестировать одинаковый raw key при смене targets/visibility,
разные неперезаписываемые canonical artifacts, корректное чтение обеих версий
targets и изменение raw key при изменении melody/timing. Общие helpers и schema
не менять.

## Приемка и тесты
- identity/positive/negative/inverse/transposition range; tritone and boundary;
  несколько tracks, percussion, constant-pitch/empty track если schema допускает;
  detached source и target-free graph.
- Track statistics и availability совпадают с independent rebuild oracle;
  объявить tolerance лишь там, где tensor floating arithmetic требует ее.
- Исторические transposition tests/fingerprints проходят без rewrite ожидаемых
  значений ради нового поведения.
- Chords changed/removed/extended → одинаковые raw duration/bars/beats/features,
  bytes/fingerprints mandatory raw projection; target masks/reasons корректны.
- Run existing tests/experiments/test_phase9eb5a_transposition.py,
  test_phase9eb5f_transposition_correctness.py, профильные HookTheory unit tests
  и tests/test_repository_contract.py; новые tests дополнительно.
- В отчете отделить инженерное воспроизведение от неизвестного влияния на C2.
  Следующая фаза — отдельно зарегистрированный эксперимент по новому contract,
  не автоматический rerun.
