# WS4 — frozen evaluator для POP909-CL

## Цель и фаза
Небольшой инструмент будущей оценки переноса corrected C0/C2 на POP909-CL без
дообучения. Здесь реализовать API/CLI и проверить synthetic checkpoints/graphs.
Реальную оценку, чтение обученных весов/корпуса и выбор C0/C2 не запускать.
Это root/quality protocol, не все 18 heads и не music-quality score.
Общие правила: [README](README.md).

## Проверенные исходные APIs
- experiments/analysisgnn/corrected_model.py:
  CorrectedAnalysisGNNModel.predict(raw_graph_batch), CorrectedModelOutput,
  corrected_model_contract и model_state_fingerprint.
- harmonic_event heads root/quality routed на raw beat.
- multitask_contract.py: TASK_BY_ID/get_vocabulary — порядок/размер классов;
  corrected root 37, quality 17, а не public AnalysisGNN dimensions.
- corrected_training.load_checkpoint — resume API. Он восстанавливает optimizer,
  scheduler/scaler/sampler/RNG и использует unrestricted loading. Не вызывать его.
- Прочитать POP909_CL_ADAPTER_CONTRACT, HARMONIC_SUPERVISION,
  MULTISOURCE_TARGET_CONTRACT, EVALUATION и нынешний event-start alignment.

## Протокол и реализация
1. Additive model-only loader. Поддержать явно документированную corrected
   schema. CPU torch.load(weights_only=True); нет fallback на unrestricted
   pickle. Проверять ожидаемый artifact SHA/size, schema/model contract,
   полные keys/shapes/dtypes, finite values, model-state fingerprint.
   Strict load в новый model, eval mode, без optimizer. Constructor RNG
   изолировать, caller Python/NumPy/Torch RNG сохранять на success/failure;
   checkpoint RNG не применять. Проверки сейчас только на файлах, созданных
   synthetic tests. Если реальный historical envelope несовместим с restricted
   loader, явно отразить gap; не обещать поддержку без проверки.
2. Root crosswalk: из spelled labels registry получить PC 0..11. Softmax по
   всем 37 → сумма вероятностей enharmonic classes → argmax по 12.
   Не применять argmax37→PC. Проверить conservation/class order/tie policy;
   mapping и registry fingerprint входят в versioned eval protocol.
3. Quality: subset9 = M,m,o,+,D7,M7,m7,/o7,o7, сопоставление через явные
   interval templates с corrected names. Argmax всех17 ДО сравнения subset9.
   Predicted outside9 при gold inside9 — ошибка, не исключенный пример.
   Ground truth sus2/sus4/mM7/+7 в первой версии unsupported с отдельным count.
   Никаких renormalization по9, 17→15 paper mapping или tuning по метрикам.
4. Event alignment: сначала raw predictions, затем source chord event.start
   → единственный raw beat [start,end), exact RationalTime. Один event —
   одно наблюдение, не span-average/nearest onset/распространение на весь аккорд.
   Несколько source events в beat остаются отдельными наблюдениями с одним
   prediction; report multiple_events_same_beat/conflicting labels и resolution
   limit. Missing/ambiguous containing beat явно unavailable.
5. Native primary view и metadata_removed secondary view заданы до метрик.
   Удаление означает ТОЛЬКО:
   note.velocity, note.program, note.channel, track.program, track.channel → None
   до graph build, с полной closure производных features/availability,
   включая track.mean_velocity.
   Tempo, meter, source_track_index, track IDs, is_percussion, все музыкальные
   pitch/onset/duration и topology-driving timing сохраняются. Не расширять
   metadata removal до всего metadata. Построить graph заново через public API,
   не обнулять произвольные tensor columns. View fingerprints раздельные.
6. POP channel0 уже содержит melody+accompaniment, channel1 target-only chord
   blocks. Не включать channel1 в raw notes и не синтезировать accompaniment из
   gold. Targets могут присоединяться только после predictions.
7. Independent root/quality masks, intersection только для joint accuracy.
   Accuracy и observed-true-class macro-F1 для каждой задачи, per-class support,
   confusion и включенные class IDs. Outside-subset quality prediction сохраняет
   FP/extra confusion category и FN истинному классу; нельзя выкидывать ошибки.
   Отдельные coverage denominators events/pieces/groups, missing/ambiguous/
   unsupported/unaligned. Empty denominator = null с reason.
   Не называть mean двух F1 corrected_primary_macro_score.
8. Versioned deterministic report: protocol/view/registry/model bindings,
   counting/alignment policies, synthetic provenance и access observations.
   CLI smoke по умолчанию не открывает локальные данные. Real-data execution
   hook может требовать explicit inputs/registered run; в этой волне только
   synthetic --smoke. TEST запрещен до source/cache/checkpoint reads.

## Владение файлами
Создать src/music_critic/experiments/pop909_cl_frozen/ с отдельными contracts,
checkpoint, projection, views, evaluation, run модулями по необходимости.
Создать tests/experiments/test_pop909_cl_frozen_{checkpoint,projection,views,evaluation}.py,
docs/POP909_CL_FROZEN_EVALUATION.md,
docs/workstreams/2026-09-05/WS4_RESULT.md.
Обновить собственный STATUS и при необходимости append-only WS4 ADR.
Shared vocabularies, adapters, graph core, corrected_training, other evaluator
contracts и terminal records read-only. Metadata_removed принадлежит WS4.

## Приемка
- Enharmonic-sum counterexample против argmax-first, conservation и registry drift.
- Full17 quality winner outside9 остается ошибкой; независимые masks/zero counts.
- Rational beat boundaries, fractional onset, end-exclusive, multiple events.
- Targets changed/removed, annotation-track modified → raw fingerprints,
  candidate IDs/logits unchanged. Запускать logits в eval без stochastic dropout.
- Metadata removal closure, неизменные raw musical events, deterministic repeated
  view и source canonical nonmutation.
- Synthetic strict checkpoint roundtrip и hash/schema/shape/keys rejection;
  запрещенные resume/RNG/optimizer calls monkeypatch to raise, caller RNG stable.
- Batch-order invariant агрегаты, TEST guard до reader callbacks.
- Новые CPU tests и относящиеся corrected model / checkpoint evaluation regressions,
  включая tests/experiments/test_phase9eb5c_corrected_model.py и
  tests/evaluation/test_checkpoint_evaluation.py без real-data options.
Handoff честно разделяет implementation readiness и еще не измеренное zero-shot
качество. Следующая фаза — отдельно зарегистрированная TRAIN/VALIDATION оценка
после получения/верификации нужных checkpoints.
