# WS2 — theory→SSL transfer и совместные objectives

## Цель и фаза
Ограниченный bridge между corrected Phase 9E theory model и существующим
Phase 7–8 SSL перед Phase 10. Один backbone должен получать градиенты от
SSL и размеченных музыкальных задач. Результат — API и маленький synthetic
CPU harness, не production trainer и не доказательство защиты от forgetting.
Общий контекст и ограничения: [README](README.md).

## Проверенные исходные APIs
- `experiments/analysisgnn/corrected_model.py`: CorrectedAnalysisGNNModel,
  local encoder, hierarchical context, sequence_decoder и 18 task_heads.
- `ssl/model.py`: MaskedGraphSSLModel, prepare_mask_binding и public forward;
  full-view stop-gradient targets и cosine objectives сохранить, EMA не добавлять.
- `ssl/transfer.py`: export_pretrained_encoder_state требует
  HierarchicalHeterogeneousBaseline и не принимает corrected model. Не ослаблять
  старый validator и не объявлять полный checkpoint encoder export.
- `experiments/analysisgnn/corrected_training.py`:
  align_target_sidecars_after_prediction, corrected_supervised_loss,
  FrozenClassWeights. Исторический loss 8 primary/10 auxiliary остается тем же.
- SSLRawSample, collate_ssl_samples: строить synthetic batches без runtime
  factories реальных корпусов. build_source_free_fixture читает frozen B5B
  class weights; вместо этого создать явно synthetic uniform weights/labels.

## Реализация
1. Новый strict bridge переносит полные проверенные группы:
   local_baseline.encoder, context_encoder.pooling, context_encoder.transformer,
   context_encoder.fusion. Сверить точные фактические пути state_dict в source и
   destination, документировать явный mapping.
2. Сохранить corrected архитектуру: width 128, local 3, Transformer 2 слоя,
   4 attention heads, FFN factor 4, dropout 0.1. Уменьшать только размер графа
   fixtures; совместимость нельзя доказывать искусственно упрощенной моделью.
3. До первой мутации проверить model/feature contracts, точные sets keys,
   shapes, dtypes, finite tensors и fingerprints. Никакого strict=False,
   silent filtering или частичной загрузки. Отказ оставляет destination без изменений.
4. Исходный source model неизменен. Retained theory branch сохраняет копию
   всех весов и конфигурации BiGRU/18 heads; SSL-only mask token, representation
   decoder, projectors/predictors остаются свежими. Ненужные legacy heads в SSL
   container исключить из active optimizer inventory.
5. Обе ветви внутри harness используют одни и те же Parameter objects общей
   части. Две независимые копии с одинаковыми числами не подходят. Shared objects
   не должны alias исходный source model. Optimizer inventory содержит каждый
   параметр ровно один раз.
6. API принимает подготовленный SSL batch/binding и необязательный отдельный
   labeled raw batch/sidecar/weights. Theory targets присоединяются после raw
   prediction. SSL восстанавливает representation до отдельного corrected
   BiGRU; теория продолжает использовать BiGRU и heads.
7. Режимы: SSL-only, theory-only, mixed с явно заданными lambda_ssl и
   lambda_theory. Нет theory batch/labels — нет соответствующего term, а не
   class 0. Возвращать component losses, total, counts/diagnostics и parameter
   inventory. Не писать dataset/scheduler/epoch loop/checkpoint manager.
8. Transfer report явно различает transferred, retained, fresh, inactive groups,
   их ключи/fingerprints; provenance synthetic_in_memory. Не выдумывать hashes
   несуществующих checkpoint files. Коэффициенты fixtures не production tuning.

## Владение файлами
Создать:
- src/music_critic/ssl/theory_warmstart.py;
- src/music_critic/ssl/theory_retention_harness.py;
- tests/ssl/test_theory_warmstart.py;
- tests/ssl/test_theory_retention_harness.py;
- docs/WS2_THEORY_SSL_HARNESS.md;
- docs/workstreams/2026-09-05/WS2_RESULT.md.
Обновить собственный STATUS и при новом contract decision append-only WS2 ADR.
Существующие transfer.py, model contracts, C0/C2 runners, vocabularies, adapters,
graph core и другие workstreams read-only. Shared edit сначала координатору.

## Приемка
- Точный transfer, сохранность BiGRU/heads, source immutability, fresh SSL groups.
- Atomic rejection missing/extra keys, wrong shape/dtype/contract/hash/nonfinite.
- Shared identity и отсутствие optimizer duplicates.
- Finite nonzero gradients по активным группам в трех режимах, отсутствие
  градиентов inactive branches; не требовать nonzero каждой scalar weight.
- Stop-gradient targets; восстановление train/eval modes после target pass.
- Один mixed CPU optimizer step меняет shared backbone; raw-only step не
  меняет theory-only параметры. Fixtures не мутируются.
- Замена/удаление sidecars не меняет raw logits; masks меняют только loss.
- Новые tests, существующие tests/ssl/test_transfer.py и относящиеся
  SSL/model/corrected-loss regressions; точный список/results в handoff.
Следующая фаза — отдельный registered warm-start experiment с checkpoint
provenance и измерением сохранения theory performance. Не запускать его здесь.
