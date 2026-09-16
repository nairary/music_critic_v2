# Задача 2 — Logit fusion: интеграция и контролируемый пилот

Цель: довести уже реализованный harmonic fusion до проверяемого bounded paired training/evaluation workflow. Не переписывать существующую механику с нуля.

Основа: docs/HARMONIC_LOGIT_FUSION_ABLATION.md, docs/PERFORMANCE_REPLAY.md, docs/experiments/PERFORMANCE_REPLAY_C2_RESTORATION_REVIEW.md, docs/POP909_CL_TRANSFER.md; harmonic_logit_fusion.py и harmonic_fusion_checkpoint.py. Сейчас есть 109 CPU acceptance tests, model-only checkpoint и +216886 parameters. Это не production resume и не выполненный научный пилот.

1. Интегрировать opt-in wrapper в отдельный bounded training runner, optimizer/scheduler/RNG/sampler resume и новый строгий checkpoint contract. Старый corrected runner/loader/defaults сохранить. Проверка целостности, полных классов, shapes и contracts обязательна; никаких strict=False, unsafe pickle fallbacks или молчаливого игнорирования fusion weights.
2. В обеих ветках одинаково продолжать одну выбранную и побайтово проверенную C2. Контроль тоже получает всё дополнительное обучение. Новые одинаковые optimizer/scheduler states для общих параметров; это fresh continuation, не exact historical resume. Сохранить B2, 18-head loss, masks, raw graph и historical C2 transposition. Не добавлять WS1 feature-complete transformation, другой loss или source mixture.
3. Реализовать fusion-aware evaluation binding для имеющегося POP arithmetic, не ослабляя старый loader. Выбор checkpoint и tuning — Dilemma VALIDATION. POP native primary/secondary views фиксированы; его последующая оценка не становится новым независимым zero-shot benchmark.
4. CPU acceptance: zero-init identity, ненулевые cross-head gradients, старый путь unchanged, deterministic interrupted/uninterrupted continuation с optimizer/RNG/sampler и scheduler, strict corrupt/mismatched checkpoint rejection, per-head alignment и evaluator действительно использует nonzero fusion. Запустить нужные существующие регрессии.
5. Проверить фактическую готовность performance C2-equivalence gate по актуальным records, а не выводить её из C0 speedup. Если CUDA gate отсутствует, подготовить конкретную ограниченную команду и ресурсный план для очереди координатора. Не запускать GPU/SSH самостоятельно.
6. До будущего paired pilot предложить и зафиксировать P updates, LR/warmup/decay, cadence и stopping rule; обосновать ресурсами, не произвольными «1000» и не результатами одной ветки. Обе ветки имеют одинаковые updates/draws; учитывать дополнительное время/параметры отдельно. Основные результаты: прежний eight-head macro-F1, 18-head breakdown, joint harmonic correctness/consistency и class confusions, с отдельным cadence/metrical monitoring. Не обещать fusion исправление cadence.

Готовность: CPU-tested runner/resume/eval, reproducible configs и конкретный gated GPU пакет. ID префикс EXP-FUSION-*. Внесение runtime default или baseline promotion, полный 120k, новый decoder, multi-corpus/SSL и TEST вне этой задачи.

