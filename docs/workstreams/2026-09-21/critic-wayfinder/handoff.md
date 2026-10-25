# Передача карты в mc-to-spec и mc-to-tickets

Актуализация после команды реализации: [v1 реализован](implementation.md),
synthetic CPU приёмка пройдена. Следующая работа — конкретный data/runtime/run
protocol; прежние записи о неготовом коде ниже описывают этап спецификации.

Статус22сентября: **[спецификация v1 готова](spec.md) для инженерной реализации**.
Ниже сохранены история и границы исходной карты. Актуальные contracts и
приёмка, включая добавленную локализацию, находятся в `spec.md`; код и run
admission пока не выполнены. Данные для musical-error localization отсутствуют:
первая local head обучается находить вмешательство, не гарантированную ошибку.
[Карта решений](README.md). Phase9E evidence → ограниченная Phase12 feasibility.
Не менять accepted архитектуру только потому, что карта рекомендует вариант.

## Актуализация22сентября: multilevel aspect critic первым

Пользователь выбрал несколько уровней, двунаправленный контекст,
rhythm/harmony/melody/structure и взвешенный итог. Harmony-only beat scalar
больше не основной вариант. Актуальное предложение:
[три recurrent пути, обмен векторами и aspect readouts](../../../research/MULTISCALE_ASPECT_CRITIC_DESIGN_20260922_RU.md).
Точная конструкция пока не принятая спецификация. Следующие seams для неё:
source/stage-aware feature pyramid, raw track-event/slice/bar sequences,
context fusion, aspect masks и reference-score policy. Existing beat probe
остаётся контролем; PLL — более поздний отдельный путь.

Weak aspect pairs позволяют начать без listener labels. Нужны work/component
split, учёт overlap с backbone TRAIN, отдельные operator-family holdouts и
размер допустимого TRAIN expansion.20works — development/feasibility, не
достаточная база для надёжной многокомпонентной модели. Человеческая оценка
остаётся последующей независимой проверкой. Raw track не равен голосу/партии,
а local-order не равен форме целого произведения.

Остаток около7300₽ сообщён пользователем; C2 artifact byte-verified, temporal
binding ещё надо закрыть. Прежние бюджеты не подтверждают вместимость нового
дизайна. Ниже сохранена историческая probability/small-probe decomposition;
S4 human labels и E2 probability больше не blockers первого weak critic.

## Исторические материальные выборы перед исполнением

Подтверждено: human labeling — пользователь и 1–2 слушателя; raw MIDI/exact time/
masks/provenance/TEST lock; Cloud вместо 3090; сейчас только документы.

Открыто: принять ли первый pitch-set-given-rhythm охват и harmony-aspect;
выбрать lineage-safe 24-work evaluation scope/жанры и practical effect margin;
подтвердить конкретный available checkpoint и Cloud funds/quota; выбрать какие
из E1–E7 и с каким конвертом исполнять. Не требуется спрашивать заново о наличии
слушателей. Бюджетный выбор делается по готовому протоколу, не вслепую.

## Исторические срезы probability/small-probe спецификации

| Срез / статус | Поведение и seam | Наблюдаемая приёмка | Зависимости |
|---|---|---|---|
| S0 source binding / ready to specify | Evidence manifest для выбранного checkpoint/source и отдельной target projection | Несовпадение schema/hash не даёт продолжить; stale/unknown видимы | D1, D3; доступ к байтам ещё неизвестен |
| S1 masked raw input / ready to specify | `MaskedConditionalInput`, `BlockQuery`, separate target sidecar, новый versioned graph view | Hidden-content substitutions не меняют predictor input; graph-before-target; rational boundaries | D2, D3; окончательное conditioning выбрать до реализации |
| S2 normalized decoder / ready to specify | Separate `ConditionalBlockDistribution.log_prob`, Bernoulli baseline, optional AR | На fixture mass=1; correct teacher forcing; deterministic block schedule; empty/unavailable contract | S1, D4 |
| S3 probability score/localization / ready to specify | Per-level scores/support/normalizers/mask IDs; explicit pooling | Repeated scoring stable; overlap/empty support; metadata/track robustness report с допусками; no full-view cache | S2, D4 |
| S4 human pair schema / ready to specify | Pair order, ties/uncertainty, exact spans, rater/renderer/provenance, group split | Pair swap; no sibling split; no name/operator features; fixed rater aggregation; missing aspect mask | D3, D5; независимая от S1 работа |
| S5 frozen critic readout / blocked on S0, S4 | C2/C3/raw-summary controls, aspect+pair predictions и local pooling; theory heads сохранены | Synthetic pair overfit; pair-swap expected outcome; pool coverage; absent labels masked | S0, S4, D4 |
| S6A native frozen gate / not authorized | Memory, equivalence, resume + current cost binding | Документированный E1A; CPU tests не выдают GPU admission | S0, S5 и Cloud проверки; не S1–S3 |
| S6B native probability gate / not authorized | Отдельный masked runtime admission | Документированный E1B; контроль target visibility и memory | S0–S3 и Cloud проверки |
| S7 experiments / not authorized | E2–E7 отдельными задачами с новыми ExperimentRecord IDs | Terminal artifacts, comparison limits, human holdout, stop caps | A: S4–S6A; B: S4+S6B; combination: обе выбранные ветки; выбор пользователя |

Это decomposition hints, не созданные issue/tasks и не экспериментальный реестр.
После принятия ограниченного scope `mc-to-spec` оформляет public contracts,
negative/availability behavior и compatibility; `mc-to-tickets` делает локальные
самостоятельно проверяемые issues с acyclic blockers. Engineering, native gate
и science evidence не объединять в один критерий «готово».

Не копировать старый SSL harness, теряя BiGRU/temporal. Не загружать полные
checkpoints в обзор документов. Не исправлять незаметно historical transpose/PC
targets и не использовать gold roles/segmentation. Для roadmap расширения
пользоваться D6, а не включать всё в первый implementation ticket.

## Завершение исходной карты21сентября

Новые документы: evidence review, этот handoff, README и семь decision files;
обновлён STATUS. Реализация не менялась, experiment/ledger/DECISIONS не менялись:
новых runs/imports и принятой architecture decision нет. Legacy logic не
инспектировалась/не переиспользована. Проверки этой задачи — документационные
ссылки/границы/dependency DAG и независимая проверка содержания, не unit tests
модели и не научный результат. Следующий этап — конкретная mc-to-spec по выбранному
первому срезу, затем mc-to-tickets, без автоматического запуска опытов.
