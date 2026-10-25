# D4 — Вероятность, обучаемый критик и fragility

Статус: **непроверенные гипотезы**. Зависимости: [backbone](01-backbone.md),
[объекты/маски](02-objects-masks.md), [targets](03-data-targets.md).
Владелец будущих измерений: назначить перед запуском. [Карта](../README.md).

## Актуальный основной вариант — multiscale aspect critic

Последующее уточнение пользователя22сентября отвергает beat→scalar как основной
readout. Требуемое направление: совместные note/event, sounding slice, beat,
bar/span representations, двунаправленный контекст, четыре aspect scores и
взвешенный итог. Предложенная конструкция и проверенные code paths находятся в
[исследовании](../../../../research/MULTISCALE_ASPECT_CRITIC_DESIGN_20260922_RU.md).

Рекомендуется сохранить frozen encoder и обучать надстройку: stage-aware
feature pyramid; shared per-track-event BiGRU; sounding-slice BiGRU; bar BiGRU;
ограниченный обмен vector messages между уровнями; отдельное attention pooling
по аспектам. Скаляры появляются после обработки контекста. Новые track-time и
span sequences не выдаются за уже готовые pretrained states. Точные размеры и
gates ещё предложение, не утверждённая реализация.

Начальные выходы: harmony/rhythm/melody/local-order. Whole-form quality требует
более длинных данных, arrangement dropout пока диагностический. Weak labels
применяются только к заявленному аспекту пары; остальные неизвестны, а не
обязательно неизменны.20works достаточно для проверки механики, но для обучения
богатой надстройки предложено расширение TRAIN works существующими операторами.
Автоматическая генерация и обучение не запускались. Human labels проверяют
соответствие вкусу позже, а не блокируют первую weak-supervision реализацию.

Общий score использует явные веса и предварительно фиксированные reference
scales аспектов, сообщает coverage и scope. Reference percentile — относительная
шкала, не человеческая калибровка качества. Changes и PLL остаются отдельными
ветвями; masked PLL не может использовать full-view feature cache.

## Исторический beat-only readout — теперь только контроль

Этот более ранний вариант сохранён как дешёвый baseline; он **не является
актуальным предложением основного критика**. Его harmony-only scope и feature
interface не ограничивают новую конструкцию выше.

Пользователь хочет использовать уже подготовленные искажения. Человеческая
разметка не является blocker первого диагностического/weak-supervision пилота.
Существующие original/variant пары допускаются как **синтетические слабые
предпочтения**, с assumption `original > edited`; это не gold человеческого
качества. Для первой harmony head использовать пригодные гармонические пары
из46 подготовленных альтернатив, не смешивать ритм/удаление партии/порядок
тактов в один предполагаемый harmony target. Остальные семьи можно прогнать
как descriptive diagnostics. Проверить фактический inventory/валидность перед
выполнением; малое число20works ограничивает переносимость результата.

Предлагаемый интерфейс `FrozenCriticFeatures` содержит только контекстные
beat embeddings `[B,128]`, sample ownership, raw exact beat times, покрытие
фрагмента и binding checkpoint/source/graph. Он не содержит target labels,
operator ID, имени файла, признака original/edited или известной области правки.
В C2 это `encode(raw).fused.embeddings['beat']`; в temporal/C3 брать **финальные**
beat states соответствующего model output, после temporal refinement.
Эти пути проверены по текущим `corrected_model.py` и `temporal_slices/model.py`;
для иного C3 source нужен собственный adapter/binding. Не считать base encode
финальным temporal representation.

Encoder в `eval()`, его параметры `requires_grad=False`; features один раз
извлекаются без autograd и кешируются отдельно для каждого original/variant.
Optimizer владеет только readout. Два заранее фиксированных кандидата:

- линейный baseline128→1:129параметров;
- малый `Linear(128,32) → GELU → Linear(32,1)`:4161параметр.

Одна и та же голова применяется к каждой raw доле. Она возвращает `u_b`;
`s(X)=sum_b coverage_duration_b*u_b / sum_b coverage_duration_b` — относительный
балл фрагмента. Нет поддержки → unavailable. Частичные крайние доли учитываются
по точной длительности внутри clip; score не суммируется с числом событий.
Локальные `u_b` можно показывать как вклад readout, но при clip-level supervision
это ещё не проверенная локализация музыкальной ошибки. Парный loss
`softplus(s(edited)-s(original))`; случайно менять стороны пары вместе с target.
Не требовать линейной зависимости score от ручки силы искажения: perceived
severity не установлена. На inference нужен один MIDI, reference не обязателен.
Нулевая точка и масштаб score не являются оценкой качества0–100 или
калиброванной вероятностью предпочтения слушателя.

Sampling сначала по произведению, затем по его парам; все варианты одного
произведения остаются в одном head-train/dev/evaluation group. Не использовать
число вариантов как число независимых произведений. L2/early stopping и выбор
линейной/MLP версии только по dev; raw-summary baseline выявляет простые
плотностные/диапазонные shortcuts. Отложенные пары из inspected suite остаются
development evidence; новые works/operator families и люди нужны для заявления
о переносе музыкального предпочтения. Старые18 heads сохраняются диагностикой,
их confidence/CE не подменяет новое score. Changes — отдельная99-output задача
и не превращается в critic scalar автоматически.

**PLL не входит в этот первый readout.** Следующая самостоятельная ветка учит
на разрешённой обычной TRAIN-музыке normalized decoder скрытого PC-set/блока
по видимому контексту: сначала12 Bernoulli, при полезности — малый AR.
Targets берутся из raw нот, а не из исторической93-class auxiliary vocabulary.
Нужен новый mask-aware input до всего encoder и проверка скрытых производных
признаков/удерживаемых нот; full-view feature cache первой головы повторно
использовать для PLL нельзя. Score — нормированный по поддержке composite
conditional log score, не полная вероятность MIDI. Теоретическая confidence
при видимом target и расстояние embeddings не являются PLL.

Сначала сравнить ranking score и PLL раздельно; позже разрешена калибровка
комбинации на dev с фиксированной проверкой добавочной пользы на других works.
Добавление PLL требует собственных реализации, обучения, masked forwards и
бюджета; это не флаг в MLP и не автоматически оплаченный этап текущего плана.
Полный условный/масочный контракт ниже и в [D2](02-objects-masks.md).

## Сравниваемые пути

| Путь | Обучение / данные | Что способен показать | Риск | Самый малый различающий тест |
|---|---|---|---|---|
| A: frozen multiscale aspect critic | Multilevel states, новый recurrent/fusion context, masked weak aspect pairs; human labels позже | Aspect ranking на declared scope и его соответствие человеческим оценкам | Мало независимых works, genre/operator shortcuts | Raw summaries, beat probe и multilevel pooling без новых GRU; held-out work groups |
| B: masked conditional decoder | Обычная TRAIN музыка, raw-derived targets, proper log loss | Условную совместимость projection с окружением | Rarity/density bias; masked-view shift | Prior vs Bernoulli vs small AR; context controls; NLL/calibration |
| C: fragility | Сначала без обучения; изменение representation/predicted distributions/scores | Чувствительность к музыкальному вмешательству | Большая реакция бывает и на полезную правку | Human-judged edits + benign transforms + support-matched deltas |
| D: A+B/C combination | Малый regularized calibrator на dev labels | Добавочную preference пользу signals | Selection/multiple-comparison overfit | Best individual vs fixed combination на untouched works/families |
| Latent SSL / JEPA | Detached representation target, cosine/distance | Другую гипотезу representation learning | Нет normalized likelihood; прежние null results | Только отдельный matched scratch/theory-only/SSL протокол |

Не нужно выбирать один путь заранее. A и B — ограниченные параллельные baseline
ветки с общим human evaluation; C — диагностика. D запускается только после
полезности хотя бы одного сигнала. Исполнение каждого имеет отдельный лимит.

## Что именно является вероятностью

Пусть `Y_B` — выбранная в D2 **проекция** музыкального блока, `X_vis` — разрешённый
masked input, `C` — явные условия (включая rhythm/query skeleton первой ступени).

`qθ(Y_B | X_vis, C)` должно быть нормированным распределением на фиксированном
target space. Для одного 12-bit set:

- Independent Bernoulli: произведение 12 вероятностей; нормировано на 4096sets,
  но не моделирует зависимость pitch classes.
- 4096-way softmax: моделирует целый set; длинный хвост и редкие комбинации.
- Small autoregressive decoder: произведение условных вероятностей битов;
  фиксированный порядок PC0…11, затем raw event order в блоке. Teacher forcing
  раскрывает только предыдущие target tokens **decoder**, не encoder; это часть
  chain-rule factorization, а не bidirectional ответ в контексте.

Рекомендуемый первый контроль — Bernoulli, затем AR12bits/event+между events.
Большой sequence decoder не нужен до проверки малого. Для variable-length
note/onset/full-track variant понадобятся vocab для exact timing или явная
квантизация с потерями, EOS/count, voicing, factorization и новый admission.
Set decoder даже при normalization может давать массу музыкально/ритмически
нереализуемым наборам относительно данного skeleton; это вероятность projection,
не генеративная валидность raw score. Restrictions support требуют нормирования.

Для level `l`:

`S_l(X) = sum_B w_B log qθ(Y_B | X_vis(B), C_B) / sum_B w_B n_B`.

`n_B` — число predicted binary variables (12×events), также выводить сумму и
per-event score. `w_B` и детерминированный block schedule фиксируются до
evaluation. Веса конечны и неотрицательны; общий weighted support положителен.
При нулевой поддержке возвращается `unavailable`, а не score=0.
Сначала отдельные event/span/track scores; их перекрывающаяся сумма
— composite/pseudo log score, **не log p(всего произведения)**. Whole-track
агрегация из локальных blocks не равна whole-track conditional distribution.

Публикация [Salazar et al., ACL2020](https://aclanthology.org/2020.acl-main.240/)
использует masked token conditional scores для PLL. Расширение к нашим
музыкальным блокам выше — проектное предложение; статья не доказывает согласие
с музыкальными предпочтениями. Existing cosine SSL (`ssl/objective.py:192–252`,
`ssl/multilevel.py:967–975,1061–1068`) не становится PLL после смены названия.

Probability training: unweighted proper BCE/CE на доступных raw targets, с
piece/group-balanced sampling. Class-weighted theory CE не переносится молча
как оценка natural probability. Если меняется sampling prior, назвать distribution,
сохранить reweighting/calibration protocol и проверить на validation.
Temperature/calibration выбирается только на dev-validation, без final holdout.

## Preference и локализация

Сохранить 18auxiliary heads; для critic использовать отдельные aspect readouts и
pooling с coverage masks. Начать с harmony compatibility и pairwise logistic
objective; ties как soft0.5, uncertainty отдельно с маской/снижением веса, не
превращать «не могу оценить» в отрицание. Logistic output обозначать
`q_A = P(A>B) + 0.5 P(tie)` — ожидаемый исход пары, не strict win probability.
Pair-swap меняет `q_A` на `1-q_A`. Если нужна отдельная вероятность ничьей,
вводится отдельный three-outcome model; это не свойство binary logistic head.
Подробности labels/метрик — [D5](05-human-evaluation.md).

Локальная карта PLL surprise — гипотеза location, а не экспертная ошибка.
Проверять против human marked spans и matched benign edits, а не только координат
искусственного вмешательства. Attention maps не являются причинным объяснением.

## Fragility без подмены качества

Сравнивать embeddings/distributions/score на одинаковой наблюдаемой поддержке.
Деление на число изменённых событий или длительность задаётся до evaluation;
сохранять raw и normalized delta. Для удаления/ритмической правки разные graph
slots требуют exact matching и explicit unmatched coverage. Нет поддержки —
`unavailable`, а не нулевая реакция.

Track permutation/renaming — тест желаемой устойчивости. В existing backbone
track ordinal positions влияют на Transformer (`models/hierarchy.py:828–862`),
поэтому точная invariance не гарантирована. Измерять изменения и заранее задать
допуск по аспекту; exact canonicalization была бы отдельным изменением контракта.
Transposition, tempo и
instrument change не универсально нейтральны для вкуса: для harmony-only аспекта
фиксировать условный смысл/транспозиционную equivariance и human uncertainty.
Оценивать полезные human edits, neutral и ухудшающие примеры, скрывая operator ID.

## Когда путь закрывать

Вероятностный путь не превосходит smoothed prior/rhythm-metadata control на
held-out NLL или не использует meaningful context — закрыть текущий decoder,
не делать quality claims. A не превосходит raw-summary baseline — готовое
representation пока не помогает данному preference task. C различает лишь
плотность/длину/следы оператора — оставить технической диагностикой. D не даёт
held-out gain/calibration — сохранить простой отдельный путь. Малый пилот с
широким интервалом даёт «неопределённо», не универсальную бесполезность.
Конкретные bounded stages/лимиты находятся в [D7](07-budget-gates.md).
