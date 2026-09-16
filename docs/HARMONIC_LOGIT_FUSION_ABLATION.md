# Harmonic logit fusion — mechanics and proposed ablation

Дата: 2026-09-07. Статус: **первый implementation increment прошёл CPU acceptance;
научное сравнение остаётся предложенным, запусков нет**.
Область: отдельная ограниченная абляция после corrected Phase 9E; без изменения
замороженного baseline, продвижения в production или обещания улучшения.
Этот документ не заменяет ADR-108, ADR-C0C2-001 и существующие C0/C2 контракты.
Первоначальный документ был только предложением. После следующего запроса
пользователя координатор начал отдельную реализацию механики и CPU tests.

## Первый implementation increment

Объём — отдельные `HarmonicLogitFusion` / `HarmonicFusionModel`, новый output
contract и строгий model-only checkpoint. Старые corrected model, trainer,
loader, loss, граф и performance-путь не меняются. Production trainer/resume,
CUDA acceptance, параметры scientific pilot и fusion-aware POP evaluator
остаются следующими интеграционными задачами. Сохранение optimizer state в
synthetic test не является готовым production resume.

В замороженном реестре 12 гармонических голов имеют суммарно 518 logits.
Описанные матрицы без bias добавляют **216 886 параметров** к исходным
3 661 936: +5,9227%, всего 3 878 822. Это подсчёт архитектуры, не замер
скорости или качества. Disabled wrapper не создаёт fusion Parameters и
возвращает исходный output; его стандартный `state_dict` имеет префикс
`base.`. Исходный `base.state_dict()` и старые checkpoint keys не изменяются.

Новая сериализация идентифицирует именно base + fusion и не выдаёт wrapper
за старую corrected model. Она хранит только веса и контракт; RNG вызывающего
кода сохраняется, optimizer или сохранённый training RNG не восстанавливается.

Здесь нет нового обучения на реальном корпусе. Независимое review и root-gate
завершены: **109 passed, 0 failures/skips, 2 прежних Torch JIT deprecation
warnings, 8.29s**. Это 65 новых fusion/checkpoint cases и 44 regression cases
для исходной corrected model, POP checkpoint loader и repository contract.
`git diff --check` прошёл.

Добавлены файлы:

- [harmonic_logit_fusion.py](../src/music_critic/experiments/analysisgnn/harmonic_logit_fusion.py)
  — слой, wrapper, отдельный output и полный model/class contract;
- [harmonic_fusion_checkpoint.py](../src/music_critic/experiments/analysisgnn/harmonic_fusion_checkpoint.py)
  — snapshot и strict CPU model-only loader;
- [model tests](../tests/experiments/test_harmonic_logit_fusion.py) и
  [checkpoint tests](../tests/experiments/test_harmonic_fusion_checkpoint.py).

На fixture включённая zero-init fusion сохранила logits, все 18-head losses,
каждый исходный gradient tensor и RNG при stochastic train forward. При
ненулевой связи gradient quality loss достигал root logits/head и общего
encoder даже после отключения прямого quality path в тесте. Missing и
unsupported targets по-прежнему маскируются прежним loss. Проверены пустые
кандидаты, неодинаковые размеры samples, границы/дубликаты global IDs,
совпадение межголовного alignment и FP32/finite guards. Это не утверждение
о точной CUDA-траектории будущего обучения.

Model-only roundtrip сохраняет base + ненулевые fusion weights. Изменённые
class labels, schema, keys, shapes, dtypes, NaN/Inf и SHA отклоняются;
старый loader отвергает новый формат. Отдельный synthetic test подтвердил
совпадение следующего AdamW шага после восстановления модели и внешнего
optimizer state; loader сам optimizer не строит. Legacy не использовался.

## Вопрос и текущая опора

Помогает ли обмен предсказаниями между гармоническими головами при том же
backbone, данных и бюджете обучения? Текущий corrected model выдаёт 18
независимых FP32 MLP-предсказаний после общего encoder и onset BiGRU. Targets
присоединяются после raw-only forward. Двенадцать задач читают одни и те же raw
beat-кандидаты: `local_key`, `tonicized_key`, `root`, `bass`, `primary_degree`,
`secondary_degree`, `quality`, `inversion`, `roman_numeral`, `pitch_class_set`,
`harmonic_rhythm`, `pedal`. `cadence` читает onset, пять остальных задач — note.

Основания: [архитектура](ARCHITECTURE.md),
[corrected model](../src/music_critic/experiments/analysisgnn/corrected_model.py),
[реестр задач](../src/music_critic/experiments/analysisgnn/multitask_contract.py),
[разбор C0/C2](experiments/C0_C2_120K_REVIEW.md).
Наблюдаемый выигрыш C2 относится к исторической транспозиции, не к fusion;
причина cadence-регресса ещё не установлена.

## Минимальный вариант

Пусть `z_h` — исходные FP32 logits головы `h` на каждом raw beat. Предлагается
один одновременный слой коррекции:

```text
z'_h = z_h + W_h · concat(z_k for k in harmonic_heads if k != h)
W_h.shape = [C_h, sum(C_k for k != h)]
```

`W_h` инициализируются нулями; bias отсутствует. Порядок голов и классов
фиксируется fingerprint реестра. Используются исходные logits, без argmax,
температуры, дополнительной нормализации, dropout, attention или рекурсии.
Все поправки вычисляются из одного набора исходных logits, а не поочерёдно из
уже исправленных. Размеры выводятся из зафиксированного реестра.

Это намеренно минимальный cross-task baseline. Он **не реализует официальную
attention-fusion AnalysisGNN и не является её воспроизведением**. Новые факты
об официальной модели или результаты из публикаций здесь не утверждаются.

Fusion работает до alignment и не принимает labels, masks, confidence,
sidecars или gold segmentation. Межголовной `detach` отсутствует: loss одной
головы может обучать другую через её предсказания. Отсутствующая разметка
головы не выключает её raw prediction и не становится отрицательным target.
Шесть остальных выходов wrapper возвращает без изменений; после обучения
общего backbone их качество всё равно может измениться.

## Изоляция от baseline

- Будущая реализация — отдельный opt-in wrapper/module и отдельные tests,
  например `experiments/analysisgnn/harmonic_logit_fusion.py`. Общие model,
  heads, graph, training, performance и frozen profile файлы не изменяются.
- Старый default путь не создаёт fusion Parameters, не расходует новый RNG,
  сохраняет state keys, contracts, initialization и вычисления побитово.
  Выключенный wrapper непосредственно возвращает исходный результат.
- Включённый wrapper получает новый model/profile/checkpoint contract с
  исходным model fingerprint, полной fusion-конфигурацией, реестром и весами.
  Он не экспортируется под `CorrectedAnalysisGNNModel@1.0.0`.
- Импорт базовых весов строгий и с provenance. Нельзя скрывать отсутствующие
  или дополнительные ключи через `strict=False`, изменять старые checkpoint
  bytes либо выдавать новую ветку за exact resume исторического C0/C2.

## Пара и условия, фиксируемые до запуска

Первый научный запуск — короткий paired pilot **после** принятия performance
пути и его отдельной эквивалентности для выбранной C2-транспозиции. Ускорение
C0 само по себе этот gate не закрывает. Подготовка кода и synthetic tests может
идти параллельно; документ не разрешает запуск реальных данных.

| Условие | Предлагаемая фиксация для обеих веток |
| --- | --- |
| Контроль A | Та же исходная модель и тот же backbone, без fusion; она тоже проходит весь pilot |
| Ветка B | Копия того же состояния плюс описанная zero-initialized fusion |
| Начальное состояние | Один заранее выбранный и побайтово проверенный corrected checkpoint; предпочтительно выбранный C2 best после проверки provenance; одинаковые base tensors |
| Продолжение | Новый эксперимент, свежие одинаковые optimizer/scheduler/scaler states для общих параметров; не восстановление старого training state |
| Обучаемые модули | В обеих ветках весь исходный backbone, BiGRU и 18 heads; только B имеет дополнительные fusion Parameters |
| Seed/RNG | Seed 17 для первого описательного pilot; одинаковые отдельные dropout/sampler domains; создание fusion сохраняет RNG исходной модели |
| Split | Замороженный B3 component split 1295/162/162; TRAIN для updates, Dilemmadata VALIDATION для оценки; TEST закрыт |
| Batch | B2, те же полные raw pieces, границы батчей и отсутствие gradient accumulation |
| TRAIN schedule | Один сохранённый упорядоченный поток record/component/shift IDs исторического C2-профиля; одинаковое число предъявлений и границы эпох |
| Inputs | Одинаковые graph/feature/vocabulary/sidecar/transposition fingerprints; WS1 complete transposition не добавляется в эту пару |
| Loss | Прежние per-head masks, supported TRAIN class weights и reduction; `L_primary + 0.25 * L_auxiliary`, все 18 heads; без нового consistency loss |
| Precision/update | FP32, прежние finite guards, clipping 1.0 и applied-update правила; никаких ослабленных проверок |
| Выбор checkpoint | Только одинаково запланированная identity Dilemmadata VALIDATION и прежний primary score; final и best сохраняются раздельно |

Перед регистрацией численно зафиксировать короткий applied-update budget `P`,
число предъявлений `2P`, полный график LR/warmup/decay, AdamW-параметры,
validation/checkpoint cadence, seed domains и правило остановки. Не выбирать
эти значения по результатам одной ветки и не обрезать молча старый 120k
scheduler. **Пока эти поля, checkpoint SHA/size/state и runtime/source hashes
не заполнены, протокол не готов к запуску.** Бюджет и schedule выбираются один
раз после performance gate, до результатов pilot.

Здесь matched budget означает равенство updates, draws и base forward/backward
проходов. Дополнительная fusion имеет собственные параметры и вычислительную
стоимость: записывать их, wall time, peak memory и failed/skipped attempts;
не объявлять два варианта равными по FLOPs или времени без измерения. Контроль
не заменяется исторической C2-метрикой без такого же дополнительного обучения.

## Оценка и решение

На Dilemmadata VALIDATION сохранить исходную точку, final и выбранный best;
проверить прежние primary score, harmonic joint и отдельные метрики всех 18
голов. Сохранить per-class confusion/support и per-component predictions.
Особое внимание cadence, включая IAC/PAC/HC и остальные доступные классы:
она не входит в primary, и рост primary не отменяет её ухудшения. Не менять
loss weights для исправления обнаруженного регресса внутри этой же пары.

Оценку unseen tuples подписывать по фактическому словарю TRAIN-экспозиций;
native-unseen не объявлять augmentation-unseen. При all-shift диагностике
фиксировать eligibility и отдельно общий набор произведений. Эти проверки
не означают оценки музыкального качества.

Положительный pilot поддержит дальнейшую проверку добавленного слоя, но не
докажет преимущество между seeds или пользу именно межголовных связей отдельно
от дополнительной ёмкости. Для последнего нужен отдельный заранее описанный
контроль добавленных параметров без cross-task обмена. Null/negative outcome
сохраняется; автоматического расширения бюджета или baseline promotion нет.

## POP909-CL как отдельная проверка переноса

Сначала полезно получить исходную C0/C2 transfer-оценку без нового обучения.
[WS4](POP909_CL_FROZEN_EVALUATION.md) сохраняет synthetic-only API. Отдельный
[TRAIN/VALIDATION runner](POP909_CL_TRANSFER.md) для старых C0/C2 уже реализован
и проверен на fixtures; реальные source/checkpoint artifacts ещё предстоит
привязать и оценить в зарегистрированных запусках.

Оценка fusion дополнительно требует **новой версии model loader и evaluation
contract**, которая явно проверяет base + fusion веса, shapes, class order и
fingerprints. Старый strict corrected loader и его контракт не ослабляются.
Нельзя подать wrapper под старый fingerprint или незаметно отбросить fusion
weights. Root37→PC marginalization, full17 quality predictions/subset9 gold,
native/metadata_removed views, независимые masks и exact event→beat alignment
фиксируются до результатов; сравнивать с контрольной веткой на тех же данных.

POP не используется для настройки fusion, LR, loss, размера pilot, выбора
checkpoint или входного view. При разработке решения по результатам POP это
уже development evidence, и потребуется новая независимая transfer-проверка.
TEST не читается и не используется ни на одном этапе предложения.

## Bounded implementation gates

Будущие source-free CPU tests должны проверить:

1. Старый default и disabled wrapper: точные logits/loss/gradients/state/RNG
   без fusion keys; enabled zero-init: совпадение базовых logits и gradients.
2. Все 12 голов совпадают по sample/entity indices, типу, порядку и offsets,
   а не только числу строк; неверные метаданные/widths отклоняются. Покрыть
   неодинаковые длины произведений, пустые beat candidates и перестановку batch.
3. После ненулевой fusion loss A достигает logits B и backbone. На первом
   zero-init шаге нулевой cross-task gradient ожидаем; сами `W_h` обучаемы.
   Parameters входят в optimizer ровно один раз, между samples связи нет.
4. Изменение/отсутствие targets не меняет raw forward; unavailable/unsupported
   targets остаются masked; остальные шесть выходов wrapper неизменны.
5. Несколько synthetic updates, finite guards и собственный checkpoint/resume
   сохраняют состояние и следующий шаг; неправильные contracts/keys/dtypes
   отклоняются до мутаций. Старый loader не принимает новый формат.

CUDA/runtime equivalence, memory and timing gates имеют собственные границы
доказанности; synthetic overfit не считается музыкальным результатом.

До любого существенного запуска выполнить [EXPERIMENT_LOGGING](EXPERIMENT_LOGGING.md):
отдельный planned ID каждой ветки/оценки, связь с исходным checkpoint/run,
новые output roots, неизменяемые artifacts и terminal records, включая failures.
Этот предложенный дизайн не является experiment record и не регистрирует запуск.

Не входят в первый implementation increment: чтение реальных datasets/checkpoints/TEST,
официальная реплика AnalysisGNN, изменение графа/разрешения beat, CLS/tokenizer,
SSL/PDMX обучение, PLL/fragility, обучение на деградациях, настройка cadence,
новые зависимости, legacy inspection/import, изменение performance-кода,
автоматическое production promotion. Полный scientific pilot требует отдельной
интеграции trainer/resume/evaluator и регистрации до запуска.
