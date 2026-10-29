# Review постановки перед реализацией

25 октября 2026. T08, review существующего дизайна и сохранённых результатов.
Вердикт: начинать паспорт данных и минимальный контролируемый baseline можно;
полный C3/H/NBF/box pipeline пока не готов к обучению как одна система.
Ниже рекомендации, а не принятое изменение default architecture или новый бюджет.

## Что обосновано

Сохраняются отдельные наблюдаемые ноты, конкретные HarmonyEvent и связи участия;
функция задаётся в контексте, неизвестные labels маскируются, H предсказываются
из raw music, oracle выделен отдельно. Это полезное ядро
[spec](spec.md) и [domain note](../../../research/HARMONIC_EVENT_GRAPH_DESIGN_20261025_RU.md).
Иерархия типов, временное вложение и музыкальная редукция остаются разными задачами.

| Утверждение | Основание и доступность | Вердикт |
|---|---|---|
| C3 сохраняет состояния отдельных узлов | Текущий `PLLContextEncoder._encode_graph`, прочитан | supported |
| Pilot readout ограничен парой endpoint states | Текущий `RelationHead`, общая 64d projection | supported |
| Именно это ограничение вызывает наблюдаемые ошибки | Нет контролируемого опыта с refiner | unsupported |
| Full-pretrained TransE дал mean Recall@5 0.5630, но always-linked во всех seeds | REVIEW001 и связанный comparison; SHA/size проверены | descriptive; один split, weak labels |
| 67 произведений достаточно для всех четырёх модулей | Field-slot counts не дают joint/per-family/class support | unverifiable до task passport |
| NBF и box улучшат музыкальный анализ | Есть методологические основания, музыкального сравнения нет | unsupported до отдельного опыта |

Проверены canonical fingerprint
[REVIEW001](../../../experiments/records/EXP-LLM-RELATION-REVIEW-001.json),
SHA/size его `comparison.json` (18349 bytes) и compact archive (168121 bytes).
Внутренние файлы архива и remote checkpoints в этом review повторно не проверялись.
Сохранённые результаты: existence global macroF1 0.2874 у full-pretrained TransE
против 0.3737 у constant-none; 96.90% dev negatives приходятся на три works.
Новые метрики на корпусе не вычислялись; это повторное чтение terminal evidence.

## 1. Главный пробел: обучение выбору цели

В текущем [runtime](../../../../src/music_critic/relation_probe/runtime.py)
пара известна при `positive or not source.label`: для linked source есть
положительные tails, а для none source — отрицательные. Остальные tails linked
source остаются unknown, что соответствует контракту неполной разметки.

Однако функция `score(source, target) = a(source)` может правильно разделить
все такие размеченные пары, вообще не различая targets. Это неоднозначность
supervision; она не доказывает, что обученная TransE фактически выбрала этот
обход. Existence collapse также не локализован одним этим наблюдением.

Перед заявлением о качестве адресного reasoning нужен subset с известными
positive и reviewed-false candidates **у одного source**. Их можно получить
из уже доступных явно отрицательных фактов или полностью проверенного набора
исходящих связей в фиксированном candidate domain; наличие таких scopes ещё
нужно установить. Случайные неуказанные tails нельзя превращать в negatives.
Если поддержки нет, адресный contrastive loss и доказательный ranking experiment
откладываются; остаётся слабый описательный pilot или отдельная разметка.

В baseline добавить source-only score и raw time/pitch proximity; tie-breaking
фиксируется и не использует targets. Pairwise preference accuracy на известных
positive/negative парах одного source проверяет именно выбор цели. Positive-only
Recall@k остаётся допустимым ограниченным отчётом, но не заменяет такую проверку.

## 2. NBF и Query2box пока следует сравнить как разные пути ответа

[NBFNet](https://arxiv.org/abs/2106.06935) учит представления пар через агрегацию
путей; [Query2box](https://arxiv.org/abs/2002.05969) представляет запросы boxes
и оценивает сущности в общем embedding space. Это не готовый интерфейс
`query-conditioned candidate states → box`.

Минимальный предлагаемый контракт:

```text
C3 + raw-derived H candidates -> shared entity states z[v]
  A: query refiner(G, z, source, relation) -> scores indexed by candidate IDs
     -> min/product over the same candidate IDs for conjunctions
  B: inductive box query head(shared z, query program) -> box -> scores on z[v]
```

Путь A близок по представлению ответов к
[GNN-QE](https://proceedings.mlr.press/v162/zhu22c.html); product здесь оператор
fuzzy scores, не автоматически калиброванная вероятность совместного события.
Путь B обязан вычислять embeddings новых событий из музыки, без lookup table
по ID произведения/ноты. Оба используют одинаковый universe, masks и query set.
Дальнейший совместный вариант возможен, но требует общей системы координат,
явного bridge/loss и самостоятельного сравнения. Box остаётся компактным
представлением множества и тоже может терять структуру; это проверяемый tradeoff.

## 3. Сначала ограниченные H-кандидаты, затем свободный detector

Неполные H labels не дают честного background для unmatched slots. Одно лишь
masking не ограничивает количество дубликатов и не идентифицирует objectness.
Для первого пути предлагается детерминированный bounded набор interval
candidates по raw timing, с явными правилами длины, дедупликации и бюджета.
Это кандидаты, а не утверждение, что каждая временная ячейка является аккордом.
Сначала измеряется их coverage и учатся membership/attributes там, где есть
targets. Свободные slots, число H и learned boundaries — отдельный шаг при
полных H scopes либо явно обоснованной схеме partial supervision.

Единицу H фиксировать до экспорта: конкретный интервал/участники в выбранном
view, включая повторяющиеся одинаковые аккорды и допустимые перекрытия.
IoU matching должен учитывать scope/view и не сливать одновременные гипотезы.

## 4. Граф должен содержать доступные на inference данные

Первые связи NBF: raw temporal/containment и геометрическая incidence между
нотами и H-кандидатами. Последняя означает попадание в интервал, не chord-member.
Такой опыт проверяет структурированное извлечение отношений из музыки;
семантический KG completion с готовыми теоретическими edges — другая постановка.

Нужна таблица каждого входного канала: raw / predicted / query-given / oracle,
источник, train/eval availability, границы времени и view. Gold target, inverse,
derived equivalent, target-dependent candidate pruning и cached pooled features
не должны раскрывать ответ. Удаление одной стрелки не устраняет все эти пути.
Основной reasoner обучается на тех же типах proposals/features, что на inference;
oracle labels/objects — явно отдельное условие. Если позже добавятся predicted
semantic edges, нужны predictions вне обучающих folds либо отдельная проверка
разрыва качества train/inference; эту сложность проще не вводить в первый опыт.

Пока предлагается offline analysis с объявленным окном и доступной будущей
музыкой; результат нельзя автоматически трактовать как causal critic генерации.
Gold K в вопросе допустим только как явно заданное условие; для raw-only анализа
тональный контекст должен предсказываться. IDs anchors маршрутизируют запрос,
не становятся обучаемыми идентификаторами произведений.

## 5. Проверять пользу структуры, а не только дополнительных параметров

Минимальный reasoner comparison: frozen C3 + static MLP/TransE; сопоставимый
query-conditioned decoder без propagation; query-conditioned refiner с typed
propagation. Общие candidates, features, source queries, exposure и masks.
Число параметров и compute публикуются отдельно: равенство updates их не уравнивает.
Диагностическое отключение группы рёбер после обучения показывает чувствительность;
для утверждения о пользе группы нужен также обученный сопоставимый контроль.
Изменения только refiner edges не доказывают роль топологии уже внутри C3.

Для box сравнения нужны запросы, где каждый конъюнкт по отдельности оставляет
неверные кандидаты, а пересечение их исключает. Такие distractors должны быть
подтверждены labels, не придуманы по отсутствию разметки. Простые запросы,
решаемые одним time-overlap, годятся для smoke, но мало различают reasoning.
Held-out combinations отделяются от новых works; число сгенерированных запросов
не увеличивает число независимых произведений. Shared-witness DAG остаётся позже.

## Рекомендуемый порядок и условия старта

| Шаг | Результат | Условие перехода |
|---|---|---|
| 0. Task passport | Нормализация словаря; per-family/class/view support; same-source negatives; H/query completeness; общий input contract | Есть поддержанная задача, masks и оценочный universe |
| 1A. Узкий relation pilot | Frozen C3, фиксированные note candidates, static/conditioning/propagation controls | Нужная адресная supervision подтверждена; H для этого опыта не обязательны |
| 1B. H path | Bounded raw interval candidates, membership/attributes, oracle и predicted режимы | Coverage, дубликаты и supervised отрицания учтены |
| 2. Composite answers | Min/product против inductive box head | Есть различающие AND queries и проверенная совместная разметка |
| 3. Adaptation | Joint finetune; scratch C3 отдельным сопоставимым условием | Есть сигнал, оправдывающий обучение большего числа параметров |

1A и 1B — раздельные оси: дорогой H detector не нужен для первого теста
query-conditioned propagation на нотах. Если same-source negatives отсутствуют,
приоритет получает поддержанный membership task или подготовка reviewed subset;
матрица запусков не создаётся автоматически из всех возможных комбинаций.

До GPU запуска фиксировать один primary endpoint, минимально полезную разницу,
допустимую ошибку пустого ответа, candidate coverage, selection на inner-fit,
paired seeds, update/wall/VRAM caps и правила остановки. Для relation pilot
предлагается query-macro Recall@5 совместно с обязательным existence/empty-answer
guardrail; нельзя объявить успех по одному recall при always-linked поведении.
Числовые gates выбираются до оценки новых условий, а не под их результат.

Неопределённость считать по независимым works/families, seeds показывать отдельно.
Bootstrap по тысячам нот завысит уверенность; три seeds не заменяют новые семьи.
Старые 12dev уже использованы, остаются exploratory holdout. Global TEST закрыт;
pretraining raw overlap неизвестен. Раздельный отчёт native/LLM origin и небольшой
независимо reviewed набор нужны для будущих заявлений о музыкальной правильности.
Контроль транспозиций — полезный следующий тест; объёмный новый словарь,
hyperbolic geometry и полноценную редукционную иерархию сейчас добавлять незачем.

## Scope review

Проведён один независимый read-only review модульных стыков и плана сравнения;
его существенные замечания включены выше. Код, данные, terminal records, default
architecture и legacy не изменялись. Corpus payloads/weights не открывались,
обучение, inference, новая GPU сессия и платные ресурсы не запускались.
Перед следующим substantial admission/run нужен новый planned record согласно
[EXPERIMENT_LOGGING](../../../EXPERIMENT_LOGGING.md). Этот review не является
новым экспериментом или повторной оценкой модели на датасете.
