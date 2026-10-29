# AutoSchA и пятый вариант построения H

27 октября 2026. T08/HIERGRAPH, исследовательский план. Пользователь хочет проверить
все четыре обсуждавшихся H detector и добавить пятый вариант. Это фиксирует состав
сравнения; реализация, численный run protocol и обучение здесь не выполнены.

## Вопрос и границы доказательств

Можно ли строить H через обучаемое объединение элементов нотного графа, сохраняя
детальные состояния для F/f/P/p, и лучше ли это других способов обнаружения H?
Ни AutoSchA, ни имеющиеся H/member результаты не отвечают на этот вопрос за нас.

Рабочая копия: isolated harmonic-event-pilot, HEAD
`287bf856643f9e555352e1661a7bf0563772178a`, накопленный WIP. Original release checkout
не изменяется. Источники: первичные статьи, публичное приложение AutoSchA, существующий
[паспорт current67](../experiments/HIERGRAPH_SUPERVISION_20261027_RU.md), текущие
encoding contracts. Новый корпусный прогон и доступ к TEST не проводились.

## Что переносим из опубликованных методов

[AutoSchA, §3](https://arxiv.org/html/2512.18232#S3) учит маски сохранения исходных
нот на глубинах: directed multirelational GNN, глобальный контекст, sigmoid scores,
threshold isolation и мягкое взвешивание состояний. BCE использует depth labels;
монотонность поощряется штрафом. Это не генератор H и не готовый parent-child parser.

Создание групп технически ближе к
[DiffPool](https://arxiv.org/abs/1806.08804): learned assignment переводит состояния
узлов в состояния групп и может индуцировать граф между группами. Поэтому E5 ниже —
наша адаптация learned graph coarsening, не воспроизведение AutoSchA и не новая
разметка шенкеровской глубины.

Проверенный [supplement](https://github.com/stephenHahn88/AutoSchA_Supplement)
содержит PDF, README и LICENSE, без реализации. В
[appendix A.2–A.3](https://raw.githubusercontent.com/stephenHahn88/AutoSchA_Supplement/main/AutoSchA_Technical_Appendix.pdf)
подбор параметров и максимумы абляций описаны через test accuracy; основной текст
также использует validation. Неясности split/selection не позволяют считать эти
числа ориентиром качества нашей модели или утверждать утечку данных.

Уточнение прежнего объяснения: отключение локальных рёбер само по себе не
доказывает обнуление всех глобальных путей информации. В нашей реализации raw
residual paths задаются явно и тестируются, а не выводятся из слова «изоляция».

## Пять вариантов, которые остаются в плане

| ID в плане | Механизм создания H | Отдельная проверяемая гипотеза |
|---|---|---|
| E1 | Оценка fixed raw span bank | Достаточен ли контролируемый перебор интервалов при измеренном coverage? |
| E2 | Boundary/segmentation head | Полезна ли совместная последовательная сегментация одного harmonic view? |
| E3 | Anchor-free endpoint head | Помогает ли свободное уточнение границ без списка длительностей? |
| E4 | Learned event queries | Помогает ли набор обучаемых запросов создавать экземпляры H? |
| E5 | Graph-assignment pooling | Помогает ли явное распределение элементов графа по H до чтения их свойств? |

E1–E5 — имена условий, не IDs выполненных экспериментов. Все пять должны пройти
реализацию, mechanics gate и bounded pilot; не исключать E3 или E5 только по
априорной симпатии к более простым моделям. Затем запланированы paired seeds
17/42/73 для всех пяти прошедших технические проверки. Неуспешные условия сохраняют
свои failure records; не выдаются за нулевое качество и не заменяются молча.

## E5: минимальная проверяемая адаптация

1. Существующий C3 и raw features дают состояния нот/времени/контекста. Сначала
   freeze policy общая для всех условий; fine-tuning C3 — последующий контраст.
2. Небольшая GNN предсказывает матрицу pooling weights S по raw graph. Число столбцов
   K — одинаково ограниченная с E4 ёмкость на блок, не число gold H. У E5 нет
   event-query cross-attention decoder: S получается из graph-assignment head.
3. Для группы j вычисляется нормированное состояние
   `H0[j] = sum_a S[a,j] * Z[a] / (epsilon + sum_a S[a,j])`.
   Отдельные головы предсказывают интервал, score и поддержанные свойства H/F.
   Мягкое S позволяет градиентам задач H/F обучать агрегацию; жёсткое удаление нот
   и недифференцируемый выбор компонент не нужны в первом варианте.
4. Raw states, IDs и геометрический доступ ко всем нотам сохраняются. После
   предсказания H создаются contextual h с явным временем. Ноты, важные для
   разрешения, не должны исчезать из-за низкого chord-membership score.

Принципиально различаются четыре величины: геометрическое попадание во временной
контекст, латентный pooling weight S, семантическое участие и chord membership.
Attention/soft assignment не становится semantic probability по названию.
Один общий коэффициент «гармоническая важность» не заменяет эти цели.

Единица a — raw note-time fragment или другой явно определённый raw time-context
элемент с сохранением physical note ID. Разбиение строится без targets по общей
временной политике. Удержанная нота может иметь разные h в нескольких H. Запрещено
навсегда назначать целую физическую ноту ровно одному H. Для минимального варианта
S имеет независимые sigmoid weights с нормировкой по входам каждой группы;
не требуется row-softmax, навязывающий единственного получателя. Пустые временные
участки получают time/context states; событие не обязано начинаться атакой ноты.
Raw grid, residual boundary parameterization, limits и обработка point/grace events
фиксируются общим implementation contract до native gate.

Интервал H не равен min/max времени выбранных нот: арпеджио, удержания и паузы
делают такую замену неверной. Exact target time сохраняется, predicted endpoints
имеют явную политику рационального представления. Одна интерпретация H должна
соблюдать temporal exclusivity; альтернативные views/scopes не смешиваются.

Из S можно получить типизированные coarse связи `A_H[r] = S.T @ A_raw[r] @ S`.
Но их отдельная польза — следующий E5+coarse контраст. В первом сравнении E1–E5
используется один downstream updater с одинаковыми правилами доступа к контексту;
E5 не получает дополнительные H-message rounds или уникальный глобальный модуль.
Иначе преимущество оценивает весь пакет изменений, а не только создание H.
Стоимость S и coarse K×K матриц ограничивается tiles/capacity и измеряется.

## Откуда берётся обучение

По сохранённому паспорту: 45263 H, функция известна у11809 H (57 works),
temporal_role у74587 h, chord membership у146785 h; у1872 P (61 works) известен
тип процесса. Это counts до нового export/matching/split, не новый прогон и
не число гарантированно пригодных примеров. Weak source/LLM provenance сохраняется.

| Цель | Допустимая опора | Что нельзя подставлять вместо неё |
|---|---|---|
| Интервалы/существование H | Exact H intervals, совместимый view, отдельно known-negative scopes | Все unmatched объекты как background |
| F и свойства H | Доступные нормализованные значения на matched H | Sonority JSON целиком как случайный класс; D→T как готовый процесс |
| h/f | Контекстные подинтервалы, поля membership/degree/temporal_role/voice_role с masks | chord member как structural depth или единственный участник H |
| Pooling S | Градиент от границ и свойств H; при необходимости отдельные declared regularizers | Независимый human-gold semantic assignment, которого паспорт не установил |
| P/p позднее | Собственные интервалы/типы/participant IDs P и адресные relation endpoints | Автоматические P из соседства, F или отдельных edges |

Текущие h→H ссылки дают контекст, но не доказывают независимую семантическую
группировку. Регуляризаторы локальности, разреженности или согласованности со
временем — предпочтения модели. Разные S могут давать те же H/F, поэтому хорошее
качество выходов не доказывает правильную музыкальную интерпретацию S.
Метка temporal_role=structural может быть вспомогательной локальной задачей,
но не служит gold глубиной редукции. Такой auxiliary head одинаков для всех arms.

Matching targets с predicted H находится вне forward. Loss использует только
known fields и отдельно нормируется по задачам; правила matching, geometry gates
и masks общие. Gold boundaries, role labels, confidence и availability не подаются
в encoder, S или decoder. Partial-label ограничения одинаково применяются к E1
negative spans, E2 no-boundary, E3 background, E4/E5 no-object — не только к DETR.
Предсказания H не привязываются к постоянным corpus IDs.

## Как расширить до F, P и адресных связей

F — контекстное свойство H, а не обязательно новый уровень кластеризации.
`F_j = head(H_j, neighboring_H, whole_piece_context)` с masked targets.
Состояние H должно сохранять доступ к нотам и локальному голосоведению.

После сравнения пяти H detectors один общий P-module может объединять H и h в
процессы. Для P нужен многозначный assignment: процессы могут перекрываться и
не обязаны образовывать дерево. Type/process head, временные границы и participant
head обучаются по разным маскам. Direct h↔P обязателен, включая non-chord notes.
Последующий p decoder предсказывает конкретные source/target IDs, тип и направление.
Depth-survival score не заменяет этот decoder.

Паспорт поддерживает1234 P с известным process и explicit H (32 works),634 с h
(37 works); совместный process+H+known F support —930 P в32works. Это не готовая
полная таблица participant roles: подготовка/задержание/разрешение на каждом
member не размечены равномерно. H.function и P.function остаются разными полями.

H↔P, h↔P и predicted F/f/P feedback проверяются по существующему A/B/C/C+soft
плану после выбора H constructor. Никакой обязательной монотонности «нота исчезла
из H — больше не может влиять на P» не вводится. NBF-like addressed decoding и
Query2box composition остаются отдельными последующими опытами, не частью E5-mini.

## Честное сравнение и порядок выполнения всех пяти

1. Общий consumer существующих adapters/encoders: target ontology, view/scope,
   raw time policy, known-negative scopes, family split, TRAIN-only vocab.
   Повторный подсчёт исправленного anchor coverage получает отдельный record;
   исторические91.8% не выдаются за актуальный ceiling.
2. Общая output schema H/h/F, matcher, metrics и пять model constructors.
   Проверки: target-blind forward; held-note multiple-H; exact boundary export;
   padding/empty/point events; permutation/slot matching; masked loss gradients;
   raw residual path; gradient H/F→S; cross-tile ownership и deterministic replay.
3. Один bounded mechanics/resource gate и короткий pilot на каждый вариант.
   Для каждого — заранее sealed planned record и fresh evidence root.
4. Сравнение всех пяти, paired seeds17/42/73, одинаковые admitted families,
   входной контекст, freeze policy, targets и sample schedule. Общие критерии
   checkpoint selection; actual updates/exposure и compute публикуются отдельно.
5. После этого: E5 coarse-edge ablation при технической пригодности, A/B/C,
   P и feedback, live C3, addressed reasoning/boxes — раздельные решения и бюджеты.

Основной proposed endpoint при неполной разметке — macro по независимым families
recall известных H при фиксированном бюджете предсказаний, зависящем только от raw
длительности. Однозначное one-to-one temporal matching внутри совместимого view,
порог overlap/допуск и ranking policy фиксируются заранее. Это recovery известных
events, не precision всего произведения. Количество gold H не задаёт output budget.

Вторичные: joint H+F recovery на known F при том же бюджете, conditional macroF1 F
на matched H, oracle-H interpretation, boundary error, supported h/f и duplicate
rate среди совпадений с known H. Boundary/object precision/F1 считаются лишь в
локально доказанных полных областях; остальные предсказания остаются unresolved.
Не выбирать другой primary после результатов. По каждому условию/seed показывать
coverage, raw candidate/capacity ceiling, applied updates, wall, latency, peak memory,
failed/aborted. Подбор по inner selection; exploratory dev внутри global TRAIN,
TEST закрыт, raw C3 pretraining overlap unknown. Семейные confidence intervals
не заменять bootstrap отдельных нот.

До каждого real-data gate или pilot заранее фиксируются собственные K/time
resolution, config, targets/negatives, updates, stop criteria, per-worker/total
wall caps и planned record. Наличие слова gate не освобождает запуск от бюджета.
После engineering gates и до основного сравнения запечатываются окончательные
общие optimizer/schedule, exposure/checkpoints, primary tolerance/output budget,
selection rule, min effect и total budget E1–E5. Tuning использует только разрешённый
fit/inner-selection; exploratory dev не используется для этих решений.
Результаты pilot не служат основанием для незаявленного исключения неудобных arms.
Намерение сделать все пять не означает запуска полного factorial grid.

## Итог решения и следующая работа

E5 включён в исследовательский план вместе с E1–E4. Его конкретный механизм —
graph-assignment pooling с отдельными H boundaries и сохранением raw graph.
Улучшение качества, интерпретируемость S и скорость остаются гипотезами.
Следующая реализация начинается с общего consumer/output/matching contract и
synthetic tests, затем по одному добавляет пять constructors. Новых runs, моделей
или результатов эта заметка не создаёт; experiment registry не изменён.
