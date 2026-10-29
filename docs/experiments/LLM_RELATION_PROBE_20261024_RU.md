# Первый пилот на LLM-разметке: адресуемое разрешение

Дата исходного предложения: 24 октября 2026. Пользователь25октября разрешил
автономно реализовать и запустить сравнение, экономно следить за ним. Ниже
протокол; фактические результаты принадлежат отдельным experiment records.

## Bound implementation profile — 25 октября 2026

600updates на условие; seeds17/42/73 последовательно, начиная с полного блока17.
Семь условий на seed: raw MLP и3×2 C3 modes/decoder. Общий supervisor cap21600s,
native gate cap1800s, run cap900s плюс bounded cleanup. Сначала29 synthetic/adjacent encoder
tests и независимое Standards/Spec review, затем зарегистрированный native gate.
Все full-модели имеют LR3e-4, raw/frozen heads1e-3, AdamW/weight decay1e-4,
clip1, без поиска гиперпараметров. Недообучение при лимите остаётся ограничением.

Source контексты допускаются только на точных raw slices. Окно4core bars+1halo,
2048notes/200000directed edges maximum; до32 будущих raw attack кандидатов
в8QN, без gold voice/chord inputs. Отдельно учитываются все исключения,
включая исходные текущие67 против66 геометрически допустимых raw exports.
Prepared006 исправляет findings предыдущего preparation004; все failed attempts
и предыдущий terminal export сохранены. Неоднозначные направления не объявлены
ошибкой музыкального анализа: они не поддерживаются данным конкретным decoder.

Remote supervisor останавливает и reap только собственные process groups до
terminal hashing. Мониторинг — раз в час, максимум8проверок, без LLM API calls
в обучении и без продолжения за пределом бюджета. Автоматизация останавливается
при terminal кампании, требуемом действии пользователя или достижении8проверок.

## Фактический запуск

25октября00:15:54MSK supervisor запущен на пользовательском RTX3090;
максимальное завершение25октября06:15:54MSK. Снимок исходников1532436bytes,
SHA256 `ca16645c11246af29788bc3b31116ddbb10be76d919cbf969f92930c5fd2007e`.
Prepared manifest006 SHA256
`777862335599d9eb2d3c75c87ec5782e51197e7299d5094408055b3262e44a50`.
Config и22planned records сохранены перед запуском; источник и prepared payload
сверены. Final source tests:29passed,1deprecation warning,2.08s.
Remote campaign workspace `music_critic_relation_probe_20261024/campaign001`;
монитор automation-2, hourly максимум8. Native gate001 completed за63.11s;
2114tiles,52fit/12devworks. work012/work032 исключены по raw context_track_limit;
итог64runtime works. Dev1742positive/2578none, candidate support1645/2271
исходных binary targets. Нулевые/неподдержанные запросы сохраняются в coverage;
метрики ранжирования рассматриваются вместе с этим ограничением. На первой проверке монитор подтвердил завершение всех21условия; результаты ниже.

## Завершение и проверенные результаты — 25 октября 2026

Кампания завершилась в **00:57:19 МСК**: 2485.11s (41min25s), включая gate и orchestration. Все21 training records — completed, по600updates, итого12600updates; gate8discarded updates отдельно. Failed0, aborted0, invalid0. Сумма измеренного wall отдельных run-функций2381.49s; это не GPU-active time. Процессы supervisor/последнего worker отсутствуют, собственных CUDA training процессов нет. Heartbeat automation-2 приостановлен после первой проверки; новые прогоны не запускались.

Сравнение использует **final step600**, а не выбранный best checkpoint. Три paired seeds17/42/73, один и тот же family split, cache и candidates. Основная существующая метрика — work-mean supported-class macroF1. Рядом приведён global macroF1, чтобы не скрывать неоднородность классов между произведениями. Ranking — query-mean Recall/Hits по1653 запросам,4 320 existence contexts,12 dev works. Это согласие со слабыми LLM targets внутри global TRAIN; не human accuracy.

Среднее ± sample SD по трём seeds; R/H — доли в процентах. Seeds не являются тремя независимыми разбиениями.

| Условие | Work macroF1 | Global macroF1 | Recall@1,% | Recall@5,% | Hits@5,% | Wall/run,s |
|---|---:|---:|---:|---:|---:|---:|
| Raw features / mlp | 0.598 ± 0.000 | 0.287 ± 0.000 | 6.8 ± 1.7 | 31.9 ± 7.0 | 32.0 ± 7.0 | 31.9 ± 0.1 |
| C3 frozen / mlp | 0.665 ± 0.009 | 0.370 ± 0.018 | 3.8 ± 0.6 | 22.9 ± 3.8 | 22.9 ± 3.8 | 139.3 ± 0.9 |
| C3 frozen / transe | 0.650 ± 0.020 | 0.320 ± 0.022 | 13.8 ± 0.2 | 45.2 ± 0.1 | 45.3 ± 0.1 | 139.8 ± 1.8 |
| C3 full finetune / mlp | 0.628 ± 0.037 | 0.296 ± 0.012 | 4.8 ± 0.6 | 21.2 ± 2.1 | 21.3 ± 2.0 | 120.7 ± 0.2 |
| C3 full finetune / transe | 0.598 ± 0.000 | 0.287 ± 0.000 | 15.1 ± 3.1 | 56.3 ± 4.5 | 56.4 ± 4.5 | 121.0 ± 0.3 |
| C3 с нуля / mlp | 0.699 ± 0.017 | 0.402 ± 0.051 | 2.2 ± 0.7 | 15.9 ± 2.2 | 16.0 ± 2.2 | 120.8 ± 0.2 |
| C3 с нуля / transe | 0.660 ± 0.053 | 0.308 ± 0.017 | 7.3 ± 1.2 | 29.2 ± 6.8 | 29.2 ± 6.7 | 120.4 ± 0.3 |

**Раздельный вывод по двум задачам.** У pretrained-full TransE Recall@5=56.30%, frozen TransE45.24%, random-full TransE29.21%; преимущество full над frozen положительно на каждом из3seeds. TransE выше MLP по Recall@5 внутри каждого режима и seed в этом пилоте. Это описательное сравнение конкретных процедур при600updates, не доказательство универсального преимущества геометрии или C3. В raw baseline есть только MLP; raw TransE и step0/untrained ranking отсутствуют, поэтому вклад геометрического prior отдельно не изолирован.

**Existence-head имеет серьёзное ограничение.** Full-pretrained TransE и raw MLP во всех3seeds дают всегда linked: TP1742,FP2578,TN0,FN0. Work macroF1=0.598 при этом совпадает с тривиальным always-linked baseline; global macroF1=0.287. Аналитический always-none global macroF1=2578/(2×2578+1742)=0.374. Random-MLP имеет наиболее высокий средний work macroF1=0.699, но global0.402±0.051 и не доказывает устойчивого выигрыша над этим константным контролем. Пять из12dev works имеют только linked;96.90% reviewed-none сосредоточены в work005/work003/work066. Объяснение через work-uniform sampling и конфликт/масштаб losses — гипотеза, причинно не проверенная.

**Покрытие.** От67 исходных current-v2 к66 prepared: work044 BAR_INVALID; затем64 runtime works: work012/work032 context_track_limit. Fit52works/dev12works; 2114tiles. Dev1742positive/2578none contexts. Генератор поддержал1645 из2271 исходных binary endpoint mappings (72.44%); это endpoint coverage, а не потолок query-mean Recall. Ranking оценивается на поддержанных source queries с известными gold endpoints; ответы вне candidate domain внутри этих запросов учитываются как misses. Неподдержанные source contexts не входят в ranking, отражены в runtime coverage.

### Все условия и seeds

| Seed | Условие | Updates | Wall,s | Work F1 | R@1,% | R@5,% | H@1,% | H@5,% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | C3 full finetune / mlp | 600 | 120.88 | 0.6163 | 4.75 | 21.57 | 4.78 | 21.60 |
| 17 | C3 full finetune / transe | 600 | 120.80 | 0.5981 | 18.45 | 60.25 | 18.45 | 60.31 |
| 17 | C3 frozen / mlp | 600 | 138.89 | 0.6551 | 4.45 | 22.60 | 4.48 | 22.63 |
| 17 | C3 frozen / transe | 600 | 138.74 | 0.6724 | 13.85 | 45.31 | 13.85 | 45.31 |
| 17 | C3 с нуля / mlp | 600 | 120.98 | 0.6912 | 2.18 | 17.15 | 2.18 | 17.24 |
| 17 | C3 с нуля / transe | 600 | 120.53 | 0.6868 | 8.32 | 37.02 | 8.35 | 37.02 |
| 17 | Raw features / mlp | 600 | 31.89 | 0.5981 | 6.47 | 32.12 | 6.53 | 32.24 |
| 42 | C3 full finetune / mlp | 600 | 120.69 | 0.6692 | 5.44 | 23.11 | 5.44 | 23.11 |
| 42 | C3 full finetune / transe | 600 | 120.90 | 0.5981 | 12.28 | 51.39 | 12.28 | 51.42 |
| 42 | C3 frozen / mlp | 600 | 138.71 | 0.6688 | 3.36 | 26.80 | 3.39 | 26.86 |
| 42 | C3 frozen / transe | 600 | 138.74 | 0.6384 | 13.85 | 45.07 | 13.85 | 45.13 |
| 42 | C3 с нуля / mlp | 600 | 120.64 | 0.6883 | 1.57 | 13.40 | 1.57 | 13.49 |
| 42 | C3 с нуля / transe | 600 | 120.05 | 0.5983 | 7.68 | 25.47 | 7.74 | 25.53 |
| 42 | Raw features / mlp | 600 | 31.97 | 0.5981 | 8.65 | 38.81 | 8.71 | 38.90 |
| 73 | C3 full finetune / mlp | 600 | 120.43 | 0.5981 | 4.26 | 19.00 | 4.30 | 19.06 |
| 73 | C3 full finetune / transe | 600 | 121.31 | 0.5981 | 14.46 | 57.26 | 14.46 | 57.35 |
| 73 | C3 frozen / mlp | 600 | 140.29 | 0.6715 | 3.72 | 19.21 | 3.75 | 19.24 |
| 73 | C3 frozen / transe | 600 | 141.85 | 0.6378 | 13.55 | 45.34 | 13.55 | 45.37 |
| 73 | C3 с нуля / mlp | 600 | 120.71 | 0.7188 | 2.99 | 17.21 | 3.02 | 17.30 |
| 73 | C3 с нуля / transe | 600 | 120.62 | 0.6943 | 6.02 | 25.14 | 6.05 | 25.17 |
| 73 | Raw features / mlp | 600 | 31.86 | 0.5981 | 5.29 | 24.89 | 5.32 | 24.98 |

### Integrity и границы вывода

Архив compact evidence168121bytes, SHA256 `f68c61dc0f88f20a5b120d5c3756c4878d75dd826a612384517e8e51d71b4f0b`: outer hash/size, archive paths/types и internal manifest133files проверены. Canonical fingerprints22remote records проверены, разрешённые lifecycle transitions выполнены без изменения terminal records. Проверены106 локально доступных record artifacts;63checkpoint artifacts оставлены на сервере и **не byte-verified/reloaded в этом review**. JSON results сверены с record.results и history/events.jsonl на200/400/600. Не запускались импортированные scripts или inference. Исторические данные/weights не копировались в Git.

Один cache/source/config для всех условий,600applied updates у каждого. По коду work/tile schedule детерминирован seed и shared cache; actual draw-by-draw trace не сохранён, поэтому равенство exposure не верифицировано независимым журналом. Full и frozen имеют разный forward/precompute budget и LR; равенство wall/compute не заявляется. Unknown raw pretraining overlap остаётся ограничением held-out claim. Корпусные VALIDATION/TEST не использовались в campaign; downstream-dev — внутренний global TRAIN.

| Claim | Первичная опора | Verdict |
|---|---|---|
| 21bounded run завершились по600updates | terminal records, result.json, events.jsonl, campaign.json | supported |
| Full-pretrained TransE лучше ранжирует цели на этом split/budget | matched-final common-support metrics трёх seeds | descriptive |
| Existence-head full TransE готова к надёжному применению | constant linked confusion во всех seeds | unsupported |
| C3 с нуля бесполезен | random-MLP выше по existence; цели/метрики различаются | unsupported |
| Модель освоила гармонию, улучшила музыкальную красоту или пересечения связей | таких задач/человеческой проверки не было | unsupported |

Следующий минимальный gate — отдельно проверить class/work balance и обучение existence-head, добавить raw TransE и step0 ranking контроль; затем human audit небольшой выборки разрешений. Это предложения: новые run budgets не выделены, inference/обучение не запускались. Архитектурный default не меняется.

Review record: [EXP-LLM-RELATION-REVIEW-001](records/EXP-LLM-RELATION-REVIEW-001.json). Compact evidence/расчёты сохранены в ignored monitor001 и content-addressed artifact storage.

## Задача и проверяемый вопрос

Восстановить имеющиеся LLM-метки `resolves_to` между конкретными нотными
событиями/контекстами. Проверяются полезность готовых представлений C3,
необходимость их адаптации и обучение того же encoder с нуля совместно с
relation head. Отдельное сравнение декодеров проверяет point translation
против MLP при одинаковом режиме encoder.
Это проверка атомарных связей перед обучением их пересечений.

## Замороженная исходная граница

Источник: опубликованный
[RELEASE003](records/EXP-CORPUS400-COMPLETION-015-RELEASE003.json), completed,
`dataset_published=true`. Его сохранённые `current67_field_coverage` и
`published_summary` прочитаны после успешной проверки SHA-256/размера:

- current67-field-coverage.json: 6096 байт,
  `6e474d46c7fcf9157acaf144e7ec7a4ef75297e87e03d0477e156eacdee4c95c`;
- summary.json: 12337 байт,
  `559ca05ebe3d107d5ee8c6054bdbf6cd23afd13fe2e0ead8839add50b67eb5b6`.

Опубликованная коллекция — 75 unique works: 67 current-v2 и 8 historical-v1.
Для первого пилота предлагаются 67 v2, сохраняя masks, view и origin.
Положительных resolution-contexts 10671, reviewed-none 21497, masked242768,
всего274936. Preparation:7730 linked,23115 none,244091 masked.
Известных полей notes/analysis.tension_role44640, но их распределение по
классам/произведениям этим чтением не пересчитывалось.

Все эти числа относятся к сохранённому выпуску. Они не равны уникальным рёбрам,
числу независимых примеров или измеренной accuracy. Выпуски и historical-v1
не складываются; неопубликованные candidates не добавляются.
Физическая папка published manifest называется release004: идентичность
публикации задаётся RELEASE003 record и digest, а не номером каталога.

## Вход, выход и смысл отрицаний

Вход: исходные ноты, точное время и допустимые raw features. Представление
note-context должно учитывать raw временной slice: одно состояние всей
удержанной ноты не описывает смену её роли. Gold harmony/voice/function и
целевые рёбра не становятся входами модели или генератора кандидатов.

Выходы учитываются раздельно:

1. Есть ли исходящая интерпретированная зависимость: linked / reviewed-none;
   unresolved/conflicting/unreviewed исключаются соответствующими masks.
2. На каких подтверждённых событиях она заканчивается. Несколько участников
   сохраняются; подлинно многосоставные отношения не разворачиваются в
   декартово произведение пар без семантического обоснования. Неподдерживаемые
   типы endpoints исключаются с отчётом о потерянном покрытии.

Reviewed-none — утверждение об отсутствии зависимости в определённом view и
scope. Это не универсальная отрицательная метка для любых пар. Для pair loss
нужны подтверждённые отрицания, чья область покрывает рассматриваемого
кандидата. Неуказанный tail у положительного source остаётся неизвестным,
если полнота ответов отдельно не установлена. Если такого supervised сигнала
недостаточно, pair-часть протокола уточняется до обучения, без автоматического
negative sampling по всем неразмеченным нотам.

## TransE-подобный декодер: конкретное предложение

Модель событий индуктивна: `h_i = E_theta(raw_piece, event_i, raw_slice_t)`.
Состояние объединяет соответствующие представления note/slice и допустимые
raw признаки самого события. Для обоих концов используется одна обучаемая
проекция `u_i = normalize(P h_i)` в пространство размерности64 (предложение
пилота, не измеренный оптимум). ID события нужен для alignment, а не для
обученной таблицы embeddings знакомых нот.

Для каждого именованного типа `r` обучается вектор `v_r` той же размерности.
Для начального `resolves_to`:

```text
q_i,r = u_i + v_r
d(i,r,j) = ||q_i,r - u_j||^2
s(i,r,j) = b_r - d(i,r,j) / tau_r
tau_r = softplus(a_r) + epsilon, epsilon > 0
```

Положительные пары получают высокий score, подтверждённые отрицательные —
низкий. `sigmoid(s)` — модельная оценка с отдельной проверкой calibration,
а не автоматически верная вероятность музыкального отношения. При ранжировании
одного source/relation сначала проверяется наличие зависимости отдельной
existence head; минимальная дистанция сама не доказывает существование связи.
Несколько допустимых arrivals оцениваются отдельно, без one-gold softmax.

Начальная pair-цель — weighted BCEWithLogits только по известным y/masks;
веса классов и нормировка определяются из fit, с отчётом по семействам.
Unknown пары не участвуют в supervised BCE. Normalization и ограничение/штраф
норм relation parameters фиксируются до запуска. Если подтверждённых отрицаний
недостаточно, заранее выбирается и обосновывается другой протокол; это не повод
выдать случайных неизвестных кандидатов за отрицательные.

Градиенты идут через distance в P, v_r, b_r, tau_r и, в full-режимах, encoder.
Общая проекция и один фиксированный вектор на тип оставляют понятное ограничение
translation geometry. Многослойный парный обход, напрямую вычисляющий score,
в этом условии не добавляется: иначе он может решить задачу независимо от
проверяемой геометрии. У MLP-scoring головы те же u_i/u_j; разности и произведения
их компонент вычислимы из них и не добавляют входной информации.

Это **TransE-подобный** вариант, потому что вместо исходной lookup-таблицы
сущностей используется контекстный encoder; также изменены loss и calibration.
Сам принцип translation:
[TransE, 2013](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf).
Relation-specific source/target проекции или context-dependent shift были бы
отдельными более выразительными вариантами. На этом этапе их не смешиваем.
Известное ограничение множественных ответов остаётся; это предмет сравнения,
а не обещание заранее лучшего декодера:
[TransH, 2014](https://www.microsoft.com/en-us/research/publication/knowledge-graph-embedding-by-translating-on-hyperplanes/).

## Сравнение: инициализация encoder × декодер

Пользователь предлагает включить C3 с нуля совместно со всей моделью отношений.
Расширение плана от24октября добавляет full-режимы до результатов frozen probe,
а не делает их условием, доступным только после положительного frozen результата.

| Encoder | MLP scorer | TransE-подобный scorer | Проверяемый вопрос |
|---|---|---|---|
| Pretrained C3, frozen | F-MLP | F-TransE | Есть ли нужный сигнал в готовом представлении |
| Pretrained C3, full fine-tuning | FT-MLP | FT-TransE | Помогает ли адаптация представления к новым отношениям |
| C3 architecture, random initialization, full training | R-MLP | R-TransE | Можно ли выучить представление совместно с новыми отношениями с нуля |

Во всех шести условиях projection/existence/relation heads обучаются заново;
frozen касается только готового encoder. В full-режимах обучаются все активные
слои C3, влияющие на relation loss: нотный graph encoder, hierarchy/temporal
слои и новые головы. Наследованные theory heads без соответствующих targets
не получают искусственный loss. HCM event decoder в эту систему не добавляется.
В initial full-сравнении loss/targets одинаковы; auxiliary harmony/SSL objectives
потребуют отдельной оси, одинаковой для сравниваемых инициализаций.

Обязательный простой контроль:

| Вариант | Вход и модель | Что проверяет |
|---|---|---|
| A | MLP на raw признаках пары: интервалы высоты/времени, длительности, метр и observable track | Насколько задача решается локальными подсказками |

Простая эвристика ближайшего по времени/высоте продолжения служит дополнительным
контролем. Track не объявляется голосом. F-MLP против A проверяет прибавку
представлений. MLP против TransE внутри каждой строки изолирует decoder.
F против FT проверяет адаптацию; FT против R сравнивает стратегии инициализации
с их разной предысторией. Сравнение F-MLP с R-TransE смешало бы две причины.

Одинаковы downstream fit/dev, разрешённые labels, кандидаты, порядок предъявлений
для paired seeds и критерий выбора. Параметризация голов/регуляризация и фактические
updates/draws/FLOP-equivalent accounting или доступный compute proxy фиксируются.
Предварительное обучение не считается бесплатным одинаковым прошлым опытом:
его источники и возможный overlap сообщаются. LR для pretrained и random может
различаться; равный заранее ограниченный hyperparameter-search budget и selection
rule обязательны, нельзя дать одному условию ручную донастройку по dev.

Основное качество обучаемого подхода проверяется парой FT-MLP/FT-TransE;
R-MLP/R-TransE — содержательный контроль обучения с нуля. Frozen-условия остаются
диагностикой и могут выполняться раньше, но их неуспех не закрывает full-ветку.
Learning curves и fit/dev gap показывают недообучение/переобучение; проигрыш
random при данном бюджете не доказывает принципиальную недостаточность данных.

Первая основная цель — resolution. Preparation, tension-role и объединение
нескольких ограничений остаются следующими отдельными условиями.
Никакого вывода об улучшении PLL или качестве музыки из этого пилота не следует.

## Разбиение и оценка

Внутри разрешённого глобального TRAIN создаётся family-grouped downstream
fit/dev разбиение; произведения, версии, фрагменты и транспозиции одной семьи
не расходятся между сторонами. Точное распределение определяется после
паспорта поддержки классов. Общий corpus TEST остаётся закрытым.

C3 мог видеть raw-ноты downstream-dev при прежнем обучении на глобальном
TRAIN. Отдельно фиксировать checkpoint/source overlap: без его устранения
это перенос на новые LLM-targets, а не доказательство работы на совершенно
невиданных произведениях. Нельзя подбирать encoder по результатам dev и
представлять это как независимое сравнение.

Метрики: macro-F1/precision/recall linked-versus-none, восстановление известных
участников (Recall@1/5 и наборов участников при одинаковом candidate pool),
результаты по независимым произведениям, жанрам, source/origin и availability.
Candidate recall измеряется отдельно. Неразмеченный правильный ответ не
объявляется false positive автоматически. Интервалы неопределённости считать
по семействам; одинаковые метки в повторных контекстах не дают независимых trials.

Основной reference — LLM-разметка: метрики измеряют согласие с ней. Агентский
review не создаёт human gold. Для вывода о музыкальной правильности нужна
отдельная независимая человеческая проверка. Противоречивые аналитические
виды не объединяются в один безусловный target.

## Порядок исполнения после выбора опыта

1. Зарегистрировать planned подготовку по
   [EXPERIMENT_LOGGING](../EXPERIMENT_LOGGING.md). Metadata-only инвентарь,
   затем flat named TRAIN admission с SHA/size; паспорт endpoints, масок,
   уникальных рёбер, family support, negatives и source/checkpoint overlap.
2. Реализовать отдельный dataset adapter/heads с synthetic tests; проверить
   направление связей, masks, changing note-context, multi-participant связи,
   split leakage и отсутствие gold-input leakage. Дополнительно проверить
   одинаковый экспорт контекстов, отсутствие encoder gradients в frozen и
   прохождение relation gradients во все ожидаемые активные слои full-режимов;
   random condition не загружает готовые веса. Legacy не нужен.
3. После этого оформить matched protocol/config и planned training attempts.
   Предложение для воспроизводимости — paired seeds17/42/73 с отдельными RNG
   streams encoder/head/sampling. Число updates,
   wall-time cap и checkpoint selection фиксируются до обучения после короткого
   отдельного resource gate; ETA сейчас не измерена. Новая capacity-check3090
   непосредственно перед запуском обязательна, текущая свободная карта не бронь.
4. Закрыть каждый launched attempt, сохранить hashes/logs и сравнить common
   support всех объявленных условий. Отсутствие выигрыша F над A/эвристикой
   ограничивает полезность frozen representation, но не отменяет FT/R.
   Сравнить MLP/TransE внутри каждой строки и FT/R при одинаковом scorer,
   с learning curves и учётом бюджета. При отсутствии выигрыша geometry
   предпочесть простой scorer в пределах проверенных условий. Если атомарные
   связи восстанавливаются, следующий опыт — объединение двух ограничений
   на один endpoint с явным baseline объединения scores.

## Границы выполненной задачи

Проверены только текущий код/контракты, компактная запись и сохранённые сводки;
нового TRAIN-payload census, численного обучения, checkpoint loading или TEST
доступа не было. Идентичность checkout и общий research scope:
[обзор геометрии](../research/HARMONIC_RELATION_GEOMETRY_20261024_RU.md).
Ни один результат обучения не получен. Изменены только этот план и STATUS;
реализация, зависимости, DECISIONS и legacy не менялись. Проверки относятся
к локальным ссылкам/формату документов, не к научной достаточности данных.
