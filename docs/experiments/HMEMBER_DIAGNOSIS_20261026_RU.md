# H/member: причины ошибок и проверка исправлений, 26 октября 2026

Статус: диагностики и все четыре заранее заданных контрольных обучения завершены; научная проверка результатов выполнена. Основной review: `EXP-HMEMBER-DIAG-20261026-001`. Работа только в isolated worktree `harmonic-event-pilot`; отдельный GitHub release не затронут.

## 1. Что именно проверяем

В прошлой кампании все 12 запусков дошли до 80 000 обновлений, но selection выбрал веса на 20–60k, ни разу на 80k. Средняя выбранная dev AUROC Graph = 0.9275, MLP = 0.9591; FPR около 28–40%. Эти значения относятся к членству нот в **заранее известном H**, при замороженном C3. Это ещё не распознавание гармонических функций, не поиск H в сырой музыке и не проверка Query2box.

Ночной разбор разделяет три вопроса: почему Graph расходится при одинаковом seed; почему высока доля ложных участников; насколько устойчив сам вывод об отставании Graph. Смена архитектуры и продление до следующего числа шагов не включены в протокол.

Все метки — слабая source/LLM-разметка. H задаётся oracle-геометрией. Downstream train/selection/dev находятся внутри global TRAIN; global TEST закрыт. Пересечение сырых данных с pretraining C3 неизвестно. Общий надёжный track-scope H в source не установлен: кандидаты включают все наблюдаемые pitched tracks, а labels остаются в своём H/view. Это candidate superset, а не доказанная ансамблевая гармония; unknown не превращались в nonmembers. Dev уже использовался в предыдущих исследованиях, поэтому новые сравнения exploratory.

## 2. Причины: что установлено, а что остаётся гипотезой

| Наблюдение | Проверка | Вывод и предел |
|---|---|---|
| Graph численно расходится при одном seed и порядке примеров | Одинаковые веса, входы и три повтора; отдельно GPU `index_add`, затем forward/gradient/update | Механизм недетерминированной агрегации воспроизведён; строгий режим устраняет его в коротком native тесте. Величина влияния на 80k качество не установлена. |
| Отрицательные метки получают малую массу коэффициентов усреднения objective | Точный подсчёт ожидаемого веса под действующим work/H sampler | Их ожидаемая доля коэффициентов = **7.2393%**; это не доля численного значения BCE или градиента. Это основание для контролируемого вмешательства, а не доказательство причины ошибок. |
| Порог выбирался только из 0.1…0.9 | Точный поиск на selection, неизменные 12 checkpoint, прежние dev predictions | Устранение ограничения повышает selection F1, но средний dev F1 снижается у всех четырёх heads. Одной этой правки недостаточно. |
| Итог Graph сильно зависит от малой поддержки primary metric | Декомпозиция по произведениям, без исключения примеров | work016 с одним mixed H даёт 76.6% среднего разрыва Graph–MLP; это чувствительность оценки, а не основание удалить произведение. |
| Вероятности насыщаются до ровно 1 в FP32 | Selection logits и sigmoid из одного forward | Насыщение теряет порядок некоторых кандидатов. Raw logits теперь можно сохранять; историческую метрику задним числом не меняем. |
| Дополнительные сообщения Graph могут быть избыточны поверх C3 | Анализ конструкции моделей | Рабочая гипотеза: C3 уже контекстный, Self тоже получает H. Oversmoothing и бесполезность графов этим не доказаны. |

### Воспроизводимость

`EXP-HMEMBER-DETERMINISM-GATE-001`: исторический GPU `index_add` на синтетическом наборе с разными масштабами чисел даёт max difference 0.0166015625. Это стресс-тест редукции, не размер ошибки на музыке. У Graph с одинаковыми начальными весами расхождение первого logits = 1.19e-7, итоговых весов после восьми синтетических обновлений ≈6.89e-6. MLP, TransE и Self в этом тесте совпали.

При `torch.use_deterministic_algorithms(True)` и заданном перед запуском `CUBLAS_WORKSPACE_CONFIG=:4096:8` все четыре heads совпали побитово во всех проверенных стадиях и трёх повторах. Настройки оформлены контекстом с восстановлением при выходе. Исторический default сохранён. Это гарантия проверенного теста и окружения, не разных GPU/версий Torch и не полного 80k replay.

### Данные и устойчивость метрик

В frozen cache 64 произведения: 42 train, 10 selection, 12 dev. Поддержанные цели обучения есть в 37 train works; 10 860 H имеют известные targets. 133 247 известных положительных пар и 9 517 отрицательных дают 6.666% отрицательных по простому подсчёту пар. Однако sampler выбирает сначала поддержанное произведение, затем H; loss усредняется внутри H. Под этим распределением ожидаемая масса коэффициентов отрицательных targets равна 7.2393% (`EXP-HMEMBER-LOSSMASS-DIAG-001`). Missing labels остаются masked, а не отрицательными.

Primary AUROC сначала усредняется по mixed H внутри work, затем поровну по поддержанным works. Dev: 596 mixed H в 7 works. work016 имеет 1 mixed H и 15 известных отрицательных пар; work003 — 3 mixed H и 11 отрицательных. Каждое получает 1/7 итогового веса. Их вклады в средний Graph−MLP gap −0.031593 составляют −0.024194 и −0.005126: 76.6% и 16.2% разрыва. work051 с 186 отрицательными, но без mixed H, в primary AUROC не участвует. Эти данные не удалялись и веса метрики не менялись.

Selection AUROC поддержана пятью works. В selection 2 717 из 4 289 отрицательных относятся к одному work010; в dev два works содержат около 68.3% отрицательных. Поэтому перенос одного порога между произведениями сложнее, чем порядок кандидатов внутри H. Высокая within-H AUROC совместима с большим FPR при глобальном пороге.

Общий dev: 10 475 H / 12 works / 93 530 кандидатов; 58 621 известных targets (55 656 positive, 2 965 negative), 34 909 unknown. Known coverage = 62.68%. Retrieval оценивается на 3 418 H с хотя бы одним известным positive; при высокой доле positives Recall@k требует сравнения с constant/candidate-order контролем.

## 3. Исправление порога: результат на всех старых seeds

`EXP-HMEMBER-THRESHOLD-DIAG-001`: checkpoint и выбор шага фиксированы по прошлой кампании. Новый inference только на selection: 92 076 запросов суммарно, 0 updates, 0 новых dev forwards. Для dev повторно использованы аутентифицированные сохранённые predictions. Старые метрики воспроизведены; selection AUROC replay delta = 0 для всех 12 условий.

До запуска этой диагностики зафиксированы пять policies; исторические dev результаты уже были известны: исторический порог, coarse replay, exact в [0.1,0.9], exact в [0,1], и максимум selection positive recall при empirical FPR≤0.1. Старые grid candidates сохранены; ties решаются близостью к 0.5, затем меньшим порогом. Не выбираем лучшую policy по dev.

Таблица — средние по seeds 17/42/73; F1 — pooled pair macro-F1 двух классов, recall — positive recall. Это другие агрегации, чем primary within-H/work AUROC.

| Head | Пороговая policy | Dev macro-F1 | Dev FPR | Dev positive recall |
|---|---|---:|---:|---:|
| mlp | historical | 0.8070 | 0.3532 | 0.9789 |
| mlp | exact_full | 0.8029 | 0.3166 | 0.9735 |
| mlp | fpr10 | 0.7060 | 0.1085 | 0.8918 |
| transe | historical | 0.8045 | 0.2814 | 0.9703 |
| transe | exact_full | 0.8017 | 0.2716 | 0.9683 |
| transe | fpr10 | 0.6321 | 0.0678 | 0.8230 |
| self | historical | 0.7825 | 0.3980 | 0.9760 |
| self | exact_full | 0.7744 | 0.3681 | 0.9687 |
| self | fpr10 | 0.6487 | 0.0885 | 0.8429 |
| graph | historical | 0.7772 | 0.3393 | 0.9663 |
| graph | exact_full | 0.7762 | 0.3342 | 0.9656 |
| graph | fpr10 | 0.6129 | 0.0863 | 0.7870 |

Coarse replay полностью повторяет исторические показатели. Exact bounded почти ничего не меняет. Exact full улучшает selection F1, но средний dev F1 падает у всех heads. FPR10 заметно снижает ложные срабатывания ценой пропущенных участников: для Graph recall падает примерно с 96.6% до 78.7%. Ограничение selection FPR не является гарантией dev FPR: у MLP среднее 10.85%.

Насыщение: у MLP seed17 на selection ровно score=1 имеют 34 921 positive и 300 negative. Raw-logit AUROC выше FP32-probability AUROC на 0.004569. Порог по probabilities не разделит одинаковые единицы. Это ограничение представления scores; не доказательство, что оно объясняет отставание Graph. В новой опции сохраняются logits без дополнительных forward; старые ranking/selection metrics остаются по probabilities.

Полные значения по всем условиям, seeds, policies, works, thresholds и confusion counts: `outputs/hmember_diagnosis_20261026/threshold001/results/result.json`; агрегаты — `threshold001/comparison.json`.

## 4. Проверка взвешенного loss: заранее заданный протокол

Четыре условия: MLP/plain, MLP/balanced, Graph/plain, Graph/balanced. Seed17; фиксированные 30 000 updates; без выбора длительности и без adaptive seed/objective search. Strict deterministic runtime включён для всех. Начальная инициализация внутри каждой пары, C3/cache, размеры/архитектура heads, Adam1e-3, norm cap1, work/H sampler и masks одинаковы. Selection выбирает только порог; dev оценивается один раз на фиксированном финальном checkpoint.

Вес берётся из TRAIN-only sampler expectation π=0.0723933480:

```
w_negative = 1 / (2π) = 6.9067119239
w_positive = 1 / (2(1−π)) = 0.5390215766
loss = sum_known(w_y * BCE) / n_known
```

Нормировка по числу известных targets сохранена; деления на сумму весов внутри H нет. Частота gradient clipping фиксируется, поскольку cap1 может изменить силу вмешательства. Weighted и unweighted значения самого loss напрямую не сравниваются.

Обе operating policies заданы до запуска: exact selection macro-F1 и selection FPR10; raw-logit AUROC считается диагностически из тех же forwards. Эффект взвешивания может ограничиться сдвигом калибровки, и подбор порога может компенсировать этот сдвиг. Поэтому успех требует лучшего error/recall tradeoff или ранжирования, а не просто меньших probabilities/FPR при 0.5.

Общая ожидаемая экспозиция для каждой из четырёх моделей: hash `3979e42ba507723da022bdd25884604c82c0d8bb0d6af36f763569064331b8d1`, 781 399 известных пар. Все четыре planned records созданы до запуска. Supervisor cap 2400s, worker 540s, outer 600s, cleanup≤10s; первая ошибка останавливает оставшиеся условия.

### Завершённые четыре контроля

Завершены 4/4, failed=0, aborted=0; каждый выполнил **30 000 applied updates**, всего 120 000. Финальные веса фиксированы на 30k: выбора `selected_update` здесь нет. Старт 00:56:34 UTC, завершение 01:14:19 UTC; campaign wall **1064.69s (17m45s)**. Новых evaluation queries 103 284: для каждого run 7 673 initial-selection + 7 673 final-selection + 10 475 dev. Дополнительные threshold/raw-logit calculations переиспользуют эти же rows.

Все четыре exposure hashes совпали с зарегистрированным ожиданием и между собой: 781 399 известных пар / 30 000 H-query exposures / 6 604 unique train H (60.81% от 10 860 supported H). Покрытие dev одинаково; initial-selection metrics внутри обеих plain/balanced пар совпали точно. Это подтверждает сопоставимость новой пары; старые selected checkpoints из 30k/80k кампаний не являются её matched-update контролем.

IDs: `EXP-HMEMBER-REPAIR-S17-{MLP,GRAPH}-{PLAIN,BALANCED}-001`. Все seed17. MLP имеет 187 009 обучаемых параметров, Graph — 344 321. Ни один run не обучал C3.

При основном пороге, выбранном по exact selection macro-F1:

| Head / loss | Updates | Worker wall, s | Dev AUROC | Dev macro-F1 | Dev FPR | Positive recall | Recall@1 / @5 | Clip fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp / plain | 30000 | 203.25 | 0.9472 | 0.7992 | 0.3133 | 0.9720 | 0.1303 / 0.5026 | 0.1155 |
| mlp / balanced | 30000 | 204.19 | 0.9650 | 0.7610 | 0.3383 | 0.9606 | 0.1303 / 0.5025 | 0.1365 |
| graph / plain | 30000 | 327.36 | 0.9269 | 0.7704 | 0.3929 | 0.9713 | 0.1295 / 0.5013 | 0.1199 |
| graph / balanced | 30000 | 326.26 | 0.9599 | 0.7574 | 0.3359 | 0.9588 | 0.1300 / 0.5024 | 0.1410 |

Selection показывает другую сторону переноса:

| Head / loss | Selection AUROC | Selection macro-F1 | Selection FPR | Основной threshold | Raw-logit dev AUROC |
|---|---:|---:|---:|---:|---:|
| mlp / plain | 0.9430 | 0.8743 | 0.2579 | 0.84768271 | 0.9495 |
| mlp / balanced | 0.9543 | 0.8269 | 0.3486 | 0.92407501 | 0.9654 |
| graph / plain | 0.9529 | 0.8420 | 0.3343 | 0.87973183 | 0.9274 |
| graph / balanced | 0.9362 | 0.8484 | 0.2203 | 0.94157052 | 0.9599 |

Вторая заранее заданная policy максимизирует selection recall при empirical selection FPR≤0.1. У всех четырёх фактический selection FPR = 428/4289 = 0.0997902; на dev он не фиксирован:

| Head / loss | Threshold FPR10 | Dev macro-F1 | Dev FPR | Dev positive recall |
|---|---:|---:|---:|---:|
| mlp / plain | 0.99832648 | 0.7364 | 0.1366 | 0.9214 |
| mlp / balanced | 0.99974531 | 0.6492 | 0.0816 | 0.8470 |
| graph / plain | 0.99973983 | 0.6146 | 0.1656 | 0.8269 |
| graph / balanced | 0.99618477 | 0.7081 | 0.1943 | 0.9124 |

**Интерпретация.** В основной policy Graph/balanced повышает dev AUROC на 0.033019 и снижает FPR на 5.70 процентного пункта, но macro-F1 падает на 0.013008. FP уменьшаются с 1 165 до 996, FN растут с 1 598 до 2 295. Selection AUROC одновременно падает с 0.952881 до 0.936150. Это изменение ранжирования для одного seed с разным эффектом на splits; устойчивое улучшение не установлено. Raw-logit dev AUROC также растёт 0.927357→0.959898, поэтому одними sigmoid ties прирост не объясняется.

У MLP/balanced dev AUROC растёт на 0.017830, но macro-F1 падает на 0.038214; одновременно увеличиваются и FP (+74), и FN (+637) при selection-F1 пороге. Значит class weighting не является универсальным исправлением.

В FPR10-policy у Graph/balanced есть полезный сигнал: F1 0.614617→0.708138 и recall 0.826937→0.912444. Но dev FPR тоже растёт 0.165599→0.194266. Это tradeoff при одном selection-ограничении, **не превосходство при одинаковом dev FPR**. Для MLP FPR уменьшается, но recall/F1 заметно ухудшаются.

Изменения dev AUROC balanced−plain по всем поддержанным works:

| Work | Mixed H | MLP ΔAUROC | Graph ΔAUROC |
|---|---:|---:|---:|
| work003 | 3 | +0.000000 | +0.000000 |
| work005 | 87 | -0.006633 | +0.023015 |
| work014 | 45 | -0.003871 | +0.076398 |
| work016 | 1 | +0.153584 | +0.039818 |
| work033 | 222 | -0.001019 | -0.001925 |
| work034 | 187 | +0.005978 | -0.001292 |
| work066 | 51 | -0.023226 | +0.095121 |

Graph улучшает AUROC в 4 из 7 works, ухудшает в 2, один tie. Весь агрегированный выигрыш MLP перекрывается вкладом одного work016: +0.153584/7 = +0.0219405 при общем +0.0178304; остальные шесть signed contributions суммарно отрицательны. Это иллюстрирует, почему одной итоговой AUROC недостаточно для решения.

Prespecified no-learning controls одинаковы во всех четырёх runs:

| Контроль | Dev AUROC | Macro-F1 | FPR | Recall@1 / @5 |
|---|---:|---:|---:|---:|
| all_member | 0.5000 | 0.4870 | 1.0000 | 0.1274 / 0.4945 |
| all_nonmember | 0.5000 | 0.0481 | 0.0000 | 0.1274 / 0.4945 |
| duration_only | 0.7829 | 0.5416 | 0.6567 | 0.1288 / 0.4966 |

Модели явно лучше этих controls по AUROC/F1; высокая Recall@k сама по себе слабее различает их, потому что даже constant-score candidate-order baseline даёт Recall@5≈0.4945 против ≈0.501–0.503 у моделей. Unknown-кандидаты участвуют в ranking, но не считаются известными negatives.

Проверены 4 canonical fingerprints и SHA/size 12 compact artifacts (`result`, `progress`, `operating-points`). 24 checkpoint weights и 4 больших predictions.jsonl остались remote; локальная byte verification для них отсутствует. Supervisor завершён, собственных процессов этой кампании при проверке не найдено. Compact source results: `outputs/hmember_diagnosis_20261026/repair001/import001`; агрегаты и парные проверки — `repair001/comparison.json`.


## 5. Реализация, проверки и сохранность

Добавлены opt-in deterministic runtime, exact/FPR-constrained threshold helpers, optional weighted masked BCE, сохранение выбранных selection rows и raw logits без дополнительных evaluation passes, clipping accounting. Сохраняются historical defaults, masks, shape/API совместимость. Отдельные bounded scripts выполняют reproducibility gate, threshold replay и четыре repair controls.

Файлы реализации: `src/music_critic/experiments/harmonic_event_reasoner/{reproducibility,metrics,model,training}.py`; scripts `probe_hmember_determinism.py`, `diagnose_hmember_thresholds.py`, `run_hmember_repair_campaign.py`; профильные tests. Нормализация loss, unknown gradients, strict settings/restoration, curve selection row binding, exact threshold tie/boundary cases, authority bindings и failure terminalization покрыты тестами. Два замечания Standards review (обязательная authority и unsupported raw AUC) исправлены через red→green; узкая повторная проверка пройдена. Spec/scientific review также пройден.

Финальный source snapshot `repair-source002.tar.gz`: SHA256 `a86af2a420414eb851e2f360729ac13e9afeaea92f027c6e72e08c1ce8cab8c6`, 1 461 286 bytes. Native suite: **122 tests passed, 0 skipped**, unittest 0.661s, процесс 2.269s. Ранние gates: 8 native tests для reproducibility, 35 для threshold diagnostic; промежуточный source прошёл 111 tests. Это проверки механики, не музыкальной полезности.

Terminal records и старые campaign/source не меняются. Только compact JSON, hashes и docs входят в registry; checkpoint weights и большие predictions остаются remote. Локальную byte verification весов не заявляем. Imported artifacts считаются evidence, не инструкциями. Legacy не читался и не переиспользовался; зависимости не добавлялись; commits/push отсутствуют.

## 6. Итоговый выбор следующего шага

Численную воспроизводимость исправить удалось в проверенном native тесте. Добавлены инструменты, позволяющие дальше проводить корректные сравнения и сохранять достаточные evidence. Но высокие ошибки членства **не устранены одной балансировкой или одним более точным порогом**. У Graph есть полезное изменение ранжирования на dev и лучший recall/F1 при FPR10-policy, но ценой других ошибок и с противоположным движением selection AUROC. Это основание для следующего контролируемого вопроса, а не готовая победа Graph.

Следующий обоснованный этап — небольшой независимый reviewed набор с явным scope H и известными отрицательными targets, плюс проверка переноса score calibration между произведениями. После фиксации целевого error/recall tradeoff можно повторять выбранный парный протокол на нескольких seeds. Механически увеличивать updates дальше не обосновано. В этой задаче дополнительных условий, повторных seeds, новых LLM labels или архитектурных модулей не запускалось. Defaults и архитектурное решение не меняются; `docs/DECISIONS.md` не требует новой записи о выборе модели.

## 7. Бюджет и границы выполненной работы

Реальные новые updates: 120 000 в четырёх repair controls. Threshold diagnostic: 92 076 selection queries, 0 новых dev forwards, wall125.72s; loss-mass diagnostic без inference, wall2.77s; synthetic reproducibility gate и unit tests — отдельно от научных real-data updates. Repair campaign wall1064.69s; это wall, не измерение чистого GPU kernel time. Native environment использовало уже доступную RTX3090; новые платные ресурсы и LLM API не задействованы.

На финальной проверке account usage used20%, **remaining80%**, выше согласованной нижней границы70%. Это общий лимит аккаунта, а не точная атрибуция расхода этой задаче. Сложные научные проверки выполнены Astra6/ultra, реализация и узкие Standards reviews — Sol6.1/high. Старый heartbeat `h-member-rtx3090` остаётся PAUSED; новых периодических проверок не создано.

Исходный checkout для научного руководителя, terminal истории старых опытов и legacy сохранены. Изменения и полный отчёт находятся только в isolated worktree. Неразрешённые вопросы: перенос калибровки/меток между works, достоверность source/LLM negatives и scope, устойчивость на других seeds, длинная воспроизводимость и внешняя/human evaluation. Global TEST не открывался.
