# H/member: проверка длительности обучения

## Итог и evidence review — 25 октября 2026

**Все12runs завершены по30 000applied updates** (360 000всего), failed=0, aborted=0.
Начало19:08:34UTC, конец19:50:45UTC (22:08–22:50МСК); supervisor2531.64s,
то есть42min12s. Сумма run wall2520.78s. Лимиты соблюдены, процессов этой кампании
не найдено. Automation приостановлена после первой проверки; новых запусков нет.

### Помогло ли дольше обучать

Да: средний selected dev AUROC и MacroF1 выросли у всех4heads. Ниже сравнивается
прежний фиксированный2k с pipeline выбора20k/30k по inner selection. Это результат
политики выбора длительности, а не dev-оценка всех одинаковых updates.

| Head | Old2k AUROC | Selected AUROC | ΔAUROC | Old2k MacroF1 | Selected MacroF1 | Selected FPR |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.8938 | 0.9667 | +0.0730 | 0.7158 | 0.8086 | 0.3553 |
| TransE-like | 0.8462 | 0.9626 | +0.1163 | 0.6175 | 0.7926 | 0.2989 |
| Self-only | 0.8666 | 0.9331 | +0.0665 | 0.6765 | 0.7584 | 0.3816 |
| Graph | 0.8909 | 0.9408 | +0.0499 | 0.7004 | 0.7768 | 0.3538 |

Бюджет2k заметно ограничивал результат. Особенно улучшился TransE: теперь его
AUROC0.9626 близок кMLP0.9667. Простой MLP сохраняет лучшие mean AUROC/MacroF1
в данном split. Graph0.9408 превосходит self0.9331 в среднем, но paired разница
поseeds17/42/73 теперь +0.0053/+0.0222/−0.0044; общего превосходства Graph нет.
FPR остаётся высоким:в среднем≈30–38%известных nonmembers ошибочно включены.
В9из12runs порог выбран на верхней границе прежней сетки0.9; это ограничивает
вывод о достигнутом operating point. Новая сетка/балансировка здесь не подбирались.

### Выбранные checkpoint и dev

Все условия реально выполнили30k. Выбраны30k в10runs и20k в2runs (MLP17,self73).
Dev оценён только на выбранном состоянии. Для30k-невыбранных состояний dev metrics
отсутствуют; они не восстанавливаются интерполяцией.

| Head | Seed | Applied | Selected | Wall,s | Dev AUROC | MacroF1 | FPR | R@1 | R@5 | Known coverage | Статус |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MLP | 17 | 30000 | 20000 | 175.88 | 0.9686 | 0.8015 | 0.4162 | 0.1303 | 0.5024 | 62.68% | completed |
| TransE-like | 17 | 30000 | 30000 | 178.26 | 0.9579 | 0.7979 | 0.3622 | 0.1302 | 0.5028 | 62.68% | completed |
| Self-only | 17 | 30000 | 30000 | 228.59 | 0.9563 | 0.7785 | 0.3585 | 0.1299 | 0.5019 | 62.68% | completed |
| Graph | 17 | 30000 | 30000 | 257.17 | 0.9617 | 0.7814 | 0.4132 | 0.1301 | 0.5018 | 62.68% | completed |
| MLP | 42 | 30000 | 30000 | 174.47 | 0.9612 | 0.7961 | 0.3241 | 0.1298 | 0.5025 | 62.68% | completed |
| TransE-like | 42 | 30000 | 30000 | 178.45 | 0.9603 | 0.8025 | 0.3322 | 0.1305 | 0.5032 | 62.68% | completed |
| Self-only | 42 | 30000 | 30000 | 229.80 | 0.9071 | 0.7234 | 0.4209 | 0.1301 | 0.5019 | 62.68% | completed |
| Graph | 42 | 30000 | 30000 | 256.22 | 0.9293 | 0.7859 | 0.3642 | 0.1301 | 0.5025 | 62.68% | completed |
| MLP | 73 | 30000 | 30000 | 174.93 | 0.9704 | 0.8284 | 0.3255 | 0.1304 | 0.5027 | 62.68% | completed |
| TransE-like | 73 | 30000 | 30000 | 179.70 | 0.9695 | 0.7773 | 0.2024 | 0.1307 | 0.5026 | 62.68% | completed |
| Self-only | 73 | 30000 | 20000 | 229.04 | 0.9358 | 0.7735 | 0.3653 | 0.1300 | 0.5022 | 62.68% | completed |
| Graph | 73 | 30000 | 30000 | 258.27 | 0.9314 | 0.7631 | 0.2840 | 0.1300 | 0.5021 | 62.68% | completed |

Mean±sample SD по3seeds (не доверительный интервал по works):

| Head | AUROC | MacroF1 | Wall mean,s |
|---|---:|---:|---:|
| MLP | 0.9667±0.0049 | 0.8086±0.0173 | 175.09 |
| TransE-like | 0.9626±0.0061 | 0.7926±0.0134 | 178.80 |
| Self-only | 0.9331±0.0247 | 0.7584±0.0305 | 229.15 |
| Graph | 0.9408±0.0181 | 0.7768±0.0121 | 257.22 |

### Кривые inner selection — все seeds

Порог на каждой точке выбран только наselection. Primary — mixed-H/work AUROC;
MacroF1/FPR — все известные candidates. Unique H — кумулятивный train coverage.

| Head | Seed | Update | Selection AUROC | MacroF1 | FPR | Unique train H |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 17 | 2000 | 0.9051 | 0.6230 | 0.6629 | 1448 |
| MLP | 17 | 5000 | 0.9272 | 0.7335 | 0.5817 | 2791 |
| MLP | 17 | 10000 | 0.9326 | 0.7838 | 0.4880 | 4212 |
| MLP | 17 | 20000 | 0.9440 | 0.8217 | 0.3975 | 5724 |
| MLP | 17 | 30000 | 0.9430 | 0.8736 | 0.2726 | 6604 |
| TransE-like | 17 | 2000 | 0.8332 | 0.5830 | 0.6967 | 1448 |
| TransE-like | 17 | 5000 | 0.9195 | 0.7051 | 0.6284 | 2791 |
| TransE-like | 17 | 10000 | 0.9137 | 0.7905 | 0.4416 | 4212 |
| TransE-like | 17 | 20000 | 0.9372 | 0.7716 | 0.4131 | 5724 |
| TransE-like | 17 | 30000 | 0.9434 | 0.8050 | 0.4332 | 6604 |
| Self-only | 17 | 2000 | 0.9088 | 0.6275 | 0.6304 | 1448 |
| Self-only | 17 | 5000 | 0.9327 | 0.7202 | 0.5773 | 2791 |
| Self-only | 17 | 10000 | 0.9149 | 0.7783 | 0.4780 | 4212 |
| Self-only | 17 | 20000 | 0.9506 | 0.8192 | 0.3593 | 5724 |
| Self-only | 17 | 30000 | 0.9551 | 0.8338 | 0.3474 | 6604 |
| Graph | 17 | 2000 | 0.9146 | 0.6360 | 0.6596 | 1448 |
| Graph | 17 | 5000 | 0.9418 | 0.6876 | 0.6547 | 2791 |
| Graph | 17 | 10000 | 0.9322 | 0.7947 | 0.4264 | 4212 |
| Graph | 17 | 20000 | 0.9373 | 0.7537 | 0.5624 | 5724 |
| Graph | 17 | 30000 | 0.9470 | 0.8065 | 0.4185 | 6604 |
| MLP | 42 | 2000 | 0.9140 | 0.6809 | 0.6372 | 1455 |
| MLP | 42 | 5000 | 0.9141 | 0.7274 | 0.5213 | 2838 |
| MLP | 42 | 10000 | 0.9360 | 0.7841 | 0.4479 | 4272 |
| MLP | 42 | 20000 | 0.9444 | 0.7612 | 0.5516 | 5721 |
| MLP | 42 | 30000 | 0.9534 | 0.7905 | 0.4733 | 6562 |
| TransE-like | 42 | 2000 | 0.8373 | 0.5971 | 0.7969 | 1455 |
| TransE-like | 42 | 5000 | 0.8900 | 0.7155 | 0.5694 | 2838 |
| TransE-like | 42 | 10000 | 0.9171 | 0.7759 | 0.4390 | 4272 |
| TransE-like | 42 | 20000 | 0.9350 | 0.7798 | 0.4630 | 5721 |
| TransE-like | 42 | 30000 | 0.9571 | 0.8064 | 0.4281 | 6562 |
| Self-only | 42 | 2000 | 0.8841 | 0.6311 | 0.6962 | 1455 |
| Self-only | 42 | 5000 | 0.9188 | 0.7705 | 0.4763 | 2838 |
| Self-only | 42 | 10000 | 0.9358 | 0.7953 | 0.4050 | 4272 |
| Self-only | 42 | 20000 | 0.9364 | 0.8129 | 0.4143 | 5721 |
| Self-only | 42 | 30000 | 0.9449 | 0.8102 | 0.4556 | 6562 |
| Graph | 42 | 2000 | 0.9025 | 0.6609 | 0.5481 | 1455 |
| Graph | 42 | 5000 | 0.9284 | 0.7574 | 0.4113 | 2838 |
| Graph | 42 | 10000 | 0.9378 | 0.8107 | 0.3551 | 4272 |
| Graph | 42 | 20000 | 0.9442 | 0.8051 | 0.4211 | 5721 |
| Graph | 42 | 30000 | 0.9547 | 0.8402 | 0.3712 | 6562 |
| MLP | 73 | 2000 | 0.8895 | 0.6888 | 0.6111 | 1414 |
| MLP | 73 | 5000 | 0.9097 | 0.7094 | 0.6137 | 2785 |
| MLP | 73 | 10000 | 0.9452 | 0.7878 | 0.4775 | 4218 |
| MLP | 73 | 20000 | 0.9417 | 0.8259 | 0.3875 | 5722 |
| MLP | 73 | 30000 | 0.9592 | 0.8387 | 0.3856 | 6603 |
| TransE-like | 73 | 2000 | 0.8289 | 0.6058 | 0.8058 | 1414 |
| TransE-like | 73 | 5000 | 0.9020 | 0.6928 | 0.6738 | 2785 |
| TransE-like | 73 | 10000 | 0.9320 | 0.7369 | 0.5675 | 4218 |
| TransE-like | 73 | 20000 | 0.9395 | 0.7810 | 0.4145 | 5722 |
| TransE-like | 73 | 30000 | 0.9500 | 0.7823 | 0.3164 | 6603 |
| Self-only | 73 | 2000 | 0.9132 | 0.6757 | 0.6353 | 1414 |
| Self-only | 73 | 5000 | 0.9345 | 0.7055 | 0.5577 | 2785 |
| Self-only | 73 | 10000 | 0.9337 | 0.7890 | 0.4630 | 4218 |
| Self-only | 73 | 20000 | 0.9476 | 0.7542 | 0.4875 | 5722 |
| Self-only | 73 | 30000 | 0.9460 | 0.8584 | 0.3106 | 6603 |
| Graph | 73 | 2000 | 0.8975 | 0.6689 | 0.5957 | 1414 |
| Graph | 73 | 5000 | 0.9050 | 0.7222 | 0.6048 | 2785 |
| Graph | 73 | 10000 | 0.9304 | 0.7824 | 0.3854 | 4218 |
| Graph | 73 | 20000 | 0.9360 | 0.8056 | 0.3469 | 5722 |
| Graph | 73 | 30000 | 0.9538 | 0.8175 | 0.3642 | 6603 |

Средний selection AUROC по3seeds на одинаковых updates:

| Head | 2k | 5k | 10k | 20k | 30k |
|---|---:|---:|---:|---:|---:|
| MLP | 0.9029 | 0.9170 | 0.9379 | 0.9434 | 0.9518 |
| TransE-like | 0.8332 | 0.9038 | 0.9209 | 0.9372 | 0.9501 |
| Self-only | 0.9020 | 0.9286 | 0.9281 | 0.9449 | 0.9487 |
| Graph | 0.9048 | 0.9251 | 0.9335 | 0.9392 | 0.9518 |

На30k охвачено6562–6604уникальных H из10860доступных обучающих H, против
1414–1455на2k. При work/H uniform sampling это не число полных эпох. В10runs
выбран крайний30k checkpoint; насыщение качества до30k не установлено.
Продолжение за30k из этих данных не следует автоматически и не запускалось.

### Совпадение данных и воспроизводимость

Для каждого seed на каждой контрольной точке все4heads имеют одинаковые
exposure SHA256, known-pair exposure и unique H. Полные hashes ниже. Префикс2k
совпадает с коротким опытом у всех12runs. Cache manifest и dev support одинаковы.
Для MLP/TransE/self selection AUROC на2k совпал точно с прежним. ДляGraph
наблюдались небольшие отличия:seed17 +0.00066814,seed42 +0.00004543,seed73 −0.00017665.
Причина численного дрейфа не локализована; побитовая воспроизводимость Graph
не доказана. Это ограничение сохраняется отдельно от подтверждённого schedule parity.

| Seed | Update | Known-pair exposure | Schedule SHA256 (общий всем heads) |
|---|---:|---:|---|
| 17 | 2000 | 52453 | `ab8e229e2fab01e75e1abfdc188ffc9e5a5a0f8e20e2ee49bb53a8a630e8ec8e` |
| 17 | 5000 | 130017 | `3555bf23ff4472f84b2ca23f6e7c14f905b36bf16207d21386a19ce9b128a875` |
| 17 | 10000 | 258319 | `3fa0d1ea1e4bb0636bd9217a463840accd782e747d0dc06dd2f7996e97b1577b` |
| 17 | 20000 | 521053 | `f541e7b3e574f8a8eb8a518163802ba4170022fa82f4cad0a888bceb3d98ce23` |
| 17 | 30000 | 781399 | `3979e42ba507723da022bdd25884604c82c0d8bb0d6af36f763569064331b8d1` |
| 42 | 2000 | 52371 | `4727a04b3fe11de23e2225305fb438fedfdd95efdd8fdae777f202076a467ac5` |
| 42 | 5000 | 128848 | `c0eea2e04d5ba6c3e12a51bfb56fef197a15d9508c14536152eb7774587d386e` |
| 42 | 10000 | 263009 | `03afa09da16050a2f2604bb3526d092ec8cbe9ff238554df5472259719fd7563` |
| 42 | 20000 | 529382 | `6d8a2c05ab6047e5d254d29cec5c7272cc044dc060dba51faa309cd18795a2f9` |
| 42 | 30000 | 791693 | `db8b625aa94a4b9acecc56612b545fcf5c59b9910a565469a53e351e2413a92a` |
| 73 | 2000 | 52805 | `cfb8210814bc303a5039936534c7b9c9fcdf6759b5fcd229cb3d4ad353cf32ba` |
| 73 | 5000 | 130137 | `c05566c90a940f716baa7e7fb1327c658399973c02d3d23baeb2e203fef66257` |
| 73 | 10000 | 261251 | `a0f778e4e61ddc94b6654cd74b8419ee2967cd7822e511f92a5e91b47f624147` |
| 73 | 20000 | 520643 | `61ce61472f61ef2888f5bdb3c2914bc6a2d8069cd4650c4ebb82a50077f56861` |
| 73 | 30000 | 781572 | `03fb4ba58237bb87a4475d590343b9a6439b7d942de9dd84aa7c61dffbb777b9` |

### Покрытие, controls и границы вывода

Dev:10 475H,93 530candidates,58 621known (55 656positive/2 965negative),
34 909unknown. Primary AUROC поддержан596mixed H в7families; retrieval —
3418positive H/12works. Unknown не считается negative. Negative support
неравномерен:68.26%известных negatives приходится на2works. Labels weak source/LLM,
H oracle, C3 frozen, exploratory dev внутриglobalTRAIN, TEST закрыт; raw pretraining overlap unknown.
Controls не изменились:duration-only AUROC0.7829/MacroF1 0.5416;all-member
AUROC0.5/F1 0.4870;all-nonmember AUROC0.5/F1 0.0481. Recall@5 остаётся около0.502
у всехheads и слабо различает методы.

| Утверждение | Основная evidence | Вердикт |
|---|---|---|
| Все12runs выполнили30k и selection protocol |Canonical records,result/progress/curve parity,available artifacts SHA/size | supported |
| Дополнительный budget+selection улучшил этот diagnostic |Selection curves растут, selected dev AUROC/F1 выше2k у всех12runs | descriptive |
| Graph превосходит MLP или TransE |Mean selected dev AUROC/F1 ниже, wall выше | unsupported |
| Полная сходимость, human harmonic accuracy, raw H inference |30k bound, weak labels, oracle H,7mixed dev families | unsupported |

### Сохранность и завершение

В isolated ignored `outputs/hmember_long_20261025/monitor001` получено50compact JSON.
Проверены fingerprints12runtime records и36compact artifact SHA/size
(result/progress/selection-curve).108head checkpoints и12prediction streams
остались remote; в этой проверке их bytes не скачивались/не проверялись локально.
Terminal runtime records не переписывались; planned/running продвинуты вперёд.
Closure record: `EXP-HMEMBER-LONG-REVIEW-001`. Registry/ledger обновлены.
Новых model tests, inference, corpus reads, commits/push или training здесь не было.
Legacy не использовался. Original GitHub release checkout не затрагивался.

Следующий отдельный вопрос:снизит ли class-balanced training либо заранее заданный
FPR operating point оставшиеся nonmember ошибки при сопоставимом30k budget.
В этой проверке новые условия не запускались; default architecture не меняется.

## История подготовки и запуска

25 октября 2026. T08/A, отдельная кампания после завершённого короткого пилота.
Пользователь явно разрешил выполнить ранее предложенный опыт до30 000шагов.
Разработка остаётся в managed worktree `harmonic-event-pilot`, отдельно от
параллельно готовящегося GitHub release. Commit/push не выполняются.

## Вопрос и неизменные условия

Достаточны ли2 000updates для MLP/TransE/self/graph, или качество на внутренней
проверочной выборке продолжает расти при большем охвате обучающих H?
Сохраняются64works, разбиение42/10/12, frozen C3, raw candidates, masks,
loss mean-BCE, sampler work/H, Adam1e-3, hidden64,3rounds и seeds17/42/73.
Балансировка классов и новые relation/semantic heads в этот опыт не входят.
Legacy не требуется. Исходный корпус и C3 повторно не читаются/не кодируются.

Используется прежний completed cache, manifest862 665bytes,
SHA256 `15689991fb4978a38630f1b14105d84db22a7e89e201f2acece5a41ada790b19`.
Tile bindings проверяются до использования. Старые terminal records и artifacts
остаются неизменяемыми; новые IDs: `EXP-HMEMBER-LONG-S{seed}-{head}-001`.

## Обучение и выбор длительности

Новый head стартует с исходного seed и непрерывным optimizer до30 000updates.
Старые checkpoints содержат только model state; их нельзя считать точным
optimizer/RNG resume. Поэтому первые2 000updates повторяются, их exposure
fingerprint сопоставляется с коротким опытом.

На2/5/10/20/30тысячах сохраняются веса и полные **inner-selection** metrics.
Выбор: максимальный mixed-H/work AUROC, при равенстве — более ранний update.
Порог выбранного состояния определяется только inner-selection macroF1.
На dev один раз оцениваются выбранные веса после завершения30k. Отдельно
фиксируются фактически выполненные30k и update выбранного checkpoint.
Кривые F1/FPR не меняют правило выбора после просмотра результатов.

Это learning-curve и selection эксперимент, а не matched-update comparison
выбранных моделей: разные heads могут выбрать разные updates. Совпадение
training exposure проверяется по одинаковым контрольным точкам.

## Бюджет, проверки и границы

[Config](../../configs/hmember_long_20261025.json):12runs, cap4h на кампанию;
worker890s внутри outer900s, cleanup≤10s. Повторный запуск root запрещён;
первый сбой останавливает последующие условия с terminal records.

Тесты проверяют выбор по selection, ранний tie-break, единственную dev evaluation,
загрузку выбранных весов, неизменность final training weights/schedule при
добавлении evaluation, input bindings и новые bounded IDs. Native результаты,
source fingerprint и фактическое состояние дополняются перед запуском.

Labels остаются weak source/LLM; H oracle, dev exploratory внутриglobalTRAIN,
TEST закрыт, raw pretraining overlap неизвестен. Новые платные ресурсы,
LLM API, C3 adaptation и Query2box не запускаются.

## Native admission и запуск

81synthetic tests прошёл в серверном окружении без skips:61data/metrics/trainer/
supervisor,12model,8cache. Включены8новых curve/campaign tests. AST и независимые
Standards/Spec reviews закрыты. Исправлено сохранение partial evidence при
прерывании supervisor. Dependencies не добавлялись, legacy не использовался.

Frozen source `duration-source001.tar.gz`:1 443 392bytes,
SHA256 `d250c8f437d44aa499c73dc79bcc2196d2fe1b1b8d3d3dee7b6b8d3e23807205`.
Remote evidence root:
`/home/humtech/music_critic_hmember_20261025/campaign-long001`.
Относительные runtime artifact locators разрешаются от этого root; cache
остаётся readonly в соседнем `campaign001/cache001`.

Supervisor запущен25.10.2026 19:08:34UTC (22:08:34МСК). Первый MLP seed17
превысил2 000updates; selection AUROC на2k=0.9051398291, совпадает с прежним
коротким запуском. Вывод о пользе дополнительного обучения ещё не делается.
В сохранённом launch snapshot первый run достиг22 010updates; selection
оценки на2/5/10/20k уже записаны,30k и итоговая dev evaluation ещё впереди.
Совпадение prefix schedule SHA и known-pair exposure также проверяет supervisor
перед признанием условия completed.

Обновлён heartbeat `h-member-rtx3090`:одна проверка в час, максимум4; quiet on
unchanged state, pause on terminal/error/noSSH. Старые terminal records/outputs
не изменяются, исходный GitHub release checkout не затрагивается.
