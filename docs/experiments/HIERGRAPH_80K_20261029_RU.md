# HIERGRAPH E1–E5,80k applied updates —2026-10-29

## Итог кампании —29 октября2026,11:15МСК

**Все пять условий completed, exit0, по80000 applied; всего400000 обновлений.**

Campaign wall 38240.388s (10.62h). Failed0,aborted0. Worker walls ниже относятся к result.json; supervisor accounting имеет небольшой orchestration overhead.

| Condition/seed | Applied | Attempted | Skipped | Selected | Worker wall,s | Dev macroH | JointH+F | ConditionalF | F families | Pooled H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1/17 | 80000 | 86967 | 6967 | 40000 | 10099.474 | 0.4643 | 0.1523 | 0.1640 | 12 | 2965/11212 |
| E2/17 | 80000 | 86967 | 6967 | 80000 | 6834.031 | 0.3652 | 0.1754 | 0.2253 | 11 | 6645/11212 |
| E3/17 | 80000 | 86967 | 6967 | 40000 | 6400.638 | 0.4915 | 0.1667 | 0.2193 | 12 | 1765/11212 |
| E4/17 | 80000 | 86967 | 6967 | 2000 | 7465.172 | 0.3463 | 0.1345 | 0.2518 | 12 | 1761/11212 |
| E5/17 | 80000 | 86967 | 6967 | 2000 | 7423.114 | 0.3206 | 0.1346 | 0.2220 | 12 | 1406/11212 |

### Selection curve: family-macro known-H recovery

| Applied | E1 | E2 | E3 | E4 | E5 |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0.3613 | 0.3057 | 0.3731 | 0.4315 | 0.3687 |
| 5000 | 0.3437 | 0.3085 | 0.3881 | 0.2983 | 0.2972 |
| 10000 | 0.3269 | 0.3055 | 0.3988 | 0.2944 | 0.2516 |
| 20000 | 0.3594 | 0.2909 | 0.4081 | 0.2558 | 0.2937 |
| 40000 | 0.3703 | 0.3187 | 0.4393 | 0.1882 | 0.3102 |
| 60000 | 0.3621 | 0.3125 | 0.3550 | 0.2391 | 0.3242 |
| 80000 | 0.3564 | 0.3194 | 0.3670 | 0.1986 | 0.3427 |

All five exposure hashes/attempted counts match at each2/5/10/20/40/60/80k checkpoint. Final80k hash31d1b57e9e419a526bf85a1339b559880348c42d6de105a65dea6e9c91099843. Actual80000 differs from selected40k/80k/40k/2k/2k. Selected-dev is not matched-update comparison.

Coverage unchanged across all conditions: fit25591admitted+598outsidehalo+7rawexcluded=26196; selection7805+50=7855; dev11186+26=11212. Exclusions retained in primary denominator. H supports13families, jointF12; conditionalF supports12/11/12/12/12 on different recovered events.

| Claim | Primary evidence / integrity | Verdict |
|---|---|---|
| All five finished400000 updates | campaign terminal, five result/record states, exit0; compact SHA/size and six fingerprints verified | supported |
| E3 leads selected-dev macroH0.4915; E2 jointH+F0.1754 | primary results above; different selected checkpoints, one seed | descriptive |
| E4/E5 best selection at2k | seven-point curves, fixed earliest-max rule | supported; cause unknown |
| E4/E5 failure caused by overfitting or pooling collapse | curves alone do not identify mechanism | unsupported |
| Intrinsic F ranking | different recovered H/support; no common-support diagnostic | unsupported |
| Local checkpoint integrity | weights not downloaded | unverifiable |

Weak source/model labels; frozenC3/cache004,42H/Fheads,augmentationnone; exploratorydev insideglobalTRAIN,globalVALIDATION/TESTclosed,raw pretraining overlap unknown. No P/f learning or full global H/P reasoning claim. Compact24files/134239bytes retained under ignored monitor007. Checkpoint weights retainedremote without local byteverification. REVIEW-001 finalizedcompleted; earlier sections below are historical snapshots. Next decision gate: constructor selection and global H/h/P design; no new runs launched.

---


## Промежуточная проверка — 29 октября, 03:41 МСК

E1 завершён с exit0; E2 запущен последовательно. Артефакты первого terminal
условия сохранены в ignored monitor003; SHA/size компактных JSON и canonical
fingerprints проверены. Weights не скачивались и локально не проверены.
REVIEW-001 находится running: итоговое сравнение ждёт всей очереди.

| Условие / seed | Статус | Applied | Attempted | Skipped | Selected update | Wall, s | Dev family-macro H recovery | Joint H+F | Conditional F macroF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 / 17 | completed | 80000 | 86967 | 6967 | 40000 | 10099.474 | 0.4643 | 0.1523 | 0.1640 |
| E2 / 17 | running, snapshot | 10462 | 11359 | 897 | — | 1010.432 | — | — | — |
| E3 / 17 | planned | — | — | — | — | — | — | — | — |
| E4 / 17 | planned | — | — | — | — | — | — | — | — |
| E5 / 17 | planned | — | — | — | — | — | — | — | — |

H recovery усреднён по 13 dev families, F-метрики поддержаны в 12. Это weak-label
exploratory dev внутри globalTRAIN. E1 фактически обучался до80k; dev оценён
один раз на выбранном по inner-selection checkpoint40k, не на последнем80k.
Нет failed/aborted условий. Остальные результаты отсутствуют, не равны нулю.

E1 fit coverage:25 591 admitted +598 outside-halo +7 raw-excluded =26 196;
selection:7805+50=7855; dev:11186+26=11212. Exclusions остаются в primary denominator.
Oracle-H F macroF1=0.2079 — отдельная diagnostic на другом support; его нельзя
вычитать из conditional F оценки для вывода о причине ошибок.

Парные E1/E2 exposure hashes и attempted counts совпали на applied milestones: 2000, 5000, 10000.
Сравнение dev разных selected updates не будет matched-update comparison.
Архитектурное превосходство пока не оценивается; очередь не изменена.


Кампания **запущена** на существующей RTX3090. Начало29 октября,00:38:28МСК
(UTC28 октября21:38:28). Supervisor PID1193772; E1 worker group1193780.
Очередь автономна на сервере и запускает **E1→E2→E3→E4→E5** последовательно.
Наблюдаемый initial snapshot: E1 applied1812,attempted
1948,skipped_unknown136,worker wall
244.032s. Это проверенный начальный прогресс, не текущий
финальный результат. E2–E5 пока `planned` и будут стартовать после predecessor.

## Зафиксированный protocol

| Поле | Значение |
|---|---|
| Модели / seed | Пять H constructors E1–E5,seed17 |
| Target | Ровно80000 applied optimizer updates на успешное условие |
| Draw cap |160000 attempted windows; skipped unknown не вызывает Adam.step |
| Checkpoints по applied |2000,5000,10000,20000,40000,60000,80000 |
| Selection | Earliest maximum inner-selection family-macro known-H recovery |
| Dev | Один выбранный checkpoint + отдельный oracle-H diagnostic |
| Worker / whole campaign cap |21600s /111600s,outer21620s и bounded cleanup |
| Shared settings |Frozen C3/cache004,42H heads,Adam3e-4,clip1,batch1,hidden128,rounds3 |
| Data |Те же67 работ,44fit/10selection/13exploratorydev,augmentationnone |
| TEST |Закрыт; globalVALIDATION также не используется |

Неизвестные labels не отрицательные. Raw/halo exclusions остаются в метрике;
fit26196H=25591admitted+598outside-halo+7raw-excluded. Weak source/model labels,
не human gold; raw pretraining overlap неизвестен. Single-seed results будут
описательными, не доказательством превосходства конструктора.

## Execution и артефакты

Remote ready root `/home/humtech/music_critic_hiergraph_ready_20261028`.
Новая неизменяемая source006; evidence root `campaign-80k001`,
`campaign.json`, `launch.json`, `progress.json`, `records/`, `runs/E*-s17-001/`.
Artifact locators **во всех шести records относительны campaign evidence root**:
например,runs/E1-s17-001/result.json; parentprogress —progress.json.
Код не меняет прошлые source005/cache004/records. Основной checkout для GitHub
release не редактировался. Commit/push не выполнялись.

Parent **EXP-HIERGRAPH-80K-CAMPAIGN-001**, worker
**EXP-HIERGRAPH-80K-E1-S17-001** —running. Остальные E2/E3/E4/E5 —planned.
Ранее подготовленные2000-step планы не исполняются и не переписаны.
Очередь не возобновляет и не повторяет условия автоматически. Worker failure
сохраняется/финализируется перед переходом дальше. Whole process group очищается;
GPU flock унаследован timeout/worker и защищает от повторного захвата при падении
supervisor. In-flight update при внешнем прерывании отмечается unknown; сохраняется
последний подтверждённый applied/attempted count.

| Binding | SHA256 |
|---|---|
| Source006 manifest |`d8717903221e770bf13703e3beb8664301f2861f68f57bd0396fdb810424e0b1`|
| Source006 archive |`9d2fb2744dcf4ff19dc20d4fd005515db60b1ac3b3e8c231082d04c723e51c20`|
| Campaign plan |`83a22b693f67e91283541232d36322bd31aab762afa9f314a59c4db8b9d66feb`|
|80k config file |`ee75002caf404d030cad16ce7e88e6c21bd76f722abd6732f7425e470c3c804d`|
| Target manifest |`2d808660243ad68a24b29474140eebd9b9abb7e0818b0ca60e7af7fe0e0bd415`|
| Cache004 manifest |`5d4e2634bff7a9cffeea22c83e0be52325be7b4bcb89eb7321771f0bc0d45a32`|

Checkpoint config binding uses canonical semantic JSON digest; launch binding
above hashes the retained pretty-printed config file bytes. Both domains are
explicit in the runner and must not be confused when inspecting checkpoints.

## Проверки и изменения кода

- Local HIERGRAPH suite:136 tests,135passed/1CUDA-onlyskip; existing CUDA five-arm
  exact replay from preparation remains applicable to unchanged model/runtime.
- Native Linux new harness:14tests passed1.025s, без corpus/GPU обучения.
- Native harness attempts1/2 failed before test bodies: отсутствовали declared
  pytest и дополнительный registry CLI helper. Logs сохранены. Declared pytest>=8
  установлен отдельно вtest-deps001, training venv не изменён; test harness —
  отдельная копия ownsource006+проверенный CLI helper, frozen snapshot неизменен.
- Independent Standards и Spec: найдены/исправлены budget/identity gaps, orphan
  process-group cleanup и lifetimeGPUlock. No remaining blockers in reviewed scope.
- Новые applied-target/drawcap tests наблюдали red→green. Реальные subprocess tests
  проверяют stop/reap, serial order, immutable records, partialcounts и GPUlock.

Изменены scripts/run_hiergraph.py,newcampaign module и wrapper, четыре новых testfiles,
два80kconfig,шестьregistryrecords,STATUS/ledger и этотreport/spec. Переиспользован
существующий V2 runtime; модели/loss/data/cache неизменны. Legacy не читался.

## Контроль

Heartbeat **hiergraph-e1-e5-80k**, одна краткая проверка в час,максимум32.
Нет новых runs/retries/subagents/LLM calls/source reads. Штатный прогресс тихий;
уведомление —новый сбой,завершение или нужное действие. Terminal evidence review
будет зарегистрирован отдельно до анализа; actual80000 и selected_update
показываются раздельно, exposure сравнивается на одинаковых applied milestones.
Weights и большие exposure/selection/dev artifacts остаются remote.

[Spec](../workstreams/2026-10-29/hiergraph-80k/spec.md),
[80k config](../../configs/hiergraph_e1_e5_80k_20261029.json),
[campaign plan](../../configs/hiergraph_e1_e5_80k_campaign_20261029.json).
Следующий этап: завершение очереди и доказательный разбор пяти условий.


## Разбор E1 по вопросу пользователя

[Per-field результаты и объяснение selected40k](../workstreams/2026-10-29/hiergraph-80k/e1-results.md).
Family-macroH46.43%,pooled2965/11212=26.44%. ConditionalcoreF1 0.599,root0.477,
inversion0.333,function0.164; support различается. Train loss0.379→0.070→0.040
на2/40/80k,приH-selection36.13→37.03→35.64%.40k выбран по заранее заданномуH
критерию,а не как универсально лучшийпо42head. Полная очередь не изменена.


## E1 vs E2: промежуточные best40k selection checkpoints

Сравнение зарегистрировано EXP-HIERGRAPH-80K-E1E2-INTERIM-REVIEW-001. E2 ещё
обучается (57516 applied на snapshot), итоговый best может измениться.
Оба текущих best —40000 applied,43462 attempted и одинаковый exposure hash
839d126b052cabf9cd47da526268765d795b28f0e9ec20965bdf4bb78b16b8ac.
Это **selection у обоих**, не сравнение E1dev46.4% с E2selection.

| Метрика selection | E1 | E2 |
|---|---:|---:|
| Family-macro H recovery,10families |37.03% |31.87% |
| Pooled H recovery |1422/7855=18.10% |3090/7855=39.34% |
| Family-macro jointH+F recovery,7families |19.65% |13.07% |
| Conditional F macroF1 |0.256 (7families) |0.429 (5families) |

E1 выше по H recall в7из10families. Два крупных POP909 families073/792 содержат
5518из7855H (70.25%): E2 там достигает36.0%/33.7%, E1 —3.37%/3.42%.
Поэтому pooled и macro меняют порядок. E2 не нашёл H в2другихfamilies,у E1
ненулевой recall во всех10. ConditionalF нельзя считать доказательством более
сильной F-головы: набор найденных H и поддержанные families различаются.
Совместная H+F-метрика учитывает пропущенные H на одинаковом known-F support.

Пока E1 впереди по объявленному primary и jointH+F, E2 находит больше событий
суммарно. Это описательный результат seed17, не доказанное превосходство.
Протокол и очередь не изменены; E2dev отсутствует до его окончания.

## E2 completed; E3 active —29 октября2026,05:44МСК

| Condition/seed | Status | Applied | Attempted | Skipped | Selected | Worker wall,s | Dev macroH | JointH+F | ConditionalF | F families |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1/17 | completed | 80000 | 86967 | 6967 | 40000 | 10099.474 | 0.4643 | 0.1523 | 0.1640 | 12 |
| E2/17 | completed | 80000 | 86967 | 6967 | 80000 | 6834.031 | 0.3652 | 0.1754 | 0.2253 | 11 |

E3 active snapshot: applied17784,attempted19285,skipped1501; supervisor/worker alive. E4/E5 planned. Failed/aborted0.

E1/E2 exposure hashes and attempted counts match at all seven2/5/10/20/40/60/80k applied checkpoints. Final80k hash31d1b57e9e419a526bf85a1339b559880348c42d6de105a65dea6e9c91099843. Dev uses selected40k versus80k: this is descriptive, not matched-update dev. E2 selectionH31.935% at80k only slightly exceeds31.874% at40k.

Pooled devH: E12965/11212,E26645/11212; family weighting changes the ranking. ConditionalF support differs12vs11families and recovered events. OracleF macroF1 E10.2079,E20.2080.

Coverage unchanged: fit25591 admitted+598outsidehalo+7rawexcluded=26196; selection7805+50=7855; dev11186+26=11212. Exclusions retained in primary denominator. No added model conditions or protocol changes.

| Claim | Evidence / integrity | Verdict |
|---|---|---|
| E2 completed80k, E3 running | terminal record/result, exit0; active progress and PID; available compact SHA/size verified | supported |
| E1/E2 differ on selected dev | table above; seed17, selected40k/80k, same final exposure but different selected exposure | descriptive |
| E2 intrinsically better at F | different recovered sets/support; oracleF near-identical | unsupported |
| Weight bytes checked locally | .pt not downloaded | unverifiable |

Weak source/model labels, frozenC3/cache004,42H/Fheads,augmentationnone; exploratorydev insideglobalTRAIN,globalVALIDATION/TEST closed,raw pretraining overlap unknown. Compact17files/66927bytes imported under monitor005; canonical fingerprints and available compact record artifact digests checked. Weights remain remote without local byteverification. REVIEW-001 remainsrunning until fullcampaign terminal. Earlier snapshots above are historical.

## Аудит кривых обучения —29 октября2026

EXP-HIERGRAPH-80K-CURVE-REVIEW-001 completed. Только существующие selection curves и loss journals; GPU/inference/corpus0. Полные exposure не скачивались. Все пять hash chains,86967attempted/80000applied/6967skipped,проверены против terminal results; loss-file SHA/size впервые зафиксированы сейчас.

| Update | E1 selectionH,% | E2 | E3 | E4 | E5 |
|---:|---:|---:|---:|---:|---:|
| 2000 | 36.13 | 30.57 | 37.31 | 43.15 | 36.87 |
| 5000 | 34.37 | 30.85 | 38.81 | 29.83 | 29.72 |
| 10000 | 32.69 | 30.55 | 39.88 | 29.44 | 25.16 |
| 20000 | 35.94 | 29.09 | 40.81 | 25.58 | 29.37 |
| 40000 | 37.03 | 31.87 | 43.93 | 18.82 | 31.02 |
| 60000 | 36.21 | 31.25 | 35.50 | 23.91 | 32.42 |
| 80000 | 35.64 | 31.94 | 36.70 | 19.86 | 34.27 |

| Condition | Train loss,last1000 at2k | At40k | At80k | Description |
|---|---:|---:|---:|---|
| E1 | 0.3789 | 0.0696 | 0.0398 | Late near-plateau;40k→80k −1.40pp. |
| E2 | 0.4071 | 0.1072 | 0.0756 | Near-plateau;40k→80k +0.06pp despite latestmaximum. |
| E3 | 0.3550 | 0.0894 | 0.0644 | Growth to40k, substantialdrop60k, partialrecovery80k;peak→final −7.23pp. |
| E4 | 0.4723 | 0.1298 | 0.1093 | Early peak then substantial degradation;peak→final −23.29pp. |
| E5 | 0.4322 | 0.1220 | 0.0898 | Early peak,drop10k,then recovery;peak→final −2.60pp,not flat throughout. |

E1/E2 descriptions are visual/descriptive, not statistical plateau tests. Only seven selection points: line segments do not imply measured intermediate values. No all-checkpoint dev reevaluation. Training scalar averages available task losses, while selection selects thresholded/ranked event recovery; decreasing loss does not guarantee improved discovery. Overfitting/objective mismatch are hypotheses; concrete mechanism unmeasured. Constructor parameters/losses differ, so compare loss trends within condition.

Plot and compact aggregates: ignored outputs/hiergraph_80k_20261029/curve-review001/. Weaklabels,one seed,exploratoryselection withinglobalTRAIN,TESTclosed,pretrainingoverlapunknown. No change to model,completedrecords,protocol or architecture decision.
