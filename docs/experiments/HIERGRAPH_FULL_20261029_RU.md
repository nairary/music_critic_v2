# Full HIERGRAPH E3: первый полный baseline — 29 октября 2026

## Итог —29октября2026,21:00МСК

Вычислительная часть завершила **500 applied /500 attempted /0skipped** и
сохранила selected-checkpoint dev metrics. Actual wall19560.626s=**5ч26мин**,
seed17. Итоговый **experiment record имеет immutable status failed**:при записи
результатов в strict registry обнаруженtuple вместоJSONlist. Это ошибка
финализации после сохранения result.json,а не недовыполненные optimizer updates.
Parent EXP-HIERGRAPH-FULL-E3-S17-001 не переписан какcompleted.
Read-only EXP-HIERGRAPH-FULL-REVIEW-001 completed.

### Selection и выбранные веса

| Applied update | Attempted | Family-macro known-H recovery |
|---:|---:|---:|
|10|10|42.936%|
|**50:selected**|50|**51.648%**|
|100|100|43.716%|
|250|250|49.793%|
|500|500|39.824%|

Выбран50 по заранее заданному earliest maximum,финальныйdev вычислялся на этих
весах один раз. Падение последней selection не доказывает overfitting или
конкретный механизм деградации. Последние500updates и selected50 не являются
matched-update сравнением;train loss/exposure stream повторно не читались.

### Dev на selected50:family macro и coverage

| Задача | Family-macro показатель | Поддержка / pooled coverage |
|---|---:|---|
|H known recovery|**43.057%**|13семейств;1839/11212=16.402%pooled|
|Совместное H+F recovery|13.525%|12семейств,3977knownF|
|F conditional macroF1|16.726%|12семейств,только сопоставленныеH с доступнымF|
|h known context recovery|**10.337%**|13семейств;3104/70411=4.408%pooled|
|P known recovery|**72.807%**|13семейств;305/413=73.850%pooled|
|Совместное P+тип recovery|29.552%|12семейств,320known process type labels|
|P type conditional macroF1|17.125%|12семейств,только сопоставленныеP|
|Участники P:positive recall|**9.312%**|13семейств;673/26510=2.539%pooled|
|Направленные связи:positive recall|**0%**|0scored of3638known positives|

Family macro даёт каждому семейству одинаковый вес;43.057% не означает1839/11212.
Большие/плотно размеченные семейства существенно меняют pooled результат.
Поддержка разных полей/семейств не одинакова,значения условных и совместных
метрик не являются множителями одной общей вероятности.

| Поле | Conditional macroF1 | Joint field recovery | Supported families |
|---|---:|---:|---:|
|H.function|16.726%|13.525%|12|
|H.inversion|41.068%|35.435%|11|
|H.sonority.core|38.132%|33.617%|10|
|H.sonority.quality|21.813%|22.073%|5|
|H.sonority.root_pc|8.285%|8.424%|12|
|h.chord_membership|71.256%|9.569%|13|
|h.temporal_role|11.154%|1.780%|11conditional/13joint|
|h.voice_role|21.348%|2.782%|11conditional/13joint|
|h.preparation.linked|51.766%|9.066%|9conditional/13joint|
|h.resolution.linked|62.176%|6.034%|11conditional/13joint|
|P.function|12.964%|17.691%|12|

Для prepares:1498known positives,0scored,Recall@1/5=0,Hits@1/5=0,MRR=0.
Для resolves_to:2140known positives,0scored,Recall@1/5=0,Hits@1/5=0,MRR=0.
ConditionalF1 связей **недоступен**,это не измеренный0 или100% качества головы.
Наdev ни одна размеченная связь не получила полного сопоставления её endpoint
наборов с доступными prediction candidates. Это подтверждённый провал scoring
coverage;разделить вклад H/h alignment,endpoint-shape и candidate budget по
имеющимся compact aggregates нельзя. Unknown links не считались отрицательными.
Для участников conditionalF1=1.0 на673scored positive pairs;known negatives=0.
Это не global precision и не доказательство правильного обнаружения участников.

### Почему record failed при completed result

`result.json` содержитcompleted и всеdev aggregates;registry seal вызван позже.
В in-memory `dev.families[...].H.fields[...].pairs[0]` лежалPython tuple.
JSON artifact writer сериализовал его вlist,а strict registry canonicalizer
отклонилtuple: `experiment_registry.json.unsupported_type:...:tuple`.
Except-handler записалfailed record с этой причиной. Всеactual counts,
exposure hashes,selected checkpoint иdev aggregates совпадают между сохранённым
result иfailed record. Raw progress.status осталсяrunning после последнего
update;он устарел и не отменяет terminal record/result и отсутствиепроцесса.

В bounded log есть и allocator warning о неудачной попытке выделить≈718MiB при
≈129MiBfree. Fatal CUDA OOM exception в сохранённом traceback отсутствует;
terminal причина —registry serialization. Это всё же признак малого запаса
памяти. Механизм восстановления allocation этой review не профилировался.

Следующий исправляющий engineering slice:нормализовать tuple metric pairs в
JSON primitives до registry seal и добавить terminal CLI regression на реальные
по формеdev aggregates. В этой heartbeat-review код и finished record не менялись,
новых train/inference или повторной оценки не было.

### Integrity и claim matrix

- 5compact JSON сверены сremote SHA256/size;source record канонически валиден,
  fingerprint576aad080975f9bf22d2f003ec87afea447faf5d5e3ec37f3e865526bca76f5b.
- Local terminal import только running→failed;предыдущиеterminal записи неизменны.
- Пять checkpoints10/50/100/250/500 присутствуютremote,каждый61820445bytes;
  selected50 SHA b39f3ea951d17746b3cd9b49ae2e0d4fe1d601593f75e4a699b7d52f5297398e.
  В этой review их bytes/SHA локально не проверялись и weights не скачивались.
- Final exposure34e8433b2476b6ff6ad5c5da90aa7a77bf835a14e6f2267177972044054746f4
  совпал вprogress,result,failed record истрокеcurve для500updates;selected50 hash также совпал.
- Compute:500updates иdev summary —supported. Experiment finalization success —
  unsupported:recordfailed. Musical-quality/superiority claims —unsupported.
- Одна seed17,weak source/model labels,H/P неoracle,C3frozen,E3warm40k,fit43 /
  selection10/dev13 внутриglobalTRAIN,work044 excluded,fit44 vocabulary retained,
  TESTclosed,raw pretraining overlap unknown. Старый80kwindow режим имеет другой
  sampling/matching/geometry/compute budget;численное сравнение не является
  доказательством эффекта новой архитектуры.

Primary training failed1,aborted0;compute resultcompleted1. Сохранённый Gate002
failure —отдельная ранее описанная engineering попытка,не повторныйtrain.
Fullreview receipts:ignored outputs/hiergraph_full_20261029/monitor006.
GPU training PID завершён,память освобождена. Автоматизация
full-hiergraph-e3-rtx3090 **удалена**,посколькуrun terminal;новыеruns не назначены.

---

Ниже сохранён исходный протокол и исторический стартовый snapshot.


Статус: **running**. EXP-HIERGRAPH-FULL-E3-S17-001 запущен на существующей RTX3090
29 октября в15:34:53МСК. Подтверждён первый snapshot: **10 applied / 10 attempted /
0 skipped**,268.33s с начала runner. Это факт обновлений optimizer; качество
полной модели пока не измерено. Первое selection на10 applied.

## Протокол

| Параметр | Значение |
|---|---|
| Модель | Whole-piece E3 + H/h/P,3раунда,feedback,mutable geometry |
| C3 | Frozen существующий cache; backbone SHA d651511441907d4d559d0d3009c925107b9e371ca462db953f189ad4528e4a35 |
| E3 | Trainable,weight-only warmstart selected40k прежней E3; новый Adam |
| Поля | 42 H/F,5 h/f,2 P; независимые маски |
| Seed / augmentation | 17 / none |
| Данные | 43 fit,10 selection,13 exploratory dev,внутри globalTRAIN |
| Исключение | work044 целиком: отсутствующий core в старом cache004 |
| Vocabulary | Сохранён исходный fit-only vocabulary44; исключение одного fit не переобучает словарь по holdout |
| Бюджет | 500 applied whole-work updates,1000 attempted max,43200s wall cap |
| Checkpoints | 10/50/100/250/500 applied |
| Выбор | Earliest maximum selection family-macro H.known_h_recovery |
| Dev | Один раз на выбранном checkpoint; oracle diagnostic отложен |
| Кандидаты | maxH6144,maxP4096;4 overlapping H pernote×2 h slots;near4/far4 directed neighbors |
| Пределы | max h131072,participants524288,relations2097152; все физические ноты сохраняются |
| Не включено | NBFNet,Query2box,transposition |

Один update обрабатывает **целое произведение**, затем один его аналитический
вариант; старые80k были локальными окнами. Числа шагов и длительности этих
режимов несопоставимы. Слабые source/model labels не являются human gold.
Raw pretraining overlap неизвестен. Global VALIDATION/TEST закрыты.

## Подготовка и native gates

PREPARE-001 completed:66 целых raw graphs собраны без ошибки. Один неполный fit
work044 исключён явно; исходные canonical/targets/cache004 и его evidence
не изменены. Новые admission manifests и hardlinks находятся в отдельном
inputs001. V3 target shards, vocabulary и существующие C3 tensors byte-identical.

| Gate | Статус | Наблюдение |
|---|---|---|
| FULL-GATE-001 | completed | Малое work042:52.68s,peak1.91GiB,finite backward,0 updates |
| FULL-GATE-002 | failed | Малое2.93s,среднее32.80s; большое достигло явного h_capacity при8H pernote |
| FULL-GATE-003 | completed | Окончательный budget4H pernote,3размера,finite backward,0 updates |

| Gate003 work | H / h / P | Forward | Loss | Backward | Peak CUDA |
|---|---:|---:|---:|---:|---:|
| work042,small | 298 /3930 /255 |0.48s|1.43s|0.64s|1.28GiB|
| work036,median |1639 /31124 /1040|1.97s|1.53s|9.14s|6.81GiB|
| work046,largest fit |4660 /70020 /1995|4.31s|4.24s|37.00s|17.83GiB|

Work046 содержит8982 физические ноты. Эти замеры не включают selection/dev,
загрузку всех target views и первый Adam state; это engineering admission,
не точный ETA. Длительность кампании зависит от sampled works и evaluation.

## Исправления перед запуском

1. CLI больше не требует запрещённый реестром абсолютный locator. Evidence root
   и output path переносимы относительно машинного --evidence-base; actual path
   всё равно проверяется полностью. Регрессия проходит через validate_record.
2. Для больших whole-work assignments добавлен CPU float64 tensor Hungarian без
   новых зависимостей. Сохраняются максимальная мощность сопоставления, одинаковый
   cost и стабильный временной порядок; small problems используют исходный
   Fraction solver. Пограничная IoU eligibility уточняется точно. Secondary
   near-ties могут отличаться от Fraction, поэтому это новый протокол.
3. Убрано повторное H/P сопоставление внутри одного full_loss; отсутствие
   licensed negative scopes сразу возвращает пустой список отрицательных.
4. Число дальних relation neighbors теперь ограничено явно; все неподдержанные
   target addresses/relations остаются в coverage. Gate002 failure сохранён.

Изменения производительности и budget связей были совместными; уменьшение
общего времени не доказывает причинный эффект одного из них.

Native/local synthetic regression после изменений: **340 passed,1 CUDA-only
skipped,118 subtests passed**,одно существующее TorchScript warning. Focused
independent review:41 tests и250 randomized assignment parity checks passed;
Standards/Spec blockers не найдены. Новые зависимости не устанавливались.
Legacy не читался/не использовался. Основной checkout и publication не изменены.

## Evidence и контроль

Remote evidence root: configured `music_critic_hiergraph_full_20261029/campaign001`
на существующем humtech RTX3090. Frozen source=`source002` sibling.

- Source manifest SHA: fa5419f480503f0131e9380bf81eea95930a034dd77ec49f3583124a6303ccf5.
- Source archive SHA: ab48f9f499ee64065c457101429a8c95a99a52bc5c8340fa4545cd6a74829e42.
- Config SHA:2f487e60dbba6055583e853a8507d9dc84c94580378b1ac9ef393e975a091ba9.
- Target admission manifest SHA:bee187d979ed8c39d662e09e03cb0739b6bc74ebc79b2a6f067dc12054595adf.
- Cache admission manifest SHA:10ecefcb757d2cf770ae122d9ce2fa2d5dd31cfe780b27ed92335dad59a395cb.
- E3 weights SHA:31629df45632c82f0aad9a0c36c258b1d83b8085ddd935e842565c8f016e434f,
  19029219bytes; authenticated remotely by warmstart,не скачивались локально.

Launch wrapper PID1503196,deadline epoch1793320523.2674408; outer timeout43230s,
cleanup15s,GPU flock. Локальные receipts/compact results находятся в ignored
outputs/hiergraph_full_20261029/bootstrap001. Gate records и доступные result
SHA/size проверены через canonical registry reader.

Автоматизация full-hiergraph-e3-rtx3090: одна краткая проверка в час,max14,
только текущий run; без повторного обучения/LLM API/чтения корпуса. При terminal,
ошибке/недоступном SSH или лимите проверок — pause. Только actionable уведомления.
Длительный процесс на сервере не зависит от открытого SSH-сеанса.

Следующее действие: дождаться первого selection и итогового выбранного
checkpoint; terminal analysis оформить отдельным REVIEW-001 до существенного
импорта/анализа. Большие weights остаются remote; их availability и проверка
байтов не смешиваются с локальной проверкой compact JSON.
