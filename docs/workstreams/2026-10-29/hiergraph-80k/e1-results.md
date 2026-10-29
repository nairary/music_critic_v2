# E1: что показывают первые80k —29 октября2026

E1 —скоринг готового raw span bank,а не регрессор новых границ. Он получает
замороженные C3 node states,обновляет raw граф/H states и ранжирует кандидатов;
общие42 H heads предсказывают свойства и H.function. P/h/f и процессы разрешения
остаются следующими slices. Нулевой/majority baseline здесь не измерялся,поэтому
этот отчёт описывает способность выбранной модели,не причинный прирост от обучения.

E1 обучался80000 applied updates. Dev использовал один выбранный40k checkpoint.
На13 dev families family-macro H recovery=0.4643; pooled2965/11212=0.2644.
Это recall известных H при IoU>=0.5,one-to-one match,до64 predictions/4per qn,
а не accuracy или precision. Неполная разметка не лицензирует общийfalse-positive
счёт. Между families большая неоднородность: макро и pooled значения различаются.

## Свойства на найденных и размеченных H

| Поле | Family-macro conditional F1 | Matched support | Families |
|---|---:|---:|---:|
| H.sonority.root_pc | 0.4774 | 1477 | 12 |
| H.sonority.core | 0.5991 | 951 | 10 |
| H.sonority.seventh | 0.5366 | 204 | 4 |
| H.inversion | 0.3328 | 665 | 11 |
| H.function | 0.1640 | 1607 | 12 |

Core map=augmented,diminished,major,minor,power,sus4. Root —pitch class без spelling.
У функции1607 matched labels,pooled accuracy0.4997,ноfamily-macro F1 лишь0.1640.
1471/1607 ответов —tonic илиdominant; это91.5%,при86.2% этих классов в matched
truth. Дляmodal_center нет предсказаний на3 соответствующих matched examples.
Редкие функции не оцениваются надёжно по общейaccuracy. Даже отдельнаяoracle-H
диагностика F1=0.2079,но у неё другойsupport; разницу нельзя считать причинным
эффектом исправления границ. Joint H+F family-macro recovery=0.1523.

Надстройка9:10 known dev labels,6 matched,все одного positive класса. F1=1.0 здесь
не подтверждает универсальный анализ надстроек/их отсутствия. Bass_pc не имеет
known dev support,аbass_degree всего2labels. Эти выходы нельзя объявлять обученными.
Для12interval bins отдельныеF1 также не равныexact chord-set accuracy.

## Почему selected40k

Правило было зафиксировано до обучения:earliest maximum inner-selection
family-macro known-H recovery. Получено:

| Applied update | H recovery(selection) | Joint H+F(selection) | Conditional F F1(selection) | Mean train loss,last1000 applied |
|---|---:|---:|---:|---:|
| 2000 | 0.3613 | 0.1669 | 0.2612 | 0.3789 |
| 5000 | 0.3437 | 0.1262 | 0.2920 | 0.2471 |
| 10000 | 0.3269 | 0.1681 | 0.4404 | 0.1619 |
| 20000 | 0.3594 | 0.1682 | 0.4808 | 0.1085 |
| 40000 | 0.3703 | 0.1965 | 0.2556 | 0.0696 |
| 60000 | 0.3621 | 0.1305 | 0.2239 | 0.0524 |
| 80000 | 0.3564 | 0.1760 | 0.4127 | 0.0398 |

40k даёт0.3703 H recovery;80k —0.3564. Прирост40k относительно2k всего0.0090
(0.90percentage points),поэтому это максимум неустойчивой одиночной кривой,
не доказанный универсальный optimum длительности. По условнойF-метрике максимум
на20k,но набор recoveredH там отличается. Joint H+F тоже максимален на40k;
rule всё равно выбирал толькоH,а не все42heads. Dev для всех checkpoints не
оценивался:46.4% dev и37.0% selection относятся к разным наборам,не к приросту.

Training objective продолжает улучшаться,а H-retrieval не растёт монотонно.
Это совместимо с переобучением и несовпадением цели/loss с retrieval ranking;
одинseed не устанавливает причину. Loss усредняет maskedCE доступных heads,
многие из нихPCbins. E1 boundaries фиксированы; function/attribute gradients могут
менять общиеfeatures и ранжирование при уменьшении общегоloss. Полноценных
подтверждённых отрицательных окон нет; positive objectness не калибрует отсутствие.
Сатурация scores/конкретный механизм деградации здесь не измерялись.

Retained raw-bank diagnostic:unbudgeted H recovery0.9710 macro,10769/11212 pooled.
Возможность найти H среди всех окон существенно выше результата ограниченного
отбора. Этот контроль объединяет эффект ранжирования и outputbudget и не
позволяет приписать весь разрыв только качеству scoring. Полноценный анализ
причин потребовал бы отдельно спланированной ablation; текущая очередь неизменна.

## Evidence scope

EXP-HIERGRAPH-80K-E1-DETAIL-REVIEW-001 completed. Only retained existing JSON and
streamedloss were read; no GPU/inference/checkpoint or source corpus access.
Aggregate H/F exactly matches registered result. Exposure86967draws/80000updates
and hash chain validated. Original fullreport/loss bytes had no prior registered
digest; new SHA/size recorded at review time. Compact extraction hashes retained.
Weights stayremote,not locally byteverified. Source/model labels weak;
exploratorydev insideglobalTRAIN,TEST closed,raw pretraining overlap unknown.
