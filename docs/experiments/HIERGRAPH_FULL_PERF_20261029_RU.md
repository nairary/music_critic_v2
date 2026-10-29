# Full HIERGRAPH: CPU/GPU bottleneck diagnostic —29октября2026

В живом процессе подтверждён сильный простой GPU при занятом одном логическом
CPU. Предварительная подготовка данных полезна,но не устраняет измеренную
длинную паузу целиком. Точная доля matcher и autograd/запусков kernels ещё
не атрибутирована профилировщиком.

Parent:EXP-HIERGRAPH-FULL-E3-S17-001,source002 manifest
fa5419f480503f0131e9380bf81eea95930a034dd77ec49f3583124a6303ccf5.
Наблюдение не останавливало и не изменяло обучение. Завершающий snapshot:
115applied/115attempted/0skipped,selection10/50/100 completed. Диагностические
процессы не запускали модель/optimizer/GPU;parent updates им не приписываются.

## PERF-001:120 секунд неинвазивной телеметрии

61sample,период2s;actual120.034s. Процесс оставался на110 завершённых updates,
то есть обрабатывал следующий training draw,не selection. CPU≈100.02% одного
логического CPU при28 доступных. GPU mean6.46%,median9%,max19%;все61sample<20%.
Logical file reads=0bytes,physical reads=0bytes;RSS≈3977MiB и стабилен.

Это доказывает CPU-active ожидание с низкой занятостью GPU и отсутствие чтений
в этом интервале. Оно само по себе не определяет конкретную Python-функцию и
не отличает CPU preprocessing,matching или host-side autograd dispatch.
При первом коротком snapshot ещё шёл selection100; основной120s интервал уже
попал в training,что исключает объяснение всей паузы только checkpoint оценкой.

Source-bound reader и дальнейшая completed exposure row111 подтверждают
work038 для sampler cursor110. В этой строке:knownH2644,matchedH2033;
knownh10746,matchedh1418;P201/201. Медленный предыдущий update100 былwork040:
H2549/2549,h8359/1302,P37/37. Читались только эти compact support поля,
полный exposure не скачивался,source labels для этой проверки не открывались.

## PERF-002:CPU-подготовка того же fit work038

Seed17,cursor110 однозначно выбираютwork038 из sorted43 fit IDs. Один отдельный
CPU процесс использовал замороженный source002 и существующие manifests;
модель/optimizer не создавались. Данные:32окна,1781нота,1view;raw tensor payload
40868872bytes(≈38.98MiB),это не RSS и не размер полной разметки.

| Фаза | Время |
|---|---:|
| Открытие общей metadata |0.630s|
| Canonical auth/parse/validation |0.037s|
| Raw tile loading + whole-work assembly |0.325s|
| Numeric auth/decompression/parse |1.384s|
| Target views + masks/range checks |1.672s|
| Всего |4.049s|
| Подготовка входа без общей metadata |3.419s|

Это повторное чтение при существующем OS page cache,не cold-storage benchmark.
В worker была проверена та же схема,физические note IDs,fit scope и bounds.
В этой пробе основная подготовка приходится на targets,а неraw tensor loading.
Даже полностью скрыв измеренные3–4s,одну наблюдённую≥120s паузу убрать нельзя;
для этого конкретного шага это лишь несколько процентов. На другие произведения
и всю кампанию такой процент не экстраполируется.

## Кодовые кандидаты и приоритет

1. `full_matching.py:12`:Hungarian выполняет Python while/for с мелкими CPU tensor
   операциями. На work0382644 target H против2033 prediction H прямоугольное
   назначение может стать дорогим. Конкретная stack attribution не снята.
   Возможная оптимизация:ориентировать задачу меньшей стороной в строки и/или
   использовать быстрый compiled solver,проверив cardinality,cost,masks,ties.
2. `full_model.py:349`:relation features строятся в цикле по каждому H/h;
   на каждой итерации создаётся CUDA index tensor и отдельныйcat/gather.
   При десятках тысячh это много мелких kernels и узлов autograd. Следует
   сначала сформировать source/target indices,затем выполнить batched gathers.
   Это проверенная структура кода,но ещё не измеренный процент текущей паузы.
3. `full_runtime.py:99`:reader вызывается синхронно перед forward;raw cache в
   reader хранит лишь последнее произведение. Канонический разбор и target views
   повторяются при переходе к другому произведению. Bounded RAM cache и1–2CPU
   workers с prefetch следующего sample могут скрыть эту отдельную фазу.

Prefetch должен сохранять work/view order,exposure hash,mask semantics и exact
source binding. Можно кешировать статические raw inputs и target views;
matching,предсказанные H/h/P и trainable E3 states зависят от текущих весов и
не являются готовой статической очередью. GPU memory почти заполнена по
nvidia-smi(включая allocator cache),поэтому начальный prefetch следует хранить
вCPU RAM,а не держать два полных autograd-графа одновременно.

## Claim matrix

| Утверждение | Статус |
|---|---|
| В измеренном training интервале GPU простаивает при одном занятом CPU |supported|
| Простой вызван ожиданием чтения диска в эти120s |не поддерживается:reads0|
| CPU-подготовкаwork038 стоит≈3.42s без общей initialization |supported,одна warm-cache проба|
| Только prefetch устранит всю длинную паузу |не поддерживается|
| Основные кандидаты:matcher и мелкие relation/autograd operations |обоснованная гипотеза,нужна фазовая attribution|
| Конкретный общий speedup от очереди или vectorization |пока не измерен|

Records PERF-001/PERF-002 completed;SHA/size local compact artifacts сохранены в
реестре. Новые зависимости не устанавливались. CPU sampling stack profiler не
был доступен;py-spy отсутствует,никакие ptrace/security настройки не менялись.
Source002 и running campaign не изменены;нового обучения/inference нет.
Следующий implementation slice:фазовые timers+batch relation features и bounded
CPU preparation queue;корректность и speedup проверять отдельно перед применением
в новом source/продолжении. Текущий frozen500-update run продолжает свой протокол.
