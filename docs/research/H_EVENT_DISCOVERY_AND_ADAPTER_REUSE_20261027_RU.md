# Построение H и переиспользование существующих adapters

27 октября 2026. T08/HIERGRAPH: статический аудит опубликованного кода и исследование
вариантов detector. Это уточнение предложения, без нового корпусного прогона,
training/inference, TEST, внешних платных ресурсов или изменения production defaults.

> Продолжение от27октября: пользователь выбрал сравнение **всех четырёх**
> вариантов ниже с пятым graph-assignment pooling. Актуальный состав, supervision
> и порядок — [в плане AutoSchA/E1–E5](AUTOSCHA_HIERARCHICAL_POOLING_PILOT_20261027_RU.md).
> Предложенный в конце этого обзора сокращённый subset не является текущим планом.

## Проверяемые вопросы и результат

1. Выбор H из заранее заданных окон не установлен как лучший метод. Это удобный
   контроль, ограниченный покрытием банка. Следует сравнить learned boundaries и
   event-query detector до фиксации механизма H для A/B/C graph-update опытов.
2. Query2box не определяет сам по себе число новых H и их временные границы.
   Его геометрия множеств ответов может дополнить membership/constraint heads.
3. Унификация датасетов и значительная часть численного кодирования targets уже
   реализованы. Формулировка «создать адаптер» была слишком широкой: нужен consumer
   существующих контрактов для новых model outputs и ограниченная semantic projection.

## GitHub: что действительно опубликовано

Проверены GitHub API `commits/main` и `git ls-remote origin refs/heads/main`:
оба вернули `645922d4db282ff0c3a06c1cf82ecc9d0594f5c7`; commit date 25 октября 2026,13:45:55UTC.
Исследованы committed blobs этого SHA через git show, без подмены незакоммиченным
WIP основного или isolated checkout. Основной checkout не изменялся.

| Готовый слой | Что сохраняет/делает | Проверенный опубликованный код |
|---|---|---|
| Source adapters | HookTheory, POP909-CL, Dilemmadata и PDMX → canonical/raw и допустимые targets | [adapters](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/adapters/hooktheory.py#L1479), [POP909-CL](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/adapters/pop909_cl.py#L1599), [Dilemmadata targets](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/adapters/dilemmadata_targets.py#L950), [PDMX](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/adapters/pdmx.py#L266) |
| Общие sample/target контракты | IDs, exact spans, masks, confidence и provenance | [SampleTarget / TargetBundle](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/tasks/multisource.py#L55) |
| Общая загрузка | Canonical cache, identity/availability checks, attachment targets без изменения raw graph | [IndexedMultiSourceDataset](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/tasks/loading.py#L613) |
| Базовое кодирование и collator | Categorical/multilabel targets, deferred open vocab, tensorized alignment | [encoding](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/tasks/encoding.py#L107), [collator](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/tasks/collator.py#L141) |
| Гармонии, h и p | TRAIN vocab, IDs/intervals/views/note indices, masks/OOV, directed endpoints и participants | [build_vocabulary / encode_enrichment](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/tasks/event_theory_enrichment.py#L234) |
| P и другие объекты | Process/motif/phrase/section/ensemble semantics, member IDs, intervals, states, masks/OOV | [build_object_vocabulary / encode_object_fields](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/tasks/whole_work_annotation.py#L440) |
| Контекстные подинтервалы нот | Exact context intervals, H IDs, provenance и отдельные physical links | [DenseHarmonicBatch](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/tasks/dense_harmonic_targets.py#L537) |
| Raw временные опоры | Начала/концы нот, начала/концы долей; без доступа к targets | [raw_time_sources / candidate_time_sets](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/data/raw_time_candidates.py#L46), [readout slots](https://github.com/nairary/music_critic_v2/blob/645922d4db282ff0c3a06c1cf82ecc9d0594f5c7/src/music_critic/data/raw_time_readout_slots.py#L112) |

Это давняя инфраструктура: multisource contract `eae8b17`, collation `2d73d40`,
loading `2c39a0d` — 26 июля; Dilemmadata sidecars `127913d` и supervision `0f45802` —
17 августа. Enrichment/object/dense modules включены в published snapshot `8f6768b`
от25октября. Это подтверждение существующего кода, не новый тест runtime/качества.

Оставшиеся различия интерфейсов:
- IndexedMultiSourceDataset принимает DilemmadataTargetCacheIndex; enrichment/object
  APIs существуют отдельно. Их composition для current67 и model batch требует wiring.
- encode_enrichment умеет кодировать целый typed JSON как class ID. Это не означает,
  что уже выбрана нужная факторизация sonority на root/quality/reading или что
  source-native common projections покрывают каждую authored interpretation.
- Ни численные targets, ни raw time slots не реализуют новый H/P detector или
  matching его предсказанных объектов с target objects.

Нужно переиспользовать CanonicalPiece, TargetBundle, существующие encoding/validation,
context intervals, masks/OOV/provenance. Новый слой ограничить model-facing assembly,
prediction-to-target matching, нужными semantic projections и masked losses.
Не создавать параллельную универсальную систему ingestion/vocabulary/sidecars.

## Варианты построения H

| Подход | Что предсказывает | Практические ограничения |
|---|---|---|
| Fixed raw span bank | Оценка заранее предложенных интервалов | Простое сравнение, но recall ограничен банком; множество близких spans |
| Boundary/segmentation head | Вероятность смены H и свойства последовательных сегментов | Естественно для одного согласованного harmonic view; overlapping scopes требуют отдельных каналов/decoder |
| Anchor-free endpoint head | Для временной позиции — класс и смещения до начала/конца | Не нужен список длительностей; всё ещё есть временное разрешение, duplicate handling и sampling policy |
| Learned event queries | Набор H с временными границами, состояниями и участием нот | Требуются matching, ограничения числа объектов, обучение confidence при неполной разметке |

Для совместной сегментации и chord labeling существует прямой музыкальный precedent:
[Masada & Bunescu, segmental CRF](https://arxiv.org/abs/1810.10002).
Он оценивает согласованную последовательность сегментов; это альтернатива независимым
window-positive scores, хотя сам также использует пространство допустимых сегментов.

[ActionFormer](https://arxiv.org/abs/2202.07925) использует multiscale states и
предсказание класса/границ по времени без predefined anchor windows.
[DETR](https://arxiv.org/abs/2005.12872) использует learned object queries и matching;
[TadTR](https://arxiv.org/abs/2106.10271) переносит такой принцип на временные события
с уточнением границ. Их результаты получены на визуальных задачах; перенос на MIDI
и weak current67 — наша непроверенная гипотеза.

Предлагаемый для сравнения query-вариант:

```text
Raw MIDI → существующий encoder → states нот/времени/контекста
                                 ↓
                    обучаемые H-запросы
                                 ↓
             H: start/end + состояние + soft участие
                                 ↕
           общие n/h/H/P updates + F/f/p heads
```

H-query — рабочий обучаемый вектор, получающий содержание через внимание к
конкретной пьесе. Он не обозначает заранее заданный аккорд, фиксированное окно
или корпусный H-ID. Модель предсказывает границы; состояния нот сохраняются для
дальнейшего обмена. Равенство attention и истинного membership не предполагается:
для семантического участия нужна отдельная предсказываемая величина и supervision.

Число queries задаёт ёмкость. Для длинной пьесы нужен bounded local/whole-piece
режим, владение событиями на границах блоков и измерение пропусков по capacity;
несколько десятков queries на всё произведение не считаются достаточными заранее.
Для единственного выбранного H-view потребуется согласованный non-overlap decoder;
разные scopes/альтернативные views/P не обязаны образовывать одну последовательность.

Стандартный DETR no-object loss нельзя перенести автоматически: unmatched prediction
не является false при неполных targets. Negative scopes, mask policy и полнота
структурной разметки проверяются до обучения. При частичных spans отдельно задаётся
matching с h подинтервалами. Exact target times сохраняются; float features/loss
не заменяют RationalTime и не используются для exact equality. Разрешение и
преобразование predicted times в выходной формат должны быть явно заданы.

## Роль Query2box

В [Query2box](https://arxiv.org/html/2002.05969v2) box представляет множество
ответов в embedding space; проекции отношений и intersection отвечают на составной
запрос. Это не временной bounding interval и не готовый генератор новых H.
Оригинальная модель учит embeddings сущностей KG. Для новых MIDI нужен inductive
вариант с embeddings из нашего encoder.

Наша возможная адаптация: H-query создаёт latent box для soft оценки подходящих
участников; отдельная голова предсказывает время. Box-операции затем объединяют
условия функции/участия/связей. Это отдельная гипотеза относительно обычного scorer,
не стандартное применение статьи и не причина автоматически называть membership
составным логическим reasoning. Выборка queries/answers и known negatives требует
собственного протокола. Raw geometric доступ к non-chord нотам сохраняется.

## Исправление интерпретации предыдущего anchor census

[Аудит current67](../experiments/HIERGRAPH_SUPERVISION_20261027_RU.md) сообщил91.8%
H с обеими границами на его raw anchors. Статическая проверка нашла расхождение:
скрипт добавлял start_qn и end_qn при наличии, тогда как CanonicalBeat/CanonicalBar
хранят start_qn + duration_qn. Готовый raw_time_candidates.py уже вычисляет концы
долей правильно. Следовательно,91.8% характеризует историческую сетку конкретного
счётчика и не является проверенным покрытием существующего raw helper.

В счётчик добавлено вычисление метрических концов по duration и regression test.
До исправления:23 census tests,4 ожидаемых сбоя подслучаев. После:23/23 passed
(независимый повтор root:0.068s); streaming11/11 passed,0skipped. Проверка механики
не является новым измерением покрытия корпуса. Auditor SHA256 после исправления:
`8063b4729470b11c255cb0faf9eec18c90aa4bc00d7b72f52b84a8d67d0d4d53`.
Источник запуска AUDIT-002 сохранён в immutable outputs/source001; terminal record,
корпус и старый отчёт не переписаны. Новый corpus replay не проводился, исправленная
численная coverage неизвестна. Возможное совпадение конца доли со следующим началом
не позволяет заранее утверждать ни изменение, ни сохранение91.8%.

Эта ошибка не относится к field availability, P/h joins или их опубликованной parity;
их кодовые пути не менялись. Ограничение окна-кандидата остаётся свойством класса
методов, но величину эффекта нашего исправленного банка нужно измерить отдельно.

## Следующее решение

Предлагается оставить fixed-window bank контролем и сравнить небольшой boundary
head с learned H-query head при общем encoder/context, одинаковой supervision и
ограниченном budget. Для текущего single-view H boundary head проще; H-queries
лучше соответствуют долгосрочному запросу на набор сущностей и overlapping scopes,
но это структурное соответствие, а не доказанная точность/скорость.

После выбора detector отдельно проверять A/B/C graph updates, затем box membership/
logical-query добавление. Не смешивать изменения генератора H, hierarchy updater и
Query2box в одном сравнении без абляций. Реализация и запуск нового detector этим
обзором не выполняются. Переиспользование существующих adapters — обязательная
отправная точка; слово «адаптер» далее означает узкий model supervision consumer.
