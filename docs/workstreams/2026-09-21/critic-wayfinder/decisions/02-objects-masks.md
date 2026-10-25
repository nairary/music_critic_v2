# D2 — Что скрыто, что предсказывается и какой контекст разрешён

Статус: **рекомендуемый контракт; выбор охвата ещё не принят**.
Зависимости: нет. Владелец подготовки: агент. [Карта](../README.md).

## Первая ступень и последующее расширение

| Объект | Предсказываемое содержимое | Видимый контекст | Что оценка ещё не покрывает |
|---|---|---|---|
| Созвучие в raw event | 12-bit sounding pitch-class set | Наблюдаемый rhythm/query skeleton; музыка до/после, иногда другие tracks | Octave, voicing, multiplicity, articulation, «красоту» |
| Track×event | PC-set выбранной raw дорожки | Одновременные другие дорожки + её горизонтальный контекст | Семантическую роль или независимый voice |
| Track×1–2 bars | Последовательность PC-sets в блоке | Другие tracks внутри блока; контекст до/после | Полный rhythm/count likelihood |
| Все tracks×1–2 bars | Совместная последовательность PC-sets | Музыка вне блока | Полную вероятность raw MIDI |
| Semantic chord / chord sequence | Source-native chord labels, отдельно от actual notes | Скрытое соответствующее музыкальное содержимое + окружение | Узнавание видимого аккорда не равно completion; label spaces различны |
| Полная track | Позже PC sequence, затем pitches/octaves/durations/count | Остальная музыка + явно указанная часть track skeleton | Первый вариант при видимом skeleton не является whole-track likelihood |
| Полный onset / fragment | Позже attack/release/count/time и pitches | Fixed raw метрические query slots + внешняя музыка | Нужны новый decoder, поддержку variable length и graph rebuild |

Рекомендация первой ступени — первые четыре строки. Это уже многоуровневая
условная оценка, не PLL одиночных нот. Event-query задаётся raw timing,
не gold chord boundary. На первой ступени точные времена и note-count skeleton
объявлены условием; не утверждать качество ритма на основе такого score.
Семантический chord decoder полезен отдельным диагностическим контролем после
source-semantic gate; не использовать его как замену likelihood звучащих нот.

## Горизонтальный и вертикальный контекст

Для одинаковых targets сравнить full context, только внешний горизонтальный,
только одновременные другие tracks, shuffled/no musical context. Сохранять
одинаковые query times, metadata, block budget и score normalization; отмечать
изменение распределения masked inputs. Это проверка полезности контекста в
постановке, а не доказательство причинного музыкального понимания.

Основной контекст — всё произведение, которое уже обрабатывает baseline;
целевой блок сначала 1bar, затем 2bars. Отдельно ограничить доступный внешний
контекст до ±1/±4bars только при измеренной необходимости/стоимости. Не менять
одновременно context window, decoder, backbone и loss. Для длинных произведений
детерминированные окна и boundary masks задаются до inference; cropped context
помечается в выходе. Это не новая semantic segmentation.

## Новый mask-aware вход

Существующий SSL masked view нельзя подключить к C3 одним флагом:
`TemporalSliceModel` вызывает `base.encode(graph.base)`, а corrected encode не
принимает overlay. Warmstart harness переносит local/hierarchy части C2,
не доказывая поддержку temporal+BiGRU masked path.

Предлагаемый будущий seam: `MaskedConditionalInput` + `BlockQuery` + отдельный
target sidecar. Маскирование применяется **до** local GNN, pooling, Transformer,
BiGRU и temporal stage. Нельзя передать decoder full-view embedding cache.
Формат и точные имена API — предмет mc-to-spec, не готовая реализация.

Проверить все каналы скрытого содержимого:

- Pitch, PC, octave, relative pitch, track mean/std/min/max, derived intervals.
- Порядок simultaneous notes, IDs, source indices и `next_in_track`, если порядок
  зависит от pitch. `ssl/hierarchy_leakage.py:53–56` прямо допускает rank leakage
  в старом SSL; новый строгий контракт должен его устранить или явно обусловиться
  на rank. Рекомендация — устранить, не ослаблять заявленную задачу.
- Все вхождения удерживаемой ноты, включая outside-block context и её сводки.
  Если одна нота звучит в target block, видимая копия её pitch вне блока раскрывает
  target. Её скрытая identity closure фиксируется в mask manifest.
- Starts/holds/ends, exact onset/offset, gaps, node/edge counts и query creation.
  Для pitch-given-rhythm они разрешены; для hidden rhythm/whole-content — ответы,
  которые удаляются до перестройки графа. Fixed metric queries не строятся из
  скрытых атак/окончаний. Отсутствие target в MIDI после удаления не равно rest label.
- Theory values, availability masks, annotation IDs/границы, semantic roles и
  экспертные voices не могут влиять на predictor input/query selection.

Новый граф сохраняет exact rational time и raw-MIDI путь. Удаление/маскирование
не изменяет original canonical/artifacts; производный view имеет свою версию.
`track_id` означает raw membership, а не melody/accompaniment/voice.

## Самая маленькая проверка и STOP

Synthetic pairs отличаются только скрытыми pitch values, chord composition,
длительностью или onset — в зависимости от объявленного mask regime. Требовать
идентичных predictor-visible bytes/embeddings inputs/query schedule при равном
разрешённом условии; target sidecars должны различаться. Проверять attacks,
sustains, releases, octave duplicates, track permutation, empty tracks и grace.

Любое необъявленное раскрытие target блокирует PLL-claim и native gate.
Не чинить утечку лишь в loss после полного encoder pass. Эти проверки имеют
нулевую облачную стоимость; реальные budgets — [D7](07-budget-gates.md).
