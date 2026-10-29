# Факторизованный HIERGRAPH: контракт, преобразователь и экспорт, 27 октября 2026

Эксперимент `EXP-HIERGRAPH-FACTOR-EXPORT-001` завершён. Один зарегистрированный
CPU-проход подготовил 67 произведений и прошёл последующую проверку всех
численных артефактов. Обучение, GPU, LLM API, global VALIDATION/TEST не запускались.

## Реализованное поведение

Аккорд проецируется в root/core/seventh, независимые надстройки, добавления,
альтерации, явные пропуски, бас/его ступень и обращение. `Fmaj9/G`, `Gm11no9`,
`E7(b9,no3)` и другие поддержанные формы разбираются без создания новых
физических нот. Неуказанное остаётся masked. Конфликт исходных компонентов
маскируется локально; зависимые обращения учитывают окончательные root/bass.
Самостоятельно размеченная относительная ступень баса не требует абсолютной
высоты. Altered fifth ограничивает допустимое second inversion.

Исходные H/h IDs, exact intervals, views, note associations и relation endpoints
сохранены. Новые процессы, голоса или границы H не выводились. Полные исходные
значения и provenance доступны по проверенным SHA/size bindings; компактный
экспорт сохраняет per-component rules, states, origins/confidences и OOV.
Контракт: [factorized-chord-targets](../workstreams/2026-10-27/hierarchical-graph-analyzer/factorized-chord-targets.md).

## Данные и разбиение

- 67 работ; 44 fit / 10 selection / 13 dev внутри global TRAIN.
- Предыдущие family assignments сохранены; словарь построен только по fit.
- 134 исходных canonical/bundle файла, 3 182 172 659 байт; 201 read operation
  из-за повторной проверки canonical при кодировании. Reports/MIDI не читались.
- Сохранены 45 263 H и 274 936 h, включая удержанные ноты в нескольких H.
- Все 134 projection/numeric gzip-файла сверены по SHA/size. Все 67 numeric
  JSON прочитаны, arrays/masks/IDs/source references/fit bounds проверены.
- 60 полей экспортировано; 16 прошли минимальный технический fit-support gate,
  44 deferred. Это не доказательство достаточности разметки для качества модели.

## Покрытие

Доступность ниже — exact+coarsened semantic projection по всему current67,
до возможного selection/dev OOV. «Семьи» — число независимых source families
с доступным значением. Это не точность музыкального распознавания.

| Поле | Доступных значений | Семьи | Минимальный gate |
|---|---:|---:|---|
| `H.sonority.root_pc` | 14827 | 62 | eligible |
| `H.sonority.quality` | 6954 | 30 | eligible |
| `H.sonority.core` | 12509 | 58 | eligible |
| `H.sonority.seventh` | 1973 | 34 | eligible |
| `H.sonority.extension.9` | 88 | 6 | eligible |
| `H.sonority.extension.b9` | 30 | 3 | deferred |
| `H.sonority.extension.#9` | 31 | 1 | deferred |
| `H.sonority.extension.11` | 36 | 4 | deferred |
| `H.sonority.extension.#11` | 31 | 2 | deferred |
| `H.sonority.extension.13` | 25 | 2 | deferred |
| `H.sonority.extension.b13` | 6 | 1 | deferred |
| `H.sonority.addition.9` | 66 | 8 | deferred |
| `H.sonority.omission.3` | 20 | 4 | deferred |
| `H.sonority.omission.5` | 49 | 5 | deferred |
| `H.bass_pc` | 22 | 8 | eligible |
| `H.bass_degree` | 22 | 8 | eligible |
| `H.inversion` | 9694 | 56 | eligible |
| `h.chord_membership` | 146785 | 61 | eligible |
| `h.chord_degree` | 0 | 0 | deferred |

Для natural ninth: 80 положительных и 8 явных отрицательных значений. Большинство
остальных extension/modification полей имеют лишь положительные примеры или
узкую поддержку. Их отсутствие в аннотации не превращено в отрицательные метки.
Eligibility bass-полей также не означает большой датасет: абсолютный/относительный
бас имеют лишь 22 доступных события. Веса и состав первых model losses ещё
нужно выбирать по этому паспорту до обучения.

Нормализованный core поддержан в 12 509 событиях/58 семьях; старый basic quality
сохраняется отдельным target (6 954/30). Это разные представления, не измерение
улучшения модели. У H ещё много unavailable sonority и неаккордовых/свободных
описаний: коллекции высот не принуждаются к трезвучной интерпретации.

Авторский h.chord_degree сохранён в source evidence, но его 142 038 известных
исходных значений не объявлены автоматически major-reference. В этом экспорте
`degree_frames={}`, поэтому h degree losses недоступны. Требуется отдельное
проверенное source-specific admission, а не угадывание convention.

OOV event targets: `{"harmonies:H.bass_degree": 5, "harmonies:H.bass_pc": 5, "harmonies:H.sonority.alteration.#9": 31, "harmonies:H.sonority.alteration.b5": 8, "harmonies:H.sonority.core": 1, "harmonies:H.sonority.extension.#9": 31}`.
Object OOV: `{"harmonic_process:P.function": 4}`.
Raw значения и причины недоступности сохранены; vocabulary не расширялся на dev.

## Ресурсы и проверки

Wall: 663.663 s (11.06 min).
Суммарный worker CPU: 578.797 s; parent CPU отдельно
не измерен. Максимальный наблюдённый process-group RSS:
3.442 GiB при лимите 6 GiB.
Проверка готовых numeric artifacts: 19.946 s.
Attempt один, completed; failed/aborted=0. Output tree:
129,188,391 bytes (ignored).

Synthetic suites: **135 tests: 133 passed, 2 skipped**. Проверены chord parser
29, projection49(1skip), export15, lazy facade8(1skip), census23, stream11.
Spec и Standards review обнаружили и помогли исправить зависимые masks,
altered-fifth inversion, coarsening eligibility и duplicate identity leakage;
исправления независимо перепроверены. Pure imports обычные, без fake dependencies.
Два skipped связаны с отсутствующими Torch/PyG/mido; tensor runtime не объявлен
проверенным. Legacy не использовался, новых зависимостей нет.

## Границы выводов и следующий этап

Подтверждены исполнимость преобразования, целостность/геометрия экспорта,
fit-only IDs и измеренное покрытие. Не подтверждены human musical correctness,
качество новой модели, полнота разметки, польза иерархии или generalization.
Grammar/ontology учитывают retained inventory всех67, downstream dev остаётся
exploratory; raw pretraining overlap неизвестен. TEST закрыт.

Следующий этап — общая модельная механика и пять H constructors с единым
контрактом, после отдельного native gate/протокола. P/feedback, live C3,
NBF-like и Query2box сохраняются последующими опытами.

Основные артефакты: `outputs/hiergraph_factorized_20261027/attempt001/summary.json`,
`verification-receipt.json`, `test-receipt.json`, `export001/manifest.json`,
`vocabulary.json`, `heads.json`; snapshots и большие target-файлы вне Git.
