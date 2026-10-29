# Интервальный HIERGRAPH V3: реализация и экспорт current67, 27 октября 2026

`EXP-HIERGRAPH-INTERVAL-EXPORT-001` completed. Реализована отдельная интервальная
проекция и новый CPU-экспорт тех же 67 TRAIN-произведений. Разделение 44 fit /
10 selection /13 dev сохранено побайтно; словарь строится только по fit. Обучения,
GPU и LLM API не было. Original release, исходные annotations, V2 экспорт и
terminal records не изменялись.

## Что реализовано

`ChordIntervalProjection@1` вычисляет `(tone_pc - root_pc) % 12`, разбирая явные
PC/MIDI и поддержанные названия нот с октавой. В `HiergraphProjection@3.0.0`
добавлены 37 независимых полей: состав интервалов аккорда, абсолютный состав
наблюдаемого созвучия, его интервалы и отдельный наблюдаемый root. Всего 97 полей.
`HiergraphVocabulary@3`/`HiergraphNumericTargets@3` не смешиваются с V2; config и
workers проверяют полную версию/контракт до чтения payload.

Корень аккорда и корень наблюдаемого среза независимы. Literal root извлекается
только из связанного с тем же H/span/участниками доказательства. Конфликтующие
scope не дают относительных меток, но проверенный абсолютный состав сохраняется.
Нет подстановки тоники/баса, inference корня или создания нот.

В составе аккорда нули разрешены только проверенным полным authored template;
частичный список даёт positive/unknown. Точное имя аккорда для интервальной цели
необязательно. Для наблюдаемого созвучия полнота проверяется по canonical
pitched-note IDs, pitches и exact half-open span без внутренних границ нот;
неопределённый pedal scope исключается. Старые PC-поля остаются positive-only.
Ноль в новом observed target не означает nonchord/omit/no. Bin 6 объединяет b5/#11,
bin 2 — 2/9; исходное написание и прежние точные факторы сохраняются.

[Контракт](../workstreams/2026-10-27/hierarchical-graph-analyzer/factorized-chord-targets.md)
содержит TODO `HIERGRAPH-ENHARMONIC-001`. Отдельный renderer названия, транспозиции,
обучение и пять H constructors здесь не реализованы.

## Покрытие новых целей

Числа ниже относятся к semantic projection по current67 до возможного OOV при
fit-only numerical encoding. «Полные 12» — события со всеми известными bins;
true/false — число отдельных известных булевых значений, не число аккордов.
Группы могут пересекаться; складывать их события нельзя.

| Группа | События с метками | Полные 12 | true / false | Семьи |
|---|---:|---:|---:|---:|
| `H.chord.interval_pc.` | 10309 | 6994 | 34291 / 60928 | 45 |
| `H.observed.interval_pc.` | 3212 | 3212 | 9793 / 28751 | 10 |
| `H.observed.pitch_class.` | 11096 | 11096 | 28610 / 104542 | 10 |
| `H.observed.root_pc` | 3212 | — | — / — | 10 |

Из 97 scalar fields 47 проходят минимальный fit-support
gate, 50 deferred; среди новых 37 eligible
31. Это не 97 отдельных моделей и не гарантия
достаточности данных. Детализация позволяет выбрать masks/losses следующего пилота.

| Новое поле | Доступные H | Семьи | Минимальный gate |
|---|---:|---:|---|
| `H.chord.interval_pc.0` | 10303 | 45 | eligible |
| `H.chord.interval_pc.1` | 7025 | 33 | eligible |
| `H.chord.interval_pc.10` | 7725 | 37 | eligible |
| `H.chord.interval_pc.11` | 7032 | 35 | eligible |
| `H.chord.interval_pc.2` | 7307 | 35 | eligible |
| `H.chord.interval_pc.3` | 7774 | 42 | eligible |
| `H.chord.interval_pc.4` | 9265 | 43 | eligible |
| `H.chord.interval_pc.5` | 7166 | 33 | eligible |
| `H.chord.interval_pc.6` | 7050 | 37 | eligible |
| `H.chord.interval_pc.7` | 10151 | 45 | eligible |
| `H.chord.interval_pc.8` | 7001 | 32 | eligible |
| `H.chord.interval_pc.9` | 7420 | 33 | eligible |
| `H.observed.interval_pc.0` | 3212 | 10 | deferred |
| `H.observed.interval_pc.1` | 3212 | 10 | deferred |
| `H.observed.interval_pc.10` | 3212 | 10 | eligible |
| `H.observed.interval_pc.11` | 3212 | 10 | eligible |
| `H.observed.interval_pc.2` | 3212 | 10 | deferred |
| `H.observed.interval_pc.3` | 3212 | 10 | eligible |
| `H.observed.interval_pc.4` | 3212 | 10 | eligible |
| `H.observed.interval_pc.5` | 3212 | 10 | deferred |
| `H.observed.interval_pc.6` | 3212 | 10 | eligible |
| `H.observed.interval_pc.7` | 3212 | 10 | eligible |
| `H.observed.interval_pc.8` | 3212 | 10 | deferred |
| `H.observed.interval_pc.9` | 3212 | 10 | deferred |
| `H.observed.pitch_class.0` | 11096 | 10 | eligible |
| `H.observed.pitch_class.1` | 11096 | 10 | eligible |
| `H.observed.pitch_class.10` | 11096 | 10 | eligible |
| `H.observed.pitch_class.11` | 11096 | 10 | eligible |
| `H.observed.pitch_class.2` | 11096 | 10 | eligible |
| `H.observed.pitch_class.3` | 11096 | 10 | eligible |
| `H.observed.pitch_class.4` | 11096 | 10 | eligible |
| `H.observed.pitch_class.5` | 11096 | 10 | eligible |
| `H.observed.pitch_class.6` | 11096 | 10 | eligible |
| `H.observed.pitch_class.7` | 11096 | 10 | eligible |
| `H.observed.pitch_class.8` | 11096 | 10 | eligible |
| `H.observed.pitch_class.9` | 11096 | 10 | eligible |
| `H.observed.root_pc` | 3212 | 10 | eligible |

## Проверка совместимости и происхождения

- Все 134 исходных canonical/bundle файла проверены:3 182 172 659 bytes,
  201 read operations; только разрешённый current67 TRAIN. Global VALIDATION/TEST
  не читались. Reports/MIDI и recursive source references экспорт не открывал.
- Проверены SHA/size всех 134 target artifacts; декодированы все 67 numeric JSON.
  Проверены прямоугольность arrays, типы, masks/-1, диапазоны fit-vocabulary IDs,
  note indices, source bindings, manifests/config/split и frozen340-file snapshot.
- Сохранены 45 263 H,274 936 h и все исходные object/relation identities/геометрия.
  С V2 совпали 4,153,825
  прежних envelopes+mappings и 21,718
  addressed relations. Исключены только ожидаемые метаданные projection version;
  values/masks/origin/confidence/rules сравнивались.
- Сохранены все 35,746 старых положительных
  PC-меток; разбор октовых tones добавил 46,353
  положительных PC-меток. Это рост покрытия парсера, не измерение качества модели.
- С предыдущим SHA-bound audit построчно совпали все 7 949 observed составов,
  все 65 literal roots и их 12 bins. Все 7 884 аудированных неизвестных корня
  остались masked. Значения сверены поwork/H/span, не только по общему числу.
- Artifact/record integrity не является независимой музыкальной экспертизой.

## Тесты, ресурсы, ограничения

**174 synthetic tests: 172 passed,
2 skipped,0 failures/errors.** Новые 33 helper tests плюс
projection 53 (1 skip),chord 29,export 17,lazy 8 (1 skip),census 23,stream 11.
Skipped: optional full ontology dependencies и отсутствующий mido; native tensor
runtime не объявлен проверенным. Independent Standards/Spec reviews passed.
Ошибки scope/root precedence, pedal handling, completeness и typed proof были
воспроизведены регрессиями и исправлены до frozen export. Legacy не использовался,
новых dependencies/commit/push нет.

Export wall 894.463s (14.91min), worker CPU
782.058s, peak observed process-group RSS
4.734GiB при лимите 6 GiB.
Numeric verification 29.316s, semantic parity
128.887s. Output tree 200,007,865bytes,
ignored. Один completed attempt; failed/aborted 0; training/GPU/LLM API 0.

OOV: `{"harmonies:H.bass_degree": 5, "harmonies:H.bass_pc": 5, "harmonies:H.sonority.alteration.#9": 31, "harmonies:H.sonority.alteration.b5": 8, "harmonies:H.sonority.core": 1, "harmonies:H.sonority.extension.#9": 31}`. Object OOV:
`{"harmonic_process:P.function": 4}`. Эти значения не добавлялись
в словарь по selection/dev; unavailable/unsupported не превращались вfalse.

| Утверждение | Первичная опора | Вердикт/ограничение |
|---|---|---|
| V3 экспорт исполним и целостен | execution,134 artifact bindings,67 numeric checks | Supported: инженерная проверка |
| Старые семантические цели/геометрия сохранены | V2 registry-bound manifest + pairwise parity | Supported с явным version-metadata исключением |
| Literal roots извлечены в нужную область смысла |65 точных audit correspondences | Supported для этих weak labels |
| Доступны новые интервальные метки | per-field/family support и masks | Supported coverage; не качество гармонического понимания |
| Новая модель лучше анализирует музыку | Обучение/оценка не проводились | Unsupported |

Observed composition во многом детерминирована самими MIDI нотами и oracle H.
Её предсказание — вспомогательная задача, не доказательство понимания функций,
напряжения или разрешения. Разметка source/LLM weak; dev exploratory внутри
TRAIN, grammar учитывает текущий inventory, raw pretraining overlap unknown.
TEST закрыт. h degree reference convention по-прежнему не допущен автоматически.

Следующий этап — общая графовая модель и пять вариантов построенияH с едиными
задачами/масками; отдельные oracle-H и predicted-H оценки. Нужен native runtime
перед нейронными smoke/training. Сохранённые артефакты находятся в
`outputs/hiergraph_intervals_20261027/attempt001/`: summary,verification/parity,
manifest/vocabulary/heads и frozen source; большие target files вне Git.
