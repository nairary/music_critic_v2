# HIERGRAPH: выходной словарь v0 и правила проекции

> Current continuation: [factorized chord targets V2](factorized-chord-targets.md) adds
> core/seventh/modifications and source-bound export; preserves all source H/h
> geometry. Earlier V1 implementation statements below are historical.

27 октября 2026. T08/S1, спецификация первого implementation slice.
Implementation update: [pure projection consumer v1.0.0](projection-implementation.md)
реализован и проверен synthetic tests; original source schemas и terminal evidence
сохранены. Native admission, family split и numeric training export ещё не закрыты.
Описания «будущего mapping» ниже относятся к исходной спецификации; точные
реализованные parsers/aliases и ограничения перечислены в implementation note.

Основа: [кампания](campaign.md), [пять H constructors](../../../research/AUTOSCHA_HIERARCHICAL_POOLING_PILOT_20261027_RU.md),
[DATA_CONTRACT](../../../DATA_CONTRACT.md), [паспорт current67](../../../experiments/HIERGRAPH_SUPERVISION_20261027_RU.md).

## Что закреплено

Одна общая модель хранит n/h/H/P/context states и отдельные свойства F/f,
адресные рёбра p, с прямыми H↔P и h↔P. Сетка E1–E5 меняет способ построения H;
словарь, source projections, availability, matching и разрешённый контекст общие.
C3/существующие adapters и numerical contracts переиспользуются. Полная модель,
новый export и training этой спецификацией не реализованы.

Существующую разметку используем как source evidence. Её значения не переписываются
в удобные классы: над ними строится явно описанная model-facing projection.
Это не новая универсальная система ingestion или замена T07/T08 schemas.
Raw MIDI inference не требует ни одного из перечисленных targets.

## Источник и статус словаря

Рабочая копия: isolated harmonic-event-pilot, HEAD
`287bf856643f9e555352e1661a7bf0563772178a` с накопленным WIP.
Использован завершённый [AUDIT-002](../../../experiments/records/EXP-HIERGRAPH-SUPERVISION-AUDIT-002.json),
без нового corpus scan. Canonical record загружен через registry.load_record.
Для retained `outputs/hiergraph_supervision_20261027/audit002/scan001/class-inventory.json`
повторно сверены382611 bytes и SHA256
`a680498309fd642052ceb9dbce3b92b23f768cf474393d28180a2cc36e1d7e25`.
Ни terminal record, ни inventory не изменяются.

Списки ниже — наблюдавшиеся значения заданного release и принятая начальная
онтология, не универсальный исчерпывающий словарь музыкальной теории. Все значения
четырёх перечислений F/temporal_role/voice_role/P.process в inventory имеют тип str.
Это не утверждение, что source schema запрещает другие формы: validator принимает
и словари с label. Такая форма требует отдельной явной проекции.

Различаются:
- **Онтология:** смысл поля и устойчивые строковые semantic IDs.
- **Projection rules:** допустимые преобразования source values с evidence/loss states.
- **Experiment vocabulary:** плотные численные IDs, порядок, trainability и fingerprint
  конкретного fit split. Общий current TRAIN не равен внутреннему fit: selection/dev
  не добавляют классы, aliases или statistics в этот export.

Не удалять редкий класс только из-за малого количества примеров. Класс из онтологии
без fit support сохраняется как unseen-in-fit; active numerical vocabulary может
не иметь его ID. Он не сливается с other и не становится false. OOV/coverage
публикуются отдельно от in-vocabulary качества; известные, но неподдержанные цели
нельзя скрывать из общего отчёта. После просмотра dev словарь не расширяется.

## Поля первого словаря

Имена H/h/F/P ниже обозначают model-facing смысл, а не переименование source paths
или уже зарегистрированные task IDs. Stable task IDs будут привязаны в consumer
к существующим scoped registry/encoding APIs, с versioned projection identity.

| Поле выхода | Source evidence | Тип и решение v0 |
|---|---|---|
| H interval / event score | H start/end, view, отдельная полнота scope | Геометрия/score, не словарь названий аккордов; no-object не выводится из отсутствия строки |
| H.function (F) | harmony/analysis.function | Отдельная categorical голова, семь IDs ниже |
| H.key.tonic_pc / mode | harmony/analysis.key и совместимые explicit fields | Композиция компонентов с независимыми masks; существующая local-key семантика |
| H.sonority.root_pc / quality / pitch_classes | harmony/analysis.sonority | Раздельные projected поля; не класс из целого JSON или свободного label |
| H.inversion / harmonic bass | analysis.inversion и доказанные source компоненты | Basic inversion и бас различаются; special bass descriptions не дополнительные инверсии |
| h.chord_membership | notes/analysis.chord_membership | Boolean; false — известный ответ, missing — mask |
| h.chord_degree / key_degree | Одноимённые contextual поля | Раздельные spelled degrees относительно root H и tonic; compound information сохраняется |
| h.temporal_role (f) | notes/analysis.temporal_role | Отдельная categorical голова,13 IDs ниже |
| h.voice_role (f) | notes/analysis.voice_role | Отдельная categorical голова,9 IDs ниже |
| P.process | harmonic_processes/analysis.process | Отдельная categorical голова,12 IDs; обучение после первого H сравнения |
| P.function | harmonic_processes/analysis.function | Отдельный namespace и словарь; не копия H.function |
| p.type + addressed participants | T07 relations, preparation/resolution decisions | Тип и списки конкретных endpoint IDs; IDs не классы |

Первая сетка E1–E5 обязательно использует один H/F contract. Базовые h/f входят
в этот же словарь; точный active-head manifest утверждается до пилотов и одинаков
для всех пяти. Наличие поля в v0 не включает его loss автоматически. P/p и
обратные F/f/P гипотезы остаются следующими контролируемыми этапами.

### F: семь исходных значений без новых слияний

`tonic`, `dominant`, `predominant`, `subdominant`, `linear_connection`,
`coloristic`, `modal_center`.

Перенос exact по известной строке. Не отождествлять predominant и subdominant,
не объявлять их отношением родитель/ребёнок. Их разграничение отражает текущую
разметку и требует музыкальной проверки; сейчас оно не стирается. F не выводится
автоматически из root, качества созвучия или Roman degree. H разных произведений
с одной F остаются разными экземплярами.

### f: временная роль и роль голоса — разные поля

Temporal role:
`structural`, `arpeggiation`, `passing`, `neighbor`, `ornamental_repetition`,
`pedal`, `appoggiatura`, `anticipation`, `suspension`, `escape`,
`changing_tone`, `retardation`, `mixed`.

Voice role:
`melody`, `harmonic_support`, `bass`, `rhythmic_support`, `countermelody`,
`inner_voice`, `pedal_support`, `mixed`, `ornamental`.

Скалярный target каждого поля трактуется как выбранная автором категория в данном
context/view, не доказательство, что в музыке все такие свойства взаимоисключающие.
Нота может одновременно иметь membership=true, chord_degree=3 и voice_role=melody.
Нет общего softmax по всем свойствам. mixed — известное авторское значение, не
корзина неизвестного. structural не равняется chord_member=true или глубине
шенкеровской редукции. analysis.voice вроде voice_001 — scoped ID для связей,
не глобальный класс голоса; track, voice и voice_role не взаимозаменяемы.

### P и p: отдельные цели следующего этапа

P.process:
`progression`, `prolongation`, `cadence`, `loop`, `sequence`, `tonicization`,
`pedal`, `chromatic_connection`, `vamp`, `harmonic_stasis`, `modulation`, `mixed`.

В текущем P.function встречаются `mixed`, `dominant`, `tonic`, `linear_connection`,
`predominant`, `coloristic`, `modal_center`, `nonfunctional`, `subdominant`.
Они сохраняются отдельно от F гармонии. P.function не назначается всем участникам.
Перекрывающиеся процессы и прямые h participants допустимы; единое дерево не требуется.

Первые planned addressed types — `prepares`, `resolves_to`, с исходным направлением
и scope/view. Остальные relation types сохраняются без обязательного включения в
первый loss. Linked/none — directed decisions с отдельными endpoint targets,
а не классы из целого dict с relation_ids/reason. Подготовка — входящее отношение,
разрешение — исходящее. Generic P member_ids не создают labels роли «разрешающий».
Многоместные source/target lists не разворачиваются в независимые binary positives.

## Составные поля: что можно нормализовать, а что ещё не готово

### Тональность

Переиспользуется существующая factorized local-key семантика: tonic_pc0..11,
mode major/minor/unknown/other и независимые availability masks. unknown в
совместимом сериализованном mode не превращается в обучаемый музыкальный ответ:
неизвестный компонент остаётся masked. other допустим лишь для известного режима
вне грубой проекции, с сохранением исходной записи и объявленной потерей деталей.

В retained inventory есть E-flat major и Eb major. Их разрешённая spelling
проекция может дать одинаковые tonic_pc=3 и mode=major; raw spelling сохраняется.
Это пример будущего deterministic mapping, не результат нового export. Значения
вроде A, B dorian, B-flat blues reference и E_centered_chromatic_tonality нельзя
обрабатывать догадкой о major/minor. Для каждого формата нужен явный parser/mapping
с независимой доступностью компонентов. Непонятная строка остаётся unsupported
для проекции, хотя исходная аннотация по-прежнему known.

### Созвучие, обращения и набор высот

Raw sonority сохраняется целиком. Structured root/root_pitch_class, quality и tones
проецируются раздельно; их взаимные противоречия не разрешаются молча. Свободные
label вроде literal_key_collection и observed_trumpet_sonority обозначают разные
виды чтения; не следует считать их обычными chord-quality classes.

Для source-native AN/DLC уже есть registry-bound root/quality/inversion/PC-set
projection. Она не является готовым parser произвольного analysis.sonority.
PC set из root+quality-template и authored tones — отдельные evidence routes:
не подменять один другим. Конкретное authored-tone mapping (MIDI integer, spelled
pitch, pitch class, root-relative degree) требует явной грамматики и reference frame.
Отсутствие высоты в неполном/неясном списке не даёт negative multi-label target.
До доказанной полноты множества использовать доступность по компонентам или
оставлять set head deferred; не присваивать всем неупомянутым pitch classes нули.

Basic inversion сохраняет существующие root/first/second/third. root_position
и first_inversion могут иметь явные aliases; ninth_in_bass, nonchord_Bb_bass,
pedal_bass не становятся четвёртым/пятым/шестым обращением. Эти raw значения
сохраняются; basic inversion для них недоступна без отдельного доказательства.
Нижняя звучащая MIDI-нота и гармонический бас не автоматически одно поле.

### Ступени и Roman

DenseHarmonicTargets уже различает chord_degree, key_degree и source_note_degree;
generic degree использует1..7 с bbb..### в major-reference системе. Применяем
эти определения там, где source reference и написание действительно совместимы.
Семантику arbitrary analysis.* строки нельзя вывести только из совпадения символов.

Compound degrees9/11/13 не стираются до2/4/6. Предлагаемая явная проекция #9,
если source convention подтверждена, сохраняет generic #2 и compound9 вместе
с исходным #9. Это ещё не реализованный mapping. При отсутствии явного compound
evidence он остаётся unavailable, не отрицательным. Аналогично нельзя превращать
b3 в3 только потому, что текущая тональность минорная.

Roman сохраняется losslessly и пока deferred как самостоятельная head. Следующий
parser должен сохранять main degree, alteration, quality/inversion и secondary
target в поддержанной грамматике. V/V не равно V; необычные labels не становятся
новыми одноразовыми классами. Его implement/admission — отдельный slice.

## Состояния и правила потерь

Не смешивать source decision state, projection state и experiment support:

| Случай | Что сохраняем | Участие в loss |
|---|---|---|
| Known value и exact projection | Исходное значение, provenance, projected value | Да, если поле/класс допущены fit manifest |
| Coarsened projection | Именованное правило и information_loss | Только если эта проекция заранее разрешена |
| Unreviewed / unresolved / conflicting / missing | Причину и исходное состояние | Mask; не отрицательный класс |
| Not applicable | Явную неприменимость поля | Не semantic class первого пилота; applicability отдельно |
| Known false / directed none | Отрицательный факт в точном scope | Только в соответствующей бинарной/адресной задаче |
| Unsupported / ambiguous / invalid mapping | Исходную known аннотацию и diagnostic | Проецированное поле masked; malformed identity/geometry fail validation |
| Unseen-in-fit / OOV | Известный semantic/raw value и отсутствие active ID | Не дополняет vocabulary, loss masked, coverage отдельно |

Никакого автоматического other для редких, неизвестных или непонятных значений.
Typed JSON сохраняет различие false/0/строк/списков; их нельзя сливать строковым
кастом. Unknown confidence не равна нулю. Derived pitch lookup остаётся слабой
производной аннотацией, не независимым human gold. Маски, source labels и confidence
поступают в targets/loss/evaluation, но не в raw graph/обычный forward.

## Переиспользование существующего кода

- [Event vocabulary/encoding](../../../../src/music_critic/tasks/event_theory_enrichment.py):
  scoped deterministic typed values, masks, OOV, provenance и numeric IDs.
- [Object vocabulary/encoding](../../../../src/music_critic/tasks/whole_work_annotation.py):
  independent object_type namespaces, member IDs, states и missing/OOV.
- [Dense contextual semantics](../../../../src/music_critic/tasks/dense_harmonic_targets.py):
  spelled degrees, predicates, compound evidence и reference frames.
- [Common projections](../../../../src/music_critic/tasks/dilemmadata_common.py):
  existing key/root/quality/inversion semantics и mapping states.
- [Open/closed encoding](../../../../src/music_critic/tasks/encoding.py):
  deferred open strings; не создавать ID в batch/worker.

Existing event builder замораживает целый canonical JSON как class; новые поля
требуют узкого projection consumer до этой стадии. Builder проверяет current TRAIN,
но не внутренний fit: caller обязан передать только admitted fit families.
Current event encoder не считает полностью новые field names, отсутствующие в
vocabulary; consumer должен явно учитывать их как deferred/unseen, не терять молча.
Object encoder отдельно сохраняет новые поля/types. TRAIN-only APIs не объявляются
готовым режимом corpus VALIDATION/TEST; наша первая selection/dev находится внутри
global TRAIN с отдельным family admission.

## Критерии завершения ближайшего slice

1. У каждого active field есть source path, тип, reference frame, semantic IDs,
   projection version, masks/OOV и примеры exact/coarsened/unsupported.
2. На synthetic fixtures: разные raw descriptions с одной разрешённой проекцией
   дают одинаковый target при сохранении provenance; контекстно разные смыслы не
   объединяются; неизвестное никогда не становится false или ordinary other.
3. Независимые tests: spelling/PC loss, mode mask, #9/compound сохранение,
   false против0, field-local mixed, P.function против H.function,
   scoped voice IDs, many-to-many endpoints и смена h у held note.
4. Fit-only vocabulary deterministic и self-fingerprinted; известный новый класс
   в selection не меняет class order. Вклад OOV и неподдержанных mappings виден.
5. Новый зарегистрированный preparation run даёт support по fit/selection/dev
   families после projections/masks, conflicts и loss coverage. Порог исключения
   полей/классов нельзя придумывать после просмотра dev качества.
6. Все E1–E5 читают один output contract и active-head manifest. Implementation,
   native admission и результаты обучения закрываются раздельно.

Исходная документационная задача закрыла структуру словаря, начальные finite IDs
и правила недоступности. Следующий implementation slice уже добавил bounded
aliases/parsers и pure consumer; его точный scope и проверки описаны в
[implementation note](projection-implementation.md). Остаются native admission,
утверждённые family split/source conventions и проверенный numeric training export.
Миграция источников, обучение, TEST, новые LLM labels и бюджет GPU не добавлены.
