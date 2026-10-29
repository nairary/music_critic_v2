# Factorized chord targets and current67 preparation — 2026-10-27

Status: V2 and V3 exports complete. `EXP-HIERGRAPH-INTERVAL-EXPORT-001` passed
all artifact, numerical and semantic-parity checks. [V3 immutable report](../../../experiments/HIERGRAPH_INTERVAL_EXPORT_20261027_RU.md).
`EXP-HIERGRAPH-FACTOR-EXPORT-001` completed for all67 admitted works. This is the
T08/HIERGRAPH S1 converter/export slice; candidate coverage, neural implementation
and training remain later work. [Immutable experiment report](../../../experiments/HIERGRAPH_FACTOR_EXPORT_20261027_RU.md).

## Implemented interval extension V3

`HiergraphProjection@3.0.0`, `HiergraphVocabulary@3` and
`HiergraphNumericTargets@3` add 37 fields for 97 total. `ChordIntervalProjection@1`
and its grammar/scope rules participate in the projection fingerprint.

| Field group | Source and permitted negatives |
|---|---|
| `H.chord.interval_pc.0..11` | Reconciled chord root + declared component tones; negatives only with admitted, matching complete authored template |
| `H.observed.pitch_class.0..11` | Complete source list authenticated against exact canonical active-note geometry; no root required |
| `H.observed.interval_pc.0..11` | Same observed list plus its independently bound direct or literal root |
| `H.observed.root_pc` | Root of observed slice only; never copied from outer structural root, key or bass |

Every absent/unadmitted source retains masks. Relative bins collapse octave and
spelling with recorded loss; actual source text and exact named factors survive.
All12 observed bins are available only after exact source-ID, tone and interval
agreement, no internal note boundary and no extra uncertain pedal scope. Literal
attributes additionally bind H identity, participants, root/quality and matching
known same-scoped metadata. Explicit scope contradictions mask relative targets;
independently authenticated absolute composition may remain available.

The pure helper uses a reusable rational interval index, so no repeated full
note scan per H is needed. No source schema, H/h geometry, original annotations
or previous immutable exports are mutated. Old `H.sonority.pitch_class.*` stays
positive-only, now also parsing supported octave-bearing tones. Existing precise
extension fields do not become coarse bins. V3 config and worker gates reject
stale contracts before payload reads; fitted vocabulary remains fit-only.

Tests: **174 total, 172 passed, 2 optional-dependency skips**. Independent Standards
and Spec reviews passed after fixes for separate roots, partial metadata, scope
conflicts, pedal uncertainty and complete-template proof. These validate mechanics,
not musical label correctness. Exported coverage is reported separately.

Measured current67 coverage:10,309 interpreted chord-interval H (45 families),
including6,994 complete12-bin sets;11,096 complete observed absolute sets and
3,212 observed relative/root sets (10 families). Of97 fields,47 meet the minimal
fit-support gate (31new); this is not proof of adequate independent data or model
quality. All original H/h/object geometry and addressed relations were retained;
all65 audited literal roots and7,949 observed sets matched. Weak source/LLM labels
and exploratory downstream dev remain explicit. See the V3 report for true/false,
family/split support, version exclusions, OOV and resource accounting.

The sections below preserve V2 and design history. Display-name rendering,
augmentation, native tensor integration and five H constructors are not part of
this converter slice; `HIERGRAPH-ENHARMONIC-001` remains open.

## Contract and implementation (V2 history)

`HiergraphProjection@2.0.0`, `HiergraphVocabulary@2` and
`HiergraphNumericTargets@2` are opt-in new identities. Their task namespace is
`hiergraph.v1.*`. The original source schemas, annotations, previous exports,
terminal experiment records and old model checkpoints are unchanged. The earlier
[pure consumer report](projection-implementation.md) describes the V1 slice;
this document records the subsequent factorization and admission work.

`tasks/chord_structure_projection.py` owns a pure bounded grammar, scoped quality
aliases and their fingerprint. `tasks/hiergraph_projection.py` integrates the
factors with existing event/object validators and encoders. Existing common/dense pitch, degree and basic-quality semantics are reused;
numerical IDs are newly fitted through the existing event/object encoders. There is no whole-JSON category, inference from
physical pitch sets to a chord interpretation, or source-to-function heuristic.

| Factor | Semantics |
|---|---|
| root_pc | Absolute root pitch class; spelling remains in source evidence |
| core | major, minor, diminished, augmented, sus2, sus4, power |
| seventh | b7, 7 or bb7 when explicitly licensed by the supported convention |
| extension | Independent 9/b9/#9/11/#11/13/b13 indicators |
| addition | Independent 2/4/6/9/11/13/#11 indicators retaining add semantics |
| alteration | Explicit b5/#5/b9/#9/#11/b13 modifications |
| omission | Explicit absence of degrees 1/3/5/7/9/11/13 |
| bass_pc / bass_degree | Independently known bass and its spelled relative degree |
| inversion | Basic root/first/second/third only when evidence licenses it |

Unmentioned factors are unavailable, never automatic false labels. Explicit
`no9` can provide negative evidence for the natural ninth as well as a positive
omission indicator. `alt` does not expand into an invented collection of altered
tones. Extended symbols license their stated highest extension and conventional
seventh; they do not assert all inner degrees are sounded. `6` is distinct from
`13`, `add9` from `9`, and `sus4` from an added fourth/eleventh. Unknown third does
not establish a power chord. Multiple explicit alterations can coexist.

Examples of the grammar (these are semantic examples, not model results):

- `Fmaj9/G`: F root, major core, seventh 7, extension 9, bass G/degree 9;
  no invented fourth inversion or nonchord/pedal role.
- `Gm11no9`: G root, minor core, b7, extension 11, explicit omitted ninth.
- `E7(b9,no3)`: E root, major core, b7 and b9, explicit omitted third;
  no physical G-sharp participant is created.
- `C7#5/G`: natural G does not license a second inversion of the explicitly
  altered fifth. `C7#5/G#` does.

Root, quality and label sources are reconciled per component. A disagreement
masks the affected factor; final root/bass conflicts also mask their derived
bass degree and inversion. Independently authored relative bass degree needs
no absolute pitch-class labels. Directly authored inversion keeps independent
availability. Unsupported prose preserves raw evidence and a diagnostic.

The legacy `H.sonority.quality` is retained as a separate compatibility target.
Its basic dictionary is not silently widened to extended composite classes.
The 60-field ontology includes the previous scoped H/h/P fields and the new
factors; all five H constructors must share its contract and fitted vocabulary.

## Уточнение следующего этапа: интервальный состав — 2026-10-27

**Статус:** пользователь согласовал временно не различать энгармоническое
написание для новой грубой цели. Ниже — контракт направления следующей
реализации; текущие V2 код, словари, экспорт и terminal evidence не изменены.

Предлагаемая дополнительная цель — 12 независимых признаков присутствия
интервала `d = (tone_pc - root_pc) % 12` в размеченном составе аккорда.
Корень должен быть известен и непротиворечив; тональность для этой операции не
обязательна. Неизвестный корень маскирует все 12 признаков. Точный нейминг
аккорда и выбор `6/13`, `add9/9` не блокируют интервальную цель. Это отдельная
версионируемая цель, а не переименование существующих extension/omission полей.

В первом варианте bin 6 объединяет b5/#11, bin 2 — 2/9; запись ступени,
октавное расположение и функция из одного bin не восстанавливаются. Исходные
spelling, label, provenance и уже доступные точные факторы сохраняются.
Неоднозначность не должна превращаться в произвольный выбор #11 вместо b5.
Даже при известном корне не предполагаем, что его высота обязательно входит
в список: rootless состав допустим. Терцовый скелет из грубого множества —
отдельная детерминированная проекция с явными правилами неоднозначности.

Смысл и полнота списка задают допустимую цель:

| Смысл исходного списка | Допустимое использование |
|---|---|
| Полный состав аккорда в данном H/view | Наличие и отсутствие интервалов: 0/1 |
| Частичный состав аккорда | Известные присутствия; остальное masked |
| Ноты гаммы/тональности | Контекст тональности; не chord-component target |
| Все звучащие ноты окна, включая мелодию | Отдельный observed-sonority target; не автоматический состав аккорда |
| Неустановленный смысл | Сохранить источник; chord-component target masked |

Полноту проверяем в том же scope. В authoring API уже есть
`AuthoredNoteTemplate.members_complete`; при true валидатор требует равенства
members и sonority.tones. Этот факт можно использовать после связывания с
исходным H/view, но он не объявляет все sonority в корпусе полными аккордами.
Ноль означает отсутствие интервала в объявленном полном составе; это не
автоматическая метка намеренного `omit/no` и не доказательство отсутствия
соответствующей ноты во всех дорожках. Конфликтующие source interpretations
не объединяем молча. Полноту/противоречия нельзя решать по dev-метрикам.

**TODO HIERGRAPH-ENHARMONIC-001:** после первого пилота вернуть различение
написанных ступеней (b5/#11, b3/#9 и т. п.), октавного расположения и
контекстного названия; до этого не заявлять их распознавание по coarse bins.
Повод вернуться к TODO — добавление задач точных ступеней/голосоведения или
интерпретации альтераций. Исходное написание не удалять при аугментации.

Перед реализацией: определить source-specific смысл/полноту и provenance;
добавить synthetic проверки равенства энгармонических PC, rootless наборов,
совместной транспозиции корня/тонов, unknown masks и разделения list scopes.
Реальный пересчёт поддержки и новый экспорт требуют отдельного experiment ID.
Числа завершённого FACTOR-EXPORT-001 описывают его прежний маршрут проекции,
а не исчерпывающий подсчёт всех возможных надстроек по исходным tones.

## Производное название аккорда — решение 2026-10-27

Пользователь делегировал выбор соглашения о точном названии. Для следующей
реализации выбираем детерминированное отображаемое название из поддержанных
структурных факторов. Оно не является обязательной отдельной flat-class
задачей и не участвует в восстановлении интервального состава как gold-вход.
Исходное authored название сохраняется отдельно; доступные source symbol
факторы не отбрасываем. При неоднозначности coarse bins выводим корень и
интервальный набор, без произвольного выбора enharmonic степени или 6/13.
Правила renderer и его версия должны быть явными при реализации.

Значения label вроде literal_key_collection и observed_trumpet_sonority —
основание проверить смысл исходных списков, но сами по себе не доказывают
ошибки и не разрешают массово исключать записи. Для подтверждённого состава
аккорда вычисление интервалов прямое. Проверка scope определяет, можно ли
называть результат составом аккорда; проверка полноты определяет допустимость
отрицательных меток. Ни одна проверка не требует угадывать красивое название.
Этот шаг фиксирует решение; renderer и новая проекция ещё не реализованы.

## Проверенный смысл двух групп sonority — аудит 2026-10-27

[EXP-HIERGRAPH-SONORITY-SCOPE-002](../../../experiments/HIERGRAPH_SONORITY_SCOPE_20261027_RU.md)
завершён: проверены все 7 949 H двух групп в девяти произведениях. Их tones и
source_event_ids точно совпали с активными canonical MIDI нотами на точных
полуоткрытых интервалах. **literal_key_collection означает нажатые клавиши,
а не гамму** в проверенных 7 150 строках. Предыдущее предположение о наборе
нот тональности для этой source route опровергнуто.

У этих 7 150 нет размеченного корня. Из 799 observed_trumpet_sonority у 65
есть дополнительное completion012_literal_reading с literal root/quality и
полным observed-slice scope; у остальных 734 корня нет. Наружный
root_unresolved не должен скрывать валидное дополнительное чтение. Все 65
явно сохраняют structural_root_asserted=false; они не размечают функцию.

Следующая проекция использует root+tones там, где известен совместимый корень,
сохраняя различие literal observed и structural scopes. Для 7 884 выбранных
root-unresolved H доступен абсолютный наблюдаемый состав; тонику/бас не
подставляем вместо root. Это не причина выбрасывать ноты/события. Zero в полном
observed наборе не означает semantic nonmembership или намеренный omit/no.
Списки других source routes этим аудитом не проверены. Текущий V2 export и
его исторические метрики не менялись; расширение конвертера остаётся следующим
этапом. Старый гипотетический пример с гаммой остаётся только общим граничным
случаем, не описанием этих записей.

## Harmonic time and note trajectories

This preparation preserves the original H IDs, exact intervals, analytical
views, note IDs, h contexts, object members and relation endpoints. It neither
merges H events nor creates a new H for a transient note. A held MIDI note may
retain multiple h contexts under successive harmonies.

A chord factor describes an interpretation, not a physical realization. Existing
harmony_id/note_id links and directed evidence remain available so later models
can jointly learn chord structure, note roles and voice-leading processes. The
converter does not equate `sus` with a temporal suspension, infer semantic voices
from tracks, or manufacture a P/resolves_to label from an extension. A new
extension-to-note edge is not asserted when the source does not identify it.

Author builder metadata calls note degrees an author-declared pitch map; it does
not prove a shared major-reference convention. Therefore this export uses
`degree_frames={}`. Raw note degree values and source identities are retained,
but projected h.chord_degree/h.key_degree and compound losses stay masked. This
restriction does not prevent parsing explicitly supported chord-symbol degrees.
A later source-specific admission may recover compatible note-degree targets.

## Export and provenance

`experiments/hiergraph_export.py` and `scripts/prepare_hiergraph_targets.py` provide
registered, CPU-only, exclusive-output preparation. `tasks.__init__` now resolves
its 184 existing exports lazily, permitting normal imports of pure annotation
code without Torch/PyG. Heavy APIs retain their real dependency errors. No
packages were installed; tensor/model integration is still a separate gate.

Input is the authenticated AUDIT-002 current67 admission. Payload access is restricted to134 distinct files: 67 canonical and
67 bundles; metadata is also read and canonical files are rechecked during encoding; no report/MIDI/source pointers are followed. Each file
is SHA/size checked before JSON decoding, identity and exact geometry are
validated, and only current TRAIN is accepted. The sole null native split uses
its previously authenticated work066 current-TRAIN authority. Duplicate piece
identities or canonical hashes across work aliases are rejected before reads.

Each work runs in a bounded subprocess. A projection cache and numerical output
are deterministic gzip JSON, with typed values, independent masks, OOV counts,
coarsening exclusions, route diagnostics and source origins/confidences. Large
raw source data remain in their original SHA/size-bound files rather than being
copied into every target. The compact reference includes source semantic identity
and exact input bindings; consumers must retain access to those evidence files.

The split preserves prior H/member and relation-probe family assignments:
**44 fit / 10 selection / 13 dev**. All are inside global TRAIN. Existing dev
families are not reassigned to fit. The newly admitted family absent both prior
manifests goes to fit. These are exploratory partitions, not untouched TEST;
raw C3 pretraining overlap remains unknown.

Vocabulary construction streams fit projections only. All fields are exported;
`vocabulary.active_fields` is the numerical export selection, not permission to
activate every loss. `heads.json` separately reports eligibility based on
post-coarsening fitted support (at least three fit families and two fitted
classes). Positive-only pitch sets and P heads stay deferred for the initial H
slice. Rare/unseen classes are reported rather than merged. Selection/dev never
expand vocabulary. This eligibility gate is an engineering default, not proof of
adequate sample size or final model-loss configuration.

## Registered execution

- Record: `EXP-HIERGRAPH-FACTOR-EXPORT-001`; parent `EXP-HIERGRAPH-SUPERVISION-AUDIT-002`.
- Evidence: `outputs/hiergraph_factorized_20261027/attempt001/` (ignored).
- Frozen source: source001, 338 Python source/test files, source-manifest.json.
- One CPU worker; 600 s worker, 3600 s campaign, 3660 s outer watchdog.
- RSS cap 6 GiB, per-work output cap 1 GiB, total output cap 12 GiB.
- Source payload bound: 134 distinct files, 3,182,172,659 bytes. Canonical files
  are checked again during numerical encoding; read operations are counted.
- GPU, training, inference, LLM API, global VALIDATION/TEST: excluded.

## Verification and review

Synthetic checks cover parsers, false/unknown, source preservation, geometry,
component conflicts, altered-fifth inversion, explicit relative bass degree,
source references, vocabulary determinism/OOV, coarsening, split/identity leaks,
path safety, gzip integrity, immutable output, cap failures and a complete tiny
subprocess project-fit-encode flow. Tests validate mechanics, not musical truth.
Independent Spec and Standards review findings were fixed with regressions.

Regular neural/tensor API execution is not claimed: Torch/PyG and mido are not
installed in the local interpreter. Pure APIs now use ordinary package imports;
no fake dependencies or test-package bootstrap are needed.

## Result

Completed67/67 with44 fit,10 selection,13 dev. The export retains45,263 H and
274,936 h, with60 fields and16 minimally eligible heads. Core is available for
12,509 H/58 families; seventh for1,973/34. Most extended factors remain sparse or
positive-only. Absolute/relative bass have only22 available events. Eligibility
is not evidence of adequate learning support. Author-declared note degrees stay
masked pending compatible reference-frame admission.

All134 projection/numeric artifacts were SHA/size verified; all67 numeric JSON
files decoded with identity, geometry counts, vocabulary bounds and mask checks.
Synthetic checks:135 tests,133 passed,2 skipped. One preparation attempt completed,
no failed/aborted attempts. Wall663.663s, peak process-group RSS3.442GiB under6GiB.
Output tree129,188,391bytes. No model-quality or annotation-completeness claim.
The detailed report records OOV and limitations. Grammar was informed by retained
current67 inventory, so downstream dev is exploratory; global TEST remained closed.

Next phase after export review: shared graph/matcher/heads/trainer mechanics,
then a separately registered native gate and five H-constructor pilots. P
feedback, live C3, NBF-like decoding and Query2box remain later experiments.
