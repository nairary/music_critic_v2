# HIERGRAPH: deterministic projection consumer, 27 октября 2026

Статус: implementation slice T08/S1 готов к дальнейшей интеграции; native
admission, downstream family split и численный training export не выполнены.
Контракт: [output-vocabulary v0](output-vocabulary.md).

## Изменённые файлы и границы

- `src/music_critic/tasks/hiergraph_projection.py`: pure target consumer,
  component parsers, field manifest, frozen fit vocabulary и CPU encoding.
- `src/music_critic/tasks/harmonic_projection_primitives.py`: вынесенные без
  изменения DEGREES, common spelling grammar и common quality intervals.
- `src/music_critic/tasks/dense_harmonic_targets.py` и `dilemmadata_common.py`:
  используют те же определения из общего модуля; native schemas, mapping IDs,
  шаблоны, порядки и формулы не меняются.
- `tests/tasks/test_hiergraph_projection.py`: synthetic public leaf-API tests.
- `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`, `docs/STATUS.md`,
  `campaign.md`, `output-vocabulary.md` и этот документ: состояние реализации.

Consumer вызывает существующие `validate_enrichment`, `build_vocabulary`,
`encode_enrichment`, object vocabulary/encoder и directed-decision validation
из whole-work API. `projection_encoding_specs(vocabulary)` лениво связывает
именно fitted numerical IDs с существующим `encoding.TargetEncodingSpec`.
Он не подменяет ontology order численными IDs и не включает новые головы в loss.

Нет универсального ingestion adapter, нового forward, H constructor или
запуска обучения. Пять E1–E5 должны получать один `projection_contract()` и один
замороженный vocabulary/active-fields fingerprint; constructor-specific веток
в consumer нет. Их neural integration и проверка общего manifest ещё впереди.
Legacy checkout не читался и не изменялся; V1 runtime/теоретические эвристики
не переиспользовались. Правило I/iii/vi → общая функция не добавлялось.

## Поддержанные правила v1.0.0

| Выход | Поддержано | Что остаётся недоступным |
|---|---|---|
| H.function, h.temporal_role, h.voice_role, P.process/P.function | Exact finite strings в собственных namespaces, включая все 7/13/9/12 IDs v0 | Неизвестные категории и dict.label; нет редукции в other |
| H.key.tonic_pc | Literal pitch grammar common; scoped `E-flat`/`E-sharp` и Unicode accidentals; explicit integer tonic_pc 0…11 | Неразобранные ключевые описания, bool/числовая строка вместо explicit PC |
| H.key.mode | Literal major/minor; перечисленные dorian/phrygian/lydian/mixolydian/locrian/blues reference → coarsened other | Unknown/missing отдельно masked; неизвестный mode не other; modal-mixture prose deferred |
| H.sonority.root_pc, H.bass_pc | Structured root/root_pitch_class, bass/bass_pitch_class и explicit analysis root/bass поля | Chord-symbol/free label parsing, numeric root без объявленного PC поля, вывод баса из нижней MIDI-ноты |
| H.sonority.quality | Common nine standard qualities и явная scoped alias table | Incomplete/extended/free prose, major_core, literal_key_collection, arbitrary labels |
| H.inversion | root/first/second/third и четыре authored aliases | Special bass descriptions; third inversion при известном triad quality masked как ambiguous |
| H.sonority.pitch_class.0…11 | Positive-only evidence из spelled pitch list либо explicit `tones_format=midi/pitch_class` | Нули из отсутствия тона, root+quality template как замена authored tones, неявный формат чисел, octave-bearing/relative-degree strings |
| h.chord_membership | Только typed bool; false — известный ответ | Integer 0, missing и unresolved не false |
| h.chord_degree/key_degree | Объявленный совместимый major-reference frame, bbb…### и 1…7/9/11/13 | Необъявленная convention, отсутствие bound H, arbitrary degree prose |
| h.*degree.compound | #9 → generic #2 + compound 9; аналогично 11/13 | Отсутствие compound evidence остаётся unavailable, не отрицательным |
| Directed decisions/p | Incoming prepares, outgoing resolves_to, exact source/target lists, explicit reviewed none | Wrong view/direction, self-links, contradiction none/positive; многоместные endpoints не раскладываются в пары |

`project_field()` проверяет source decision state через существующий whole-work
validator. Нормализация ступеней требует явного `degree_frames` для конкретного
уже проверенного source convention. Параметр не доказывает совместимость реального
корпуса; этот шаг не объявляет все current67 degrees автоматически admitted.

Приоритет key components: explicit tonic_pc, затем explicit tonic spelling,
затем key description; explicit mode, затем key mode. Внутри sonority explicit
PC предшествует spelling. Два доступных несовпадающих компонента дают ambiguous.
Явный unavailable компонент более высокого приоритета блокирует fallback;
conflicting evidence также маскируется. Если supported explicit PC известен,
неразобранный менее приоритетный текст не стирает этот компонент: его собственные
states/diagnostics остаются в `component_routes` и raw source. Потери менее
приоритетного spelling route не приписываются exact explicit-PC route.

Information loss зависит от реально выбранного пути: spelling → PC теряет
spelling, MIDI → PC теряет octave, explicit PC → positive presence не теряет ни
того, ни другого. Positive-unlabeled policy не является утверждением о полноте
множества. Шаблоны common quality используются здесь только для проверки
cardinality обращения; pitch-set negatives из них не создаются.

## Публичный интерфейс и сохранение evidence

`projection_field_manifest()` задаёт для каждого поля stable
`hiergraph.v0.<field>` task ID, scope/source paths, тип, frame, semantic IDs,
projection version, sentinel/mask и supervision regime. Все головы по умолчанию
неактивны. `projection_contract()` включает этот manifest и таблицы в fingerprint.

```python
# piece/bundle уже authenticated и admitted внешним loader.
projection = project_hiergraph_targets(
    piece, bundle, family_id=admitted_family_id,
    degree_frames=reviewed_source_degree_frames,
)
# Только после утверждения downstream split и active-head manifest:
vocabulary = build_projection_vocabulary(
    admitted_fit_projections,
    fit_families=approved_split_fit_families,
    split_fingerprint=approved_split_sha256,
    active_fields=sealed_active_fields,
    allow_coarsened_fields=sealed_allowed_coarsenings,
)
numeric = encode_projection(
    piece, projection, vocabulary, split_fingerprint=approved_split_sha256,
)
```

Это пример API с внешними approved bindings, не готовый split/config current67.
API требует явно переданные fit families и SHA-256 split identity, отвергает
чужую family при fitting, дубли source identity и изменение fingerprints.
Selection/dev используют тот же frozen vocabulary. Caller отвечает за реальную
семью и утверждённый split: строка SHA сама по себе не доказывает admission.
Global TRAIN не является внутренним fit; корпусные VALIDATION/TEST не поддержаны.

Projection хранит source целиком, его semantic fingerprint, family ID, конфиг
reference frames, target-only shadow bundle и per-component mappings. Исходные
IDs, views, exact intervals, scoped voice IDs, participant lists, raw spelling,
origin/confidence/evidence и author receipts сохраняются без перезаписи.
Numeric result сохраняет raw source, mappings/deferred и addressed relations
наряду с существующими event/object arrays. Это target/evidence output;
никаких raw graph/model input features consumer не строит и не меняет.

Projected envelopes имеют `origin=source_derived`; upstream native/source/LLM
origin остаётся в source. Single-source confidence переносится, включая 0;
при нескольких source fields остаётся null, без придуманной калибровки.
Source-native, model-authored и deterministic derived targets не объявляются gold.

Vocabulary хранит active fields и явно разрешённые coarsenings. Без разрешения
coarsened field не попадает в loss vocabulary; соответствующие исключения
считаются отдельно. `fit_support` сохраняет все ontology classes активного поля,
count, dense active_id либо null и unseen_in_fit. Новая selection категория
даёт OOV/-1/false и не расширяет порядок; новые event field names тоже считаются,
хотя underlying event encoder сам их опускает. Object encoder уже делает это.

Numeric coverage публикует exact/coarsened/unsupported/ambiguous/invalid/missing/
masked для всех projected полей, отдельно от OOV и coarsening exclusions.
Остальные source paths, Roman, voice identity и другие object types сохраняются
как deferred. P namespaces проверены, но включение P/p loss в первый H pilot
этим кодом не утверждено; p остаётся отдельным addressed evidence output.

## Проверки и review

Рабочая копия — isolated harmonic-event-pilot; сравнение с HEAD
`287bf856643f9e555352e1661a7bf0563772178a` не означает весь WIP этой задачи.
Для двух существующих файлов extraction дополнительно сравнивается с
снимками на начало задачи; остальные накопленные изменения исключены из scope.
Index пуст, stage/commit/push не выполнялись. Shared extraction дополнительно
сверён с task-start expressions: 238 spelling cases, 49 degrees и 10 quality
templates совпадают; пять Python source/test файлов компилируются без ошибок.

Локальный Python:
`/Users/aavalsamakin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`.
С `PYTHONPATH=src`:

- `-m unittest discover -s tests/tasks -p test_hiergraph_projection.py -q`:
  41 tests, 40 passed, 1 skipped.
- `-m unittest discover -s tests -p test_hiergraph_supervision.py -q`:
  23 passed, 0 skipped.
- `-m unittest discover -s tests/experiments -p test_supervision_stream.py -q`:
  11 passed, 0 skipped.

Red/green: первый новый тест падал из-за отсутствующего consumer module, затем
прошёл. Остальные контракты добавлялись поверх этого интерфейса; ошибки canonical
fixture были исправлены до использования результатов. Независимые Spec и
Standards passes нашли fixes для explicit masked priority, object
namespace collision, route-specific tone loss, inversion cardinality и nested
ambiguous component propagation. Каждый случай имеет regression test.
Финальные независимые verdicts: **Spec pass; Standards — no actionable findings**.
Оба reviewer повторно проверили финальный nested-conflict delta; новый suite
независимо воспроизведён как 40 passed / 1 skipped.

Идентичности рассмотренного исходного кода (SHA-256):

| Файл | SHA-256 |
|---|---|
| `src/music_critic/tasks/hiergraph_projection.py` | `62c4a36a4087b77c57e22822810a12795da9cacc108ad712489381897a0a00e2` |
| `src/music_critic/tasks/harmonic_projection_primitives.py` | `5b21ce612ff364441bf66681cdaad00ab29cb6b7ad9136f5cf5d5d6ff2abd8e9` |
| `src/music_critic/tasks/dense_harmonic_targets.py` | `d571715450d2ef1f879110287ede312921a756595eedf82ff85ab026d56a190f` |
| `src/music_critic/tasks/dilemmadata_common.py` | `2136377ca907bee66f17d865e99d6274faa0c803ee39c6fc87a85d2a91ec8a5d` |
| `tests/tasks/test_hiergraph_projection.py` | `2e164925582426d42abad7532c0403b49784ce5133fb12bcb165469b6cd0423d` |


В Python отсутствуют существующие pytest/Torch/PyG зависимости. Тесты исполняют
реальные pure leaf modules через явный test-only package bootstrap, обходящий
eager `tasks/__init__`; fake Torch или подмена collaborators не применяются.
Один regular `TargetEncodingSpec` integration test поэтому skipped; обычный import
полного tasks facade, tensor/native admission и старый pytest suite здесь не
валидированы. Новых dependencies не добавлено. Это engineering evidence, не
проверка музыкальной правильности, качества модели или native производительности.

Registry check: 453 records valid; существующий ledger `render --check` valid.
Новые experiment records не создавались, ledger не переписывался: выполнялись
только static/retained-inventory inspection и synthetic tests. AUDIT-002 остаётся
completed; его terminal record и inventory не изменены. Retained inventory
повторно сверён: 382611 bytes, SHA-256
`a680498309fd642052ceb9dbce3b92b23f768cf474393d28180a2cc36e1d7e25`.

## Что остаётся

Сначала утвердить downstream family fit/selection/dev split и source-compatible
conventions/active-head manifest. Затем planned record до чтения payloads и один
ограниченный preparation/export pass current67 global TRAIN с fresh ignored root,
post-projection masks/support/coverage и registry finalization. Не подменять этот
шаг vocabulary по всему global TRAIN. Этот этап не запускался и не считается
закрытым. Regular dependency/native integration также остаётся отдельной проверкой.
Новая модель, пять H constructors, matching/loss и training — следующие slices.
