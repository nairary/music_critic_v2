# Смысл sonority.tones: проверка спорных групп — 2026-10-27

Основная запись: `EXP-HIERGRAPH-SONORITY-SCOPE-002`. Предшественник
`EXP-HIERGRAPH-SONORITY-SCOPE-001` сохранён как failed. Родители:
`EXP-HIERGRAPH-SUPERVISION-AUDIT-002`, `EXP-HIERGRAPH-FACTOR-EXPORT-001`.

## Ответ

В проверенных **7 949 записях / 9 произведениях** поле tones перечисляет
реальные одновременно активные MIDI-ноты. **Списков гаммы в этих двух группах
не обнаружено.** Прежняя догадка, что literal_key_collection означает набор
нот тональности, опровергнута: key здесь обозначает нажатую клавишу.

У 7 150 literal_key_collection записей root_pitch_class=null и
reading=root_unresolved. У 799 observed_trumpet_sonority перечислено точное
наблюдаемое созвучие; у 65 из них есть дополнительный literal root/quality
в attributes, у 734 такого чтения нет. Всего 7 884 из выбранных записей не
имеют ни прямого, ни дополнительного literal root. Тональность известна
во всех 7 949, но она не заменяет корень каждого отдельного аккорда.

Поэтому алгоритм root + tones пригоден. Проблема выбранных групп — прежде
всего отсутствие корня/структурной интерпретации, а не список нот гаммы.
Различать надо наблюдаемую вертикаль и интерпретированный аккорд; для
подтверждённого состава с корнем дополнительная LLM-разметка интервалов не нужна.

## Источники и проверка

Из прежнего authenticated current67 admission выбраны ровно 9 works с двумя
label в retained class inventory. Проверены SHA-256 и размер **27 файлов**:
9 canonical, 9 bundle, 9 reviewed reports, суммарно **964 308 434 байта**.
Canonical split=train, bundle current_split=train; piece IDs и canonical hashes
совпали. Global VALIDATION/TEST не открывались. Исходный release не изменялся.
Дополнительные author recipe файлы, бинарный MIDI, сеть и модели не читались.

Первичные основания: поля sonority.value/attributes, исходные rationale и
unknown_reason, source_event_ids и точные canonical onset/duration. Спеллинги
с октавами преобразованы в MIDI pitch, B#3→C4 и границы проверены synthetic
тестами. Сравнение сделано для каждого H на точных полуоткрытых интервалах.

- Во всех **7 949** tones точно совпали с множеством активных MIDI pitches.
- Во всех **7 949** source_event_ids точно совпали с активными note IDs.
- Внутри каждого интервала нет внутренних атак/окончаний pitched notes.
- Во всех **7 150** keyboard-записях key_pitches и key_active_note_ids
  согласованы с canonical; possible_pedal_held_note_ids пусты.
- Все 65 literal readings совпали по H ID, интервалу, root, quality,
  note IDs, pitches/PC и metadata полей extensions/inversion.

Здесь active означает canonical key-release geometry. Это не акустическое
измерение сустейна/затухания и не независимое подтверждение музыкальной функции.

## Все проверенные работы

| Работа | Спорных H | Дополнительных literal roots |
|---|---:|---:|
| work017 — Fats Waller — Honeysuckle Rose | 799 | 65 |
| work035 — Aki阿杰 — 东风志 | 838 | 0 |
| work037 — Tfboys — 青春修炼手册 | 913 | 0 |
| work038 — 云菲菲 — 殇雪 | 1722 | 0 |
| work039 — 亦心Elissa — 走江南 | 1572 | 0 |
| work040 — SHE — 他还是不懂 | 6 | 0 |
| work058 — 乌兰托娅 — 高原蓝 | 849 | 0 |
| work062 — 五月天 — 我不愿让你一个人 | 721 | 0 |
| work063 — 任贤齐 — 伤心太平洋 | 529 | 0 |

Не следует переносить эти доли на все 45 263 H или все 67 произведений:
выборка целевая, по двум именам. Оставшиеся группы в этом аудите не проверялись.

## Три конкретных примера

1. **Aki阿杰 — 东风志, work035, harmony:00001**, интервал
   109/96–261/160 четвертных долей: tones=[Bb2,F5], MIDI=[46,77].
   label=literal_key_collection; root_pitch_class=null; reading=root_unresolved.
   Тональность D minor, но из неё не следует корень этого двузвучия.
   Source attributes прямо перечисляют два key-active note IDs и key_pitches;
   unknown_reason описывает exact key-active frame и отсутствие pedal survivors.

2. **Та же работа, harmony:00016**: label=C_major,
   root_pitch_class=0, reading=complete_collection,
   tones=[C3,C4,E4,E6,G3]. После снятия октавных удвоений интервалы от C
   равны {0,4,7}. Это положительный пример маршрута root + declared tones.
   Он взят как именованный контроль из того же bundle и не входит в 7 949.

3. **Honeysuckle Rose, work017, harmony:0002**, интервал 4–9/2:
   tones=[D4,G4,Bb4,D5]. Внешнее reading остаётся root_unresolved, но
   attributes.completion012_literal_reading задаёт root PC=7, quality=minor,
   интервалы {0,3,7} и structural_root_asserted=false. Extensions=[];
   inversion=second_inversion; target_scope=instantaneous_complete_pitch_class_sonority.
   Это полное наблюдаемое G-minor созвучие в данном срезе, без утверждения
   структурной гармонической функции. Ранее добавленное чтение необходимо
   учитывать, а не пропускать из-за внешнего root_unresolved.

## Что дают 65 дополнительных чтений

| Literal quality | H |
|---|---:|
| major | 19 |
| minor | 18 |
| diminished | 14 |
| major7 | 8 |
| dominant7 | 6 |

У 51 extensions пусты, у 14 непусты. Все 65 ограничены observed-slice scope;
structural_root_asserted=false и function_asserted=false. Полный observed
набор допускает метки присутствия/отсутствия интервалов именно для этого
наблюдаемого созвучия. Это не универсальное доказательство полноты абстрактного
гармонического аккорда или намеренного omit/no.

## Решение для дальнейшего преобразователя

1. Удалить предположение «literal_key_collection = гамма» для проверенной
   source route. Сохранять её как точный key-active набор, без потери нот.
2. Для размеченного корня и состава вычислять root-relative интервалы;
   учитывать scoped literal roots из attributes наряду с прямыми полями.
3. Различать observed-созвучие и структурную гармоническую интерпретацию.
   Отсутствие структурной функции не запрещает учить наблюдаемый состав.
4. Для 7 884 root-unresolved записей сохранять известные абсолютные высоты;
   не подставлять тонику/бас как root. Алгоритмический выбор нового root будет
   отдельной производной гипотезой с provenance, а не исходной gold-меткой.
5. Не превращать активные ноты окна автоматически в semantic chord membership.
   Границы этих H описывают вертикальные срезы с постоянным активным составом;
   их качество как целевых структурных гармонических границ здесь не доказано.

Это уточнение следующей реализации. Новый target export, переоценка общего
покрытия надстроек, миграция V2 и обучение в эту задачу не входят.

## Проверки, ресурсы и ограничения

Synthetic проверки диагностических функций: **10 passed**. Независимый review пройден: проверены retained primary rows/metadata и анализ
без повторного чтения корпуса.
Extraction worker wall: 3.657 s;
основной анализ: 3.045 s.
Это время исполняемых стадий, не всего интерактивного аудита. Peak worker RSS:
1,284,079,616 bytes. GPU/training/LLM API: 0.
Полные артефакты находятся только в ignored
outputs/hiergraph_sonority_scope_20261027/attempt002/.

Попытка 001 завершилась до извлечения данных из-за проверки не того split-поля:
проверялось bundle.split, тогда как здесь записано current_split. Она сохранена
с verified read receipts; попытка 002 явно проверила canonical train и
bundle current_split=train с точными идентичностями. Дополнительно сохранён
технический сбой сборки retry record (filename identity у record-planned.json);
до исправления повторного чтения source payload не было.

Отклонение от планового количества примеров: extractor сохранил по два
именованных control H вместо одного; отчёт использует первый. Дополнительных
исходных файлов это не потребовало. Weak source/LLM interpretation остаётся
weak; совпадение pitches не является human gold или доказательством полноты
всей гармонической разметки. Raw pretraining overlap неизвестен.
