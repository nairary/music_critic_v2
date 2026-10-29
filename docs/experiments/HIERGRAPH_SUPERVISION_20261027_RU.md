# Паспорт супервизии HIERGRAPH: опубликованные current67

Дата: 27 октября 2026. Кампания [HIERGRAPH-20261027](../workstreams/2026-10-27/hierarchical-graph-analyzer/campaign.md), этап S1.
Основной аудит: [EXP-HIERGRAPH-SUPERVISION-AUDIT-002](records/EXP-HIERGRAPH-SUPERVISION-AUDIT-002.json).
Предшествующая неполная попытка сохранена как [AUDIT-001](records/EXP-HIERGRAPH-SUPERVISION-AUDIT-001.json).

## Вывод для реализации

Данные поддерживают исследовательский прототип с общими состояниями n/h/H/P,
задачами свойств H, функций F, ролей f, типов P и адресных p. В прежнем H/member
использовалась лишь часть доступных целей. Особенно полезны совместные метки:
81 291 h с membership и функцией родительского H; 634 P с известным типом и
валидными h-участниками; 930 P с известным типом, явным H и известной функцией H.
Это основание проверить обучение иерархии, а не доказательство достаточного
объёма для надёжного whole-piece анализа или превосходства архитектуры.

Перед trainer нужен контракт нормализации targets и masks. Самые существенные
пробелы: scope H по дорожкам/группам, единые адресные роли участников P, полнота
отрицательных примеров и совместимость разных описаний sonority. Границы H
записаны, но proposed raw candidate bank ещё не оценён целиком.

## 1. Что проверено

Выбран опубликованный `EXP-CORPUS400-COMPLETION-015-RELEASE003`; его current-v2
manifest физически находится в `completion015/releases/release004/`.
Это 67 works / 67 family IDs: classical 19, jazz 7, pop 25, rock 16;
dilemmadata 19, pdmx 23, pop909_original 25. В accepted75 ещё 8 исторических v1
работ, исключённых из этого аудита. Следующий неопубликованный release не использован.

До чтения составлен flat admission. Для всех 201 файлов — canonical, bundle,
reviewed report каждого work — подтверждены SHA-256 и размер, всего 3 184 283 342
байта. Проверены piece/dataset/canonical bindings, current TRAIN, отчётные views,
числа объектов и field states. Все 31 поля опубликованной coverage-сводки
совпали с повторным подсчётом; report row/state mismatches = 0.

Обнаружено 174 123 raw notes, 45 263 H, 274 936 h, 2 335 P и 21 718 записей
отношений. h — контекст ноты внутри анализа, поэтому h больше, чем физических нот.
Точный runtime графов C3 и numeric vocabulary этим аудитом не валидировались.
67 работ в census не означают автоматически 67 принятых model-runtime входов.

## 2. Доступность основных целей

Known означает доступное символическое значение, а не human gold или готовый
numeric target. Проценты считаются по всем объектам соответствующего слоя.
«Работ» — число works хотя бы с одной известной меткой, не независимых примеров.

| Цель | Known / всего | Доля | Работ |
|---|---:|---:|---:|
| H: тональность | 43 647 / 45 263 | 96.4% | 65 |
| H: sonority — структурированное описание | 29 339 / 45 263 | 64.8% | 66 |
| F: функция H | 11 809 / 45 263 | 26.1% | 57 |
| H: functional_system | 42 397 / 45 263 | 93.7% | 62 |
| H: Roman | 16 745 / 45 263 | 37.0% | 56 |
| h: chord membership | 146 785 / 274 936 | 53.4% | 61 |
| h: ступень в аккорде | 142 038 / 274 936 | 51.7% | 61 |
| h: ступень в тональности | 227 383 / 274 936 | 82.7% | 62 |
| f: temporal_role | 74 587 / 274 936 | 27.1% | 67 |
| f: voice_role — melody/bass и т. п. | 136 181 / 274 936 | 49.5% | 65 |
| h: preparation, включая явное none | 30 845 / 274 936 | 11.2% | 67 |
| h: resolution, включая явное none | 32 168 / 274 936 | 11.7% | 67 |
| P: тип процесса | 1 872 / 2 335 | 80.2% | 61 |
| P: функция процесса | 1 021 / 2 335 | 43.7% | 53 |
| P: tension_role | 559 / 2 335 | 23.9% | 48 |

Необязательный `voice_role` отсутствует у 21 039 h, явно unresolved ещё у 117 716.
В исходном inventory optional fields учитываются при наличии; полные знаменатели
и число отсутствующих полей восстановлены в `analysis-summary-v2.json`.
Всего обнаружено 39 разных layer/field paths, включая восемь сверх обязательных 31.

Другие уровни также представлены: 1 871 ensemble, 2 679 phrases, 2 907 motifs,
774 form sections. Например, interactions известны у 1 539 ensemble, texture —
у 1 735, phrase function — у 2 005. Форматы interactions/transformations — списки;
это не готовые взаимоисключающие классы для одной softmax-головы.

## 3. H и h: границы, состав, контекст

У всех 45 263 H есть положительный exact интервал. Только 27 867 начал H (61.6%)
совпадают с атакой ноты. Среди опор из начала/конца ноты, beat/bar boundaries,
начала/конца пьесы поддержаны 43 358 начал и столько же концов (95.8% каждое);
обе границы поддержаны у 41 548 H (91.8%). У 3 715 H хотя бы одна граница
не совпадает с этой сеткой при exact matching. Причины несовпадений отдельно
не исследованы. Это anchor coverage, ещё не recall ограниченного банка spans:
ограничения длительности/числа кандидатов могут снизить его дальше.

У 31 из 67 views объединение H-интервалов равно длительности пьесы. Меньшая
длина объединения в остальных не доказывает ошибку: возможны паузы или иная
область анализа. Нет основания маркировать все unmatched spans отрицательными.
Публикация помечает 7 works как full_work_review_complete и 0 как
complete_for_declared_scope; полноту gold-разметки всех 67 утверждать нельзя.

274 808 h валидно привязаны к H, 128 не имеют H. Все 174 123 ноты покрыты
объединением своих контекстов; 31 894 ноты имеют контексты нескольких H.
273 848 h в точности равны пересечению note×H; ещё 960 привязанных контекстов
занимают подинтервал этого пересечения. Адаптер должен сохранить эти подинтервалы
или определить явную политику агрегации с masking конфликтов; их нельзя молча
считать идентичными полному note×H.

Ни у одного H не найдено явного `participant_scope/track_ids/part_ids/voice_ids/member_ids`
в проверенных top-level/attributes позициях. Это ограничение интерфейса разметки,
а не доказательство отсутствия scope в авторском тексте. 192 H явно имеют
`interpretation_scope=contextual_structural_reduction`. Поэтому H нельзя
автоматически отождествить с буквальным вертикальным аккордом всех дорожек.

44 works имеют несколько raw tracks, 23 — один. Source voice/staff присутствуют
у 55 582 нот в 19 works. Track, нотный голос и семантическая роль melody/bass
различаются. Данные позволяют вход по всей пьесе, но не дают готовой полной
supervision для отдельного H каждой дорожки и произвольных групп инструментов.

## 4. Что обозначают словари и как обучать свойства

F уже имеет компактный словарь: tonic 4 860 / 54 works; dominant 4 266 / 49;
predominant 1 300 / 30; linear_connection 722 / 24; subdominant 432 / 21;
coloristic 203 / 14; modal_center 26 / 2. Редкие классы требуют отдельного отчёта.
Совместно sonority+key+F известны у 10 142 H в 56 works.

`sonority` — dict во всех 29 339 known случаях: tones присутствуют всегда,
root — в 7 992, root_pitch_class — в 18 372, quality — в 7 767, label — в 21 474.
Эти числа по ключам пересекаются, не складываются. 220 проекций `dict.label`
смешивают `literal_key_collection`, `literal_pitch_collection`, `major`,
`C_major`, `observed_trumpet_sonority` и другие описания разных аспектов.
160 из 220 встречаются в одном work. Для Roman таких проекций 694, из них 498
в одном work. Необходим разбор root/quality/tones/reading с per-field masks и
отдельной онтологией чтения; 220/694 строки не являются готовыми классами аккордов.
Поле functional_system хорошо заполнено, но его style-подобные значения
(common_practice/tonal_pop_rock/tonal_jazz) не заменяют функцию F и могут отражать жанр.

Для f есть temporal_role: structural 41 536, arpeggiation 19 551, passing 5 262,
neighbor 4 261, suspension 161 в 18 works, retardation 14 в 4 works. Также есть
voice_role: melody 48 512 в 65 works, harmonic_support 39 470 / 42,
bass 30 103 / 52, countermelody 4 177 / 27, inner_voice 1 523 / 9 и другие.
В отличие от voice_role, 149 значений `analysis.voice` вроде voice_001 — локальные
ID голосов. Их следует использовать для внутрипьесной группировки/связей с
инвариантностью к перестановке ID, а не глобальной классификации «голос №1».

Membership: 134 181 true, 12 604 false, 128 151 masked. Known-negative имеются
в 39 works, positives — в 61. 90 840 меток получены правилом
`explicit_authored_pitch_lookup` из authored pitch interpretation; это не
независимая человеческая проверка. Совместно membership+F доступны у 81 291 h
в 53 works; membership+temporal_role — у 60 936 в 58 works.
Известные F, P.process, temporal_role — source/LLM enrichment; confidence для
этих known полей не задана. Не следует придумывать confidence или reliability weights.

## 5. P: процессы и их участники

| Тип P | Known примеров | Работ |
|---|---:|---:|
| progression | 687 | 51 |
| prolongation | 334 | 29 |
| cadence | 309 | 38 |
| loop | 127 | 19 |
| sequence | 105 | 13 |
| tonicization | 96 | 20 |
| pedal | 67 | 17 |
| chromatic_connection | 46 | 12 |
| vamp | 37 | 5 |
| harmonic_stasis | 33 | 12 |
| modulation | 26 | 11 |
| mixed | 5 | 2 |

Все 2 335 P имеют проверяемых участников: 1 569 используют только H, 658 — только h,
12 — H и h, 96 — raw n. Есть 143 048 member references, dangling IDs не найдены.
С учётом дополнительных harmony_ids явные H-ссылки имеются у 1 661 P.
При известных process labels: валидные explicit H имеются у 1 234 P в 32 works,
h-участники — у 634 P в 37 works, explicit H с известной F — у **930 P в 32 works**.
Последнее значение после scope/interval проверки, не ungated 932.

Эта структура поддерживает прямые H↔P и h↔P. Однако generic member_ids не назначают
участникам роли «подготовка/задержание/разрешение». У 17 P в трёх works найдены
role-like paths, в основном local_target_ids внутри fields; это инвентарь формы,
не проверенная таблица ролей. Требуется отдельный role contract. P.function и
H.function — самостоятельные цели; значение P.function часто mixed, и его нельзя
копировать каждому входящему H.

Поддержка process-классов в таблице маргинальная. Её распределение внутри 930
совместных P→H→F примеров ещё не вычислено; stratified split по этим данным пока
не задан. Новый паспорт модели должен пересчитать классы после всех joint masks.

## 6. p: адресные зависимости и отрицательные примеры

| Тип отношения | Валидных известных positive записей | Работ |
|---|---:|---:|
| answered_by | 90 | 11 |
| cadences_to | 154 | 26 |
| contrasts_with | 149 | 41 |
| doubled_by | 1 313 | 22 |
| imitated_by | 60 | 9 |
| interacts_with | 4 | 2 |
| modulates_to | 7 | 4 |
| prepares | 6 531 | 65 |
| repeated_as | 1 059 | 64 |
| resolves_to | 9 270 | 67 |
| supports | 1 215 | 23 |
| tonicizes | 97 | 11 |
| transformed_into | 1 623 | 51 |

Для разрешения отмечены 10 671 linked h, 21 497 явных none (36 works), остальные
242 768 h masked. Для подготовки — 7 730 linked, 23 115 none (39 works),
244 091 masked. Все none содержат reason и пустые relation_ids; противоречий с
известными положительными рёбрами того же источника/получателя и view не найдено.
Эти none применимы в указанном аналитическом контексте, не ко всем возможным парам.
Все known relation rows в данном census положительны; отсутствие записи не negative.

Linked contexts, edges и references имеют разные знаменатели: 10 676 валидных
resolution references принадлежат 10 671 linked contexts, потому что счётчик
идёт по relation_ids каждого контекста. Самих валидных resolves_to edges — 9 270.
При many-to-one одна relation может упоминаться несколькими h; нельзя смешивать
эти числа или выдавать отношение references/contexts за accuracy.

В endpoints встречаются n→n, h→h, h→n, n→h, P→H и многоместные отношения.
Source/target lists надо сохранять; декартово разворачивание не создаёт независимую
binary gold-разметку. p_binary и p_valid_positive — отдельные агрегаты, их
пересечение не измерено. 7 modulates_to в 4 works и 4 interacts_with в 2 works
недостаточны для обещаний устойчивого отдельного классификатора.

## 7. Следующий реализационный шаг

1. Зафиксировать target adapter: exact intervals/views/scope, local IDs,
   subinterval h, structured sonority/key/roman, optional missing и OOV.
2. Начать с компактных masked целей: F, key/нормализованные свойства H,
   membership/degree, temporal_role и voice_role, P.process; prepares/resolves_to
   как отдельные адресные задачи с известными none и корректными endpoints.
   Остальные поля сохраняются в формате, без требования сразу обучать 20 голов.
3. Для P учить тип/участие на допустимых сопоставленных кандидатах, отдельно
   диагностировать oracle spans. Не объявлять все возможные P отсутствующими.
   Gold участники, F/f и availability masks не поступают в обычный forward.
4. Перед модельным сравнением построить raw candidate bank, measured H/P/endpoints
   recall, class support после masks, family split и frozen vocabulary. Политики
   negative scopes и primary endpoint фиксируются до обучения/dev selection.
5. Затем реализовать общие n/h/H/P states и сравнения из campaign. Автоматического
   training launch в этом аудите нет. Данные для отдельных трековых/групповых H и
   uniform P participant roles требуют уточнения/разметки отдельным этапом.

Новый S1 census завершает инвентаризацию доступности, но ещё не закрывает весь S1:
нормализованный training export, split, OOV и bounded candidate recall остаются
следующей работой. Общая архитектура и production defaults не изменены.

## 8. Достоверность, исполнение, ограничения

| Утверждение | Статус и основание |
|---|---|
| Повторно измеренная доступность current67 | Подтверждена: 201 SHA/size, identity joins, 31 published field-state parity |
| Существуют совместные H/h/P targets | Подтверждено в указанных scope/interval masks; не все комбинации классов |
| Можно исследовать multitask hierarchy | Обоснованное направление; generalization/качество ещё не измерены |
| Все H — полные глобальные аккорды; все P роли известны | Не поддержано текущим контрактом и coverage |
| Музыкальная правильность, human preference, model superiority | Не проверялись этим аудитом |

AUDIT-001 остановлен после 66 works, wall 152.76 s: собственная проверка требовала
native canonical split=train, хотя действующий контракт допускает null при
current TRAIN authority. У Couperin native split=null, но current train и отсутствие
known heldout family conflict подтверждены exact prior admission
SHA `548750f33743cd5d437438fb52820d0f90dad5b497d7224e1100061f91aee285` и bindings.
Исходные байты не менялись. Сохранены failed record/partial artifacts; повтор — новый ID.

AUDIT-002 обработал 67/67, exit0, wall 153.78 s,
без timeout. Native train 66, native null 1 отражены раздельно; bundle.current_split
проверен train во всех 67. 20 census tests и 11 streaming tests passed, skips0;
потоковое чтение stdlib, без новых зависимостей. Независимый review сверил суммы
per-work и смысл метрик. GPU, обучение, inference, LLM API, VALIDATION/TEST не использованы.
Предобучение C3: raw overlap неизвестен; split этого audit не устраняет это ограничение.

Полные outputs остаются ignored: `outputs/hiergraph_supervision_20261027/audit001/`
и `audit002/`. Основные evidence: immutable admission, execution-plan/result,
read-log, per-work, field/class inventory, analysis-summary-v2, source snapshots,
null-split-train-authority и test logs. Compact hashes и claim boundaries — в records.
Original release checkout и legacy не менялись; legacy logic не использовалась.
Commit/push не выполнялись.

## Приложение: обязательные поля

| Поле | Known / всего | Доля | Работ |
|---|---:|---:|---:|
| ensemble/analysis.interactions | 1 539 / 1 871 | 82.3% | 67 |
| ensemble/analysis.parts_voices | 1 523 / 1 871 | 81.4% | 65 |
| ensemble/analysis.tension_role | 406 / 1 871 | 21.7% | 42 |
| ensemble/analysis.texture | 1 735 / 1 871 | 92.7% | 67 |
| form/analysis.repetition_contrast | 774 / 774 | 100.0% | 67 |
| form/analysis.section | 767 / 774 | 99.1% | 67 |
| form/analysis.tension_role | 243 / 774 | 31.4% | 37 |
| harmonic_processes/analysis.function | 1 021 / 2 335 | 43.7% | 53 |
| harmonic_processes/analysis.process | 1 872 / 2 335 | 80.2% | 61 |
| harmonic_processes/analysis.tension_role | 559 / 2 335 | 23.9% | 48 |
| harmony/analysis.extensions | 12 295 / 45 263 | 27.2% | 54 |
| harmony/analysis.function | 11 809 / 45 263 | 26.1% | 57 |
| harmony/analysis.inversion | 11 040 / 45 263 | 24.4% | 59 |
| harmony/analysis.key | 43 647 / 45 263 | 96.4% | 65 |
| harmony/analysis.roman | 16 745 / 45 263 | 37.0% | 56 |
| harmony/analysis.sonority | 29 339 / 45 263 | 64.8% | 66 |
| harmony/analysis.tension_role | 6 301 / 45 263 | 13.9% | 54 |
| motifs/analysis.identity | 2 907 / 2 907 | 100.0% | 67 |
| motifs/analysis.tension_role | 1 519 / 2 907 | 52.3% | 38 |
| motifs/analysis.transformations | 2 907 / 2 907 | 100.0% | 67 |
| notes/analysis.chord_degree | 142 038 / 274 936 | 51.7% | 61 |
| notes/analysis.chord_membership | 146 785 / 274 936 | 53.4% | 61 |
| notes/analysis.key_degree | 227 383 / 274 936 | 82.7% | 62 |
| notes/analysis.preparation | 30 845 / 274 936 | 11.2% | 67 |
| notes/analysis.resolution | 32 168 / 274 936 | 11.7% | 67 |
| notes/analysis.temporal_role | 74 587 / 274 936 | 27.1% | 67 |
| notes/analysis.tension_role | 44 640 / 274 936 | 16.2% | 67 |
| notes/analysis.voice | 125 844 / 274 936 | 45.8% | 67 |
| phrases/analysis.function | 2 005 / 2 679 | 74.8% | 61 |
| phrases/analysis.question_answer | 492 / 2 679 | 18.4% | 29 |
| phrases/analysis.tension_role | 699 / 2 679 | 26.1% | 42 |
