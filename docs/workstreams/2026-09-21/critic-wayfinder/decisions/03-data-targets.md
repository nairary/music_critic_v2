# D3 — Доступные данные, единица независимости и семантика targets

Статус: **частично установлено; source readiness имеет пробелы**.
Зависимости: нет. Владелец: агент. [Карта](../README.md).

## Источники и пригодность

| Источник | Что действительно доступно / использовано | Использование в первом маршруте | Блокер расширения |
|---|---|---|---|
| Dilemmadata corrected | 1619 records, split1295/162/162; основное supervised C0/C2/C3 | Existing TRAIN raw music для completion; VALIDATION только по новому протоколу | 1507 известных components; не считать каждую запись независимой; source-native partial targets |
| POP909-CL | 908 records,701/101/106; canonical caches, six target families; frozen C0/C2 evaluated root/quality | Позже перенос без дообучения как отдельный registered run | Channel1 target-only; старый аудит не нашёл referenced raw TRAIN MIDI в проверенных roots |
| Original POP909 | MELODY/BRIDGE/PIANO — другой view; локальная интеграция не подтверждена | Позже melody/accompaniment и track interaction | Adapter, raw availability, license/lineage; общий split с CL; BRIDGE не song-form |
| HookTheory | Melody + TheoryTab annotations; ранний mixed SSL; corrected adapter существует | Отдельная melody-conditioned completion после rebuild | Historical cache2B.1 не равен corrected2.0.0; raw rebuild/requalification; не наблюдаемый accompaniment |
| PDMX | Published large MusicXML source; adapter/cache foundation; не доказано full-scale SSL | Не блокирует первый прототип; позже обычная музыка для SSL | Release/license/filter/dedup/TRAIN manifest и streaming admission |
| Genre suite | 20 dev works/149 alternatives, provenance/player | Interface, operator и listening development | Нет независимого blind test и систематических preference labels |

Источники таблицы и hash-verified компактные аудиты перечислены в
[evidence review](../../../../research/CRITIC_WAYFINDER_EVIDENCE_20260921_RU.md).
Ранние 719 records не смешиваются с corrected1619. Source counts, note slots,
annotation counts и независимые groups — разные знаменатели.

## Существенное новое ограничение: DLC pitch_class_set

В `multitask_contract.py:1337,1387,1444` source `chord_tones` сериализуется и
декодируется буквально; `transposition.py:630` затем трактует элементы как PC.
Повторный аудит REAUDIT-002 установил fifth-coordinate semantics и missing
local-key conversion: 40/42 accepted groups в 10 TRAIN pieces отличаются от
absolute-PC set. Main, worktree146b и C3-pair имеют одинаковый SHA этого файла.
Это не corpus-wide error-rate и не измерение вреда backbone; primary8 не
включает эту auxiliary-голову, но representation обучалось совместно.

Следствия:

- Новое `raw_sounding_pcset` выводится из наблюдаемых MIDI pitches, имеет свой
  semantic ID и **не использует** исторический categorical vocabulary этой головы.
- Для нового обучения source-theory PC нужны versioned decoder/targets,
  transposition/classes/masks/cache manifests и отдельный run. Старые runs
  сохраняются неизменными. Исправление не реализуется этой картой.
- Frozen backbone разрешён как проверяемая гипотеза с известной provenance
  оговоркой. Нельзя объявить его испорченным или здоровым без измерения.
- Предсказанная старая PC-голова не становится semantic ground truth для PLL
  или preference features. При необходимости исследовать её отдельно.

## Семантические границы

`roman_numeral` — simplified184-class target. Existing hybrid decoding не
дало надёжной замены direct head; reversible factorization — новая постановка.
Changes — modifiers, а не события подготовки/задержания/разрешения.
Static `chord_tone/is_root/is_bass` уже обучаются: дублирующая membership-голова
не добавляет временной разметки. AN spelling-membership и DLC tonal membership
не сводятся молча к `pitch%12 in set`; DLC added tones имеют отдельное значение.

Для note×slice targets нужны exact interval/source-row/tie/repair bindings,
half-open sounding predicate и маска неизвестных промежутков. Source rows могут
не представлять полную harmony timeline. Отсутствие label не означает ошибку,
отсутствие weak-screen match не является отрицательным expert target.

## Самая маленькая подготовка

Перед любым новым raw processing: bind существующий TRAIN manifest и known
components, сначала группировать связанные произведения, затем делать windows,
transpositions и intervention siblings. Новые human evaluation works проверять
против **всего известного pretraining provenance**, не только preference split;
неизвестное cross-corpus родство сохранять как unknown. Не читать TEST payloads
для dedup; использовать разрешённые membership/lineage metadata.

1. Offline synthetic target fixtures: rest, sustained note, chord, octave
   doubling, meter change, grace, empty/removed track, exact boundaries.
2. Отдельная source-readiness задача: immutable manifests и безопасные locators,
   raw availability и версии; не восстанавливать все корпуса ради первого пилота.
3. Первый completion subset — существующий authenticated Dilemma TRAIN, без
   source-theory targets на predictor path. Raw snapshots не дают preference labels.

STOP: неизвестный split, target-derived geometry, mismatched hash или неопределённый
source meaning блокирует затронутый view. Не блокировать другие подтверждённые
views. PDMX processing и восстановление POP/HookTheory получают отдельные задачи,
а не скрытую часть обучения критика. Минимальные gates — [D7](07-budget-gates.md).
