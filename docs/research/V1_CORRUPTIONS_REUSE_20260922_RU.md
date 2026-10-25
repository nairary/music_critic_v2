# Искажения legacy V1: что применимо к текущему критику

Дата: 22 сентября 2026. Вопрос: какие механизмы старого Teacher/Observer
расширяют нынешние H1/H2/R1/R2/M1/M2/S1 и пригодны для независимой проверки?
Это исследование исходников, не эксперимент и не принятый план нового обучения.

## Источники и границы

Legacy V1: `nairary/Fine-tune-text2midi-llm-with-gnn-theory-critic`, ветка
`sections`, commit `2d8281f31cc9ad9c8fecaf332da0c61e0e949415`. Указанного
`/home/str/...` checkout на этой машине нет. Через GitHub API прочитаны 53 файла,
каждый проверен по Git blob SHA; копии доступны только для чтения в
`outputs/legacy_corruptions_review_20260922/source/`, SHA256/размеры — в
`source-manifest.json`. Код не импортировался и не исполнялся.

Это версия tracked source. `docs/legacy_snapshot.json` фиксирует также старые
untracked launch scripts и runtime archive, которых в GitHub snapshot нет.
Поэтому нельзя утверждать, что конкретный старый checkpoint обучался на всех
перечисленных режимах. Composed configs и usage logs конкретного запуска здесь
не проверены. V2 HEAD `287bf856643f9e555352e1661a7bf0563772178a`, dirty checkout;
текущие генераторы и контракты проверены отдельно от исторического HEAD.

Основные ссылки на pinned source:

- [реестр 32 операторов](https://github.com/nairary/Fine-tune-text2midi-llm-with-gnn-theory-critic/blob/2d8281f31cc9ad9c8fecaf332da0c61e0e949415/src/dataloader/song_corruptions.py#L2830);
- [full config: 17 режимов](https://github.com/nairary/Fine-tune-text2midi-llm-with-gnn-theory-critic/blob/2d8281f31cc9ad9c8fecaf332da0c61e0e949415/configs/dataloader/theory_aware_full.yaml#L20);
- [старый graph backend](https://github.com/nairary/Fine-tune-text2midi-llm-with-gnn-theory-critic/blob/2d8281f31cc9ad9c8fecaf332da0c61e0e949415/src/dataloader/utils_graph.py#L639);
- [OOD evaluator](https://github.com/nairary/Fine-tune-text2midi-llm-with-gnn-theory-critic/blob/2d8281f31cc9ad9c8fecaf332da0c61e0e949415/scripts/eval_teacher_ood_corruptions.py#L34).

## Что действительно реализовано

32 song-object режима распределяются так. Названия — идентификаторы исходников,
описания ниже — поведение кода, а не доказанные музыкальные дефекты.

| Семейство | Реализованные режимы | Смысл и ограничения |
|---|---|---|
| Форма: 5 | `adjacent_section_swap`, `non_adjacent_section_swap`, `section_duplicate`, `section_drop_keep_silence`, `section_drop_and_close_gap` | Перестановка соседних/несоседних разделов, повтор, удаление с тишиной или со сдвигом продолжения. Используются готовые section spans и перестраивается timeline. |
| Границы разделов: 2 | `section_entry_non_tonic_substitution`, `section_exit_non_dominant_substitution` | Замена корня первого тонического / последнего доминантового аккорда раздела. Зависит от source sections и упрощённого root-based функционального правила. |
| Мелодия и гармонический контекст: 7 | `strongbeat_nonchord_note`, `borrowed_melody_conflict`, `borrowed_kind_toggle_without_melody_change`, `melody_semitone_add_clash`, `melody_suspension_clash`, `melody_alteration_clash`, `melody_omit_core_tone_conflict` | Локальное рассогласование мелодии с аккордом/заимствованным ладом либо изменение add/sus/alter/omit компонентов аккорда относительно сохранённой мелодии. |
| Бас: 1 | `inversion_bass_continuity_conflict` | Выбор обращения с большей заданной эвристикой неустойчивости/скачков подразумеваемого баса относительно соседних аккордов. Это PC-based эвристика, не измерение реального абсолютного голосоведения. |
| Время и пропуски: 7 | `note_onset_shift`, `chord_onset_shift`, `strong_weak_beat_flip`, `drop_note_from_onset`, `drop_chord_from_onset`, `duration_stretch_shrink_note`, `duration_stretch_shrink_chord` | Сдвиги атак отдельной ноты/аккорда, перенос между сильной/слабой позицией, превращение события в паузу, изменение длительности. Сильная доля/совпадение с аккордом часто влияют на выбор места. |
| Последовательность функций: 1 | `functional_progression_violation_strict` | Замена среднего корня в ограниченных шаблонах T–PD–D, PD–D–T, T–D–T, T–PD–T. Меняется `root_id`, не реальные голоса сопровождения. |
| Дополнительные локальные высотные: 4 | `out_of_key_note`, `local_semitone_fragment_shift`, `octave_leap_violation`, `semitone_from_bass_or_chord_tone` | Замена высоты вне принятого лада, сдвиг 2–4 нот на ±полутон, изменение октавы отдельной ноты, полутоновый конфликт с подразумеваемым басом/верхним тоном аккорда. |
| Названные benign: 5 | `transpose_with_tonic_shift`, `merge_repeated_melody_notes`, `split_long_melody_note`, `melody_octave_shift`, `drop_tonic_seventh_on_strong_beat` | Глобальная транспозиция; объединение повторных нот; деление длинной; октавный перенос мелодии; упрощение тонического расширенного аккорда. Безопасность последних четырёх зависит от контекста/аспекта. |

`applied_resolution_violation` зарегистрирован отдельно как **placeholder**:
готовой реализации нет (`song_corruptions.py:2865,2887`).

Кроме этого, более старый graph backend содержит пять режимов:
`note_sd_replacement`, `chord_root_replacement`, `chord_type_replacement`,
`swap_neighboring_chords`, `onset_mismatch`. Они меняют ID в тензорах, строки
аккордовых признаков или onset-edge и отдельные временные поля. Это не полноценные
преобразования MIDI с повторным построением согласованного графа. Для V2 можно
обсуждать музыкальную идею замены/перестановки, но не переносить graph mutation.

**Активный full config содержит 17, а не 32 режима:** семь section-операторов,
семь локальных melody/harmony режимов, bass-continuity и два onset-shift.
TRAIN/VAL создаются фильтрами одного HookTheoryDataset с одним набором modes
(`src/training/train_teacher.py:416–426`). По этому маршруту семейства между
TRAIN/VAL не разделены. Доступность кода не доказывает экспозицию checkpoint.

## Что взять в первую очередь

| Приоритет | Адаптируемая идея | Что она добавляет к нынешней задаче | Условие переноса |
|---|---|---|---|
| 1 | `duration_stretch_shrink_note`, затем длительность согласованной группы сопровождения | В текущих семи операторах нет изменения длительности. Проверяет протяжённость звучания, перекрытия и паузы при сохранённых атаках/высотах. | Exact rational duration, ограничение диапазона, явная политика overlap/ties, несколько масштабов правки. Не всякое удлинение хуже. |
| 1 | `inversion_bass_continuity_conflict` как изменение реальных нот баса/расположения | Более локальная и контекстная задача, чем перенос баса на тритон: аккорд может сохранить pitch-class set, а линия баса измениться. | Проверенные партии/согласованные группы нот; абсолютные интервалы с соседями; не считать любое обращение ошибкой. Это существенная новая адаптация, не копирование `inversion_id`. |
| 1 | `note_onset_shift`, `chord_onset_shift` | Точечное нарушение согласования атак вместо фиксированного сдвига всей партии на половину четверти. | Raw metric grid, переменные позиции/масштабы, одно событие или явно связанная группа. Механизм родственен R1: это ближний перенос, а не сильное доказательство OOD. |
| 2 | `local_semitone_fragment_shift`, `strongbeat_nonchord_note`, `semitone_from_bass_or_chord_tone` | Небольшая контекстная правка вместо порчи всей партии; подходит для проверки локального ответа критика. | Менять абсолютные MIDI pitches; теория/партии только на стороне генератора с provenance. Родственно существующим pitch-shift негативам, не объявлять независимым семейством по одному новому имени. |
| 2 | `melody_suspension_clash`, `melody_alteration_clash`, `melody_omit_core_tone_conflict`, `melody_semitone_add_clash` | Локальное изменение состава созвучия при сохранении мелодии; полезны для harmony, если реально меняются звучащие ноты. | Нужен materializer аккордовых правок в наблюдаемые ноты, корректные роли/voicing. Добавления требуют нового patch-контракта. Нельзя изменять один theory label и называть это raw corruption. |
| 2 | `functional_progression_violation_strict`, boundary substitutions | Проверка гармонического продолжения/стыка фрагментов вместо одной диссонантной вертикали. | Source-faithful условие функции/границ; реальная перестройка нот; контекстная проверка человеком. V1 root-only правила — кандидаты, не универсальные нормы. |
| Позже | Non-adjacent swap, duplicate, drop/close-gap | Повторы, пропуски и дальние связи. | Нынешнее окно ≤16 тактов не доказывает качество полной формы. В raw режиме можно работать с явно метрическими блоками; нельзя называть их gold sections. Insert/delete-time требуют новых timeline/patch контрактов. |

`drop_note_from_onset` полезен как локальный вариант удаления с учётом контекста,
но сам механизм удаления уже есть в R2. `octave_leap_violation` близок к M1;
ещё один большой октавный скачок мало расширяет семейство. Глобальная порча,
максимальный скачок или всегда самое заметное место не должны быть единственными
вариантами: нужны контролируемые масштабы и выбор места, с записью применимости.

## Допустимые изменения нужны отдельным контролем

Наиболее чистый кандидат — общая транспозиция всех pitched notes в допустимом
регистре, с согласованными metadata. Начать с проверки стабильности harmony и
rhythm score, не с требования точного равенства всех аспектов и звучания.

Названия `STRICT_BENIGN_CORRUPTIONS` в V1 нельзя принять без проверки:
merge/split удаляет/добавляет атаки и меняет артикуляцию; melody octave меняет
регистр, пересечение голосов и фактуру; удаление септимы меняет гармоническую
окраску. Это кандидаты для контекстно размеченных пар, а не безусловные ties.

В проверенном V1 Teacher loss original всегда выше corrupted, с binary targets
1/0 (`src/training/teacher_losses.py:152–189,409–419`); benign/applied mask туда
не передаётся. Добавить benign в общий TRAIN corruption list означало бы
штрафовать их. Неуспешная генерация тоже может вернуть неизменённую копию;
такую пару нельзя отправлять в ranking loss как отрицательную.

В текущем V2 это тоже не готовая обучающая возможность:
`critic/targets.py:83–87` допускает только ±1 для доступного предпочтения,
а 0 означает masked unknown. Для ties/invariance нужен отдельный явный контракт
и loss либо пока только диагностическая метрика. При benign edit локальная
метка «здесь было вмешательство» остаётся истинной — она не означает ухудшение.

## Что на самом деле означает OOD в V1

В `eval_teacher_ood_corruptions.py:34–39` OOD list состоит из четырёх высотных
операторов. YAML `configs/dataloader/theory_aware_ood.yaml:20–27` содержит другой
список из семи: три из них плюс strong/weak flip, два duration и functional
progression; octave-leap отсутствует. В ablation TRAIN config octave-leap есть.

Скрипт по умолчанию выбирает `mode-set=default`, не `ood`, и `split=test`.
Он не сверяет семейства с обучающей экспозицией checkpoint. Dataset route
применяет оператор с `corruption_cfg={}` (`:240–253`); PairCorpus route фильтрует
режимы только при явном `--modes`, поэтому один `--mode-set ood` там не отбирает
OOD пары (`:703–711`). Benign rank-success остаётся original > edited вместо
метрики стабильности. Тесты evaluator содержат синтетические заданные scores,
а не результат оценки реального checkpoint. Такие результаты в прочитанной
source-выборке не предъявлены; это не утверждение, что их нигде не существует.

Полезно перенести **идею** независимых наборов и отчёт применимости:
`n_total`, `n_applied`, `applied_rate`, accuracy только по реально изменённым
парам, отдельные причины отказа. Balancer V1 учитывает successful applications
(`corruption_balancer.py:26–38`), но фиксированный V2 dev manifest предпочтительнее
изменяющейся на лету validation с общим состоянием sampler/balancer.

Для следующего V2 протокола предлагается заранее отделить:

1. Другие произведения с теми же операторами — существующий ID head-dev.
2. Новые параметры/места известных механизмов — отдельная проверка чувствительности.
3. Полностью исключённые механизмы — operator-OOD; карту семейств фиксировать
   до обучения, не делить близкие названия формально на train/test.
4. Допустимые и неоднозначные изменения — отдельные controls, с человеческими
   предпочтениями/равенством/неуверенностью там, где нет надёжного правила.

Для сохранённого checkpoint1000 разумно **сначала** адаптировать малый независимый
набор duration + реального bass/voicing + допустимой глобальной транспозиции,
зафиксировав контракты и человеческую проверку применимости. Это предложение,
не выполненная генерация или оценка. Поскольку результаты затем повлияют на
разработку, такой набор станет development OOD; для окончательного вывода нужен
отдельный ещё не использованный holdout. Официальный TEST не открывать.

## Инженерные границы и итог

`NotePatch` V2 уже выражает pitch/onset/duration/delete существующих нот
(`experiments/music_interventions.py:39`). Однако `critic/targets.py:208–219`
сохраняет duration/tracks/bars/beats/tempo/meter/key-signature context и отвергает
добавленные note IDs. Повтор раздела, split note, добавленный тон аккорда,
close-gap и изменение key metadata нельзя просто провести через текущий
VerifiedPatch: нужны отдельные проверяемые операции с lineage и masks.

Если eligibility гармонических правок берётся из DLC annotations, использовать
исправленный source-native decoder из `DATA_CONTRACT.md`, а не историческое
буквальное прочтение `chord_tones` как абсолютных pitch classes. Это сведения
генератора/проверки; они не становятся входами raw MIDI критика.

Не переносить: float/epsilon timing V1, обязательные gold chord/section inputs,
root-ID арифметику как универсальную гармонию, graph-only mutations, автоматическое
«изменено = хуже», неизменённые отрицательные пары, implicit TEST defaults.

Поддержано кодом: V1 имеет более богатый набор механизмов, часть доступна для
концептуальной адаптации; существующий OOD evaluator не обеспечивает строгий
holdout. Не установлено: какое качество эти операторы дадут V2, наличие
запоминания, человеческая предпочтительность каждой правки или превосходство V1.

Реализация/обучение/рендер/TEST не запускались. Проверки этой задачи: Git blob и
SHA256/размеры 53 файлов, AST-инвентаризация 32 registry entries и одного placeholder,
сопоставление call sites/configs, независимое чтение activation/OOD route и
проверка ссылок/whitespace новых документов. Legacy unit tests не исполнялись.
