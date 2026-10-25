# D5 — Человеческие оценки, независимые произведения и критерий прототипа

Статус: **пользователь подтвердил себя и ещё 1–2 слушателей; протокол предложен**.
Зависимость: [данные/lineage](03-data-targets.md). [Карта](../README.md).

## Три разных результата

1. Engineering: MIDI/input/masks/decoder правильны и воспроизводимы.
2. Conditional modeling: NLL/calibration и зависимость от допустимого контекста.
3. Critic: согласие с человеческими аспектными предпочтениями и локализация
   на новых works/intervention families. Первые два не доказывают третье.

20 работ/149 вариантов уже просмотренного suite — development. Автоматическое
`original > distortion` запрещено как gold. Пять classical TRAIN clips нельзя
называть novel-to-backbone. Drum stem электронной части отсутствует в MIDI:
в основной MIDI-only оценке слушатель слышит только общую воспроизводимую
MIDI-rendered информацию; drum-bed версия остаётся отдельным listening stratum
и не используется как безусловная target truth для MIDI-критика.

## Минимальная разметка и заморозка

Предлагаемые размеры — пилот и ограниченный holdout, не обещание мощности.

| Этап | Минимум | Роль / человеко-время |
|---|---|---|
| Pilot rubric | 12dev works×4 pairs=48 pairs; минимум 2 оценки/пару,6 скрытых повторов | Согласие, ties, понятность аспекта; примерно 2–4 часа на слушателя с повторами/заметками |
| Frozen-probe labels | Те же 12 works:8train/4dev-validation, split по lineage | Дешёвая feasibility оценка;4validation groups слишком мало для окончательного вывода |
| Новый evaluation cohort | Не менее 24 новых works×4 pairs=96 pairs,2 оценки/пару | Около 3–5 часов на слушателя; final model/thresholds уже заморожены |

Первый pilot можно собрать из существующего development suite, добавив
независимые human edits и нейтральные пары; не изображать его слепым тестом
для автора/пользователя, видевшего examples. Новые 24 works подбираются отдельно
после lineage проверки: broad style coverage, но вывод относится только к
представленному набору. Если предпочтителен один жанр, scope сужается явно.
Число 24 — floor для первого bounded результата; при широких CI требуется новый
заранее согласованный сбор, без последовательного добора «до значимости».

Оценщики 2 или 3: каждый pair имеет минимум 2 независимых ответа; третий получает
заранее случайно заданное пересечение и disagreements для отдельной adjudication.
При двух сохранять disagreement без принудительного консенсуса. Пользовательские
оценки и оценки других людей показывать отдельно: это не генеральная популяция.

## Схема пары

`work_group_id`, source/lineage hashes, split, candidate IDs, exact edit span,
renderer/version/gain, observed MIDI coverage, aspect, rater pseudonym, randomized
A/B order, answer{A, B, tie, uncertain/unavailable}, confidence, marked suspect spans,
optional reason. Не показывать исходность, operator name/severity, filenames и
model scores. Одно произведение и все его versions/windows остаются в одной группе.
Source IDs не подаются модели как признаки качества.

Первый вопрос слушателю: **какой вариант гармонически лучше согласован с
окружением?** Отдельно общий preference; не выводить его из гармонии. Разметки
ритма/голосоведения добавляются только при понятной инструкции и компетенции,
не как обязательные догадки. Отдельные поля tie и uncertainty предотвращают
подмену отсутствия информации ничьёй.

Пары должны включать operator edits, independently human-edited candidates,
ожидаемо benign controls и случаи «правка лучше original». Не предполагать их
результат заранее. Не менее одной семьи interventions не используется в обучении
и настройке. Human edits в dev и final имеют разные заранее разделённые
author/process families; human-edit path в целом не объявляется unseen, если он
был в dev. Финальная модель выбирается только на dev. Автор правки не даёт её
primary blind rating; при недостатке независимых оценок pair остаётся недоступной
для соответствующего claim.
При невозможности набрать независимые families соответствующий claim unavailable.

## Метрики и критерии

Primary для критика: group-macro pairwise log loss на aspect preferences с
soft tie targets. Для каждой пары берутся ровно два заранее назначенных независимых
оценщика: mean от значений A=1, B=0, tie=0.5; uncertainty не число. Если одного
ответа нет, primary pair unavailable, coverage публикуется. Пары имеют равный вес
внутри work, works — равный вес в итоговой метрике. Hidden repeats используются
только для reliability; targeted adjudication и дополнительные оценки третьего
слушателя — отдельный sensitivity report, не перевзвешивание primary. Дополнительно
decisive pair agreement, Brier/calibration,
tie coverage, unavailable fraction и per-rater consistency. Симметричный no-info
predictor, raw summaries и лучший отдельный signal — обязательные baselines.
Bootstrap кластеризовать по work/lineage, не по 149 вариантам или тысячам нот;
при 2–3 raters популяционную rater-uncertainty надёжно не оценить — показать
отдельные rater/leave-one-rater результаты, где размер данных позволяет.

Локализация: overlap human marked intervals, top-k recall и false-positive rate
на benign controls. Координата edit — только technical annotation, не доказанная
область музыкального дефекта. Выводить coverage и недоступные случаи.

Дополнительные срезы: unseen works, unseen family, human edits, style,
length/density/polyphony, track count/metadata, unusual-but-human-preferred cases.
Не подбирать score sign или веса по final cohort. Если ranking отражает
редкость, но противоречит людям, likelihood остаётся диагностикой.

Предлагаемый acceptance gate: на frozen24-work evaluation lower95% work-bootstrap
bound улучшения primary над лучшим заранее выбранным baseline >0; для practically
useful gain дополнительно заранее согласовать margin (первый ориентир +5percentage
points decisive agreement). Это **проектный порог**, не измеренный результат.
Не выбирать победителя по одной из множества вторичных метрик постфактум.
Если CI пересекает ноль — «неопределённо»; если эффект ограничен знакомым
оператором — narrow development result. Полезность для GRPO этим не проверяется.

STOP до обучения preference: rubric не воспроизводится, слишком мало decisive
labels, source leakage или отсутствует независимый cohort. Уточнить аспект и
разметку; не заменять людей ярлыком «искажено». Согласованность может быть низкой
из-за нескольких допустимых вкусов, а не плохих слушателей.
Бюджеты E4–E7: [D7](07-budget-gates.md).
