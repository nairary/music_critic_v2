# D6 — Changes и путь к остальным музыкальным аспектам

Статус: **после первого bounded critic; отдельные готовности уже установлены**.
Зависимости: [targets](03-data-targets.md), [objectives](04-objectives.md),
[human evaluation](05-human-evaluation.md). [Карта](../README.md).

Уточнение22сентября переносит rhythm/harmony/melody/local-structure в первый
основной critic по [D4](04-objectives.md); они больше не отложены целиком.
Конкретная [multiscale конструкция](../../../../research/MULTISCALE_ASPECT_CRITIC_DESIGN_20260922_RU.md)
пока предложение. Расширение ниже относится к более сильным targets/claims,
голосам, длинной форме и arrangement, а changes сохраняет отдельный pilot.

## Changes: готовые targets, ещё не установленная польза

99 TRAIN atoms,1011 TRAIN/130 VALIDATION records подготовлены и независимо
проверены. 45 atoms имеют <10 annotations,24 — один component. Loss/метрики должны
учитывать masks, annotation-weighting и common versus expanded support.
Source modifiers не формируют новые mandatory inference nodes или gold timing.

Альтернативы: оставить 18 heads при critic probe; добавить changes как вспомогательную
задачу; использовать предсказания changes как отдельно ablated critic features.
Рекомендуется первая, затем вторая только после источников/версий. Нет результата
старого 3090 протокола и нет нового доступа к 3090. Не переносить его 80kbudget в Cloud.

Самый малый следующий опыт, если будет выбран: frozen temporal embeddings,
99-atom masked multilabel probe против TRAIN prevalence control; ≥128 доступных
TRAIN records и до 32 VALIDATION, до 1000 updates, максимум 2 T4 часа. Это feasibility,
не польза backbone/critic. Допустимо сохранить 99 outputs, но метрики атомов без
subset support показывать `—`; supported-class macro сопровождается списком и
числом доступных классов. 128/32 records не обеспечивают оценку всех 99 atoms.
Для causal changes-benefit нужен **отдельный** paired
partial-finetune с/без changes, одинаковые batches/draws/seeds/PC-target regime,
до 2000 updates/arm и суммарный cap4 T4 часа. Primary8, per-atom PR/F1/calibration,
common-support theory retention и human critic endpoints раздельно. Не менять
одновременно erroneous PC decoder и добавление changes.

STOP: нарушения source/masks, few-positive overfit, потеря common-support
retention без critic gain; при неопределённости короткого бюджета не делать
универсальный вывод. Расширять только после отдельного решения и нового run ID.

## Что уже известно и чего не надо исследовать повторно

Source-scoped voice feasibility измерена в worktree146b и повторена REAUDIT-003:
usable source links36,456AN/942,732DLC; строгий неоднозначность-masked cohort
35,941/879,174. Это не production canonical importer/head. Strongest raw heuristic
на этом source cohort уже достигает 90.632%AN /73.895%DLC micro; такой baseline
нельзя заменить случайным выбором. Source/canonical cohorts и их знаменатели
различаются. Это TRAIN source-clean diagnostics, не holdout evaluation. В 003
не применялся текущий whole-piece предел 4096 нот. Нужны отдельный canonical/window
admission и подготовка разрешённых VALIDATION relation targets до pair probe.
Не повторять весь census ради очередного обзора.

Существующий static membership не равен note×time membership и не равен
consonance. Weak preparation/resolution screen не является экспертным gold.
Сохранённые реальные контрпримеры уже показывают это; нужны listener/expert
labels, а не самоподтверждение правила. Подробный evidence pointer — в
[обзоре](../../../../research/CRITIC_WAYFINDER_EVIDENCE_20260921_RU.md).

## Карта расширения с минимальными различающими gates

Все budgets ниже — предложения будущего feasibility-screen; не текущие запуски.

| Направление | Доступная постановка / альтернатива | Минимальный следующий gate | Продолжать / закрыть |
|---|---|---|---|
| Melody/voice | Pairwise восстановление source-scoped next-note среди известных raw notes; позже completion новой ноты | Сначала canonical/window/sidecar fixtures $0 и VALIDATION target gate; затем 32 TRAIN/16 VALIDATION components,≤1000 updates,≤2 T4h frozen pair probe | Превышение strongest heuristic на hard strata и сохранение masks; иначе не объявлять voice understanding |
| Preparation–resolution | Note×slice intervals; отдельная expert semantic задача | До 40 TRAIN карточек,≥2 аннотатора,≤3 ч/чел; без GPU | Если rubric/precision неустойчивы, оставить weak descriptor и не обучать gold classifier |
| Rhythm/onset | Вероятность attack/release/count/time, а не pitch при видимом skeleton | Новый hidden-time graph fixture gate $0; потом 64 TRAIN/16 VALIDATION,≤1000 updates,≤2 T4h | Context model против meter/density prior; leakage или отсутствие gain закрывает этот decoder |
| Track interaction | Сначала arbitrary track×span completion; original POP roles отдельно | Существующие D2/E2 маски; для role extension32 разрешённых TRAIN records после adapter gate,≤1 CPU ч | Advantage simultaneous context; role shortcuts через index/program/order блокируют role claim |
| Structure | Repeat/span order/long context compatibility; semantic section labels отдельно | До 24 новых human pairs на≥12 works,≤2 ч/слушатель,≤1 T4h scoring | Human agreement beyond local density/length controls; bar-swap detection не доказывает form |
| Whole-track content | Pitch-given-rhythm track score → variable-length note/rhythm decoder | После D2/D4:16 TRAIN/8 VALIDATION admitted works,≤500 updates,≤2 T4h | Normalized decoder/EOS, exact target visibility, memory/latency; иначе ограничить scope span |
| PDMX scale | Corpus расширение ordinary-music SSL | Metadata/license/lineage gate, затем≤100 TRAIN-designated records,≤1 CPU ч; без learning | Accepted schema/coverage/dedup manifest → отдельный 1kprotocol; не считать inventory готовым TRAIN |
| JEPA/latent SSL | Representation hypothesis, не PLL | Позже matched theory-only/SSL/scratch continuation,≤1000 updates/arm,≤4 T4h | Только механика/ранняя тенденция; масштабный quality claim требует нового протокола |
| VAE | Отдельный generative/latent decoder | Сначала spec normalized observation model и tiny synthetic overfit $0 | Без ясной проверяемой цели не включать в critical path; latent distance не quality |
| Audio teacher | Версионированный renderer → teacher → MIDI surrogate | До 20dev clips×2renderers после отдельного выбора teacher и цены | Renderer-shift и human agreement отдельно; платный teacher budget пока unknown, запуск заблокирован |

Ни named form, ни semantic voice, ни melody/accompaniment labels не становятся
обязательным raw-MIDI input. Развитие может идти по аспектам; не требуется
одновременно завершить все строки, чтобы доказать ограниченную гармоническую
полезность первого критика.
