# Создание задач и координация — 2026-09-08

Координатор: `01a06e25-7dba-7430-86da-202cacede488`.
Три запроса create_thread приняты приложением. Пока возвращены client IDs: подготовка worktrees не равна подтверждённому началу исполнения.

| Задача | ТЗ | Client ID |
| --- | --- | --- |
| Проверить контекстное понимание C2 | [ТЗ](01_CONTEXT_DIAGNOSTICS.md) | `client-new-thread:cd5fefb3-b8b7-40a0-a534-60ef11fa32f4` |
| Подготовить эксперимент с logit fusion | [ТЗ](02_LOGIT_FUSION.md) | `client-new-thread:0c8b6cb5-9bbb-4ec8-adce-12acf0fa651d` |
| Подготовить структурное SSL и голосовые связи | [ТЗ](03_VOICES_STRUCTURAL_SSL.md) | `client-new-thread:1a7bf880-8215-44b6-bea5-47dcdd7bf986` |

Общий снимок: 752 файла, 13 124 819 bytes. SHA-256 manifest:
`9491ad6acece3cae0bdf5b15ed7b6017f7dee366359b6d758f349c6e25e3afdf`.
Исходная ветка codex/develop, HEAD `da8c9dcd999a4b522f87f9a20bf3506ad26497b7`, dirty working tree.
Локальный ignored пакет: `outputs/coordination/research_parallel_20260908/`.
Снимок содержит текущие source/docs/tests и четыре ТЗ, не содержит datasets, checkpoints, retained outputs или .git. Fixture-данные, уже входящие в source tree, сохранены.

Проверки координатора: 752/752 hash+size; 3/3 bootstrap checks (точная установка в изолированный worktree, отказ поверх dirty worktree, отказ при неверном manifest hash); git diff --check passed.
На этом шаге меняются четыре ТЗ, этот документ и STATUS; shared implementation не изменяется. Legacy не использован; model/corpus/TEST/GPU запусков нет.

Текущее сопоставление actual thread IDs, receipt bootstrap и review хранится в ignored `coordination.json`. Client ID нельзя передавать в read_thread/wait_threads; координатор разрешает actual IDs по list_threads и проверяет проект, полное название и worktree. До разрешения не создавать дубликаты.

Каждая задача отправляет один пакет review координатору. Дальнейшая разработка автономна в согласованном ТЗ; дорогие запуски проходят единую очередь. Раздельные абляции и experiment records сохраняются; объединение изменений — отдельная проверка после результатов.

Создан отдельный heartbeat `music-critic`, ACTIVE, каждые 10 минут, с назначением в текущую задачу координатора. Он разрешает actual IDs, проверяет bootstrap и новые результаты, выдаёт корректировки и ведёт одну GPU-очередь. Уведомления только при значимом изменении; после проверки первого этапа всех трёх задач heartbeat приостанавливается. Старый `music-critic-v2` performance heartbeat остаётся PAUSED.
