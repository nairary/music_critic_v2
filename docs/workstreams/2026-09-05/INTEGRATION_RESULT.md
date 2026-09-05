# Итог четырёх направлений — 2026-09-05

Этот отчёт сохраняет итог приёмки изолированной версии. Последующее разрешение
пользователя и перенос в локальный main отражаются в [MAIN_INTEGRATION.md](MAIN_INTEGRATION.md).

**Все четыре реализации приняты после независимого ревью и совместной проверки.**
Семь замечаний исправлены; открытых замечаний этой волны нет. Общий результат:
**814 passed, 6 skipped, 2 deselected, 2 warnings in 55.38s**.
Периодическая проверка остановлена после завершения приемки.

Объединённый код находится в отдельной локальной копии:
`/private/tmp/music-critic-v2-ws-integration-20260905`, ветка
`codex/ws-integration-20260905`, base/HEAD
`f0e0886161c9f67786011508006db61c803c1752`.
Изменения не закоммичены и не перенесены в main. В основном checkout изменены
только STATUS и документы координации. Исходные четыре worktree сохранены.

## Что готово

| Направление | Реализованное поведение | Приёмка |
| --- | --- | --- |
| WS1 | Отдельная полная транспозиция пересчитывает все признаки высоты; HookTheory raw timeline/cache отделены от аккордовой разметки с новым version contract | [WS1 review](WS1_REVIEW.md) |
| WS2 | Строгий theory→SSL перенос, общие параметры backbone, сохранение BiGRU/18 голов, три режима objectives и корректные masks | [WS2 R2](WS2_REVIEW_R2.md) |
| WS3 | PDMX source manifest, группировка до split/windows, точные окна, ограниченная обработка, quarantine/cache/resume и raw SSL loader | [WS3 R2](WS3_REVIEW_R2.md) |
| WS4 | Frozen POP909-CL root/quality crosswalk, независимые masks/coverage, два raw views и ограниченная загрузка model-only весов | [WS4 R3](WS4_REVIEW_R3.md) |

Код и тесты каждого направления соответствуют рассмотренным ревизиям в
[COORDINATION.json](COORDINATION.json). Из общих исполняемых файлов менялся
только разрешённый HookTheory glue в tasks/corpus.py. STATUS и append-only ADR
сведены с сохранением исходной истории. Корневые WS_RESULT.md исполнителей
остались в их worktrees; отдельные WS1/2/3/4_RESULT.md перенесены без конфликта.

Координатор добавил
`tests/experiments/test_workstream_synthetic_integration.py` в общей копии:

1. Сгенерированный PDMX → два cached raw окна → complete +2 транспозиция,
   совпадающая с независимым canonical rebuild → новый mask binding → SSL
   backward/optimizer step. Shared backbone обновляется; source model,
   theory-only параметры и исходные графы сохраняются.
2. Сохранённая theory-ветвь → новый синтетический model-only artifact → frozen
   POP909 loader/evaluator. Веса, native/metadata_removed views и piece reports
   совпадают; загруженная модель независима, заморожена, caller RNG сохранён.

## Совместная проверка

Первый общий прогон выявил нарушение import boundary в WS4:
1 failed, 812 passed, 6 skipped, 2 deselected in 53.72s. Его ревизия, log и
patch сохранены в [INTEGRATION_REVIEW_R1.md](INTEGRATION_REVIEW_R1.md).
Исправление не ослабляет тест: синтетическая MIDI-конструкция перенесена в
новый adapter helper с сохранением точных байтов fixture.

Финальный прогон на исправленной общей копии: **814 passed, 6 skipped,
2 deselected, 2 warnings in 55.38s**, exit 0. Все новые обязательные CPU tests
и оба сквозных сценария прошли. Шесть skips требуют CUDA; две deselections —
существующий two-worker shared-memory test и один CUDA evaluation test.
Они не учитываются как passed. Два предупреждения — upstream Torch JIT
deprecation. Полная suite репозитория не заявляется пройденной.

Использована точная объединённая выборка из
[INTEGRATION_TEST_SELECTION.json](INTEGRATION_TEST_SELECTION.json), без повторяющихся
selectors и без real-data integration modules. Команда — указанный ниже Python,
`-m pytest -q -p no:cacheprovider`, затем все 29 selectors в порядке JSON и:

```text
-k 'not test_dataloader_num_workers_zero_and_two_have_schedule_parity and not bounded_evaluation_uses_concrete_cuda_runtime'
```

Working directory — интеграционная копия; interpreter:
`/private/tmp/music-critic-v2-integration-venv/bin/python`.
Environment: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Полный финальный log: `/private/tmp/music-critic-ws-integration-r2-tests.log`.
git diff --check, JSON validation, соответствие перенесённых файлов source
manifest и неизменность frozen model/loss/SSL-transfer/experiment paths проверены.

## Идентичность результата

[INTEGRATION_CODE_MANIFEST.json](INTEGRATION_CODE_MANIFEST.json) связывает все
33 изменённых/новых файла из src/tests/configs с их SHA-256. CLI entrypoint
scripts/preprocess_pdmx.py дополнительно связан полным patch и WS3 source
manifest; полный список из 34 файлов дан в
[MAIN_INTEGRATION_CODE_MANIFEST.json](MAIN_INTEGRATION_CODE_MANIFEST.json).
SHA-256 compact sorted JSON этого manifest:
`5a77bb232e0a2c9e65a0547da866f5e068fc9cbade21655a032788476b72ae0e`.
Документы отделены от этого fingerprint, чтобы финальное описание не ссылалось
на собственный hash. Полный reviewable patch:
`/private/tmp/music-critic-ws-integration-final.patch`; его SHA находится в
соседнем `music-critic-ws-integration-final.patch.sha256`.

## Границы и следующий шаг

Это завершённая инженерная волна на синтетических CPU fixtures. Она не
показывает улучшение музыкального качества, пользу транспозиции, отсутствие
forgetting, готовность реального PDMX или фактическую совместимость C0/C2
checkpoints. Historical C0/C2 runners/profiles, текущий C0 и закрытый TEST не
затрагивались. Реальные корпуса/обученные checkpoints, SSH и legacy repository
не открывались. Legacy логика не переносилась; использованы действующие V2 API.
Новых зависимостей и научных experiment records нет: выполнялись ordinary tests.

Следующий репозиторный шаг — рассмотрение готового общего diff и отдельное
решение о его переносе в main. Следующая научная волна требует собственного
зарегистрированного протокола, проверенной provenance реальных данных/весов и
явных критериев оценки. Она здесь не запускалась.
