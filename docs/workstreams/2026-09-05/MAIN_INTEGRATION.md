# Перенос принятой версии в локальный main — 2026-09-05

Пользователь разрешил следующий шаг после общей приёмки четырёх направлений:
интеграцию в main. Выполнен обычный локальный merge без squash/rebase.
Публикация на GitHub и новые эксперименты не выполнялись.

Исходный main: f0e0886161c9f67786011508006db61c803c1752.
Кандидат: a628af4bb72358f0ca99ad1a8c04fcf4a78cc845, ветка
codex/ws-integration-20260905 в /private/tmp/music-critic-v2-ws-integration-20260905.
Дерево кандидата: f1ea3c896a10a5f6b773ed50f97fb1419e4b804e.
Подготовленное дерево merge в main точно совпало с деревом кандидата.
После проверки изменены только пять итоговых документов; принятый код неизменен.
Идентификатор финального merge фиксируется во внешнем
/private/tmp/music-critic-main-integration-20260905/final.json и в истории main;
сам документ входит в этот commit и не содержит собственного SHA.

Независимый preflight подтвердил все 63 файла полного рассмотренного patch,
33 src/test/config fingerprints и CLI script SHA через исходный WS3 и source
manifest. Удалений, дополнительных изменений и изменения frozen paths нет.
Все 18 существующих координационных документов main совпадают с кандидатом;
текущий STATUS полностью содержится в расширенном STATUS кандидата.
Их hashes сохранены в /private/tmp/music-critic-main-integration-20260905/pre-main.json.
Перед merge все 19 файлов сохранены в именованный recovery stash
db327d8329ecc17b8de625462df8ff96e56f6da8; совпадение содержимого проверено.
Повторное применение stash не требуется: правки уже включены в кандидата.
Stash, исходные worktrees, patch и изолированная копия сохранены.

[MAIN_INTEGRATION_CODE_MANIFEST.json](MAIN_INTEGRATION_CODE_MANIFEST.json)
содержит все 34 implementation/test/config/CLI файла. Код не меняется по
сравнению с принятым результатом; дополнение manifest уточняет охват CLI.
Совокупный SHA256 manifest:
929c064ee2abb9acb007836bd7aa00f20edc39cdadb0971fc074d9aabd4f329e.

Непосредственно в main повторена та же выборка из 29
[тестовых селекторов](INTEGRATION_TEST_SELECTION.json): **814 passed,
6 CUDA-only skips, 2 deselected, 2 upstream warnings in 52.87s**, exit 0.
Перед запуском проверен импорт music_critic из src именно основного checkout.
Команда с окружением сохранена в
/private/tmp/music-critic-main-integration-20260905/main-gate-command.json;
полный лог — /private/tmp/music-critic-main-integration-20260905/main-gate.log.
Исключены существующие проверки двух worker-процессов с shared memory и
конкретного CUDA runtime; предупреждения относятся к upstream Torch JIT.
Это выбранные synthetic unit/contract regressions, не весь suite и не remote CI.
Предыдущая изолированная приёмка и неуспешные попытки сохранены в истории отчётов.

Legacy не используется; C0, реальные данные/checkpoints, TEST, SSH, зависимости
и научные experiment records не затрагиваются. Legacy-логика не переносилась
и не менялась на этом шаге. Публикация локального main остаётся отдельным шагом.
Следующий научный шаг требует отдельного зарегистрированного протокола,
как указано в [итоговом отчёте](INTEGRATION_RESULT.md).
