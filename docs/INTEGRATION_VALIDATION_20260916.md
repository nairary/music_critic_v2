# Проверка интеграции temporal/C2 — 2026-09-16

Рабочая основная версия — завершённый temporal/C2. Linear fusion, task-attention
fusion и grouped MMoE сохранены в `codex/fusion`; исходный полный снимок —
`codex/research-preserved-20260916@42254af`. Ничего из прежних результатов не
удалено. Это интеграционная проверка, не новый научный эксперимент.

## Изменения

- Общие `FrozenTrainingData` и RNG-утилиты вынесены из fusion-модулей. Форматы,
  исторические schema/error identifiers и checkpoint-state сохранены.
- Temporal runner/cache/checkpoint используют эти общие утилиты. Математика
  модели, головы, loss, sampler и расписание обучения не менялись.
- Из основной рабочей версии удалены fusion/MMoE-модули, их конфиги и тесты;
  они восстановлены в отдельной ветке. Неиспользуемый temporal-моделью MMoE-hook
  в общих heads также оставлен только на fusion-ветке.
- Шесть диагностических модулей используют уже существующий общий помощник
  CUDA memory statistics вместо обходивших его прямых вызовов reset.
- Обновлён устаревший синтетический multisource fixture после ранее принятого
  HookTheory timeline-контракта. Научные терминальные записи не переписывались.
- Добавлены проверяемая упаковка TRAIN/VALIDATION и загрузчик Object Storage.
  Новые зависимости проекта не добавлялись.

## Проверки

| Проверка | Результат |
|---|---|
| Полный Linux CPU suite, кроме отдельной native-группы | **4797 passed, 65 skipped**, 853.79 s |
| Native ownership, отдельный процесс | **62 passed**, 1.21 s |
| Совокупный Linux результат | **4859 passed, 65 skipped** |
| Основной интеграционный набор | 253 passed |
| Temporal cache/campaign/resume | 168 passed |
| Worker-тесты Mac вне sandbox | 5 passed |
| Fusion-ветка | 87 passed |
| CI policy и float32 objective-тесты после поправки | 32 passed, 1 skipped |
| Восстановление подготовленных файлов | 4375/4375, SHA256 и размеры совпали |
| Реестр экспериментов | 117 записей, valid; ledger пересоздан без изменения |
| compileall / git diff --check | успешно |

Linux-среда: Python3.12, Torch2.8.0+cpu, PyG2.7.0, NumPy1.26.4; CUDA скрыта.
Полный прогон выполнен на чистой копии `9abd1eb`; последующие изменения —
документация и порядок CI, проверенный отдельным policy-набором. Исторические
GPU-каталоги и окружения не модифицировались.

Первый Linux-прогон из tar без `.git` дал 35 failures. Причины проверены:
недоступный Git provenance; нативный supervisor требует одного главного потока,
а ранее выполненные DataLoader-тесты оставляли feeder threads; один численный
порог был меньше float32 epsilon. Повтор с Git прошёл. Нативные проверки в CI
теперь запускаются отдельным Python-процессом, без ослабления runtime guards.
Для batch-partition objective-теста используется абсолютный допуск 2 float32
epsilon вместо 1e-7: наблюдавшаяся разница составляла один epsilon.

В полном Linux-прогоне осталось шесть предупреждений: deprecation, scalar
conversion, отключённый CUDA autocast, fork из многопоточного процесса и
ошибка teardown `QueueFeederThread` в DataLoader-тестах. Последнее не подавлялось;
изолированные worker-проверки прошли, но это остаётся ограничением монолитного
тестового процесса. Новая GPU-кампания этой проверкой не разрешается.

На Mac четыре исторических побитовых fingerprint-проверки зависели от среды.
Инициализация C2 на старом main и интеграционной версии в одном Mac runtime
совпала (`aefa98449149876160a1ca543017bf1f31fdc956f469defa76c7149c9d15e43c`).
На Linux исторические проверки прошли. Ожидаемые научные fingerprints не менялись.

Полные журналы: `outputs/cloud_migration_20260916/linux-*.log`,
`integration-focused.log`, `worker-tests.log`, `fusion-branch-tests.log`.
Реестр: `c88e9176c19db17d4717b0142f08205f790b6d7a27937a582727693980510600`.

Legacy не читался, не импортировался и не использовался. Научные TRAIN/VALIDATION
evaluation и TEST evaluation не проводились; копирование сырых корпусов —
непрозрачная архивация. Следующий этап: закончить облачную передачу; затем по
отдельному решению пользователя реализовать temporal attention и пройти новые
GPU environment/equivalence/memory gates перед зарегистрированным обучением.
