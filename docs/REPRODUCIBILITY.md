# Модель → эксперимент → данные → веса

[Главная](../README.md) · [Модели](MODELS.md) · [Кампании](experiments/README.md) · [Данные](DATASETS.md)

| Объект | Идентичность | Где искать |
|---|---|---|
| Архитектура | Класс, schema, параметры | [Модели](MODELS.md), source/config запуска |
| Попытка | experiment_id, родители, retry/continuation | [records](experiments/records/), [ledger](EXPERIMENT_LEDGER.md) |
| Код исполнения | Git commit и связанный source snapshot/diff для dirty run | repository/protocol и source artifact записи |
| Данные | Источник, версия, payload SHA-256, masks, membership/split fingerprints | Манифест набора и артефакты записи |
| Веса | SHA-256, байты, роль, тип loader | Checkpoint artifact; имя best.pt недостаточно |
| Результат | Задача, источник, split, знаменатель, evaluator, бюджет | Запись и связанный отчёт |

## Получение

Датасеты и локальный архив checkpoints доступны через
[каталог Google Drive](ARTIFACTS.md), где отдельно указаны завершение
загрузки и состояние доступа. Веса не входят в обычную Git-историю.

Выбрать запись, прочитать ограничения, получить точные данные и веса,
проверить файлы, затем использовать сохранённый конфиг и совместимую среду.

```bash
shasum -a 256 /path/to/downloaded-file
```

Для prepared TRAIN/VALIDATION используется
[restore_training_data.py](../scripts/restore_training_data.py): проверяются
receipt, части, размеры и SHA-256 payload. Скачанные архивы сначала
помещаются в новый каталог; не распаковывать их поверх checkout.

Drive folder/file ID помогает найти файл, SHA-256 и размер определяют его
байты. Успех загрузки в UI и повторная проверка скачанных байтов —
разные уровни подтверждения, отмечаемые в receipt отдельно.
Изменение доступа выполняется отдельно от переноса и отражается в receipt.

## История

Терминальные records и артефакты не переписываются.
Retry/исправление/продолжение получает новый ID и родителя.
Новое расположение старых байтов записывается отдельным receipt переноса,
а не косметической правкой терминальной записи. Неизвестное происхождение
и недостающие артефакты остаются неизвестными.

Running/planned — состояние реестра, живой процесс проверяется отдельно.
Старый GPU gate не переносится на новую среду. TEST policy и фактический
access различаются по [протоколу](EXPERIMENT_LOGGING.md).

## Обновление навигации

```bash
python3 scripts/build_project_catalog.py
python3 scripts/build_project_catalog.py --check
PYTHONPATH=src python3 scripts/experiment_registry.py check --records docs/experiments/records
PYTHONPATH=src python3 scripts/experiment_registry.py render --records docs/experiments/records --output docs/EXPERIMENT_LEDGER.md --check
```

Карта генерируется из существующих записей; ручные пояснения не устанавливают
новый ranking моделей и не заменяют первичные свидетельства.
