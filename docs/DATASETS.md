# Датасеты и Google Drive

[Главная](../README.md) · [Модели](MODELS.md) · [Кампании](experiments/README.md) · [Воспроизводимость](REPRODUCIBILITY.md)

24 октября 2026 пользователь выбрал Google Drive для хранения датасетов.
[Папка датасетов без отдельных аудиорендеров](https://drive.google.com/drive/folders/1NE52nepE_89IHg4Ms7ZYZse2VeQYqSae).
После явного подтверждения пользователя включён просмотр и скачивание
по ссылке с ролью «Читатель»; редактирование закрыто.
Текущий режим и отдельная папка checkpoints — в
[каталоге артефактов](ARTIFACTS.md).
Старые копии в Object Storage сохраняются; их удаление не входит в перенос.
До смены аккаунта исторические комплекты были загружены в
[первоначальную папку](https://drive.google.com/drive/folders/1HZvq3FnpHttFbNXl3irYfN6itjf0or6F);
это отдельное расположение, не подтверждение загрузки на новый Drive.

## Проверенные комплекты этого переноса

| Комплект | Проверенный состав | Полный размер передачи с supplement и сжатыми манифестами | Помещается в пустые 15 GB |
|---|---|---:|---|
| Полный локальный снимок | 221309 файлов основного архива + 1261 supplemental input/listening file; сохранённые отдельные аудиофайлы включены | 17804378125 байт = 17,80 GB = 16,58 GiB; 17 основных частей | Нет, даже при трактовке лимита как 15 GiB |
| Отдельный symbolic core | 219181 файлов + тот же supplement; исключены 2128 отдельных аудиофайлов | 11168352213 байт = 11,17 GB = 10,40 GiB; 11 основных частей | Да; для одного этого комплекта остаётся около 3,83 GB при лимите 15000000000 байт |

Оба архива созданы локально и проверены чтением всех regular members и
проверкой каждого hardlink. Оригиналы и полный архив сохранены. Symbolic core
исключает loose WAV/MP3/FLAC/OGG/AIFF/M4A/MP4/WebM; непрозрачные mixed source
archives сохранены и могут содержать аудио. Это subset для хранения, а не
новая semantic release или изменение статусов принятия разметки.

Пользователь выбрал перенос symbolic core без отдельных аудиорендеров;
полный архив сохраняется локально. На новом выбранном Drive загрузка
**завершена по подтверждению UI**: «Загружены 17 файлов». В папке ровно
17 файлов; все имена совпали с delivery inventory. Хранилище показывает
10,4 GB из 15 GB. Полные inventories,
checksum lists и инструкции лежат с локальными частями; компактный
[receipt](datasets/GOOGLE_DRIVE_TRANSFER_20261024.json) связывает точные байты,
scope, названия частей и destination. SHA-256 отдельных payload внутри всех
архивов проверен; remote re-download verification не выполнена.
Локальные build receipts внутри
комплектов фиксируют состояние на момент упаковки (`remote_status: not_uploaded`);
текущий статус переноса ведётся отдельно в receipt выше. После последующей
загрузки отдельного архива checkpoints весь Drive показывает 13,15 GB из
15 GB; запас в таблице относится только к датасетам без этого архива.

## Карта источников и производных наборов

| Набор | Состав / назначение | Авторитетная точка входа | Версия и состояние переноса |
|---|---|---|---|
| HookTheory / HTCanon | Исходные JSON и варианты источника | [HookTheory audit](HOOKTHEORY_FIELD_AUDIT.md), адаптеры в [src](../src/music_critic/adapters/) | Имеются локальные копии; исторический raw-комплект от 16 сентября переносится отдельно от новых производных |
| Dilemmadata AN/DLC | Нативные символические факты и анализы | [контракт данных](DATA_CONTRACT.md), [adapters](../src/music_critic/adapters/) | Raw и target sidecars имеют разные bindings |
| PDMX | Символическая музыка и metadata | [pipeline](../src/music_critic/pdmx_pipeline/), [адаптер](../src/music_critic/adapters/pdmx.py) | Сохраняется фактически имеющаяся копия; полнота публичного релиза не заявляется |
| POP909 / POP909-CL | Исходники и подготовленные представления | [POP909-CL](POP909_CL_ADAPTER_CONTRACT.md), [original](POP909_ORIGINAL_FIELD_AUDIT.md) | Локальная папка data не содержит отдельного полного raw POP909; в перенос включена локальная копия pop909-raw.tar.gz из cloud_migration/history |
| Prepared TRAIN/VALIDATION | Манифест, canonical payload и targets | [упаковка](../scripts/package_training_data.py), [восстановление](../scripts/restore_training_data.py) | Исторический комплект: 1295 TRAIN + 162 VALIDATION; это версия 16 сентября, не нынешний corpus400 |
| Native / TAVERN / multigenre theory | Источники, слои анализа и численные targets | [T07](workstreams/2026-10-07/multigenre-theory-enrichment/spec.md), [targets](../src/music_critic/tasks/event_theory_enrichment.py) | Отдельные immutable releases, masks и source/admission bindings |
| Corpus400 | Отобранные произведения, materialization, авторские попытки, reviews и releases | [whole-work accounting](../src/music_critic/tasks/whole_work_annotation_v2.py), [кампания](experiments/README.md#annotation) | Число в названии — план корпуса; не счётчик полностью принятых произведений. Октябрьские версии сохраняются отдельно |
| Авторские жанровые произведения / interventions | Созданные примеры и контролируемые изменения | [кампания](experiments/README.md#authoring) | MIDI, разметка и рендеры различаются; аудиорендеры учитываются по явному scope комплекта |

## Версионирование и получение

Сохранять source, annotation и prepared отдельно:

```text
MusicCritic/
  MusicCritic-datasets-20261024/  # именованный комплект на новом Drive
  datasets/
    sources/<source>/<version>/
    annotations/<corpus>/<version>/
    prepared/<dataset>/<version>/
```

Именованный каталог создаётся текущим переносом; вложенная структура datasets —
соглашение для следующих выпусков, а не утверждение об уже созданных папках.
Каждый выпуск получает отдельные файлы и манифест: версия, состав, SHA-256,
размер, происхождение, schema, splits/membership и условия доступа.
Новая версия не перезаписывает старую. Неизвестные метки остаются маскированными;
черновик, reviewed view и принятый release имеют разные статусы.

Точные загруженные файлы и уровень проверки фиксируются в
[receipt переноса](datasets/GOOGLE_DRIVE_TRANSFER_20261024.json).
Не считать каталог на Drive полным до сопоставления receipt со всем inventory.
Недостающие remote-only источники и ограничения объёма отражаются явно.

Скачать требуемый файл из папки, проверить SHA-256 и размер по receipt.
Prepared package распаковать в новую директорию и передать существующему
restore_training_data.py; он сохраняет original manifest и выдаёт relocation
manifest. Обучение читает локальные данные, не поток синхронизации Drive.
Ключи и OAuth-токены в Git не сохраняются.

## Что входит в Git

Эта карта, компактные receipts, SHA-256/размеры, конфиги и инструкции.
Полные корпуса, generated MIDI/audio, payload/cache shards и checkpoints
остаются в игнорируемых каталогах и внешнем хранилище.
Условия использования каждого источника сохраняются; доступ по ссылке
не меняет лицензии исходных датасетов.
Оригиналы, исторические экспериментальные records и старые артефакты
не меняются ради нового расположения файлов.
