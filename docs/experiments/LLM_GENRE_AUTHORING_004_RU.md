# Пять новых произведений: партия004

`EXP-LLM-GENRE-AUTHORING-004` completed,24сентября2026.
Кампания: **10 разных произведений, по2/100 каждого жанра**.

| Жанр | Произведение и доступный материал | H | M | R | S | Сочетания |
|---|---|---:|---:|---:|---:|---:|
| rock | Bon Jovi — You Give Love a Bad Name,81такт |5|5|5|5|2|
| jazz | Fats Waller — Honeysuckle Rose,144такта, шесть труб |5|5|5|5|2|
| classical | Beethoven Op.13 II,73такта, аутентифицированный TRAIN |5|5|5|5|2|
| hiphop | Tyler, The Creator — Where This Flower Blooms,108тактов |5|5|5|5|2|
| electronic | Avicii — SOS,64такта, ансамблевая аранжировка |5|5|5|5|2|

100 отдельных вариантов +10 сочетаний +5 оригиналов = **115 полных MIDI/WAV**.
У каждого сохранены контекст, точные note_id/patches, исходные значения, rationale,
uncertainty, сырой diff и происхождение. Сочетания разобраны отдельно, механическая
совместимость не приравнивается к музыкальной независимости. Старые партии не
менялись: проверены хеши/размеры всех719 файлов манифеста003.

## Проверки и результаты

- **190 tests passed in3.43s**; точная команда ниже и `tests-003.log` сохранены.
-115 MIDI roundtrips;110 изменённых MIDI и diff совпали с допущенными планами.
-115 WAV: одинаковая длительность внутри произведения,110 отличных от оригинала
 PCM, общий gain, ноль clipping samples.
-950 пар:896 compatible,39 semantic revalidation,15 field conflicts.
-Рендер96.08s wall,95.94s CPU; упаковка15.47s. GPU/облако/paidAPI/training —0.
-ZIP582121954bytes,524members;CRC и все SHA256 совпали.
 SHA256 `0133115ad8cd06aac1a3a7d32de40963ee705fca9ce6be3d5d891351ca00794f`.
-Плеер: все пять вкладок/по22варианта; воспроизведение, переход к месту правки,
 сочетания и переключение A/B с сохранением позиции проверены в браузере.

Независимый символьный рецензент передал содержательную проверку100 одиночных
планов и10 сочетаний. Две неточности исправлены новыми файлами описаний: начальный
такт рок-аранжировки имеет4QN; интервал F5/Gb4 в jazz H5 — большая септима,
а не малая секунда. Музыкальные patches не изменялись. Финальный файл рецензента
прерван лимитом; полученные замечания сохранены, финальную привязку исправлений
к байтам сделал основной агент. Это явно отделено от независимой финальной подписи.

Отдельный code review выявил отсутствие проверки входного манифеста непосредственно
перед рендером. Исправление проверяет полноту, уникальность/безопасность путей,
размеры/SHA до создания вывода; добавлены регрессионные тесты. Standards и Spec
проверены раздельно. Изменения ограничены локальными скриптами партии и документацией.

## Ограничения

Человеческие предпочтения **unknown**: готовность означает проверенный материал,
не подтверждённое ухудшение. Некоторые альтернативные гармонии/контуры могут
понравиться слушателям больше. Критик не запускался.

PDMX metadata/raw counts расходятся, но адаптер сохранил все фактические raw notes:
rock3231/3058, jazz1863/1749, hiphop1722/1473, electronic2671/2561 (CSV/raw).
В Beethoven нулевые grace остаются вJSON и не озвучиваются. Служебные120BPM
не являются исполнительским темпом Adagio. Джаз — шесть труб без rhythm section;
SOS — партитура без ударных, не оригинальный электронный production.

Первый electronic-кандидат Sandstorm отклонён из-за внутренних Crab Rave/смешанных
тем; не засчитан, сохранён под отдельным invalid source attempt. Замена — SOS.
Каталоги подтверждают произведения/жанр, но не аутентифицируют пользовательские
аранжировки. Beethoven Op.13 считается одним произведением независимо от части.

VALIDATION/TEST musical payload закрыт. Известные metadata work aliases проверены;
полная связка названий HookTheory/POP909 неизвестна. Уже просмотренные произведения
остаются development, не OOD. Legacy не использовался; новых зависимостей нет.
Архитектурных решений не меняли.

## Артефакты и воспроизводимость

Плеер: http://127.0.0.1:8772/ .
Полный пакет: `outputs/llm_genre_authoring_004/five_genres_20_each.zip`.
Разборы, планы, снимки источников и подробные проверки:
`outputs/llm_genre_authoring_004/` и `listening/<genre>/`.
`artifact-manifest.json` связывает645файлов; точные execution commands/time/status
в `render-execution.json` и `package-execution.json`. Терминальный registry record
содержит переносимые locator, SHA и размеры основных свидетельств.

```bash
PYTHONPATH=src:scripts:outputs/llm_genre_authoring_004 PYTHONDONTWRITEBYTECODE=1 \
/Users/aavalsamakin/.codex/worktrees/5ab7/music-critic-v2/outputs/changes-venv/bin/python \
-m pytest -q -p no:cacheprovider outputs/llm_genre_authoring_004/test_materializer.py \
tests/experiments/test_music_interventions.py tests/experiments/test_genre_midi.py \
tests/experiments/test_genre_preview.py tests/experiments/test_genre_sources.py \
tests/experiments/test_genre_interventions.py tests/experiments/test_genre_suite.py \
tests/experiments/test_intervention_player.py tests/experiments/test_intervention_preview.py \
tests/experiments/test_music_intervention_pilot.py
```

Следующая партия005: новый planned record, следующие разные произведения после
metadata admission, затем контекстное авторство и независимая проверка.
