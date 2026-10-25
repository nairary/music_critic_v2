# Первый запуск и маршруты

[Главная](../README.md) · [Модели](MODELS.md) · [Данные](DATASETS.md)

## Окружение

Пакет объявляет Python >=3.11; исторические GPU/HCM-протоколы могут
требовать отдельную закреплённую среду. Новое окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-test.txt -e '.[dev,training]'
PYTHONPATH=src python -c "import music_critic; print(music_critic.__version__)"
```

CPU-проверки используют версии Torch/PyG из constraints-test.txt. Исторические
побайтовые fingerprints инициализации проверяются в исходной Linux/x86 CPU
среде; ARM/macOS не объявляется эквивалентной средой этих старых запусков.
Это отдельно от формата сохранённых весов и загрузки канонических данных.

Без внешних корпусов и весов можно загрузить маленький канонический пример:

```bash
PYTHONPATH=src python -c "from music_critic.data import load_piece; p=load_piece('tests/fixtures/data/canonical_piece_v2.json'); print(type(p).__name__)"
```

При установленном Torch/PyG из него можно построить raw-граф:

```bash
PYTHONPATH=src python -c "from music_critic.data import load_piece; from music_critic.graph import build_raw_graph; print(build_raw_graph(load_piece('tests/fixtures/data/canonical_piece_v2.json')))"
```

Эти примеры не устанавливают GPU admission или качество модели.
Графовый пример требует установленного окружения; новой GPU-проверки
в задаче навигации не выполняется.

## Шесть маршрутов

| Маршрут | Вход / конфиг | Интерфейс / результат | Ограничения |
|---|---|---|---|
| Канонические события → raw-граф | JSON выше | load_piece / build_raw_graph → события/граф | Без trained weights; [контракт](DATA_CONTRACT.md) |
| Теоретический классификатор | Вариант C0–C4, checkpoint, prepared data | [Обучение](TRAINING.md), [оценка](EVALUATION.md), [C4 CLI](../scripts/train_c4_pop909.py) → task-specific predictions/metrics | Loader/evaluator выбирается по записи |
| Парный критик | Frozen C3, пары, campaign config, critic checkpoint | [CLI](../scripts/run_critic_campaign.py), [интерфейсы](CRITIC_V1.md) → scores / pairwise outputs | Правки не заменяют независимых слушателей |
| HCM-1 | Bound config, input, masks, selected weights | [CLI](../scripts/run_pll.py), [API/команды](HCM1.md) → условные события/NLL | hcm1_reference.json — template, не trained config |
| HCM whole / HCM-2 | Полный источник, whole config, checkpoint | [Контракт и команды](HCM_WHOLE.md) → full-context outputs | Память/decoder определяет конфиг |
| Разметка | Admitted sources, overlays, reviews | [Экспорт](../scripts/finalize_event_theory_release.py), [контракт](DATA_CONTRACT.md) → index/shards | Candidate selection и semantic acceptance различаются |

Команды с реальными данными берутся из записи и контракта.
Наличие рецепта не запускает обучение, не открывает TEST и не выделяет бюджет.
Импортированные source snapshots — свидетельства; их выполнение не следует
из скачивания.

## Проверки

```bash
python3 tests/test_project_catalog.py -v
python3 scripts/build_project_catalog.py --check
PYTHONPATH=src python -m pytest -q
python -m compileall src
```

Полный pytest требует dev-зависимости и подходящей среды; GPU/integration
проверки включаются отдельно. Legacy checkout не нужен для работы V2.
make legacy-check — специальная сверка legacy snapshot.
