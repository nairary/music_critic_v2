# Каталог моделей

[Главная](../README.md) · [Кампании](experiments/README.md) · [Данные](DATASETS.md) · [Воспроизводимость](REPRODUCIBILITY.md)

Срез навигации: 24 октября 2026. Реализация и сохранённые результаты
учитываются отдельно. Доступность удалённых весов и живые процессы каталогом
не проверяются. Архитектура, запуск и checkpoint имеют разные идентификаторы.

| Семейство / вариант | Назначение | Реализация / контракт | Статус по сохранённым документам | Свидетельства и ограничения |
|---|---|---|---|---|
| Feature/local GNN, hierarchy, onset BiGRU | Ранние контроли представлений | [models](../src/music_critic/models/), [локальный](PHASE6A_BASELINE.md), [иерархический](PHASE6B_HIERARCHY.md) | Реализованы; исторические обучающие/оценочные маршруты | [Обучение](TRAINING.md), [оценка](EVALUATION.md); контроли сохраняются |
| C0 / C1 | Исправленный 18-head baseline и исторические варианты подготовки | [corrected_model.py](../src/music_critic/experiments/analysisgnn/corrected_model.py), [trainer](../src/music_critic/experiments/analysisgnn/corrected_training.py) | C0 исторически выбран; C1 отложен (ADR-111) | [Кампания](experiments/README.md#baselines); точный вариант определяет конфиг |
| C2 | Full-orbit TRAIN expansion | [full_orbit_training.py](../src/music_critic/experiments/analysisgnn/full_orbit_training.py) | Обучающие попытки и сравнения сохранены | Это также режим данных; название не означает полностью другую архитектуру. [Кампания](experiments/README.md#baselines) |
| Temporal / C3 | Точные attack/hold/release события и temporal message passing | [temporal_slices](../src/music_critic/experiments/analysisgnn/temporal_slices/), [контракт](TEMPORAL_SLICE_GRAPH.md) | Исторически выбранная temporal80k основа main | [Temporal80k](experiments/TEMPORAL_80K_FINAL_RU.md); старые имена temporal/C2 и C3 сохраняются |
| C3 attention / threepass / relation gate | Варианты временных связей и обмена сообщениями | [model.py](../src/music_critic/experiments/analysisgnn/temporal_slices/model.py) | Сохранены обучающие и оценочные попытки; варианты неравнозначны | [Обзор](experiments/C3_THREEPASS_FINAL_RU.md), [POP909](experiments/C3_POP909_FULL_20261001_RU.md); разные бюджеты/среды |
| Fusion / task attention / MMoE | Обмен между теоретическими задачами | [analysisgnn](../src/music_critic/experiments/analysisgnn/), [ветки](BRANCHES_AND_RELEASES.md) | Смешанный статус: контроли, остановки и незавершённые попытки | [Кампания](experiments/README.md#fusion); остановленные и незавершённые попытки сохранены |
| C4 / POP909 | C3 relation gate с обучением на POP909 | [c4_pop909](../src/music_critic/experiments/c4_pop909/), [CLI](../scripts/train_c4_pop909.py) | Сохранены обучение и отдельная оценка переноса | [Обучение](experiments/C4_POP909_3090_20261005_RU.md), [перенос](experiments/C4_DILEMMA_TRANSFER_20261005_RU.md) |
| SSL | Маскированное представление и encoder transfer | [ssl](../src/music_critic/ssl/), [контракт](PHASE7A_SSL_BASELINE.md) | Реализация/gates; статус каждого scientific run — в реестре | [Кампания](experiments/README.md#ssl); representation loss не равен музыкальному PLL |
| Multiscale aspect / pairwise critic | Аспектные оценки и сравнение вариантов с frozen C3 | [critic](../src/music_critic/critic/), [контракт](CRITIC_V1.md), [CLI](../scripts/run_critic_campaign.py) | Реализация и обучающие попытки; согласие со слушателями не установлено | [Кампания](experiments/README.md#critic); обучение на правках не устанавливает согласие со слушателями |
| HCM-1 | Локальное условное предсказание событий | [model.py](../src/music_critic/pll/model.py), [контракт](HCM1.md), [CLI](../scripts/run_pll.py) | Реализация, обучающие и диагностические попытки | [Кампания](experiments/README.md#hcm); NLL, генерация и предпочтения различаются |
| HCM whole / HCM-2 variants | Полный контекст и память повторов | [whole_model.py](../src/music_critic/pll/whole_model.py), [контракт](HCM_WHOLE.md) | Реализация и development-оценки; выбор универсального baseline не сделан | [Development-проверка](experiments/HCM2_RANDOM_3090_VALIDATION_20261001_RU.md); вариант задают config, decoder и checkpoint |

## Как выбрать веса

1. Открыть кампанию и конкретную запись обучения/оценки.
2. Найти experiment ID, source/config/data/split fingerprints и checkpoint artifact.
3. Уточнить, encoder-only это или полное состояние обучения; получить байты
   из внешнего хранилища и проверить SHA-256 и размер.
4. Использовать соответствующий loader и окружение.

Общего `best.pt` для проекта нет. Reference config не подменяет конфиг
обученного состояния. Перенос датасетов на Drive не означает перенос всех весов.
[Правила воспроизводимости](REPRODUCIBILITY.md).

Код в `experiments/` сохраняет пути: critic/C4 зависят от temporal модулей.
Его перенос требует отдельной миграции импортов и совместимости checkpoints.
Каталог не меняет выбор baseline.
