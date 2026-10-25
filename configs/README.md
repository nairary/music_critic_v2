# Карта конфигураций

[Главная](../README.md) · [Модели](../docs/MODELS.md) · [Воспроизводимость](../docs/REPRODUCIBILITY.md)

| Путь | Назначение | Как использовать |
|---|---|---|
| [pdmx/synthetic.json](pdmx/synthetic.json) | Синтетический пример PDMX pipeline | Проверка механики, не конфиг полного корпуса |
| [context_local_control_v1.json](context_local_control_v1.json) | Изолированный локальный контроль контекста | Вместе с соответствующим протоколом/record |
| [ssl_group/random_vs_beat.json](ssl_group/random_vs_beat.json) | Сравнение маскирования | По зафиксированному протоколу SSL |
| [ssl_group/scientific_pilot.template.json](ssl_group/scientific_pilot.template.json) | Шаблон научного пилота | Сначала конкретные data/source bindings и planned record |
| [pll/hcm1_reference.json](pll/hcm1_reference.json) | HCM-1 reference template | Не trained config и не native admission |
| [pll/hcm1_whole_reference.json](pll/hcm1_whole_reference.json) | Whole-piece reference template | Не trained config; см. HCM_WHOLE |
| [phase9eb1/](phase9eb1/) | Историческая реконструкция AnalysisGNN | Сохранённые config, environment, architecture, metrics, attestations и patch manifest; не современный профиль по умолчанию |

Часть runtime-настроек задаётся Python dataclasses/Hydra ConfigStore,
а точные resolved configs запусков находятся в связанных внешних артефактах.
Отсутствие отдельного JSON здесь не означает отсутствия конфигурации.
Исторические attestations не редактируются ради актуальных путей.
Все варианты не нужно устанавливать и запускать в одном окружении.
