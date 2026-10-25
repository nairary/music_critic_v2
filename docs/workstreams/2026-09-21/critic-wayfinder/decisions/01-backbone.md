# D1 — Backbone: переносимая исходная точка, а не новый конкурс архитектур

Статус: **источник → эксперимент**. Зависимости: нет. Владелец подготовки: агент.
[Карта](../README.md) · [проверенные результаты](../../../../research/CRITIC_WAYFINDER_EVIDENCE_20260921_RU.md).

## Фактическая база

Main содержит corrected C2 и `TemporalSliceModel`: локальный GNN 3×128,
иерархия, два coarse Transformer-слоя, top-down context, onset BiGRU и temporal
обмен с точными starts/holds/ends. C2 уже видел timing и имел attention в
Transformer. C3 neighbor attention и три прохода/gate находятся в отдельных
worktrees и имеют собственные model/output/checkpoint bindings.

Пути: `src/music_critic/experiments/analysisgnn/corrected_model.py:174–224`,
`temporal_slices/model.py:15–84` в том же пакете;
`src/music_critic/models/hierarchy.py:786–806,888–917,997–1007`;
`src/music_critic/models/onset_bigru.py:190–243`.
Все main-пути привязаны к HEAD карты с сохранением dirty-state оговорки.

Финальные temporal embeddings доступны для note/onset/beat/bar/slice. Они не
содержат temporal-updated song/track: глобальный или track score требует явно
заданного pooling или отдельного сохранения base coarse states. Не выдавать
раннее `base.encode(...).fused` за финальное C3 representation.

## Альтернативы и рекомендация

| Вариант | Польза | Риск / стоимость | Различающая проверка |
|---|---|---|---|
| C3 temporal mean, 80k B2 | Точные raw события; завершённый reference, main-код | Checkpoint может требовать восстановления; исторический runtime | Новый frozen probe/decoder против C2 на одних группах |
| Frozen C2 best115k | Проверен root/quality transfer; дешёвый контроль | Beat-resolution, нет позднего temporal refinement | Та же readout задача без объявления равенства backbone compute |
| C3 relation gate80k | Joint выше temporal mean; улучшены 10/18 голов на 80k | Primary ниже mean; B4/draws/runtime confounds; иной source | Позже тот же frozen probe, если перенос артефакта дёшев |
| C3 three-pass без gate | 60k promising snapshot | Финал после 60k недоступен | Сначала получить существующий итог, не запускать повтор |
| Новый backbone / full retrain | Может изменить representation | Самый дорогой путь, причинная гипотеза не выделена | Только если простые paths провалены с диагностикой |

Рекомендуется temporal mean как **кандидат первой проверки**, C2 как контроль;
не происходит promotion/перевыбор checkpoint этой документацией. Если mean
нельзя восстановить, использовать проверяемый C2 или доступный gate с явно
переименованным baseline и повторной фиксацией протокола, а не переобучать 80k.

## Что нужно по источникам

Уточнение22сентября для быстрого demo: локальный файл
`outputs/pop909_transfer/20260908/export-001/c2-model-only.pt` найден;
размер14844299bytes и SHA256
`a151dec0af22f25ac6ec8ad75927cf2889644284061fd7e35a129c67e836b797`
сверены с соседним `result.json` и совпали. Checkpoint не десериализовался,
совместимость текущего runtime и real-data forward не проверялись. Сохранение
артефакта позволяет начать подготовку demo без нового pretraining. Это fallback
к предпочтительному temporal checkpoint, не новый выбор победителя.

Зафиксировать manifest: checkpoint SHA/size, role(model-only/full), source
commit+dirty diff, model/schema/targets, train split, selection step, C2
transposition behavior, package versions и license/source binding.
Проверить наличие байтов и совместимость metadata без запуска модели; полная
загрузка и real-data forward принадлежат отдельному gate. Наличие locator или
локального model file не доказывает сохранность удалённой копии.

Исторические главы/результаты сохраняются. Known stale track pitch summaries
при транспозиции не исправляются внутри старого continuation. Новый masked
input contract получает свою версию; любые исправления требуют отдельного
source/run identity, а не заявления об эквивалентности старому run.

## Критерии продолжения и остановки

Сначала сравнивать frozen readouts при одинаковых данных, масках, head budget,
selection rule и evaluation groups; это измеряет полезность готовых
представлений, не причинный эффект предобучения архитектуры. Считать цену всей
проверки, forward counts и wall-time отдельно от updates/draws.

Если C3 не добавляет полезности относительно C2/raw-summary при заданном бюджете,
закрыть **C3-specific преимущество для этого readout**, сохранить лучший простой
baseline. Нулевой результат masked frozen decoder может объясняться новым
mask-domain shift; допускает один ограниченный partial-finetune test, но не
автоматическое полное обучение. Бюджеты: E1/E2/E5 в [D7](07-budget-gates.md).
