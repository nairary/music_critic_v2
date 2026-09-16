# Задача 1 — Контекстное понимание C2

Цель: выяснить, какая часть ошибок относится к локальному распознаванию созвучия, использованию контекста и решению quality-классификатора. Подготовить воспроизводимые проверки для следующих абляций.

Прочитать README этого каталога; docs/experiments/C0_C2_120K_REVIEW.md; docs/experiments/POP909_CL_C0_C2_FROZEN_REVIEW.md; docs/POP909_CL_FROZEN_EVALUATION.md; docs/POP909_CL_TRANSFER.md. Обзор доступен для чтения в основном checkout: outputs/research/music-understanding-20260908/report-source.md.
Разрешённые сохранённые артефакты брать по locator+hash из records, с основным checkout как read-only evidence root; outputs не копируются в source snapshot.

1. Сверить, что уже есть в сохранённых predictions/confusions, не запускать повторный forward ради имеющихся величин. Отдельно разобрать false D7 на POP (90 TP из 871 предсказаний в native C2) и cadence/metrical_strength на Dilemma. Не приписывать event probabilities отчёту, который содержит только counts. Если новая диагностика существенна, сначала зарегистрировать её в собственном worktree.
2. Реализовать отдельный диагностический harness с явными TRAINVAL membership, checkpoint/model/class/feature hashes и неизменным default evaluator. Определить local-only baseline на chroma/interval/raw rhythm features с обучением исключительно на TRAIN и сравнением на тех же eligible candidates. Отдельно ограниченный контекст у C2 — интервенция, не честно обученный local baseline; отразить возможный distribution shift.
3. Определить несколько семейств точных synthetic контекстных пар: одинаковое созвучие с разным гармоническим окружением; подготовка/разрешение диссонанса; сохранённая вертикаль с изменённым движением голосов; перенумерация/разбиение дорожек при тех же звучащих нотах. Не считать любую перестановку ухудшением. Functional labels после изменения контекста не переносить автоматически; явно отделить mechanically valid, hypothesis-only и musician-reviewed пары.
4. Тестировать masks, exact timing, raw-only boundaries, сохранение заявленных инвариантов, отсутствие перемешивания произведений/сплитов, детерминизм и ручные metric oracles. Подготовить повторно используемый bounded CLI/протокол; не запускать большой перебор моделей или гиперпараметров.
5. Итог: компактный отчёт «установлено / проверяемая гипотеза / неизвестно», per-class precision/recall/support и confusion, спецификация contextual suite и пакет следующего ограниченного запуска. На реальных данных uncertainty считать по произведениям/компонентам. TEST закрыт; POP не использовать для выбора улучшенной модели.

Предпочтительно отдельные modules/tests/scripts с context_diagnostics в имени; избегать изменений общего corrected forward. Новые experiment IDs: EXP-CTX-C2-*. Результаты и protocol — docs/experiments/ или docs/workstreams/2026-09-08/ в собственном worktree.

