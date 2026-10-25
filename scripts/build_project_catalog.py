#!/usr/bin/env python3
"""Render navigation from existing experiment records, without executing experiments.

Only this navigation file is writable. Terminal records and the evidence ledger
remain owned by experiment_registry.py. No third-party dependencies are needed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = (
    ("hcm", "PLL / HCM: условная вероятность музыкальных событий",
     ("EXP-HCM",), "Как использовать локальный и полный контекст?",
     "[HCM-1](../HCM1.md), [полный контекст](../HCM_WHOLE.md). NLL/PLL, свободная генерация и человеческие предпочтения — разные проверки."),
    ("critic", "Парный и аспектный критик", ("EXP-CRITIC",),
     "Можно ли обучить сравнение музыкальных вариантов на frozen C3?",
     "[Контракт критика](../CRITIC_V1.md). Искусственные правки и авторские примеры не заменяют независимую проверку человеческих предпочтений."),
    ("annotation", "Корпус и теоретическая разметка",
     ("EXP-CORPUS400", "EXP-MULTIGENRE", "EXP-DENSE-HARMONIC", "EXP-TONAL", "EXP-SAINTS", "EXP-JHT", "EXP-JAZZ", "EXP-HT-POP", "EXP-TARGET-SLOT", "EXP-HARMONIC-DATA", "EXP-RAW-TIME", "EXP-RESEARCH-DILEMMA", "EXP-CHANGES"),
     "Какие источники, события и теоретические поля подтверждены?",
     "[Данные](../DATASETS.md), [контракт](../DATA_CONTRACT.md). Полнота проверки, полнота меток и семантическое качество учитываются отдельно; неизвестное сохраняет маску."),
    ("authoring", "Авторские произведения и музыкальные вмешательства",
     ("EXP-LLM-GENRE", "EXP-GENRE-INTERVENTION", "EXP-MUSIC-INTERVENTION", "EXP-HARMONIC-REPLACEMENT"),
     "Какие контролируемые примеры и изменения пригодны для исследования?",
     "[Итог авторской кампании](LLM_GENRE_CAMPAIGN_FINAL_40_RU.md). Успех построения/рендера не равен музыкальному принятию."),
    ("c4", "C4: POP909 и перенос между корпусами", ("EXP-C4",),
     "Как работает обучение на POP909 и обратный перенос?",
     "[Обучение C4](C4_POP909_3090_20261005_RU.md), [перенос](C4_DILEMMA_TRANSFER_20261005_RU.md). Сравнения сохраняют источник, split и конкретные задачи."),
    ("c3", "Temporal / C3 / attention / relation gate",
     ("EXP-TEMPORAL", "EXP-C3", "EXP-3090-C3", "EXP-CLOUD-T4-C3", "EXP-POP-C3"),
     "Что меняют точные временные события и варианты обмена сообщениями?",
     "[Temporal](../TEMPORAL_SLICE_GRAPH.md), [обзор C3](C3_THREEPASS_FINAL_RU.md), [POP909](C3_POP909_FULL_20261001_RU.md). Исторические бюджеты и batch size различаются; это не единый рейтинг архитектур."),
    ("fusion", "Fusion / task attention / MMoE",
     ("EXP-FUSION", "EXP-TASKATTN", "EXP-MMOE"),
     "Помогает ли обмен между теоретическими задачами?",
     "[Карта исторических веток](../BRANCHES_AND_RELEASES.md). Незавершённые, остановленные и отрицательные попытки сохранены рядом с контролями."),
    ("baselines", "C0 / C1 / C2 и замороженный перенос",
     ("EXP-9EB5", "PLAN-9EB5", "EXP-POP-TRANSFER", "EXP-ROMAN"),
     "Какие контроли и преобразования данных поддержаны сохранёнными сравнениями?",
     "[Решения](../DECISIONS.md), [модели](../MODELS.md). Сопоставлять только совместимые данные, головы, seeds и бюджеты; baseline не выбирается этим каталогом."),
    ("ssl", "Самообучение и контекст", ("EXP-SSL", "EXP-CTX"),
     "Работают ли маскирование и перенос представлений без утечки целей?",
     "[SSL](../PHASE7A_SSL_BASELINE.md), [протокол сравнения](../PHASE8B2_COMPARISON_PROTOCOL.md). Реконструкция представлений не является PLL или оценкой эстетики."),
    ("runtime", "CPU / GPU / облако и производительность",
     ("EXP-9EP1", "EXP-CLOUD", "EXP-HARDWARE", "EXP-PILOT"),
     "Какие среды, ресурсы и способы исполнения были допущены?",
     "[Облачная история](../YANDEX_CLOUD_DEPLOYMENT.md). Инженерные gates и throughput не доказывают качество модели."),
    ("reporting", "Обзоры и сверка сохранённых результатов",
     ("EXP-ARTICLE", "EXP-ARCH", "EXP-POSTBREAK"),
     "Какие выводы можно восстановить из уже сохранённых свидетельств?",
     "[Полный ledger](../EXPERIMENT_LEDGER.md). Анализ сохранённых результатов не считается новым обучением."),
    ("unclassified", "Требуют классификации", (),
     "Какие новые семейства ещё не включены в ручную карту?",
     "Записи сохраняются здесь автоматически; категория — навигация, а не научное заключение."),
)

# Editorial summaries point to existing reports/decisions; no metrics or ranking
# are derived by this navigation renderer.
OUTCOMES = {
    "hcm": ("Сохранены запуски HCM-1/whole и development-проверки HCM-2; [отчёт сравнения](HCM2_RANDOM_3090_VALIDATION_20261001_RU.md) описывает разницу NLL на фиксированных масках, с различиями в истории обучения. Новый baseline этим обзором не выбирается.", "Польза полного контекста/памяти и качество свободной генерации требуют отдельных сопоставимых проверок."),
    "critic": ("[Отчёт перехода TRAIN002](CRITIC_TRAIN002_TRANSITION_20260927_RU.md) сохраняет родительский checkpoint и отдельное продолжение с оптимизацией. Прежние попытки не заменены новой записью.", "Обобщение сравнений на независимые человеческие предпочтения остаётся отдельной проверкой; статус живого обучения здесь неизвестен."),
    "annotation": ("Сохранены source-bound native/weak слои и отдельные reviews/releases. [Контракт](../DATA_CONTRACT.md) различает полную проверку произведения, известные targets и corpus-quota admission; частичное принятие не становится полной разметкой.", "Остаются семантические пробелы и непринятые попытки; рабочие статусы и маски должны сопровождать каждый release."),
    "authoring": ("[Итог кампании](LLM_GENRE_CAMPAIGN_FINAL_40_RU.md) сохраняет авторские произведения и их проверки. Контролируемые [вмешательства](MUSIC_INTERVENTION_PILOT_RU.md) исследуются отдельно.", "Готовый файл, renderer acceptance и человеческая оценка не взаимозаменяемы; перенос на реальные произведения требует проверки."),
    "c4": ("Сохранены [обучение на POP909](C4_POP909_3090_20261005_RU.md) и [обратный перенос](C4_DILEMMA_TRANSFER_20261005_RU.md). Разница между корпусами не служит выбором универсальной лучшей модели.", "Причины разницы переноса и качество на других источниках требуют отдельных контролей."),
    "c3": ("[Итог C3](C3_THREEPASS_FINAL_RU.md) описывает неодинаковые изменения по головам и разные batch/draw budgets; primary selection в этом отчёте сохраняется. [POP909-проверка](C3_POP909_FULL_20261001_RU.md) оформлена отдельно.", "Причинный эффект внимания/gate при равном бюджете и перенос между источниками не выводятся из общего списка запусков."),
    "fusion": ("Попытки fusion, task attention и MMoE сохранены вместе с остановками/контролями. [Решение интеграции](../BRANCHES_AND_RELEASES.md) исторически оставило temporal основой main, fusion — отдельной веткой.", "Совместимость и эффективность новых сочетаний с temporal требуют отдельной реализации и сравнения."),
    "baselines": ("[ADR-111](../DECISIONS.md#2026-09-02--adr-111-select-c0-and-defer-c1-after-the-paired-full-screen) фиксирует исторический выбор C0 и отсрочку C1 после paired screen; дальнейшая [интеграция temporal](../BRANCHES_AND_RELEASES.md) — отдельное решение.", "Нельзя переносить ранний выбор на другие данные, бюджет, seed или семейство HCM/critic без новой проверки."),
    "ssl": ("[Контракт SSL](../PHASE7A_SSL_BASELINE.md) фиксирует механические границы маскирования, resume и encoder transfer. [Протокол сравнения](../PHASE8B2_COMPARISON_PROTOCOL.md) сохраняется отдельно от этих проверок.", "Польза представлений для downstream и музыкального качества не устанавливается прохождением механических tests/gates."),
    "runtime": ("В [облачной истории](../YANDEX_CLOUD_DEPLOYMENT.md) сохранены загрузка/восстановление комплектов и проверки исполнения. CPU/GPU attempts и неудачные gates остаются в реестре.", "Для другой машины/версии необходим собственный admission; исторический gate не подтверждает текущее состояние устройства."),
    "reporting": ("Отдельные обзоры связывают сохранённые результаты с источниками. Их записи доступны ниже; они не заменяют исходные training/evaluation records.", "Недостающие первичные артефакты и несопоставимые условия сохраняются как ограничения; новых запусков этим индексом нет."),
    "unclassified": ("Автоматическая страховочная группа: ни одна неизвестная запись не исчезает из каталога.", "Добавить ручную карту нового семейства; научный итог до её составления неизвестен."),
}


def load_records(directory):
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if path.stem != record["experiment_id"]:
            raise ValueError(f"record identity mismatch: {path.name}")
        records.append(record)
    return records


def campaign_for(experiment_id):
    for campaign in CAMPAIGNS:
        if experiment_id.startswith(campaign[2]):
            return campaign[0]
    return "unclassified"


def cell(value):
    return " ".join(str(value).split()).replace("|", "&#124;")


def render_catalog(records, reports=()):
    records = sorted(records, key=lambda r: r["experiment_id"])
    counts = Counter(r["status"] for r in records)
    lines = [
        "# Эксперименты: карта исследовательских кампаний", "",
        "Автоматическая навигация по существующим записям; не новый анализ результатов.",
        "Статусы взяты из реестра: это не проверка живых процессов.",
        "Терминальные записи, старые конфиги и отрицательные результаты сохраняются неизменными.", "",
        "[Главная](../../README.md) · [Модели](../MODELS.md) · [Данные](../DATASETS.md) · [Полный ledger](../EXPERIMENT_LEDGER.md)", "",
        f"Всего записей: **{len(records)}**. " + "; ".join(f"{s}: {n}" for s, n in sorted(counts.items())) + ".", "",
        "| Кампания | Вопрос | Записей |", "|---|---|---:|",
    ]
    groups = {c[0]: [r for r in records if campaign_for(r["experiment_id"]) == c[0]] for c in CAMPAIGNS}
    for key, title, _, question, _ in CAMPAIGNS:
        lines.append(f"| [{title}](#{key}) | {question} | {len(groups[key])} |")
    for key, title, _, question, guidance in CAMPAIGNS:
        lines.extend(["", f'<a id="{key}"></a>', "", f"## {title}", "", question, "", guidance, ""])
        outcome, unresolved = OUTCOMES[key]
        lines.extend([f"**Сохранённый итог и решение:** {outcome}", "", f"**Открытый вопрос:** {unresolved}", ""])
        if not groups[key]:
            lines.append("Записей нет.")
            continue
        lines.extend(["<details>", f"<summary>Попытки и свидетельства: {len(groups[key])}</summary>", "",
                      "| Запись | Статус в реестре | Вид | Название |", "|---|---|---|---|"])
        for record in groups[key]:
            identity = record["experiment_id"]
            lines.append(f"| [{identity}](records/{identity}.json) | {cell(record['status'])} | {cell(record['kind'])} | {cell(record['title'])} |")
        lines.extend(["", "</details>"])
    lines.extend(["", "## Все текстовые отчёты", "", "<details>",
                  f"<summary>Отчёты в этом каталоге: {len(reports)}</summary>", ""])
    for report in sorted(reports):
        lines.append(f"- [{report}]({report})")
    lines.extend(["", "</details>", "", "## Обновление", "", "Из корня репозитория:", "", "```bash",
                  "python3 scripts/build_project_catalog.py", "python3 scripts/build_project_catalog.py --check", "```", "",
                  "Новый запуск сначала регистрируется по [протоколу](../EXPERIMENT_LOGGING.md).",
                  "Этот генератор только читает записи и не меняет их статусы, метрики или хеши.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "docs/experiments/records")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/README.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    reports = [p.name for p in args.records.parent.glob("*.md") if p.name != "README.md"]
    rendered = render_catalog(load_records(args.records), reports)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"stale catalog: {args.output}")
            return 1
        print("project catalog is current")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
