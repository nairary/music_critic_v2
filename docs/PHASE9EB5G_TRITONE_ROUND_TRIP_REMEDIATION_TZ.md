# ТЗ: Phase 9E-B5G — tritone physical inverse remediation

## Основание

Phase 9E-B5F установила статус `implementation_or_contract_defect` для
исполняемой B5A-транспозиции. Для `shift_pc=6` физический представитель равен
`+6`; вычисленный как `(-shift_pc) mod 12` inverse снова равен 6 и повторно
выбирает `+6`. Raw pitch проходит `p -> p+6 -> p+12`, тогда как semantic
tritone mapping является involutive.

Это ТЗ не реализуется в B5F. Оно задаёт границы отдельного remediation PR.

## Цель

Развести pitch-class identity и направленный физический сдвиг так, чтобы:

- forward policy сохранила замороженную 12-PC орбиту и явный tritone choice;
- inverse физического `+6` был однозначно `-6`;
- semantic spelling policy для shift-PC 6 осталась явно версионированной;
- raw graph, targets, masks, IDs, routing и topology проходили полный
  TRAIN/VALIDATION round-trip;
- checkpoint и C1 training result не переписывались задним числом.

## Обязательное решение до кода

Новая ADR должна выбрать один versioned API, не меняя значение старого B5A
контракта молча. Допустимые направления для рассмотрения:

1. отдельный `signed_semitones`/`direction` аргумент для физического inverse;
2. versioned directed-transform token, содержащий shift PC и signed delta;
3. отдельная inverse primitive, проверяющая исходный directed transform.

Нельзя использовать только shift-PC как идентификатор физического inverse для
tritone. Нельзя менять глобальное `SIGNED_BY_SHIFT_PC[6]` на `-6`: это лишь
перенесёт ту же неоднозначность на обратное направление и изменит выполненный
B5D эксперимент.

## Реализация

- Создать новый contract/version; старый B5A evidence остаётся immutable.
- Изменить только detached TRAIN view path и его явный inverse diagnostic API.
- Fail closed проверять MIDI range отдельно для forward и directed inverse.
- Не выполнять modulo wrap, octave folding, OOV fallback или split repair.
- Обновить B5C runtime binding только после принятия новой ADR.
- Добавить миграционное различие между историческим C1 B5D и новым профилем;
  не называть новый профиль прежним C1 без нового experiment ID.

## Проверки

- Все 12 shifts на source-free graph, включая крайние MIDI 0/127.
- Полные 17 484 TRAIN/VALIDATION record-shift diagnostics.
- Exact shift-zero identity.
- Exact categorical and `1e-6` continuous round-trip.
- B5A semantic mappings, all 20 heads, masks, class IDs, entity IDs, relations,
  topology, rational onset IDs and provenance.
- Direct transform versus B5C runtime batch regression.
- TEST loader/targets/metrics remain false/zero.
- Required repository `full-suite` after merge.

## Non-goals

Remediation не обучает модель, не запускает новые seeds, не реализует soft
augmentation/curriculum, не меняет heads/loss/weights/sampler/dataset/split и
не открывает TEST. После исправления сначала требуется повторный correctness
audit. Любой новый training screen оформляется отдельным последующим ТЗ.

## Acceptance

Remediation принимается только если все допустимые directed forward/inverse
пары возвращают raw graph и targets в исходное состояние, runtime совпадает с
versioned contract, старое B5D evidence остаётся помечено историческим broken
contract fingerprint, и `ready_for_soft_augmentation` всё ещё не выставляется
до отдельного checkpoint/per-shift или нового экспериментального решения.
