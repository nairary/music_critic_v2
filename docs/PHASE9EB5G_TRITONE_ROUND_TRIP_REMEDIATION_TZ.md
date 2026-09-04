# ТЗ: Phase 9E-B5G — tritone physical inverse remediation

## Основание

Phase 9E-B5F установила статус `implementation_or_contract_defect` для
исполняемой B5A-транспозиции. Для `shift_pc=6` физический представитель равен
`+6`; вычисленный как `(-shift_pc) mod 12` inverse снова равен 6 и повторно
выбирает `+6`. Raw pitch проходит `p -> p+6 -> p+12`, тогда как semantic
tritone mapping является involutive.

Это ТЗ не реализуется в B5F. Оно задаёт границы отдельного remediation PR.
Phase 9E-B5G реализует его отдельным additive contract
`AnalysisGNNDirectedTransposition@1.0.0`.

## Цель

Развести pitch-class identity и направленный физический сдвиг так, чтобы:

- forward policy сохранила замороженную 12-PC орбиту и явный tritone choice;
- inverse физического `+6` был однозначно `-6`;
- semantic spelling policy для shift-PC 6 осталась явно версионированной;
- raw graph, targets, masks, IDs, routing и topology проходили полный
  TRAIN/VALIDATION round-trip;
- checkpoint и C1 training result не переписывались задним числом.

## Принятое решение

ADR-113 выбирает immutable token с двумя полями:

```text
DirectedTransposition(shift_pc, signed_semitones)
signed_semitones % 12 == shift_pc
```

Canonical forward продолжает использовать B5A `SIGNED_BY_SHIFT_PC`, включая
`(6,+6)`. Метод `inverse()` сохраняет физическое направление и возвращает для
него `(6,-6)`. Raw graph использует `signed_semitones`, а semantic targets —
`shift_pc`. Несовместимые значения отклоняются структурированной ошибкой.

Рассматривались следующие варианты:

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

## Acceptance и результат

Remediation принимается только если все допустимые directed forward/inverse
пары возвращают raw graph и targets в исходное состояние, runtime совпадает с
versioned contract, старое B5D evidence остаётся помечено историческим broken
contract fingerprint, и `ready_for_soft_augmentation` всё ещё не выставляется
до отдельного checkpoint/per-shift или нового экспериментального решения.

B5G подтверждает ноль directed round-trip failures на всех 17 484
TRAIN/VALIDATION парах, включая 1 439 eligible shift-6 пар, и неизменность
всех 20 000 исторических C1 forward draws. Формулировка «broken B5D/C1
augmentation» отклонена: broken был только способ восстановить физический
inverse из одного pitch class. B5H отдельно разрешает новый C2 full-orbit
эксперимент; `ready_for_full_orbit_training=true`, но сам CUDA run не входит в
repository PR.
