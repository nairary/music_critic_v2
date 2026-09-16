# Проверка механики обучения — 2026-09-05

Проверена developer-версия на исходном commit
`da8c9dcd999a4b522f87f9a20bf3506ad26497b7`. Новые изменения этой задачи — только
два файла синтетических тестов и документация. Код модели, loss, trainer,
замороженные C0/C1/C2 и experiment records не изменялись. Main остаётся на
`f0e0886161c9f67786011508006db61c803c1752`.

Вывод ограничен исправностью вычислений: на обычных непустых графах не найден
дефект, делающий обучение фиктивным. Модель может выучить зависимость ответов
всех 18 голов от входа. Это не подтверждение музыкальной семантики, качества
критика, пользы транспозиции или готовности полного обучения на PDMX.
Живой C0, SSH, реальные корпуса, обученные checkpoints и TEST не открывались.

## Новые постоянные проверки

[test_training_learning_audit.py](../../../tests/experiments/test_training_learning_audit.py)
использует текущую архитектуру fixed128, настоящий raw forward, post-prediction
alignment, corrected loss и `attempt_applied_update`. Только веса классов и
данные искусственные: два маленьких графа с разными pitch patterns, два
противоположных класса для каждой головы и uniform all-supported weights.
Проверяются 80 обновлений со стандартным B5C optimizer/scheduler; это короткий
fixture-бюджет, не воспроизведение расписания C0-120k.

| Проверка | Результат |
| --- | --- |
| Суммарный loss на искусственных примерах | 3.7118868828 → 0.0079506151 |
| Правильные ответы | 6/36 → 36/36; каждая из 18 голов различает оба примера |
| Loss каждой головы | Уменьшился у всех 18 |
| Обновления | 80/80 применены; logits и gradients конечны |
| Local GNN, pooling, Transformer, fusion, BiGRU и каждая голова | Параметры изменились |
| Перестановка примеров | Ответ следует за входом, порядок batch его не задаёт |
| Одинаковые входы и противоречивые labels | Одинаковые logits; обе метки одновременно угадать нельзя |
| Исходные графы и sidecars | Не изменились |

Последние два контроля исключают решение постоянным приоритетом класса,
по позиции в batch или чтением label. Но задача остаётся намеренно простой:
она не проверяет гармоническое рассуждение, перенос на новые произведения
или качество всех настоящих target annotations.

[test_theory_ssl_learning_audit.py](../../../tests/ssl/test_theory_ssl_learning_audit.py)
добавляет четыре проверки fixed128 theory/SSL-связки:

1. Согласованное изменение скрытой ноты с перестроением графа не меняет online
   embeddings и decoder predictions побитно, но меняет full-view target
   (L2 5.6663394) и reconstruction loss (delta 0.01681554). Скрытый pitch не
   просачивается через производные признаки в проверенном случае.
2. При unsupported labels всех 18 голов theory term и его gradients отсутствуют;
   SSL продолжает работать с заданным коэффициентом.
3. Если обязательный SSL note objective недоступен, theory продолжает работать
   с исходным коэффициентом. Недоступная цель не превращается в нулевую метку.
4. Последовательность mixed → theory-only → SSL-only → обе цели недоступны
   проверяется с AdamW и ненулевым weight decay. Очистка gradients через
   `set_to_none=True` исключает обновление неактивной ветви по старому gradient.

| Режим | Изменились shared tensors | SSL tensors | Theory tensors | Inactive tensors |
| --- | ---: | ---: | ---: | ---: |
| Mixed | 470 | 34 | 93 | 0 |
| Theory-only | 452 | 0 | 93 | 0 |
| SSL-only | 458 | 34 | 0 | 0 |
| Обе цели недоступны | 0 | 0 | 0 | 0 |

В последней строке optimizer state также неизменен. Это проверка рецепта
вызывающего кода теста: production epoch loop/scheduler для новой WS2-связки
этой задачей не создаётся. Различия числа обновлённых tensors между режимами
объясняются маршрутами objective; shared identity не означает, что любая цель
обязана использовать каждый параметр.

## Подтверждённые ограничения

### Часть конечных ветвей не участвует в corrected theory loss

231 680 из 3 661 936 параметров, около 6.3%, не получают gradient от текущих
18 голов. Новая регрессия фиксирует точное множество из 25 tensors и требует,
чтобы остальные параметры оставались подключены к loss на полном fixture.

| Конечная ветвь | Tensors | Parameters |
| --- | ---: | ---: |
| `context_encoder.fusion.fusions.song` | 6 | 49 664 |
| `context_encoder.fusion.fusions.track` | 6 | 66 048 |
| `context_encoder.fusion.fusions.bar` | 6 | 66 048 |
| `sequence_decoder.bar_*` | 7 | 49 920 |

Причина структурная: [финальные fusions](../../../src/music_critic/models/hierarchy.py)
возвращают отдельные song/track/bar outputs; [BiGRU](../../../src/music_critic/models/onset_bigru.py)
возвращает обновлённые bar outputs; [corrected heads](../../../src/music_critic/experiments/analysisgnn/corrected_model.py)
читают только beat/onset/note. Это избыточные конечные ветви, а не отключение
coarse pooling, глобального Transformer или всей иерархии. Note-головы получают
контекст до BiGRU. Автоматически удалять эти слои или менять frozen architecture
во время C0/C2-сравнения нельзя; отдельное улучшение decoder требует новой версии.

### Известный empty-candidate дефект остаётся в исторических API

В [corrected_training.py](../../../src/music_critic/experiments/analysisgnn/corrected_training.py)
loss (строка 494), per-head metrics (965) и accumulator (1099) выбирают
`candidate_indices` до применения validity mask. Если note/onset кандидатов нет,
но sidecar содержит соответствующие masked/unaligned rows, sentinel index 0
приводит к `RuntimeError: index_select(): self indexing axis dim should be positive`.
Raw inference при этом работает. Ошибка может возникнуть даже при наличии
пригодной harmonic supervision.

Это ранее известный случай, уже обработанный в
[WS2 loss wrapper](../../../src/music_critic/ssl/theory_retention_harness.py).
Новое покрытие не доказывает, что такой граф встречается в фактическом C0.
Это явное падение на краевом входе, не обнаруженное молчаливое искажение обычных
обновлений. В будущей production-интеграции нужны согласованные guards для
loss и metrics; исторические fingerprints в этом аудите сохранены.

### SSL-механика не гарантирует сохранение теории или отсутствие collapse

Параметры backbone действительно общие, target representations отсоединены,
а train/eval flags восстанавливаются. Но после SSL-only обновления меняется
backbone, поэтому ответы сохранённых theory heads тоже могут измениться.
Сохранённые веса голов не гарантируют сохранённую точность.

Малый cosine reconstruction loss сам по себе не исключает collapse: постоянные
согласованные представления могут иметь малую ошибку. Anti-collapse statistics
в [objective.py](../../../src/music_critic/ssl/objective.py) являются диагностикой,
а не доказательством отсутствия collapse или отдельным штрафом. Для будущего
запуска нужны retained-theory metrics, разброс/различимость embeddings и сравнение
с исходным backbone. Нынешний аудит не запускает этот эксперимент.

## Дополнительная проверка существующего пути

- Raw-only prediction и маски/unsupported классы подтверждены тестами. Standard
  weighted CE использует `sum(w * CE) / sum(w)`, затем средние доступных голов
  и `primary + 0.25 * auxiliary`; оснований менять формулу не найдено.
- Проверены изоляция произведений в Transformer/BiGRU и padding. В отдельном
  synthetic full-plus-empty batch максимальное отличие logits от одиночного
  примера составило 4.47e-7.
- Синтетические три обновления batch=2 и `1 → save/load → 2` совпали точно по
  model fingerprint, losses, gradient norms и scheduler state. Optimizer step=3,
  sampler draws=6. Checkpoint создан тестом во временном каталоге и удалён.
  Совпадение C0/C2 budget/scheduler проверяется отдельно существующим control test.
- C0 runner собирает графы и sidecars из одного упорядоченного списка; ошибки
  перестановки в этом пути не обнаружено. Сам низкоуровневый aligner не проверяет
  piece identity, поэтому новые callers обязаны сохранять эту связь явно.
- Статический риск historical C2 resume: перед append JSONL не обрезаются до
  checkpoint boundary, поэтому после crash возможны повторные строки. Это не
  whole-run reproduction и не доказательство ошибки восстановления весов. Новый
  C0-120k runner имеет reconciliation; нынешний C2 artifact не исследовался.

## Воспроизводимость и следующий шаг

Совместная проверка 14 профильных модулей: **228 passed, 2 CUDA skips,
1 failed, 2 upstream warnings in 34.21s**, exit 1. Все шесть новых проверок
прошли. Это не полностью зелёный gate.

Не прошёл существующий
`tests/models/test_onset_bigru_decoder.py::test_decoder_configuration_is_structured_and_mlp_contract_is_bit_exact`:
hardcoded hash начального состояния старой MLP-модели не совпадает с локальным.
Expected `e86b577b4c91c55d66f68f22d5faceb078910c0219a179ea8590dbc7fe6f36d4`,
actual `7f97e02a6ad053c111eae2517e8dd99321e8dd2a974a23624e47158efafecb84`.
Отдельный запуск того же теста даёт тот же actual на `da8c9dc` (0.14s) и
`f0e0886` (0.21s) с прежними relevant model/test/dependency файлами. Source diff
в `src/music_critic/models`, этом тесте, pyproject.toml и uv.lock пуст.
Это не новая source regression и не эффект порядка нового набора тестов.
В свежем процессе повторные seed=1701 инициализации совпадают; default/explicit
MLP state и outputs также совпали. Конкретная причина межсредового расхождения
не установлена; её нельзя приписывать определённому релизу Torch.
Expected hash и зависимости не менялись, исходный тест остаётся failed.

Среда: Python 3.12.14, macOS 26.2 arm64, Torch 2.13.0, PyG 2.8.0.post1,
CPU FP32, OMP/MKL по одному потоку. Точная команда совместной проверки:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/music-critic-v2-integration-venv/bin/python -B -m pytest -q -s -p no:cacheprovider \
  tests/experiments/test_training_learning_audit.py \
  tests/ssl/test_theory_ssl_learning_audit.py \
  tests/experiments/test_phase9eb5c_corrected_model.py \
  tests/experiments/test_phase9eb5c_corrected_training.py \
  tests/experiments/test_phase9eb5d_full_training.py \
  tests/experiments/test_phase9eb5h_full_orbit_profile.py \
  tests/experiments/test_phase9eb5i_orbit_matched_control.py \
  tests/models/test_hierarchy.py \
  tests/models/test_onset_bigru_decoder.py \
  tests/ssl/test_theory_warmstart.py \
  tests/ssl/test_theory_retention_harness.py \
  tests/ssl/test_decoder_objective.py \
  tests/experiments/test_workstream_synthetic_integration.py \
  tests/test_repository_contract.py
```

Точная команда и полный log находятся в
`/private/tmp/music-critic-training-audit-20260905/gate-command.json` и `gate.log`.
Log: 4583 bytes, SHA256
`4b6756c6e6f7ac83a2c912e36bdfab6136d0e6dc11f5a2a33453b66812db63cd`.
SHA256 новых тестов:

- `test_training_learning_audit.py`: `8b8192db2d6e0542b7b285d41ac9a1b9dfe59d1249a874254f4775bdadee592f`.
- `test_theory_ssl_learning_audit.py`: `8accc9b3f0d1bea947615879f6b1d1952f7a5a7a93e46c5a9fc173c3845c7dcc`.

Это обычные unit/contract tests, не научный experiment и не весь repository suite.
Новые тесты отдельно дали 2 passed и 4 passed, по 2 upstream Torch JIT warnings.
Первый запуск нового learning fixture был отклонён validator из-за неправильно
заданных synthetic piece IDs/references; исправлен только тестовый fixture.
Эта ошибка не была дефектом модели и не использована как научное свидетельство.

Следующий инженерный шаг перед обучением на новых корпусах — production runner
со строгой связью raw/sidecar, guards для пустых графов и unavailable objectives,
проверенными шагами optimizer, checkpoint/resume и раздельными diagnostics.
Выбор C0/C2 и input-transposition версии делается после подтверждённого сравнения;
исправность перечисленной механики от положительного результата C2 не зависит.
Деградации остаются будущим средством оценки сигналов критика; переход к
контрастному обучению или заявление о готовности PLL этим аудитом не принимается.
Legacy не открывался и не переиспользовался; новые зависимости не добавлялись.
