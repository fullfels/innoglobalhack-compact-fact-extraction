# ML Engineer workspace

Обучение пока не реализовано. `configs/` — train/inference configs; `training/` — будущий SFT/export; `evaluation/` — offline runner и неизменный официальный evaluator; `experiments/` — компактные manifests/results, большие приватные runs в ignored runs/.

Serving bridge подключается к InjectedAdapter в src/pipeline/adapters.py; JSON-схемы только в src/contracts/models.py. Primary cleaning принадлежит DE. [Контракт](../docs/INTEGRATION_V1.md), [стратегия](../docs/07_ML_STRATEGY.md), [три конфигурации](../docs/08_MODEL_EXPERIMENTS.md), [сравнение](../docs/09_COMPARISON_PROTOCOL.md), [benchmark/manifest](../docs/11_BENCHMARKING.md), [веса](../artifacts/README.md).

Обычный CI без моделей/GPU. Real acceptance отдельно: загрузить экспортированный артефакт → bridge → DE normalizers → output gate → HTTP на согласованном real fixture → full-val с hashes → CPU/GPU benchmarks. scripts/smoke.py проверяет MOCK и не является проверкой качества real-модели. Не создавайте фиктивный training loop или вторую Fact schema.
