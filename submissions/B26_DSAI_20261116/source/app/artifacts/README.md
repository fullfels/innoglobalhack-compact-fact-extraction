# Model artifacts

Веса не входят в Git. Разместите их в `artifacts/models/<model_id>/` или задайте MWS_MODEL_DIR для read-only mount /models. Mount сам по себе не подключает модель.

Manifest: model_id, base revision, adapter/merged, logical parameter count, dtype/quantization, tokenizer/chat template, hashes файлов, training config+seed, dataset version, prompt/normalizer hashes, dependencies и license/provenance. Передача через согласованное приватное хранилище; LFS/Releases — после проверки размера, квот и разрешения распространения.

Подключение через ModelResources и InjectedAdapter: [контракт](../docs/INTEGRATION_V1.md). Реального Qwen bridge пока нет. Последнюю стабильную версию сохранять отдельно.
