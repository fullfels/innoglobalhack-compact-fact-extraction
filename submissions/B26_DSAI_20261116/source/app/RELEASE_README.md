# Qwen3-0.6B v23 → существующий FastAPI

Это инструкция к новой локальной сборке. UPSTREAM_README.md и унаследованные docs описывают историческое состояние команды; актуальный статус этого пакета находится в соседнем `../README_START.md`. Исходный API и frontend расширены, второй HTTP backend не создавался.

## API

- `POST /extract` — один транскрипт с метаданными, ответ `{prediction, meta}`.
- `GET /health` — ready/failed и явное is_mock.
- `GET /model/info` — model_id, revision/hash, dtype/context и manifest_id.
- `/` — страница с вводом, фактами, источниками и временными интервалами.

В запросе допустимы call_id, started_at, turns и разрешённые необязательные метаданные. Поле facts не принимается сервером. JSON для evaluator — только prediction; envelope с meta не подставляется вместо prediction. Неуспешная генерация/ошибка источника не превращается в успешный пустой список. При частичном отбрасывании отдельных фактов сохраняются предупреждения; их учитывать в анализе.

## Реальный запуск в уже подготовленном окружении

Нужны Python с совместимым PyTorch, `requirements.runtime.lock`, base из фиксированной revision и выбранный export. Torch специально не ставится автоматически из этого requirements, чтобы не заменить CUDA-сборку Colab.

Пример переменных для Linux/Colab после завершения GPU-работы (подставить реальные пути и значения):

```bash
export MWS_MODE=real
export MWS_PIPELINE_FACTORY=ml.qwen.serving:factory
export MTS_MODEL_DIR='/private/models/export'
export MTS_MODEL_BASE_PATH='/private/models/base'
export MTS_MODEL_DEVICE=cpu
export MWS_MODEL_ID='<из model_manifest.json>'
export MWS_ARTIFACT_SHA256='<sha256 model_manifest.json>'
export MWS_TORCH_VERSION='<фактически установленная версия torch>'
export HF_HUB_OFFLINE=1
python -m scripts.release_preflight --check-only
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

`packaging_verified` не означает model_loaded. После запуска проверить health ready и настоящий /extract. CPU использует FP32, CUDA — FP16; native численные результаты каждого устройства нужно измерять. Поддерживаются точные значения cpu/cuda; cuda:N явно отклоняется, пока для него не измерен корректный профиль.

Один экспорт включает адаптер, не базовые веса. `prepare_base.py` использует pinned Qwen/Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca. Материализованная база не зависит от symlink-кеша и проверяется по base_identity.json. Никакие токены HF или персональный HF_HOME с credentials в комплект сдачи не копируются.

## Docker

В `.env.release` или окружении задать MWS_TORCH_VERSION, MWS_EXPORT_DIR, MWS_BASE_DIR, MWS_MODEL_ID, MWS_ARTIFACT_SHA256. MWS_TORCH_VERSION — проверенная доступная версия, не слово latest; её нужно выбрать по фактическому runtime и проверить native. Значение из Colab с CUDA-суффиксом нельзя вслепую устанавливать из CPU wheel index.

```bash
docker compose --env-file .env.release -f compose.release.yaml config --quiet
docker compose --env-file .env.release -f compose.release.yaml build
docker compose --env-file .env.release -f compose.release.yaml run --rm extractor --check-only
docker compose --env-file .env.release -f compose.release.yaml up -d
```

Для GPU требуется совместимая CUDA-версия Torch, соответствующий MWS_TORCH_INDEX_URL и настроенный Docker/NVIDIA runtime. Добавить `-f compose.release.gpu.yaml`. GPU-совместимость не выводится из наличия GPU в Colab; native запуск контейнера требуется отдельно. CPU index по умолчанию не превращается в CUDA-build автоматически.

База и export монтируются read-only; сервер один. Сначала проверить образ, затем сохранить его digest и полный `pip freeze` из контейнера. Основные версии зафиксированы, но граф транзитивных ML-зависимостей и точный Torch ещё нужно сохранить по реальной сборке. Не обещаем воспроизведение непроверенного image только по Dockerfile.

## Оценка и ресурсы

`ml.qwen.serving:evaluation_factory(config)` возвращает Pipeline с честно применёнными model_dir, base_path, device, precision, manifest_sha256 и model_id. Неизвестные ключи отклоняются. `benchmark_factory(config)` дополнительно загружает модель и возвращает существующий BenchmarkTarget с путями base+adapter, чтобы не считать только размер LoRA. Quantization допускается только на CUDA и требует установленных bitsandbytes/измерений; штатный HTTP factory использует выбранный FP16/CPU FP32.

Для официального383 в текущем эксперименте использовать уже выпущенный `validate_export.py`: он явно загружает неизменённый evaluator2.1 и каталог. Старый универсальный src.experiments runner оставлен как upstream-инструмент; прежде чем использовать его вместо v23-проверки, нужно явно проверить передаваемый каталог и точную методику. Сырые/normalized/serving результаты не смешивать.

В приложении используются правила v23, включая источник, ограниченную нормализацию дат и подтверждение договорённостей. Отдельные DE normalizers не подключены автоматически: добавление нового постпроцессинга изменяет систему и требует повторного сравнения. Предупреждение об отсутствии отдельного DE tuple не означает, что отключена проверка схемы или внутренние правила v23.

## Источники runtime-команд

- PyTorch: https://pytorch.org/get-started/locally/ — выбор сборки по устройству.
- HF: https://huggingface.co/docs/huggingface_hub/guides/download — snapshot_download и fixed revision.
- Docker: https://docs.docker.com/compose/how-tos/gpu-support/ — GPU reservation и требования к хосту.

Эти инструкции не являются подтверждением успешного Docker/native inference на данной машине.
