# MWS CompactFactExtraction — прототип

FastAPI + Pydantic v2, статический HTML/CSS/JS без Node-сборки. **По умолчанию MOCK**, обученной модели нет. Подготовленные синтетические примеры сопоставляются буквально; неизвестный текст → 422 `MOCK_EXAMPLE_NOT_SUPPORTED`, без фиктивного успешного пустого ответа.

## Запуск

Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.lock
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

UI: http://127.0.0.1:8000 · OpenAPI: http://127.0.0.1:8000/docs

```bash
pytest -q
python scripts/smoke.py http://127.0.0.1:8000
docker compose up --build
```

Docker smoke: `python scripts/smoke.py http://127.0.0.1:8000`. После проверки `docker compose down`. Контейнер работает непривилегированным пользователем, веса не скачивает. Порт compose 8000; измените mapping при конфликте.

## API

`POST /extract` принимает JSON звонка (см. примеры UI), возвращает `{prediction, meta}`. Только `prediction` сериализуется строкой в официальный predictions JSONL. `GET /health`: process_alive/model_state/ready/is_mock. `GET /model/info`: режим, nullable metadata, лимиты. Схемы доступны в `/openapi.json`.

Вход не принимает `facts` и неизвестные поля. 413: тело >256 KiB; 422: неверный JSON/схема/неизвестный mock-пример; 429: занят единственный inference worker; 503: отсутствующие артефакты/ошибка inference/output. Логи не содержат транскриптов и входных ошибок. Даты без timezone не переводятся в UTC.

`MWS_MODE=real` без внедрённого адаптера даёт readiness 503 и extraction 503. Молчаливого fallback в mock нет. Публичная идентичность берётся из ModelDescriptor адаптера; Settings/MWS_MODEL_ID не могут переименовать mock в настоящую модель. Model path и tokenizer передаются программно в ModelResources, не публикуются API.

## Интеграция

См. [docs/INTEGRATION_V1.md](docs/INTEGRATION_V1.md). Реальные веса, DE normalizers и семантическое качество пока не реализованы. Чужие `src/data`, `src/normalizers`, `src/ml`, evaluator и dataset не создавались и не изменялись.

Единый проверочный snapshot и запуск после передачи ML-артефактов: [первый real inference, 10/383 calls и A/B/C](docs/FIRST_REAL_INFERENCE.md). Для real API используется `python -m scripts.real_smoke --help`; обычный `scripts/smoke.py` проверяет только mock. Текущий статус PR/review и повторных проверок: [INTEGRATION_STATUS.md](docs/INTEGRATION_STATUS.md).


## Командный репозиторий

Private: https://github.com/fullfels/innoglobalhack-compact-fact-extraction . Владелец/FS: fullfels; ML: QFa1; DE: Eduard-Sadykov (существующий write-доступ проверен). ML bridge находится в draft #53, real checkpoint и DE правила ещё не подключены. Актуальные blockers и отдельные readiness checks: [QWEN_ACCEPTANCE_STATUS](docs/QWEN_ACCEPTANCE_STATUS.md).

```bash
git clone https://github.com/fullfels/innoglobalhack-compact-fact-extraction.git
cd innoglobalhack-compact-fact-extraction
```

Архитектура: CallInput → Extractor → ordered DE Normalizers → Output validation → FastAPI → статический frontend. Raw→validation/preprocessing→train-ready→SFT→export пока ожидает реализации DE/ML; интерфейсы уже общие. Входы не содержат gold facts.

| Каталог | Назначение |
|---|---|
| src/contracts | Единственная исполняемая JSON-схема |
| src/pipeline | Extractor/Normalizer protocols, orchestration, output gate |
| src/api, frontend | Backend и демонстрационный интерфейс |
| data | Правила получения данных, manifests, независимые fixtures |
| ml | Будущие configs/training/evaluation/experiment manifests |
| artifacts | Инструкция хранения весов вне Git |
| tests, scripts | Python/JS regression tests, mock и real HTTP smoke |
| .github | Существующий CI, ownership и PR/issue templates |
| docs | Спецификация, план, handoff и backlog |

## Навигация

[Старт](docs/START_HERE.md) · [Требования](docs/01_CASE_ANALYSIS.md) · [Спецификация](docs/02_TECHNICAL_SPEC.md) · [Архитектура](docs/03_ARCHITECTURE.md) · [Контракт v1](docs/INTEGRATION_V1.md) · [Роли](docs/04_TEAM_RESPONSIBILITIES.md) · [Roadmap](docs/05_ROADMAP.md) · [Backlog](docs/06_BACKLOG.md) · [CSV](docs/tasks.csv) · [ML](docs/07_ML_STRATEGY.md) · [Эксперименты](docs/08_MODEL_EXPERIMENTS.md) · [Сравнение](docs/09_COMPARISON_PROTOCOL.md) · [Тесты](docs/10_TESTING_AND_METRICS.md) · [Ресурсы](docs/11_BENCHMARKING.md) · [Риски](docs/12_RISKS.md) · [Demo](docs/13_DEMO_AND_PITCH.md) · [Неизвестное](docs/14_OPEN_QUESTIONS.md).

[CONTRIBUTING](CONTRIBUTING.md) · [Командный workflow](docs/TEAM_WORKFLOW.md) · [Данные](data/README.md) · [Веса](artifacts/README.md) · [ML handoff](ml/README.md) · [Результат реализации MVP](docs/IMPLEMENTATION_REPORT.md).

Исходный датасет и материалы организаторов получать отдельно по data/README; в Git только код, документация, hashes и независимые fixtures. Секреты, private logs и checkpoints не публиковать. Model volume не заменяет реализацию ML bridge. Подробная настройка GitHub и ограничения — docs/GITHUB_SETUP_REPORT.md.


## Regression tests R01–R03

```bash
pytest -q                         # external evaluator/Docker checks may skip
MWS_DOCKER_TESTS=1 pytest -q        # enables Docker sentinel; official evaluator supplied separately
node --test tests/frontend.test.cjs # 5 UI controller cases, Node.js 22+
```

Node нужен только для тестов, frontend по-прежнему работает без сборки. Sentinel-тест копирует только tracked sources во временный каталог, добавляет фиктивные credentials/weights и проверяет их отсутствие в context/image; настоящие секреты не читает. CI запускает Python+Docker и JS regression tests.

Detached inference failure логируется один раз с request_id/module/error_code/error_type, без сообщения исключения. Изменение диалога очищает результат; ответ начатого ранее запроса не отображается после изменения ввода. DE/ML contracts не менялись.
