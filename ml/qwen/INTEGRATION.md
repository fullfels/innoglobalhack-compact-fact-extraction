# ML bridge → существующий Pipeline (часть T20)

Проверяемая база: FS snapshot `640dc29e6fc49869110fd291e3d445210d9d0aeb` (`fs/deep-integration-audit`). В старом main lifecycle ещё нет. Этот ML PR не меняет shared contracts, DE normalizers, API или HTTP backend.

`QwenBridge(engine)` преобразует CallInput → generated facts → существующий CandidateBatch. Input allowlist сохраняет original idx/speaker/text и started_at, gold не принимает. Неизвестное время вызывает QWEN_UNRESOLVED_TIME; механического started_at fallback нет. Speaker/source mismatch вызывает ошибку вместо переписывания роли. Quote выравнивается в той же исходной реплике; перенос к другой реплике запрещён. JSON duplicate keys/NaN/partial output и context/output overflow — явные ошибки, не успешные пустые facts.

Независимые synthetic tests:

```text
python -m pytest -q ml/qwen/test_bridge.py
```

Тесты не загружают веса и не измеряют качество модели. Архивная диагностика в `ml/experiments/2026-11-14-qwen35-diagnosis/diagnosis.json`: raw10 уже содержат по1факту,83–114output tokens при max4096. Это свидетельство поведения генерации, а не parser slicing. Причинная роль prompt не доказана без ablation.

## Factory для FS

```text
MWS_PIPELINE_FACTORY=ml.qwen.runtime:factory
MTS_MODEL_DIR=/models/export
MTS_MODEL_DEVICE=cpu
```

Factory возвращает общий Pipeline с ManagedAdapter. Loader один раз проверяет export manifest/hashes, загружает pinned base+adapter/tokenizer, делает structural startup smoke. Lifecycle и health остаются у FS. Serving работает local_files_only=True: сначала отдельно разместить base weights точной revision в HF cache, adapter/tokenizer/prompt/catalog в export. Для GPU установить MTS_MODEL_DEVICE=cuda и CUDA PyTorch; dependencies `ml/qwen/requirements.txt` дополняют FS dependencies.

Текущий FS Docker image ещё должен включить ML module/dependencies. FS может добавить `ml/` в образ или mount-код; этот PR Dockerfile не редактирует. Base/checkpoints/каталог/корпус и private raw logs остаются вне Git.

`model_manifest.json` создаётся реальным training/export job и требует artifact_state=trained_adapter_reload_verified, quality gates passed, base revision и hashes. Не существует запасного fake checkpoint. До передачи такого артефакта реальная загрузка не объявляется готовой.

## Необходимые DE решения

- `temporal_resolver(raw_fact, call)` может вернуть только обоснованную абсолютную дату либо ошибку/None. Неизвестное историческое время не заменять датой звонка. Original timezone сохраняется.
- `confirmation_resolver(candidate, call)` возвращает тот же Fact с проверенными confirmation/evidence_turns. Он должен семантически связать принятие с предложением, проверить отрицания и альтернативы. Любая клиентская реплика не считается подтверждением.
- Без resolver agreement остаётся unknown с warning и отфильтровывается общим Pipeline. Это известное ограничение, не решённая confirmation-задача. Сырые и serving-метрики в эксперименте должны считаться отдельно.
- DE catalog/time/state/polarity/grounding stages подключаются FS в установленном порядке. Согласованные object.id между звонками не изобретаются bridge.

Требуются реальные checkpoint reload, HTTP/Docker acceptance и CPU/GPU benchmark. Успешный fake-engine unit test их не заменяет. Обучающий пилот передаётся отдельным T09 PR; основной T10 только после измеренного успешного пилота.
