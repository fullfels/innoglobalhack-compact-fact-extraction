# B26 DSAI — исходники CompactFactExtraction

ЭТО АРХИВ КОДА. Он не содержит веса, готовый Docker-образ и START_WINDOWS.cmd. Для запуска без сборки скачайте отдельный B26_DSAI_CompactFactExtraction.zip, распакуйте его целиком и прочитайте его README.md.

## Структура кода

- app/: исходники единого FastAPI-сервиса, UI, адаптера Qwen и постобработки.
- app/Dockerfile.release: Dockerfile НАСТОЯЩЕГО приложения с PyTorch и Qwen bridge.
- app/Dockerfile: унаследованный MOCK-профиль, не использовать для итоговой модели.
- app/requirements.runtime.lock и app/requirements.lock: Python-зависимости; Torch устанавливается отдельно через Docker build argument.
- app/compose.release.yaml: параметризованный release-профиль; без переменных окружения не запускается.
- training_and_evaluation/: код и блокноты этапов обучения, полного оценивания и ресурсного пилота.

## Сборка выбранного CPU-образа из исходников

Из папки app/:

```sh
docker build -f Dockerfile.release --build-arg TORCH_VERSION=2.14.1 --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t compactfact-qwen3-v23:catalog-v1 .
```

Для сборки нужен интернет. Сборка не скачивает базу Qwen/адаптер и не запускает обучение. После сборки используйте compose.yaml и models/ из полного комплекта: из его корневой папки `docker compose -p mts-compactfact -f compose.yaml up -d --no-build --wait --wait-timeout 300`.

Если вместо готового compose используется app/compose.release.yaml, заполните все обязательные MWS_* переменные по app/RELEASE_README.md. Пути к базе и export должны указывать на реально распакованные models/base и models/export. Веса не должны попадать в Git.

Для воспроизведения обучения нужны соответствующие замороженные payloads из training_packages полного комплекта и GPU. В учебных блокнотах указаны нужные им пакеты. Один этот архив исходников не является набором весов/обучающих данных. Исторические README в подпапках описывают прежние этапы; главным руководством по структуре этого ZIP является данный файл.

## Текущий результат и ограничения

Выбрана Qwen/Qwen3-0.6B (revision c1899de289a04d12100db370d81485cdf75e47ca) + LoRA +500. Действующий factory: ml.qwen.serving:factory. Нормализатор только тарифный, tariff-catalog-v1; календарная коррекция выключена. Итоговый CPU-профиль — float32, GPU-оценка — FP16.

На всех 383 официальных звонках после нормализации: P=0.6736, R=0.4363, temporal=0.3260 (n=497), state=0.4575 (n=623), grounding=0.8534. Temporal/state пороги не пройдены; temporal ниже границы рассмотрения 0.4206. Все критерии выполненными не заявляются. Официальный evaluator не менялся. Полные сырые результаты и ресурсные протоколы находятся в reports/ полного комплекта.

Это частная передача организаторам. Публикация в GitHub, загрузка на облачный диск и отправка сообщения организатору автоматически не выполнялись.
