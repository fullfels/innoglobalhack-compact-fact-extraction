# Воспроизводимый benchmark ресурсов

Owner FS, adapter/config от ML, input strata от DE. Измерения приложением пока не выполнены. Результаты baseline по ресурсам отсутствуют.

## Среда

Один фиксированный CPU host и один фиксированный GPU host; по возможности CPU/GPU на одной машине. CPU report: модель CPU, sockets/cores, threads, RAM, ОС, governor/power policy, engine, precision. GPU: модель, count=1, VRAM, driver/CUDA, power limit, clock policy/MIG/MPS и sharing. Указать container digest, dependency lock, commit, model/catalog/tokenizer hashes, seed. На macOS ARM результаты CPU не переносить на x86, MPS не называть CUDA или CPU.

Три конфигурации должны быть доступны на каждом заявленном устройстве. Если выбранный quant engine не поддерживает CPU, не подставлять CPU-цифры другой модели как эквивалентные: отдельный CPU профиль (например, full precision) со своим ID и eval либо Plan B на поддержанном engine. Отсутствующие замеры null с reason, это пробел выполнения кейса, не нулевая стоимость.

## Порядок

1. После model process startup измерить load time отдельно от inference. Холодный старт 3 раза, явно указать cold process и состояние файлового кэша; не утверждать cold disk cache без контроля.
2. Прогреть 10 train-derived звонков с разной длиной, отдельно от измеряемых 383. CUDA synchronize до старта и после завершения model execution. E2E CPU wall timer охватывает preprocessing, model, normalizers и validation; HTTP latency отдельно включает serialization/queue/network.
3. Primary latency: concurrency=1,batch=1, все 383 val. Измерять wall time каждого звонка и весь run. Не запускать обучение/соседние workloads на том же GPU. При отсутствии выделенного hardware указать sharing как confounder.
4. Три повтора с одинаковым набором, чередовать порядок A/B/C: ABC, BCA, CAB. Результаты каждого повтора и сводка median-run; число quality runs и resource repeats различать. Если детерминированные predictions изменились — сохранить и исследовать, не усреднять незаметно.
5. Отдельный throughput профиль batch=4 (либо максимально допустимый фиксированный для всех), concurrency явно указан; не смешивать с batch=1 latency. Слишком большой runtime → общий фиксированный стратифицированный subset до запуска, например 100 calls по длинам, одинаковый для всех; quality остаётся на полном val, benchmark помечен subset.
6. Peak CUDA allocated/reserved сбросить перед измерением, фиксировать max_memory_allocated/reserved; NVML process/device peak с sampling interval, например 100 ms, отдельно; RSS process tree и CPU peak с измерителем, указать частоту. Process-level peak включает weights; incremental request allocation дополнительно, не вместо total.

## Формулы

mean=Σti/N; p50/p95 по сохранённому массиву с указанием interpolation метода. Throughput=N/wall_seconds×3600 calls/hour; токены/сек дополнительны, не замена звонков. Allocation GPU-seconds/1000 = Σ(GPU_count×allocated_wall_sec)/N×1000; при одном постоянно выделенном GPU это wall_seconds×1000/N, включая простои/CPU-фазы внутри запроса. Отдельно GPU kernel/event time не считать той же метрикой. GPU utilization×wall — другая приблизительная величина, не доля оплаченной GPU.

Если обещается «малая доля GPU», нужны подтверждённые MIG/MPS/allocator правила; batch=1 сам по себе не означает 0.1 GPU. CPU-only имеет 0 выделенных GPU-seconds, но это не доказательство бесконечного speedup или бесплатности: CPU time/latency/memory обязательны. Энергию и денежную экономию не оценивать без измерений/тарифов.

Disk weights = фактическая сумма необходимых inference файлов, отдельно tokenizer/config/base+adapter; shared base не забывать в B. Параметры считать логические, tied weights не дублировать, dtype/битность отдельно. Указать GB или GiB последовательно.

## Сравнение с baseline

Запросить у организаторов total/active parameters точного glm-5.3, engine/dtype/hardware, input/output token budgets, batching, GPU-seconds и протокол. Ratio params=Pbase/Pfinal; ratio GPU=(GPU-sec/call)base/(GPU-sec/call)final только при совместимом протоколе. Нет denominator — ratio=null и «соблюдение ресурса не подтверждено». Новая воспроизведённая большая модель — отдельная база, не подмена официальной. Чтение сохранённых baseline predictions не измеряет LLM.

## Файлы и графики

`resources_raw.csv`: run_id, repeat, call_id, length_bucket, input_tokens, output_tokens, windows, e2e_ms, model_ms, queue_ms, rss_bytes, cuda_peak_allocated, cuda_peak_reserved, nvml_peak, status. `resources.json`: mean/p50/p95, throughput, GPU-sec/1000, load times, peaks, conditions и missing reasons.

ML строит: temporal×peak memory; state×peak memory; Q×GPU-sec/call; latency×Q; disk weights×Q; grouped bar всех официальных метрик. CPU/GPU не смешивать на оси без device labels. На диаграммах точки A/B/C, error bars ресурсов, пороги и Pareto, одинаковые единицы; неизвестный baseline resource не рисовать в выдуманной координате. Для temporal рядом показывать recall/matched denominator. Выбор финала объяснить фактическими измерениями.

## Experiment manifest v1

Каждый run содержит `manifest.json`, immutable после завершения; missing поля null+reason. Внутри допустимы вложенные objects:

```json
{
  "schema_version": "1", "run_id": "A-eval-001", "model_id": "audit-pending",
  "checkpoint": {"base": null, "revision": null, "artifact_sha256": null},
  "training": {"config_path": null, "config_sha256": null, "seed": 42},
  "dataset": {"split": "val", "version": "raw-v1", "sha256": "fb0dd73bba4b642ca18538751ce4a64d40a569aa7edd1a2a2b595af3c6b6ca83", "n_calls": 383, "id_list_sha256": null},
  "inference": {"engine": null, "version": null, "dtype": null, "quantization": null, "batch": 1, "concurrency": 1, "do_sample": false, "max_new_tokens": 2048, "prompt_sha256": null, "postprocess_sha256": null},
  "hardware": {"cpu": null, "ram_bytes": null, "gpu": null, "vram_bytes": null, "driver": null},
  "evaluation": {"script_sha256": "dffec9b8b20c5a8a101a33a27b1672f2b6195f00ab60f021a1c14b31eeeacabd", "metrics_path": null},
  "resources": {"raw_path": null, "memory": null, "latency": null, "gpu_seconds_per_1000": null},
  "metrics": null, "predictions_sha256": null, "git_commit": null,
  "dependency_lock_sha256": null, "container_digest": null, "timestamp_msk": null,
  "status": "planned", "missing_reasons": {"metrics": "experiment not run"}
}
```

Этот пример описывает структуру и стадию planned, не выполненный эксперимент. Runtime и manifests версионировать в Git без весов и секретов; predictions/resources достаточно компактны для артефактного хранилища с SHA. Любое изменение prompt/catalog/postprocessing — новый run_id, а не перезапись старых метрик.
