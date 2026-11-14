# Команда и статусы

Owner/FS: fullfels. ML: QFa1. DE: Eduard-Sadykov — username подтверждён пользователем, существующий write проверен GitHub API 2026-11-14. Доступ не расширялся, повторные приглашения не отправлялись. Invitation не равен активному доступу. Историческая настройка: GITHUB_SETUP_REPORT.

QFa1 и Eduard-Sadykov имеют подтверждённый write. CODEOWNERS отражает ML и DE paths, canonical normalizers — src/normalizers. Shared src/contracts проверяют FS и DE/ML интеграторы по PR template, даже без автоматического назначения. CODEOWNERS сам не защищает ветку.

Source backlog: [tasks.csv](tasks.csv) и [backlog](06_BACKLOG.md). Состав/DoD в CSV, оперативный статус в Issues. Синхронизация по точному [Txx] и Task ID, без duplicates/сброса human статусов; github_task_map.json сохраняет ссылки. Плановые сроки не означают фактическое выполнение.

Milestones: Contracts & Data; Mock E2E; Real Model Integration; Model Experiments; Final Evaluation & Release. Labels area/priority/type — фильтры; status:ready/in-progress/in-review и blocked — workflow, closed(completed) — Done. Projects только дополнительный вид, если права доступны.

T16/T17/T18 закрыты после проверки кода, 66 тестов, manual browser report и успешного remote CI. T02 требует review DE/ML; T19 остаётся частично выполненной. Не закрывать задачи только по отчёту: сначала проверить код/тесты и commit. Остальные задачи не становятся Done от создания каталогов.

DE: invitation→clone→area:data→T03–T05; scripts/data для будущих transforms, src/normalizers для normalizers. ML: invitation→clone→area:ml→T06–T09, затем SFT/bridge; ml/README. FS: принимать PR, следить за shared schema и CI, T19/T20. Недоступные normalizers/веса не подменяются фиктивными модулями.

[CONTRIBUTING](../CONTRIBUTING.md), [README](../README.md).

Первый real E2E и ожидаемые артефакты: [INTEGRATION_HANDOFF](INTEGRATION_HANDOFF.md). Реальная проверка GitHub: [GITHUB_SETUP_REPORT](GITHUB_SETUP_REPORT.md).
