# Настройка командного репозитория — фактический отчёт

Репозиторий: https://github.com/fullfels/innoglobalhack-compact-fact-extraction ; owner fullfels; private; default main. Репозиторий создан в существующем проекте, без удаления компонентов или переписывания истории (история до настройки отсутствовала). Дата: 13 ноября 2026 по времени команды.

## Сохранение и интеграция

Исходный mock MVP и подробные документы сохранены. Новые изменения приложения, появившиеся параллельно настройке, проверены чтением и локальными тестами; код приложения этой задачей не переписывался. README и .gitignore расширены для командного использования; исходный CI не дублировался. Добавлены contribution/PR/issue templates, CODEOWNERS, data/ML/artifact инструкции и идемпотентный importer backlog.

Исполняемая схема только src/contracts/models.py. Реальные Qwen/DE normalizers не реализовывались. Shared Extractor/Normalizer и ModelResources/ModelDescriptor используются через существующий pipeline; перед real acceptance требуется фактический артефакт и bridge. Точный текущий контракт сверять по коду: INTEGRATION_V1 и IMPLEMENTATION_REPORT содержат исторический отчёт первого MVP и не заменяют новые regression tests.

## Доступ

fullfels — owner/admin. QFa1 — invitation 343520563 принят; GitHub collaborators API подтверждает активный write. Доступ WRITER к приватному Project также назначен через API. Data Engineer — username неизвестен, invitation не отправлен. Три активных участника пока не подтверждены. CODEOWNERS содержит fullfels и принятого QFa1: ML paths и shared adapter/contracts. Контракты требуют review затронутых DE/ML участников.

## Backlog / Projects

Источник 45 задач — неизменный docs/tasks.csv. Импорт scripts/sync_github_backlog.py сохраняет ID, описание, role owner, reviewer, priority, dependencies, DoD, estimate, deadline и milestone; повторный запуск не перезаписывает существующий human статус. GitHub assignee только действующим участникам. Итоговая карта — docs/github_task_map.json после импорта.

После отдельной подтверждённой пользователем авторизации workflow/project создан приватный Project https://github.com/users/fullfels/projects/1 и сохранён layout Board. Шесть статусов: Backlog, Ready, In Progress, In Review, Blocked, Done; поля Role (DE/ML/FS) и Priority (P0/P1/P2). Доска связана с репозиторием. Поля статуса для фильтра: status:ready / status:in-progress / status:in-review / blocked / closed.

## Проверки

- 26 исходных тестов прошли до изменений; затем 66 тестов текущего checkout прошли (включая параллельно добавленные regression tests).
- compileall src/scripts — PASS.
- Docker build — PASS; старт отдельного контейнера, health, POST extract с реальным HTTP — PASS. Финальный образ innoglobalhack-final-check пересобран после исправлений; scripts/smoke.py по HTTP на 18090 — PASS.
- compose config — PASS. Рабочий compose deployment пользователя не изменялся.
- Относительные документационные ссылки — без отсутствующих targets.
- Перед staging выполнен pattern scan токенов/private keys и размер файлов: совпадений секретов и файлов >5MB нет. Это ограниченная проверка, не гарантия отсутствия любых секретов.
- Raw dataset, PDF кейса, model weights, .env, caches, .venv и private logs не включаются в Git.

Remote CI commit 65e4ee9: success, check contracts-and-docker. Подтверждённый run: https://github.com/fullfels/innoglobalhack-compact-fact-extraction/actions/runs/43505148492 . Зависимости, pytest, compileall, Docker build/start/HTTP smoke — все шаги успешны.

Branch protection не включена. Автоматический approval review первоначально отклонил обязательный approving review до принятия invitation QFa1 из-за риска блокировки merges; позднее invitation был принят, этот конкретный риск устранён. Более безопасная настройка только запрета force push/delete (без обязательных PR/review) была проверена отдельно: GitHub вернул HTTP 403 «Upgrade to GitHub Pro or make this repository public». Тариф/visibility не менялись. PR/review/no-force/delete-main пока командная договорённость; это не техническая защита. После принятия второго участника и при доступном тарифе включить 1 approving review, разговоры resolved и уже существующий contracts-and-docker check.

Импортировано 45 Issues, повторный importer создал 0 новых. Пять milestones, 19 проектных labels (плюс штатные GitHub). T16/T17/T18 закрыты по проверенному коду, тестам, Docker и remote CI; остальные 42 открыты. T02 требует DE/ML review; T19 частично реализована, не Done.

## Работа команды

DE: принять invitation, clone, area:data и T03–T05, read data/README.md. ML: принять invitation QFa1, clone, ml/README.md, T06–T09 и общий bridge. FS: main принимает PR только после reviewer+CI; shared contracts согласовываются с интеграторами, не менять схемы в отдельных каталогах.

Для повторного импорта: `python scripts/sync_github_backlog.py` (dry-run), затем с аутентифицированным gh `python scripts/sync_github_backlog.py --apply`. Не хранить GH_TOKEN в repo; текущая сессия CLI использует временный конфиг вне проекта. На другом компьютере установить официальный gh и выполнить gh auth login либо использовать свой SSH/HTTPS доступ.


## Итоговые ссылки и checkout

- Main: https://github.com/fullfels/innoglobalhack-compact-fact-extraction/tree/main
- Issues: https://github.com/fullfels/innoglobalhack-compact-fact-extraction/issues
- Board (private): https://github.com/users/fullfels/projects/1
- Начальный commit: 65e4ee9c25d4bd9b76d9f1e9c9dff4b898fc9f93. Новые отчёты и карта Issues добавляются отдельным commit без переписывания этого root commit.
- Origin: https://github.com/fullfels/innoglobalhack-compact-fact-extraction.git
- Локальная параллельная ветка fix/review-p1-integration сохранена; main интегрирует изменения обычным fast-forward.

Фильтры доски: `role:DE`, `role:ML`, `role:FS`, `priority:P0`; фильтры Issues — area labels/assignee. До username DE его Issues остаются без assignee, с owner=DE в body. Новые scopes были подтверждены пользователем стандартным device flow; credentials находятся вне проекта и не включены в Git.
