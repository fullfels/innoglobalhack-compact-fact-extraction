# Настройка командного репозитория — фактический отчёт

Репозиторий: https://github.com/fullfels/innoglobalhack-compact-fact-extraction ; owner fullfels; private; default main. Репозиторий создан в существующем проекте, без удаления компонентов или переписывания истории (история до настройки отсутствовала). Дата: 13 ноября 2026 по времени команды.

## Сохранение и интеграция

Исходный mock MVP и подробные документы сохранены. Новые изменения приложения, появившиеся параллельно настройке, проверены чтением и локальными тестами; код приложения этой задачей не переписывался. README и .gitignore расширены для командного использования; исходный CI не дублировался. Добавлены contribution/PR/issue templates, CODEOWNERS, data/ML/artifact инструкции и идемпотентный importer backlog.

Исполняемая схема только src/contracts/models.py. Реальные Qwen/DE normalizers не реализовывались. Shared Extractor/Normalizer и ModelResources/ModelDescriptor используются через существующий pipeline; перед real acceptance требуется фактический артефакт и bridge. Точный текущий контракт сверять по коду: INTEGRATION_V1 и IMPLEMENTATION_REPORT содержат исторический отчёт первого MVP и не заменяют новые regression tests.

## Доступ

fullfels — owner/admin. QFa1 — приглашён с write (invitation id 343520563), принятие проверяется отдельно. Data Engineer — username неизвестен, invitation не отправлен. Три активных участника пока не подтверждены. CODEOWNERS активирует только владельца до принятия приглашений. Контракты требуют review затронутых DE/ML участников.

## Backlog / Projects

Источник 45 задач — неизменный docs/tasks.csv. Импорт scripts/sync_github_backlog.py сохраняет ID, описание, role owner, reviewer, priority, dependencies, DoD, estimate, deadline и milestone; повторный запуск не перезаписывает существующий human статус. GitHub assignee только действующим участникам. Итоговая карта — docs/github_task_map.json после импорта.

GitHub Projects API недоступен текущему CLI token: отсутствует read:project. Доска не создана; Issues, milestones и labels — рабочая альтернатива. Новые account scopes автоматически не запрашивались. Поля статуса для фильтра: status:ready / status:in-progress / status:in-review / blocked / closed.

## Проверки

- 26 исходных тестов прошли до изменений; затем 64 теста текущего checkout прошли (включая параллельно добавленные regression tests).
- compileall src/scripts — PASS.
- Docker build — PASS; старт отдельного контейнера, health, POST extract с реальным HTTP — PASS. Первоначальная сборка предшествует параллельным изменениям приложения; финальная remote CI проверяет committed snapshot.
- compose config — PASS. Рабочий compose deployment пользователя не изменялся.
- Относительные документационные ссылки — без отсутствующих targets.
- Перед staging выполнен pattern scan токенов/private keys и размер файлов: совпадений секретов и файлов >5MB нет. Это ограниченная проверка, не гарантия отсутствия любых секретов.
- Raw dataset, PDF кейса, model weights, .env, caches, .venv и private logs не включаются в Git.

Remote CI и protection фиксируются ниже только после фактической проверки; локальная зелёная проверка не считается GitHub Actions успехом.

## Работа команды

DE: принять invitation, clone, area:data и T03–T05, read data/README.md. ML: принять invitation QFa1, clone, ml/README.md, T06–T09 и общий bridge. FS: main принимает PR только после reviewer+CI; shared contracts согласовываются с интеграторами, не менять схемы в отдельных каталогах.

Для повторного импорта: `python scripts/sync_github_backlog.py` (dry-run), затем с аутентифицированным gh `python scripts/sync_github_backlog.py --apply`. Не хранить GH_TOKEN в repo; текущая сессия CLI использует временный конфиг вне проекта. На другом компьютере установить официальный gh и выполнить gh auth login либо использовать свой SSH/HTTPS доступ.
