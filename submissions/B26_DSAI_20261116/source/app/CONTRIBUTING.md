# Совместная разработка

```bash
git clone https://github.com/fullfels/innoglobalhack-compact-fact-extraction.git
cd innoglobalhack-compact-fact-extraction
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.lock
pytest -q
git switch -c data/train-ready-v1
```

Нужен принятый invitation в private repo. Выберите существующий T-ID в Issues; owner записан в body даже без assignee. docs/tasks.csv — источник состава backlog; Issues — оперативный статус. Не создавайте второй независимый backlog.

Короткие ветки: data/..., ml/..., fs/... или feat/..., fix/..., docs/.... Обычные изменения не делать прямо в main: это командное правило действует и без серверной защиты. Один task на PR, автор не единственный reviewer.

```bash
git add path/to/changed-file
git diff --cached
git commit -m "Txx: describe resulting behavior"
git fetch origin
git merge origin/main
# Разрешите конфликт по смыслу, git add <file> и git commit.
# При сомнении git merge --abort сохраняет исходную ветку.
pytest -q
git push -u origin HEAD
```

Откройте PR в main, заполните template, свяжите Issue, запросите другого участника. CI должен реально пройти. Изменения shared contracts требуют DE/ML review затронутых компонентов, updates fixtures/OpenAPI/INTEGRATION_V1. Force push и переписывание общей истории запрещены.

FS перед merge проверяет tests+Docker job, review и инструкции интеграции. Недоступную protection не представлять действующей; ограничения тарифа и приглашений в GITHUB_SETUP_REPORT. Срочный review допускается с причиной в PR. CODEOWNERS не заменяет защиту main.

Данные: [data](data/README.md). Веса: [artifacts](artifacts/README.md). Роли и статусы: [workflow](docs/TEAM_WORKFLOW.md). Real acceptance: [ML](ml/README.md).

## Ownership и серверные ограничения

| Зона | Ответственный | Пути |
|---|---|---|
| FS / Tech Lead | @fullfels | src/contracts, src/pipeline, src/api, frontend, Docker/Compose, .github |
| ML | @QFa1 (активный write) | src/ml, ml, configs/train |
| DE | @Eduard-Sadykov (активный write проверен API) | src/data, src/normalizers, data, scripts/data |

Canonical normalizers: src/normalizers согласно 03_ARCHITECTURE; старое название src/normalization в ранних onboarding notes не требует второго pipeline. Существующие каталоги не перемещать без согласования. Shared contracts — FS owner, DE/ML reviewers по затронутому interface. CODEOWNERS содержит подтверждённые @fullfels, @QFa1 и @Eduard-Sadykov; автоматический CODEOWNERS review не заменяет peer review автора.

Branch protection для private repo текущего аккаунта возвращает GitHub HTTP 403 (требуется Pro либо другой поддерживающий private protection тариф). Репозиторий остаётся private. Поэтому обязательные CI/review пока командное правило, не серверная гарантия. Целевая конфигурация: `.github/branch-protection.json`. После доступности функции admin применяет её через GitHub API к main и проверяет GET protection; до этого не объявлять main защищённой.

Первый real E2E: [integration handoff](docs/INTEGRATION_HANDOFF.md). Не создавать фиктивные collector/runner/loader для закрытия задач.
