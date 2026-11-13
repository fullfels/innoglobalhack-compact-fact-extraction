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
