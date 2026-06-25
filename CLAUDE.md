# Wemby-GM — Project Map

## Structure

```
nba analyst/
├── .env              # ANTHROPIC_API_KEY (git-ignored)
├── .gitignore
├── CLAUDE.md         # This file
├── app.py            # Trade analyst agent — schemas, agentic loop, tests
├── api_checker.py    # API health, model freshness, and latency checker
├── wemby_gm.db       # Auto-created SQLite session store (git-ignored)
└── api_checks.db     # Auto-created SQLite check history (git-ignored)
```

## Dependencies

```bash
pip install anthropic pydantic python-dotenv pytest
```

## Shell Commands

### Execute the application
```bash
python app.py
```

### Run tests
```bash
pytest app.py -v
pytest api_checker.py -v
```

### Run the API checker (single check)
```bash
python api_checker.py
```

### Run the API checker in watch mode (polls every 60 s)
```bash
python api_checker.py --watch
```

### Run the API checker — raw JSON output
```bash
python api_checker.py --json
```

### Resume a prior session by ID
```bash
python -c "import asyncio; from app import run_trade_agent; asyncio.run(run_trade_agent('SAS', 'IND', session_id='<your-session-id>'))"
```

## Key Design Notes

- Model: `claude-3-5-haiku-latest`
- Local mock roster in `MOCK_PLAYERS` — no external data API required
- Session state persisted in `wemby_gm.db` (SQLite); resume via `session_id`
- CBA 125 % salary-match rule enforced locally; violations trigger one Haiku self-correction pass
- Final payload always includes `confidence_score` + `calibration_logic` in `_meta`
