# Nybble

A personalized model that learns from your data and keeps learning as you use it.

**Coming soon.** https://nybble.ink

## Web app

A FastAPI app in `app/` that shows a gallery of MCP connectors (Gmail, Messages, Slack, GitHub and more). Click a card to connect it, then use **Finetune** at the top to pick connected sources to train on.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000. Add or edit connectors in `app/mcps.py`.

Connections and finetune jobs are held in memory for now: OAuth sign-in is simulated and `/api/finetune` only queues a placeholder job.
