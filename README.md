# Nybble

A personalized model that learns from your data and keeps learning as you use it.

**Coming soon.** https://nybble.ink

## Web app

A FastAPI app in `app/` that shows a gallery of MCP connectors (Gmail, Messages, Slack, GitHub and more). Click a card to connect it: connecting Messages or WhatsApp extracts its chats into a local SQLite database (`data/messages.db`, `data/whatsapp.db`; see `store.py`) straight away (see them under **Your data**, and use **Update data** in the card's popup to pull in new ones). Disconnecting deletes that source's database. Then use **Finetune** at the top to pick connected sources to train on.

Needs Python 3.9 or newer.

```bash
pip install -r requirements.txt
python -m app.main
```

Then open http://localhost:8000. The server only listens on this Mac (127.0.0.1) and refuses requests that come through any other hostname or from other websites. If you run uvicorn yourself, keep `--host 127.0.0.1`; never use `0.0.0.0`. Add or edit connectors in `app/mcps.py`.

Only Messages and WhatsApp can actually be connected; every other connector is marked **Coming soon** and can't be connected yet. Connections and finetune jobs are held in memory for now, Finetuning isn't built yet, so `FINETUNE_ENABLED` in `app/main.py` is `False`: the **Finetune** button is disabled and `/api/finetune` refuses jobs. Set it to `True` once training exists and the button queues jobs again.
