# Deploy to Railway (Docker)

Railway can deploy directly from the Dockerfile.

## Steps

1. Push the repo to GitHub.
2. Create a Railway project and add your GitHub repo.
3. Railway detects the Dockerfile automatically.
4. Set environment variables:

- `PORT=8000`
- `DATABASE_URL=sqlite:////data/app.db` (if using a volume)
- `MODELS_DIR=/data/models_registry`
- `UPLOAD_DIR=/data/uploads`
- `OPENAI_API_KEY` (optional)
- `DEV_MODE=false`

## Storage Notes

- Railway volumes are optional. If you need persistence for SQLite and model artifacts, attach a volume at `/data`.
- If you cannot use volumes, switch to Postgres and use a managed database URL.

## Start Command (if not using Docker)

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 app.main:app
```
