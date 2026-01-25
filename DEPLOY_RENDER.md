# Deploy to Render (Docker)

Render can run the app directly from the Dockerfile.

## Steps

1. Push the repo to GitHub.
2. In Render, create a **Web Service** and select **Docker**.
3. Set the **Root Directory** to the repo root (if asked).
4. Attach a **Persistent Disk** at `/data` (required for SQLite + models).
5. Add environment variables:

- `DATABASE_URL=sqlite:////data/app.db`
- `MODELS_DIR=/data/models_registry`
- `UPLOAD_DIR=/data/uploads`
- `OPENAI_API_KEY` (optional)
- `DEV_MODE=false`

6. Deploy. Render will build the Docker image and run it automatically.

## Notes

- SQLite is fine for demos. For production, use Postgres and point `DATABASE_URL` to your managed DB.
- If you switch to Postgres, you can remove the persistent disk requirement.

## Start Command (if not using Docker)

If you choose the Python environment instead of Docker:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 app.main:app
```
