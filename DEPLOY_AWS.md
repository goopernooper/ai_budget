# Deploy to AWS Lightsail / EC2 (Docker)

This guide uses an Ubuntu VM and Docker Compose.

## 1) Create an Instance

- Launch an Ubuntu instance (Lightsail or EC2).
- Open port **8000** in the firewall/security group.

## 2) Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

Log out and back in to apply group changes.

## 3) Deploy the App

```bash
git clone <your-repo-url>
cd ai-finance-assistant
```

Create a persistent data folder:

```bash
mkdir -p data
```

Run with Docker Compose:

```bash
docker-compose up --build -d
```

Your app will be available on:

```
http://<server-ip>:8000
```

## Optional: Nginx Reverse Proxy

If you want port 80/443:

```bash
sudo apt-get install -y nginx
```

Example Nginx site config:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then reload Nginx:

```bash
sudo nginx -t
sudo systemctl restart nginx
```

## Notes

- SQLite is fine for demos; for production, use Postgres and update `DATABASE_URL`.
- If you use Postgres, no persistent disk is required for the DB, but keep `/data` for model artifacts.
