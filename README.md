# 🌿 CropIQ

AI-powered crop disease detection for African farmers. Snap a leaf, upload it,
get an instant diagnosis and a plain-language treatment plan — across 10 major
African crops (Maize, Cassava, Rice, Yam, Sorghum, Millet, Cowpea, Plantain,
Groundnut, Cocoa).

## Run locally (Windows / PowerShell)

```powershell
cd C:\Users\olatu\Documents\cropiq-dashboard
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py app.py
```

Open http://localhost:5000 → log in with **farmer / farmer123**.

## Deploy to Railway (< 5 min)

1. Push this folder to a GitHub repo.
2. On [railway.app](https://railway.app): **New Project → Deploy from GitHub repo**.
3. Railway auto-detects Python, installs `requirements.txt`, and runs the
   `Procfile` (`gunicorn`). No extra config needed — it injects `$PORT`.
4. Open the generated URL.

Or via CLI:

```powershell
npm i -g @railway/cli
railway login
railway init
railway up
```

## Swap in the real model

The diagnosis engine lives in `diagnose()` in `app.py`. It currently uses a
lightweight Pillow color analyzer so the app runs with zero heavy dependencies.
Replace the function body with your EfficientNet-B0 inference (see the `NOTE`
docstring inside it), and add `torch` + `timm` to `requirements.txt`.

## Demo logins

| Username | Password   | Role             |
|----------|------------|------------------|
| farmer   | farmer123  | Individual farmer |
| agro     | agro123    | Field agronomist  |
