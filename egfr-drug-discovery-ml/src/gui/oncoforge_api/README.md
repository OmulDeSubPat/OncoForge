# OncoSynth API

API local FastAPI pentru interfata OncoSynth.

## Pornire

```bash
cd "D:\ONCS 2026\egfr-drug-discovery-ml"
python -m uvicorn src.gui.oncoforge_api.app:app --host 127.0.0.1 --port 8000
```

## Endpoint-uri principale

- `GET /api/health`
- `GET /api/dashboard?session_name=sesiune_curenta&limit=120`
- `POST /api/control/start`
- `POST /api/control/stop`
- `POST /api/control/reset`

## Ce expune

- overview de sesiune
- panou agenti si fluxuri intre agenti
- detaliu pentru molecula selectata
- timeline pe runde
- monitor RLVR
- biblioteca de candidati
- log worker si surse pentru costul estimat
