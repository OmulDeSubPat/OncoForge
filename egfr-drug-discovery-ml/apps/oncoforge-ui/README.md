# OncoSynth UI

Modern React + TypeScript + Vite + Tailwind frontend for the OncoSynth EGFR drug-discovery workspace.

## Start the stack

```bash
cd "D:\ONCS 2026\egfr-drug-discovery-ml"
python -m uvicorn src.gui.oncoforge_api.app:app --host 127.0.0.1 --port 8000

cd "D:\ONCS 2026\egfr-drug-discovery-ml\apps\oncoforge-ui"
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000` by default. If you need another backend URL, set `VITE_ONCOSYNTH_API_BASE`.

## Build

```bash
npm run build
```

## What is inside

- Top control bar with start, stop, step, import, and export actions
- Left multi-agent panel with visible credit assignment
- Center molecule workspace with atom selection and molecule switching
- Right metrics panel with pIC50, QED, SA, physchem, and radar chart
- Bottom timeline, RL monitor, and candidate library

## What is wired to real data now

- Session start, stop, reset, and refresh through FastAPI
- Live polling from `reports/gui_live/<session>`
- Multi-agent contribution cards from the selected molecule
- 2D and 3D molecular rendering from RDKit + MolBlock payloads
- RLVR reward and penalty traces derived from worker artifacts
- Ranking table and molecule detail linked to the real generated library
