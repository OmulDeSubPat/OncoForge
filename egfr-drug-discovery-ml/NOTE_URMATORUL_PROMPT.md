# Nota pentru urmatorul prompt

Actualizat la 2026-03-30.

## Ce a ramas ciudat sau deschis

1. Bundle-ul frontend pentru vizualizarea 3D este mare.
   - `apps/oncoforge-ui` foloseste `3dmol` pentru viewer-ul molecular 3D.
   - `npm run build` trece, dar Vite raporteaza un chunk de aproximativ `575 kB` pentru `3Dmol` si avertizeaza despre `eval` in pachetul vendor.
   - UI-ul functioneaza, dar urmatorul prompt bun ar fi optimizarea acestui viewer prin lazy loading mai agresiv sau split suplimentar pe rute/panouri.

2. Temporal split-ul a fost reparat la nivel de date, dar performanta ramane slaba.
   - Dupa backfill-ul ChEMBL, `rows_with_year` a urcat la `27209 / 32271`, iar moleculele cu informatie temporala au urcat la `15225 / 16133`.
   - Cu toate acestea, in `reports/model_performance_summary.json` split-ul temporal refacut are in continuare `RMSE 1.6969` si `R2 -1.0931`.
   - Asta sugereaza ca problema ramasa este una reala de generalizare in timp, nu doar de lipsa a anilor.

3. ExCAPE ramane in mare parte fara ani utili.
   - Repair-ul temporal a completat bine ChEMBL prin `document_chembl_id`, dar `excape_chembl20` si `excape_pubchem` sunt in continuare fara acoperire temporala practica.
   - Daca vrem un temporal benchmark si mai defensabil, urmatorul pas ar fi o strategie explicita pentru sursele fara ani sau o analiza separata pe subset-uri datate.

4. Metadata de balans pe surse pentru split-ul temporal raman limitate la nivel de molecula agregata.
   - Dataset-ul procesat folosit la training nu are `source_dataset` per rand, ci agregari precum `source_datasets`.
   - Din acest motiv, `temporal_split` poate separa bine intervalele de ani, dar nu poate raporta un balans de surse foarte informativ pe artefactul agregat actual.

5. Localizarea in romana nu este inca uniforma in toate artefactele user-facing.
   - `reports/studii_ablatie/rezumat_studii_ablatie.md` este in romana.
   - Dar `reports/isef_project_summary.md` si o parte din notebook/plots raman predominant in engleza.
   - Daca vrem aliniere stricta cu contextul din root, urmatorul pas trebuie sa fie localizarea completa a sumarului si a graficelor principale.

6. `run_generation_benchmark_suite.py` emite un `PerformanceWarning` Pandas.
   - Nu blocheaza rularea si rezultatele sunt generate corect.
   - Totusi, helper-ul de backfill produce un DataFrame fragmentat si merita curatat pentru robustete.

## Ce a fost deja reparat

- Temporal backfill pentru ChEMBL cu cache in `data/interim/chembl_document_years.csv`
- Provenienta anului si scor de incredere in artefactele curate
- RL readiness funnel prin restaurarea trasabilitatii din metadatele de actiune
- Propagarea `adaptive_action_prior` in `generation_benchmark_suite.csv`
- Reproducibilitate minima in root si manifest
- Suita de 4 studii de ablatie in `reports/studii_ablatie/`
