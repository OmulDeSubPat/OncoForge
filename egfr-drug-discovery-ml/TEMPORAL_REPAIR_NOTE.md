# Temporal repair note

- The raw ChEMBL EGFR export does not reliably carry a usable `year` column even though the fetch query requests it.
- Year repair is therefore being done additively via `document_chembl_id` and the cached document-year map in `data/interim/chembl_document_years.csv`.
- Next prompt should confirm whether we also want to backfill explicit year provenance fields for the non-ChEMBL sources, because those are currently heuristic defaults in the merge step.
- The temporal split is still source-skewed until the repaired ChEMBL years are regenerated into the processed artifacts.
