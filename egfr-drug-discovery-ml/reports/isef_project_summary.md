# OncoForge ISEF Summary

## Project Goal
OncoForge is an AI-assisted lead-optimization pipeline for EGFR inhibitors.
The system does not claim to discover finished drugs; it prioritizes chemically plausible, high-potential candidates for downstream wet-lab validation.

## Upgraded Methodology
- A multi-agent scorer now separates potency, chemistry, safety, novelty and applicability-domain checks instead of relying on a single scalar reward.
- Verified reward is combined with anti-reward-hacking audits so suspicious molecules are penalized even if they exploit a proxy metric.
- Ranking now uses multi-objective percentiles, veto logic and diversity-aware post-filtering.
- Ensemble training is evaluated on both random and scaffold splits, then retrained on the full dataset for final inference.

## Why The Multi-Agent Design Helps
- `Potency agent`: predicts pIC50 with ensemble uncertainty.
- `Chemistry agent`: scores QED, SA, Lipinski pressure and descriptor sanity.
- `Safety agent`: checks PAINS and structural alerts.
- `Novelty/applicability agent`: balances novelty against the training distribution and marketed drugs.
- `Audit agent`: flags reward-hacking patterns such as highly potent but out-of-domain or unsafe structures.
- `Protected ranker`: compares a naive proxy score against a protected score and explicitly demotes suspicious molecules.

## Model Performance
- Dataset size: `10606` molecules
- Random split RMSE / R2: `0.6601573323003216` / `0.7469832830172807`
- Scaffold split RMSE / R2: `0.7409472016209232` / `0.683374484713541`

## Audit Diagnostics
- Audit pass rate: `0.15670375259287195`
- Audit review rate: `0.6273807278898736`
- Audit fail rate: `0.21591551951725438`
- Median reward hacking risk: `0.2`
- Mean audit demotion: `664.6265321516123` positions

## Top Ranked Training-Space Molecules
| smiles | predicted_pIC50 | QED | reward_hacking_risk | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1C)C(=O)NCC2 | 9.700256082232992 | 0.5102437252597349 | 0.0 | pass | 10.859704326919331 |
| C/C=C/C(=O)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.283392742529765 | 0.6685952655904649 | 0.0 | pass | 10.761128680627383 |
| CCOc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.64526754879117 | 0.4521900852216164 | 0.0 | pass | 10.704630204489868 |
| COCCOc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1C)C(=O)NCC2 | 9.52688747004589 | 0.5184077804223949 | 0.0 | pass | 10.63950210975158 |
| C=CC(=O)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.023116678771231 | 0.6927074866124346 | 0.0 | pass | 10.637120881267496 |

## Scored Marketed EGFR Drugs
| name | predicted_pIC50 | QED | reward_hacking_risk | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| Sunvozertinib | 8.376537608843067 | 0.7062024654769844 | 0.1 | pass | 9.504761891897225 |
| Osimertinib | 8.258191405295543 | 0.652925124852874 | 0.1 | pass | 8.852089711411924 |
| Gefitinib | 7.680156426793009 | 0.5065425549922015 | 0.1 | pass | 7.771298439513362 |
| Dacomitinib | 7.631334898820142 | 0.517854527865434 | 0.1 | pass | 7.697301902149042 |
| Erlotinib | 7.857230414197513 | 0.4069913654648484 | 0.45 | review | 5.9650415939983015 |

## Diverse Generated Candidates
| smiles | predicted_pIC50 | QED | reward_hacking_risk | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.008198305968923 | 0.6673221117389069 | 0.0 | pass | 10.667454062676692 |
| OCc1ccc(-c2cc3c(N[C@H](CF)c4ccccc4)ncnc3[nH]2)cc1 | 9.309985758509136 | 0.4796017385083468 | 0.0 | pass | 10.64478586992268 |
| COc1cc(CO)ccc1-c1cc2c(N[C@H](C)c3ccccc3)ncnc2[nH]1 | 9.498957507913254 | 0.468660740099691 | 0.0 | pass | 10.619921016732578 |
| COCCOc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1Cl)C(=O)NCC2 | 9.297263540564424 | 0.4942665847544341 | 0.0 | pass | 10.58748419365508 |
| CCOc1cc(C)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.466693563673 | 0.4348209892776107 | 0.0 | pass | 10.562468473885964 |

## Market-Comparable Novel Shortlist
| smiles | predicted_pIC50 | QED | max_market_similarity | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.008198305968923 | 0.6673221117389069 | 0.3378378378378378 | pass | 10.640892816636562 |
| COc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.298643741626208 | 0.4560887691348292 | 0.2441860465116279 | pass | 10.599739837915008 |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.31124644191766 | 0.453837758950271 | 0.2111111111111111 | pass | 10.551439988530884 |
| CCOc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OC)c1 | 9.424778586077164 | 0.4080555322298783 | 0.2439024390243902 | pass | 10.579347837877297 |
| CCOc1cc(C)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.466693563673 | 0.4348209892776107 | 0.2048192771084337 | pass | 10.556316810739183 |

## Main Artifacts
- `reports/model_performance_summary.json`
- `reports/ranked_egfr_dataset.csv`
- `reports/marketed_egfr_scored.csv`
- `reports/generated_analogs_ranked.csv`
- `reports/iterative_ai_optimized_candidates.csv`
- `reports/final_diverse_candidates.csv`
- `reports/market_comparable_novel_shortlist.csv`
- `reports/technical_notebook/technical_notebook_summary.md`
- `reports/technical_notebook/technical_notebook_metrics.json`
- `reports/technical_notebook_quick/technical_notebook_summary.md`
- `reports/technical_notebook_quick/technical_notebook_metrics.json`