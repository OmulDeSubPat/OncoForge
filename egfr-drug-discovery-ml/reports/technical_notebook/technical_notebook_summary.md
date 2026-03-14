# Technical Notebook Assets

## Audit Overview
- Ranked molecules: `10606`
- Audit pass rate: `0.157`
- Audit review rate: `0.627`
- Audit fail rate: `0.216`
- Median reward hacking risk: `0.200`
- Median agent disagreement: `0.726`

## Model Validation Snapshot
- Random RMSE: `0.660`
- Scaffold RMSE: `0.741`
- Temporal RMSE: `1.154`

## Most Demoted By Anti-Hacking Audit
| rank | naive_rank | audit_demote_positions | predicted_pIC50 | QED | reward_hacking_risk | audit_status |
| --- | --- | --- | --- | --- | --- | --- |
| 8322 | 17 | 8305 | 9.380 | 0.769 | 0.150 | fail |
| 8323 | 69 | 8254 | 9.325 | 0.759 | 0.150 | fail |
| 8324 | 77 | 8247 | 9.391 | 0.708 | 0.150 | fail |
| 8332 | 130 | 8202 | 9.586 | 0.167 | 0.400 | fail |
| 8348 | 152 | 8196 | 9.532 | 0.188 | 0.550 | fail |
| 8325 | 138 | 8187 | 9.092 | 0.698 | 0.150 | fail |
| 8326 | 148 | 8178 | 9.422 | 0.595 | 0.150 | fail |
| 8331 | 157 | 8174 | 9.239 | 0.601 | 0.300 | fail |
| 8328 | 156 | 8172 | 9.330 | 0.567 | 0.150 | fail |
| 8330 | 185 | 8145 | 9.258 | 0.627 | 0.150 | fail |

## Most Promoted By Protected Ranking
| rank | naive_rank | audit_promote_positions | predicted_pIC50 | QED | reward_hacking_risk | audit_status |
| --- | --- | --- | --- | --- | --- | --- |
| 1662 | 9184 | 7522 | 6.611 | 0.351 | 0.250 | pass |
| 1657 | 7674 | 6017 | 7.094 | 0.291 | 0.000 | pass |
| 1660 | 7140 | 5480 | 7.369 | 0.208 | 0.150 | pass |
| 1645 | 6762 | 5117 | 7.318 | 0.303 | 0.000 | pass |
| 1631 | 6318 | 4687 | 7.493 | 0.269 | 0.000 | pass |
| 1643 | 6304 | 4661 | 7.245 | 0.375 | 0.000 | pass |
| 1661 | 6313 | 4652 | 7.489 | 0.261 | 0.250 | pass |
| 1658 | 6283 | 4625 | 7.216 | 0.354 | 0.250 | pass |
| 1659 | 6278 | 4619 | 7.618 | 0.347 | 0.250 | pass |
| 1637 | 6197 | 4560 | 7.347 | 0.241 | 0.000 | pass |

## Marketed Benchmark Snapshot
| name | predicted_pIC50 | QED | reward_hacking_risk | final_score |
| --- | --- | --- | --- | --- |
| Sunvozertinib | 8.377 | 0.706 | 0.100 | 9.505 |
| Osimertinib | 8.258 | 0.653 | 0.100 | 8.852 |
| Gefitinib | 7.680 | 0.507 | 0.100 | 7.771 |
| Dacomitinib | 7.631 | 0.518 | 0.100 | 7.697 |
| Erlotinib | 7.857 | 0.407 | 0.450 | 5.965 |
| Afatinib | 6.975 | 0.752 | 0.300 | 5.864 |
| Lazertinib | 7.027 | 0.355 | 0.300 | 4.427 |

## Generated Candidate Snapshot
| smiles | predicted_pIC50 | QED | reward_hacking_risk | final_score |
| --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.008 | 0.667 | 0.000 | 10.667 |
| OCc1ccc(-c2cc3c(N[C@H](CF)c4ccccc4)ncnc3[nH]2)cc1 | 9.310 | 0.480 | 0.000 | 10.645 |
| COc1cc(CO)ccc1-c1cc2c(N[C@H](C)c3ccccc3)ncnc2[nH]1 | 9.499 | 0.469 | 0.000 | 10.620 |
| COCCOc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1Cl)C(=O)NCC2 | 9.297 | 0.494 | 0.000 | 10.587 |
| CCOc1cc(C)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.467 | 0.435 | 0.000 | 10.562 |
| [2H]C([2H])([2H])Oc1ccccc1-c1cc2c(N[C@H](CF)c3ccccc3)ncnc2[nH]1 | 9.118 | 0.517 | 0.000 | 10.500 |
| OCc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(F)c1 | 9.352 | 0.413 | 0.000 | 10.473 |
| Cc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.144 | 0.510 | 0.000 | 10.435 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1ccc(C)c(F)c1C)C(=O)NCC2 | 9.234 | 0.485 | 0.000 | 10.469 |
| OC[C@@H](Nc1ncnc2[nH]c(-c3cccc(F)c3)cc12)c1ccccc1 | 8.987 | 0.511 | 0.000 | 10.418 |

## Novel Shortlist Snapshot
| smiles | predicted_pIC50 | QED | max_market_similarity | final_score |
| --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.008 | 0.667 | 0.338 | 10.641 |
| COc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.299 | 0.456 | 0.244 | 10.600 |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.311 | 0.454 | 0.211 | 10.551 |
| CCOc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OC)c1 | 9.425 | 0.408 | 0.244 | 10.579 |
| CCOc1cc(C)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.467 | 0.435 | 0.205 | 10.556 |
| COc1ccc(-c2cc3c(N[C@H](CF)c4ccccc4)ncnc3[nH]2)c(OC)c1 | 9.098 | 0.474 | 0.253 | 10.527 |
| [2H]C([2H])([2H])Oc1ccccc1-c1cc2c(N[C@H](CF)c3ccccc3)ncnc2[nH]1 | 9.118 | 0.517 | 0.228 | 10.471 |
| CCOc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.352 | 0.414 | 0.207 | 10.522 |
| [2H]C([2H])([2H])Oc1cc(OCC)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.429 | 0.406 | 0.238 | 10.479 |
| COc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OC)c1 | 9.365 | 0.445 | 0.241 | 10.462 |

## Structural Rescoring Snapshot
| smiles | closest_pose_reference | docking_rescore | shape_similarity | final_score |
| --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | Osimertinib | 0.716 | 0.572 | 10.641 |
| COc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | Osimertinib | 0.698 | 0.564 | 10.600 |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | Osimertinib | 0.682 | 0.525 | 10.551 |
| CCOc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OC)c1 | Dacomitinib | 0.635 | 0.391 | 10.579 |
| CCOc1cc(C)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | Afatinib | 0.646 | 0.444 | 10.556 |
| COc1ccc(-c2cc3c(N[C@H](CF)c4ccccc4)ncnc3[nH]2)c(OC)c1 | Afatinib | 0.665 | 0.493 | 10.527 |
| [2H]C([2H])([2H])Oc1ccccc1-c1cc2c(N[C@H](CF)c3ccccc3)ncnc2[nH]1 | Osimertinib | 0.717 | 0.607 | 10.471 |
| CCOc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | Erlotinib | 0.620 | 0.404 | 10.522 |
| [2H]C([2H])([2H])Oc1cc(OCC)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | Osimertinib | 0.666 | 0.518 | 10.479 |
| COc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OC)c1 | Osimertinib | 0.673 | 0.526 | 10.462 |