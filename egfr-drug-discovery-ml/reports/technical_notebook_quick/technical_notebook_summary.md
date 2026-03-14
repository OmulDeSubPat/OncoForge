# Technical Notebook Assets

## Audit Overview
- Ranked molecules: `1000`
- Audit pass rate: `0.244`
- Audit review rate: `0.608`
- Audit fail rate: `0.148`
- Median reward hacking risk: `0.200`
- Median agent disagreement: `0.673`

## Most Demoted By Anti-Hacking Audit
| rank | naive_rank | audit_demote_positions | predicted_pIC50 | QED | reward_hacking_risk | audit_status |
| --- | --- | --- | --- | --- | --- | --- |
| 853 | 29 | 824 | 9.384 | 0.162 | 0.400 | fail |
| 854 | 30 | 824 | 9.384 | 0.162 | 0.400 | fail |
| 864 | 146 | 718 | 9.770 | 0.280 | 0.600 | fail |
| 872 | 175 | 697 | 9.777 | 0.198 | 0.600 | fail |
| 873 | 179 | 694 | 9.725 | 0.192 | 0.600 | fail |
| 876 | 197 | 679 | 9.679 | 0.189 | 0.600 | fail |
| 883 | 242 | 641 | 9.436 | 0.236 | 0.600 | fail |
| 857 | 236 | 621 | 8.986 | 0.195 | 0.400 | fail |
| 899 | 280 | 619 | 9.314 | 0.197 | 0.600 | fail |
| 861 | 259 | 602 | 8.917 | 0.185 | 0.350 | fail |

## Most Promoted By Protected Ranking
| rank | naive_rank | audit_promote_positions | predicted_pIC50 | QED | reward_hacking_risk | audit_status |
| --- | --- | --- | --- | --- | --- | --- |
| 244 | 842 | 598 | 6.794 | 0.208 | 0.000 | pass |
| 238 | 692 | 454 | 6.954 | 0.312 | 0.000 | pass |
| 237 | 657 | 420 | 7.374 | 0.372 | 0.000 | pass |
| 235 | 608 | 373 | 7.473 | 0.241 | 0.000 | pass |
| 243 | 589 | 346 | 7.242 | 0.376 | 0.250 | pass |
| 230 | 556 | 326 | 7.538 | 0.255 | 0.000 | pass |
| 242 | 568 | 326 | 7.307 | 0.441 | 0.250 | pass |
| 241 | 563 | 322 | 7.235 | 0.469 | 0.250 | pass |
| 239 | 549 | 310 | 7.409 | 0.314 | 0.250 | pass |
| 233 | 538 | 305 | 7.502 | 0.436 | 0.000 | pass |

## Marketed Benchmark Snapshot
| name | predicted_pIC50 | QED | reward_hacking_risk | final_score |
| --- | --- | --- | --- | --- |
| Sunvozertinib | 8.405 | 0.706 | 0.100 | 9.549 |
| Osimertinib | 8.355 | 0.653 | 0.100 | 8.964 |
| Gefitinib | 7.824 | 0.507 | 0.100 | 8.259 |
| Dacomitinib | 7.703 | 0.518 | 0.100 | 7.834 |
| Erlotinib | 7.664 | 0.407 | 0.100 | 7.682 |
| Afatinib | 7.123 | 0.752 | 0.300 | 6.401 |
| Lazertinib | 6.997 | 0.355 | 0.300 | 4.456 |

## Generated Candidate Snapshot
| smiles | predicted_pIC50 | QED | final_score |
| --- | --- | --- | --- |
| C=CC(=O)Nc1nc2c(Cc3ccc(F)c(Cl)c3)ncnc2cc1/C=C/CCN1CCNCC1 | 9.312 | 0.477 | 9.514 |
| NN1CCN(CCC(=O)Nc2ccc3ncnc(Nc4cccc(Br)c4)c3c2)CC1 | 9.272 | 0.477 | 9.495 |
| CN1CCN(CCC(=O)Nc2ccc3ncnc(Nc4cccc(Br)c4)c3c2)CC1F | 9.162 | 0.512 | 9.392 |
| C=CC(=O)CC1=NC=C2N=CN=C(Nc3cccc(C)c3)C21 | 8.997 | 0.867 | 9.385 |
| C=CC(=O)Nc1nc2c(Cc3ccc(F)c(Cl)c3)ncnc2cc1/C=C/CCN1CCOCC1N | 9.186 | 0.459 | 9.383 |
| N/C=C/C(=O)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.985 | 0.657 | 9.301 |
| CC(C(=O)C=N)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.949 | 0.652 | 9.241 |
| OCC(O)CCc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.845 | 0.601 | 9.131 |
| C=CC(=O)OC1=NC=C2N=CN=C(Nc3cccc(C)c3)C21 | 8.833 | 0.672 | 9.129 |
| NCC(O)CNc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.843 | 0.512 | 9.082 |

## Novel Shortlist Snapshot
| smiles | predicted_pIC50 | QED | max_market_similarity | final_score |
| --- | --- | --- | --- | --- |
| C=CC(=O)Nc1nc2c(Cc3ccc(F)c(Cl)c3)ncnc2cc1/C=C/CCN1COCCOC1 | 9.357 | 0.466 | 0.368 | 9.575 |
| C=CC(=O)Nc1nc2c(Cc3ccc(F)c(Cl)c3)ncnc2cc1/C=C/CCN1CCOCOC1 | 9.344 | 0.466 | 0.352 | 9.549 |
| C=CC(=O)Nc1nc2c(Cc3ccc(F)c(Cl)c3)ncnc2cc1/C=C/CCN1CCOCCO1 | 9.325 | 0.466 | 0.352 | 9.536 |
| C=CC(=O)Nc1nc2c(Cc3ccc(F)c(Cl)c3)ncnc2cc1/C=C/CCN1CCOCCN1 | 9.324 | 0.458 | 0.344 | 9.526 |
| C=CC(=O)NC1=NOC=C2N=CN=C(Nc3cccc(C)c3)C21 | 9.114 | 0.819 | 0.262 | 9.515 |
| C=CC(=O)Nc1nc2c(Cc3ccc(F)c(Cl)c3)ncnc2cc1/C=C/CCN1CCNCC1 | 9.312 | 0.477 | 0.364 | 9.514 |
| O=C(CCN1CCN(CO)CC1)Nc1ccc2ncnc(Nc3cccc(Br)c3)c2c1 | 9.299 | 0.474 | 0.341 | 9.511 |
| C=CC(=O)Nc1nc2c(Cc3ccc(F)c(Cl)c3)ncnc2cc1/C=C/CCN1COCCCO1 | 9.305 | 0.457 | 0.348 | 9.508 |
| CN1CCN(CCC(=O)Nc2ccc3ncnc(Nc4cccc(Br)c4)c3c2)CCO1 | 9.241 | 0.552 | 0.360 | 9.496 |
| C=CC(=O)NC1=NNC=C2N=CN=C(Nc3cccc(C)c3)C21 | 9.138 | 0.724 | 0.265 | 9.487 |