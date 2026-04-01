# Technical Notebook Assets

## Audit Overview
- Ranked molecules: `16133`
- Audit pass rate: `0.138`
- Audit review rate: `0.647`
- Audit fail rate: `0.215`
- Median reward hacking risk: `0.200`
- Median agent disagreement: `0.765`

## Model Validation Snapshot
- Random RMSE: `0.651`
- Scaffold RMSE: `0.739`
- Temporal RMSE: `1.697`

## Feasibility Snapshot
- Feasibility pass rate: `1.000`
- Mean feasibility score: `0.831`
- Mean experimental readiness: `0.757`
- Experimental readiness ready rate: `0.650`
- Evidence arbiter mean support: `0.649`
- Evidence arbiter pass rate: `0.800`
- Cross-database mean consensus: `0.463`
- Cross-database strong rate: `0.213`
- External evidence mean support: `0.467`
- External evidence pass rate: `0.693`
- Papyrus molecules / mean support: `7323` / `0.444`
- ExCAPE molecules / mean support: `5009` / `0.430`
- PubChem mean enriched evidence: `0.251`
- PubChem strong evidence rate: `0.015`
- PubChem virtual/proxy exposure rate: `0.176`
- Mean Vina affinity: `-8.551` kcal/mol
- Best Vina affinity: `-9.269` kcal/mol
- Mean interaction support: `0.774`
- Best interaction support: `0.960`
- Prospective validation batch size: `18`
- Prospective mean acquisition score: `1.057`
- Prospective mean cross-database consensus: `0.559`
- Prospective mean external evidence: `0.572`
- Prospective mean structure-evidence support: `0.694`
- Prospective Pareto-front rate: `0.944`
- Broad analog count / mean generator priority: `60` / `0.882`
- Broad analog mean adaptive prior: `0.697`
- Broad analog cross-db pass / parent improvement: `0.700` / `0.333`
- AI-guided count / mean generator priority: `120` / `0.796`
- AI-guided mean adaptive prior: `0.603`
- AI-guided cross-db pass / parent improvement: `0.350` / `0.133`
- Iterative count / mean generator priority: `150` / `0.802`
- Iterative top mean final score: `10.353`
- Iterative mean adaptive prior: `0.386`
- Iterative cross-db pass / parent improvement: `0.213` / `0.027`
- RL top mean feasibility: `0.919`
- RL mean cross-database consensus: `0.463`
- RL mean external evidence support: `0.453`
- RL mean structure-evidence support: `0.632`
- RL ready rate: `0.450`
- RL best episode return: `7.862`
- GPU GNN best scaffold model: `multiview_reference`
- GPU GNN best scaffold RMSE: `0.739`
- GPU RL mean cross-database consensus: `0.481`
- GPU RL mean external evidence support: `0.468`
- GPU RL mean evidence arbiter support: `0.660`
- GPU RL mean structure-evidence support: `0.604`
- GPU RL best episode return: `9.363`
- GPU actor-critic mean cross-database consensus: `0.512`
- GPU actor-critic mean external evidence support: `0.505`
- GPU actor-critic mean structure-evidence support: `0.587`
- GPU actor-critic ready rate: `0.562`
- GPU actor-critic best episode return: `9.010`
- Best robust scaffold model: `multiview_ensemble`
- Best robust scaffold RMSE: `0.739` +/- `0.000`
- Reward-hacking challenge trusted pass rate: `1.000`
- Reward-hacking challenge proxy demoted rate: `1.000`
- Source holdout mean RMSE: `0.989`
- Best source holdout: `excape_chembl20` with RMSE `0.828`
- Source holdout mean recall @ top 20%: `0.723`
- Rediscovery protected recall @ top 10: `0.368`
- Rediscovery naive recall @ top 10: `0.000`
- Rediscovery protected recall @ top 20: `0.368`
- Rediscovery naive recall @ top 20: `0.053`

## Most Demoted By Anti-Hacking Audit
| rank | naive_rank | audit_demote_positions | predicted_pIC50 | QED | reward_hacking_risk | audit_status |
| --- | --- | --- | --- | --- | --- | --- |
| 12675 | 7 | 12668 | 10.194 | 0.579 | 0.150 | fail |
| 12671 | 4 | 12667 | 10.238 | 0.591 | 0.150 | fail |
| 12678 | 25 | 12653 | 9.967 | 0.591 | 0.150 | fail |
| 12669 | 29 | 12640 | 9.254 | 0.771 | 0.150 | fail |
| 12680 | 61 | 12619 | 9.303 | 0.695 | 0.300 | fail |
| 12674 | 78 | 12596 | 9.210 | 0.715 | 0.150 | fail |
| 12670 | 81 | 12589 | 9.225 | 0.649 | 0.150 | fail |
| 12672 | 85 | 12587 | 9.304 | 0.662 | 0.150 | fail |
| 12676 | 95 | 12581 | 9.366 | 0.708 | 0.150 | fail |
| 12685 | 107 | 12578 | 9.272 | 0.621 | 0.300 | fail |

## Most Promoted By Protected Ranking
| rank | naive_rank | audit_promote_positions | predicted_pIC50 | QED | reward_hacking_risk | audit_status |
| --- | --- | --- | --- | --- | --- | --- |
| 2219 | 14133 | 11914 | 6.621 | 0.351 | 0.250 | pass |
| 2217 | 10429 | 8212 | 7.350 | 0.208 | 0.150 | pass |
| 2218 | 9516 | 7298 | 7.234 | 0.274 | 0.250 | pass |
| 2190 | 9254 | 7064 | 7.474 | 0.269 | 0.000 | pass |
| 2215 | 8883 | 6668 | 7.019 | 0.362 | 0.250 | pass |
| 2185 | 8743 | 6558 | 7.349 | 0.241 | 0.000 | pass |
| 2214 | 8767 | 6553 | 7.257 | 0.352 | 0.250 | pass |
| 2216 | 8608 | 6392 | 7.562 | 0.360 | 0.150 | pass |
| 2184 | 8571 | 6387 | 7.123 | 0.202 | 0.000 | pass |
| 2178 | 8550 | 6372 | 7.200 | 0.279 | 0.000 | pass |

## Marketed Benchmark Snapshot
| name | predicted_pIC50 | QED | reward_hacking_risk | final_score |
| --- | --- | --- | --- | --- |
| Sunvozertinib | 8.464 | 0.706 | 0.100 | 9.520 |
| Osimertinib | 8.455 | 0.653 | 0.100 | 9.064 |
| Gefitinib | 7.636 | 0.507 | 0.100 | 7.874 |
| Dacomitinib | 7.646 | 0.518 | 0.100 | 7.686 |
| Erlotinib | 7.668 | 0.407 | 0.450 | 5.362 |
| Afatinib | 6.851 | 0.752 | 0.450 | 5.190 |
| Lazertinib | 6.903 | 0.355 | 0.300 | 4.257 |

## Generated Candidate Snapshot
| smiles | predicted_pIC50 | QED | reward_hacking_risk | final_score |
| --- | --- | --- | --- | --- |
| CC(Nc1ncnc2[nH]c(-c3ccc(CO)cc3O)cc12)c1ccccc1 | 9.167 | 0.432 | 0.000 | 10.417 |
| [2H]C([2H])([2H])Oc1c(OC)cccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.229 | 0.442 | 0.000 | 10.360 |
| COc1cc(-c2cc3c(NC(C)c4ccccc4)ncnc3[nH]2)c(OC)cc1CO | 9.283 | 0.425 | 0.000 | 10.402 |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.973 | 0.667 | 0.000 | 10.621 |
| CN(COCO)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.902 | 0.627 | 0.000 | 10.634 |
| CCOc1cc(C(N)=O)ccc1-c1cc2c(N[C@H](C)c3ccccc3)ncnc2[nH]1 | 9.282 | 0.428 | 0.000 | 10.329 |
| Nc1cc2cc3c(Nc4cccc(Br)c4)ncnc3cc2[nH]1 | 8.942 | 0.504 | 0.000 | 10.372 |
| CCOc1cc(C#N)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.342 | 0.433 | 0.000 | 10.460 |
| OC[C@@H](Nc1ncnc2[nH]c(-c3ccccc3O)cc12)c1ccccc1 | 9.045 | 0.444 | 0.000 | 10.349 |
| CC(C)Oc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.303 | 0.432 | 0.000 | 10.525 |

## Novel Shortlist Snapshot
| smiles | predicted_pIC50 | QED | max_market_similarity | final_score |
| --- | --- | --- | --- | --- |
| COc1cc(F)c(F)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.083 | 0.458 | 0.286 | 10.414 |
| CC(Nc1ncnc2[nH]c(-c3ccc(CO)cc3O)cc12)c1ccccc1 | 9.167 | 0.432 | 0.203 | 10.372 |
| C#COCC(Nc1ncnc2[nH]c(-c3ccccc3OC)cc12)c1ccccc1 | 9.081 | 0.462 | 0.232 | 10.510 |
| C#COCC(Nc1ncnc2[nH]c(-c3ccccc3OCC)cc12)c1ccccc1 | 9.331 | 0.418 | 0.202 | 10.547 |
| COc1cc(C#N)ccc1-c1cc2c(NC(CF)c3ccccc3)ncnc2[nH]1 | 9.015 | 0.504 | 0.231 | 10.356 |
| COc1cc(C#N)ccc1-c1cc2c(N[C@H](CF)c3ccccc3)ncnc2[nH]1 | 9.015 | 0.504 | 0.231 | 10.356 |

## Structural Rescoring Snapshot
| smiles | docking_backend | vina_affinity_kcal | interaction_support_score | docking_rescore | closest_pose_reference | final_score |
| --- | --- | --- | --- | --- | --- | --- |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCC(C)(C)F)[nH]c2c1C(=O)NCC2 | consensus_vina_reference | -9.096 | 0.960 | 0.724 | Osimertinib | 10.703 |
| C#Cc1c(F)ccc(Nc2c(-c3ccncc3OCC(C)(C)OC)[nH]c3c2C(=O)NCC3)c1C | consensus_vina_reference | -8.912 | 0.960 | 0.717 | Afatinib | 10.654 |
| C#Cc1ccc(-c2cc3c(NC(CO)c4ccccc4)ncnc3[nH]2)c(OCC)c1 | consensus_vina_reference | -8.626 | 0.907 | 0.694 | Sunvozertinib | 10.625 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1ccc(F)c(C#N)c1C)C(=O)NCC2 | consensus_vina_reference | -8.656 | 0.960 | 0.674 | Osimertinib | 10.565 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1ccc(F)c(O)c1C)C(=O)NCC2 | consensus_vina_reference | -9.237 | 0.880 | 0.776 | Osimertinib | 10.519 |
| C#Cc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OCC)c1 | consensus_vina_reference | -8.759 | 0.933 | 0.715 | Afatinib | 10.509 |
| COc1cc(C)c(C)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | consensus_vina_reference | -8.691 | 0.960 | 0.721 | Afatinib | 10.477 |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCC(C)(C)Cl)[nH]c2c1C(=O)NCC2 | consensus_vina_reference | -8.884 | 0.960 | 0.701 | Osimertinib | 10.489 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1ccc(F)c(F)c1C)C(=O)NCC2 | consensus_vina_reference | -9.263 | 0.880 | 0.779 | Osimertinib | 10.457 |
| COc1cc(F)c(F)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | consensus_vina_reference | -8.991 | 0.960 | 0.769 | Osimertinib | 10.414 |

## Feasibility Evidence Snapshot
| smiles | feasibility_score | feasibility_status | max_active_similarity | fragment_support_ratio |
| --- | --- | --- | --- | --- |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCCN(C)C)[nH]c2c1C(=O)NCC2 | 0.898 | pass | 0.841 | 1.000 |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCCOCF)[nH]c2c1C(=O)NCC2 | 0.835 | pass | 0.853 | 0.857 |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 0.869 | pass | 0.833 | 0.750 |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCCOC(F)F)[nH]c2c1C(=O)NCC2 | 0.866 | pass | 0.841 | 1.000 |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCC(C)(C)F)[nH]c2c1C(=O)NCC2 | 0.837 | pass | 0.855 | 0.833 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1ccc(F)c(F)c1C)C(=O)NCC2 | 0.882 | pass | 0.857 | 0.857 |
| C#Cc1c(F)ccc(Nc2c(-c3ccncc3OCC(C)(C)OC)[nH]c3c2C(=O)NCC3)c1C | 0.828 | pass | 0.776 | 0.857 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1ccc(F)c(O)c1C)C(=O)NCC2 | 0.861 | pass | 0.822 | 0.857 |
| COc1ccccc1-c1cc2c(N[C@H](CF)c3ccccc3)ncnc2[nH]1 | 0.830 | pass | 0.815 | 0.833 |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 0.808 | pass | 0.810 | 0.833 |

## Experimental Readiness Snapshot
| smiles | predicted_pIC50 | experimental_readiness_score | experimental_readiness_status | experimental_track | cross_database_consensus_score |
| --- | --- | --- | --- | --- | --- |
| COc1cc(F)c(F)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.083 | 0.839 | ready | benchmark_ready | 0.558 |
| COc1cc(C)c(C)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.113 | 0.825 | ready | benchmark_ready | 0.557 |
| CC(Nc1ncnc2[nH]c(-c3ccc(CO)cc3O)cc12)c1ccccc1 | 9.167 | 0.829 | ready | benchmark_ready | 0.575 |
| COc1cc(-c2cc3c(NC(C)c4ccccc4)ncnc3[nH]2)c(OC)cc1CO | 9.283 | 0.831 | ready | benchmark_ready | 0.560 |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.973 | 0.830 | ready | benchmark_ready | 0.689 |
| C#Cc1ccc(-c2cc3c(NC(CO)c4ccccc4)ncnc3[nH]2)c(OCC)c1 | 9.427 | 0.781 | ready | benchmark_ready | 0.536 |
| C#Cc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OCC)c1 | 9.427 | 0.791 | ready | benchmark_ready | 0.536 |
| CCOc1cc(CO)ccc1-c1cc2c(NC(C)c3ccccc3)ncnc2[nH]1 | 9.366 | 0.801 | ready | benchmark_ready | 0.563 |
| CCOc1cc(C)ccc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.294 | 0.788 | ready | benchmark_ready | 0.541 |
| [2H]C([2H])([2H])Oc1cc(C)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.184 | 0.797 | ready | benchmark_ready | 0.543 |

## Cross-Database Validation Snapshot
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | external_evidence_status | cross_database_independent_support_count | cross_database_status |
| --- | --- | --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.973 | 0.676 | 0.700 | pass | 3 | strong |
| CN(C)c1ncc2ncnc(Nc3cccc(Br)c3)c2n1 | 8.824 | 0.777 | 0.742 | pass | 2 | strong |
| O=C(/C=C\Br)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.941 | 0.684 | 0.706 | pass | 3 | strong |
| COc1ccccc1-c1cc2c(N[C@H](CF)c3ccccc3)ncnc2[nH]1 | 9.072 | 0.544 | 0.567 | pass | 2 | strong |
| CC(=O)N(C)c1ncc2ncnc(Nc3cccc(Br)c3)c2n1 | 8.419 | 0.650 | 0.655 | pass | 2 | strong |
| CC(C)Oc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.303 | 0.528 | 0.558 | pass | 2 | strong |
| COc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.337 | 0.589 | 0.597 | pass | 2 | strong |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.162 | 0.512 | 0.535 | pass | 2 | moderate |
| COc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.113 | 0.532 | 0.559 | pass | 2 | strong |
| CCOc1cccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c1OC | 9.383 | 0.524 | 0.544 | pass | 2 | strong |

## Prospective Validation Batch
| prospective_batch_rank | candidate_source | predicted_pIC50 | experimental_readiness_score | prospective_acquisition_score | experimental_readiness_status |
| --- | --- | --- | --- | --- | --- |
| 1 | shortlist | 9.083 | 0.839 | 1.173 | ready |
| 2 | shortlist | 9.167 | 0.829 | 1.149 | ready |
| 3 | diverse | 8.902 | 0.833 | 1.133 | ready |
| 4 | diverse | 8.942 | 0.801 | 1.111 | ready |
| 5 | diverse | 9.229 | 0.835 | 1.108 | ready |
| 6 | diverse | 9.283 | 0.831 | 1.102 | ready |
| 7 | diverse | 9.282 | 0.822 | 1.101 | ready |
| 8 | diverse | 8.973 | 0.830 | 1.082 | ready |
| 9 | shortlist | 9.081 | 0.774 | 1.082 | ready |
| 10 | optimized_readiness | 9.427 | 0.791 | 1.072 | ready |

## Verifiable RL Snapshot
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | experimental_readiness_score | rl_priority_score |
| --- | --- | --- | --- | --- | --- |
| C#Cc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OCC)c1 | 9.427 | 0.536 | 0.544 | 0.793 | 12.256 |
| CCOc1cc(F)c(OC)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.269 | 0.542 | 0.546 | 0.761 | 11.935 |
| CCOc1cc(F)c(OC)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.269 | 0.542 | 0.546 | 0.761 | 11.935 |
| CCOc1cc(C#N)c(OC)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.177 | 0.518 | 0.521 | 0.785 | 11.915 |
| COc1cc(-c2cc3c(N[C@H](C)c4ccccc4)ncnc3[nH]2)c(OC)cc1C(N)=O | 9.034 | 0.555 | 0.544 | 0.813 | 11.808 |
| COc1cc(-c2cc3c(NC(CO)c4ccccc4)ncnc3[nH]2)c(OC)c(N(C)C)c1 | 9.084 | 0.497 | 0.491 | 0.762 | 11.776 |
| COc1cc(C=O)cc(OC)c1-c1cc2c(NC(C)c3ccccc3)ncnc2[nH]1 | 8.927 | 0.574 | 0.582 | 0.730 | 11.549 |
| CCOc1cc(OC)c(F)c(OC)c1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 8.856 | 0.509 | 0.524 | 0.756 | 11.422 |
| CNc1c(O)[nH]c2cc3ncnc(Nc4cccc(Br)c4)c3cc12 | 8.497 | 0.592 | 0.642 | 0.794 | 11.421 |
| CCOc1cc(C)c(F)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.230 | 0.514 | 0.500 | 0.715 | 11.787 |

## Source Holdout Benchmark
| source | test_size | rmse | r2 | rmse_gain_vs_baseline | recall_top20pct |
| --- | --- | --- | --- | --- | --- |
| excape_chembl20 | 764 | 0.828 | 0.541 | 0.682 | 0.862 |
| papyrus | 4137 | 0.854 | 0.587 | 0.494 | 0.711 |
| bindingdb_articles | 152 | 0.914 | 0.483 | 0.635 | 1.000 |
| chembl | 5911 | 1.360 | -0.267 | 0.056 | 0.320 |

## Rediscovery Benchmark
| benchmark_name | benchmark_source | protected_panel_rank | naive_panel_rank | external_evidence_support | evidence_arbiter_support |
| --- | --- | --- | --- | --- | --- |
| Osimertinib | marketed | 1 | 124 | 0.845 | 0.803 |
| Dacomitinib | marketed | 2 | 129 | 0.980 | 0.836 |
| mifanertinib | iuphar | 3 | 123 | 0.804 | 0.792 |
| compound 56 [PMID: 8568816] | iuphar | 4 | 126 | 0.926 | 0.731 |
| Erlotinib | marketed | 5 | 130 | 0.862 | 0.821 |
| Gefitinib | marketed | 6 | 128 | 0.812 | 0.833 |
| Sunvozertinib | marketed | 7 | 121 | 0.700 | 0.722 |
| tesevatinib | iuphar | 82 | 133 | 0.701 | 0.698 |
| Afatinib | marketed | 129 | 132 | 0.650 | 0.608 |
| asandeutertinib | iuphar | 130 | 136 | 0.574 | 0.545 |

## GPU DQN RL Snapshot
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | evidence_arbiter_support | gpu_rl_priority_score |
| --- | --- | --- | --- | --- | --- |
| CN(CO)CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.887 | 0.701 | 0.757 | 0.786 | 12.350 |
| CCOc1cc(CO)ccc1-c1cc2c(N[C@H](C)c3ccccc3)ncnc2[nH]1 | 9.366 | 0.563 | 0.577 | 0.725 | 12.182 |
| CCOc1cccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c1OC | 9.383 | 0.550 | 0.550 | 0.694 | 12.139 |
| CN(COCO)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.902 | 0.695 | 0.753 | 0.652 | 12.094 |
| CCOc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.621 | 0.590 | 0.582 | 0.601 | 12.050 |
| CCOc1ccc(F)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.379 | 0.541 | 0.520 | 0.638 | 12.039 |
| COCCOc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1F)C(=O)NCC2 | 9.196 | 0.382 | 0.340 | 0.680 | 12.033 |
| COCCOc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1N(C)C)C(=O)NCC2 | 9.253 | 0.366 | 0.329 | 0.657 | 12.011 |
| O=C1NCCc2[nH]c(-c3ccncc3OCCF)c(Nc3cccc(F)c3CCO)c21 | 9.289 | 0.326 | 0.277 | 0.623 | 11.973 |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCC(C)(C)Cl)[nH]c2c1C(=O)NCC2 | 9.356 | 0.347 | 0.302 | 0.665 | 11.933 |

## GPU Actor-Critic Snapshot
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | experimental_readiness_score | actor_critic_priority_score |
| --- | --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.973 | 0.689 | 0.730 | 0.828 | 12.180 |
| O=C(/C=C\Cl)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.045 | 0.753 | 0.770 | 0.868 | 12.047 |
| C=CNCc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.534 | 0.665 | 0.733 | 0.834 | 11.859 |
| O=C(/C=C\Br)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.941 | 0.698 | 0.738 | 0.828 | 11.971 |
| COCCOc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1F)C(=O)NCC2 | 9.196 | 0.382 | 0.340 | 0.772 | 12.043 |
| CCOc1cc(C#N)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.342 | 0.519 | 0.518 | 0.785 | 12.108 |
| CC(C)Oc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.303 | 0.556 | 0.564 | 0.785 | 12.152 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1OCF)C(=O)NCC2 | 9.421 | 0.351 | 0.300 | 0.760 | 11.959 |
| CC(C)Oc1cnccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.197 | 0.473 | 0.440 | 0.766 | 11.953 |
| COCOc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.246 | 0.562 | 0.568 | 0.768 | 11.873 |