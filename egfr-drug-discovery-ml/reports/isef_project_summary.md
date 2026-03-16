# OncoForge ISEF Summary

## Project Goal
OncoForge is an AI-assisted lead-optimization pipeline for EGFR inhibitors.
The system does not claim to discover finished drugs; it prioritizes chemically plausible, high-potential candidates for downstream wet-lab validation.

## Upgraded Methodology
- A multi-agent scorer now separates potency, chemistry, safety, novelty and applicability-domain checks instead of relying on a single scalar reward.
- Cross-database validation now spans ChEMBL, BindingDB, IUPHAR, PubChem, Papyrus and ExCAPE-DB so candidate support is checked across independent public sources.
- Verified reward is combined with anti-reward-hacking audits so suspicious molecules are penalized even if they exploit a proxy metric.
- A verifiable-reward RL loop now optimizes traceable medicinal-chemistry actions instead of relying only on heuristic beam search.
- The generator is now reaction-aware and scaffold-preserving, with hard constraints applied during generation rather than only after scoring.
- GPU stages now benchmark a graph neural model and a neural DQN policy on the same candidate ecosystem as the classical pipeline.
- Candidate feasibility is scored with non-experimental evidence: active-neighbor support, scaffold support, fragment support and generation traceability.
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
- Dataset size: `16133` molecules
- Random split RMSE / R2: `0.6508829779836404` / `0.7616330143813422`
- Scaffold split RMSE / R2: `0.738737717424736` / `0.7164146865622143`
- Temporal split RMSE / R2: `1.2208873179699924` / `-0.10383355516887849`

## Audit Diagnostics
- Audit pass rate: `0.13754416413562265`
- Audit review rate: `0.6471827930329139`
- Audit fail rate: `0.21527304283146345`
- Median reward hacking risk: `0.2`
- Mean audit demotion: `1045.7554081695903` positions
- Feasibility pass rate on optimized candidates: `1.0`
- Best Vina affinity on optimized candidates: `-9.741`
- Mean interaction support on optimized candidates: `0.8292`
- Mean experimental readiness: `0.7541059849782176`
- Cross-database mean consensus: `0.467395012359698`
- Cross-database strong rate: `0.54`
- External evidence mean support: `0.4570595949242219`
- External evidence pass rate: `0.61`
- Evidence arbiter mean support: `0.6554639512045938`
- Evidence arbiter pass rate: `0.79`
- Papyrus molecules / mean support: `7323` / `0.4440460194190368`
- ExCAPE molecules / mean support: `5009` / `0.42960830412890716`
- PubChem mean enriched evidence: `0.25079777818118626`
- PubChem strong evidence rate: `0.015457788347205707`
- PubChem virtual/proxy exposure rate: `0.1759809750297265`
- Prospective validation batch size: `18`
- Prospective mean acquisition score: `0.9259371800461699`
- Broad analog benchmark count / mean generator priority: `1416` / `0.7155700761656891`
- AI-guided benchmark count / mean generator priority: `902` / `0.7204670668748937`
- Iterative benchmark count / top mean final score: `1829` / `10.367912279638961`
- RL mean cross-database consensus: `0.49694325633419306`
- RL mean external evidence support: `0.4815163487131554`
- RL ready rate: `0.1`
- GPU GNN scaffold snapshot: `multiview_reference | RMSE 0.739`
- GPU RL mean external evidence support: `0.45151283411101756`
- GPU RL best episode return: `8.083541498364047`
- Best repeated-seed scaffold model: `multiview_ensemble | RMSE 0.739 +/- 0.000`
- Reward-hacking challenge snapshot: `trusted pass 1.000, proxy demoted>=20 1.000, proxy review/fail 0.580`
- Source holdout snapshot: `best excape_chembl20 RMSE 0.828, mean recall@20% 0.723`
- Rediscovery benchmark snapshot: `protected top10 0.368, naive top10 0.000, protected median rank 130.0`

## Verifiable RL Diagnostics
- Best episode return: `7.964445588724134`
- Mean episode return: `4.484585397034608`

## GPU RL Diagnostics
- Best episode return: `7.964445588724135`
- Mean episode return: `5.333216844326996`

## Top Ranked Training-Space Molecules
| smiles | predicted_pIC50 | QED | reward_hacking_risk | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| C/C=C/C(=O)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.246218617897418 | 0.6685952655904649 | 0.0 | pass | 10.749370548939222 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1C)C(=O)NCC2 | 9.64492553317416 | 0.5102437252597349 | 0.0 | pass | 10.698916496302727 |
| CCOc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.621394722166604 | 0.4521900852216164 | 0.0 | pass | 10.645090707610736 |
| [2H]C([2H])([2H])Oc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.386549939172664 | 0.4855433472748625 | 0.0 | pass | 10.549651298502416 |
| CNc1ncc2ncnc(Nc3cccc(Br)c3)c2n1 | 8.963399705573321 | 0.7684706048512752 | 0.0 | pass | 10.547735227339848 |

## Scored Marketed EGFR Drugs
| name | predicted_pIC50 | vina_affinity_kcal | interaction_support_score | docking_rescore | final_score |
| --- | --- | --- | --- | --- | --- |
| Sunvozertinib | 8.464192202834752 | -8.383 | 0.76 | 0.7835104889027267 | 9.520210184855385 |
| Osimertinib | 8.454906685137393 | -8.79 | 0.8133333333333334 | 0.8377011954524191 | 9.064268118923406 |
| Gefitinib | 7.63617631407553 | -8.852 | 0.9333333333333332 | 0.8481458156764541 | 7.873645049847415 |
| Dacomitinib | 7.646033286866971 | -8.664 | 0.8133333333333334 | 0.8233868445370008 | 7.68615424526805 |
| Erlotinib | 7.668325671346465 | -8.175 | 0.96 | 0.7600710128516652 | 5.36156209009101 |

## Diverse Generated Candidates
| smiles | predicted_pIC50 | QED | reward_hacking_risk | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| C=CC(=O)N(C)c1ncc2ncnc(Nc3cccc(Br)c3)c2n1 | 8.570758754753866 | 0.695342574059154 | 0.0 | pass | 10.264301906743505 |
| Fc1cc2cc3c(Nc4cccc(Br)c4)ncnc3cc2[nH]1 | 8.645927367368346 | 0.5440922450062603 | 0.0 | pass | 10.228485585816088 |
| Cc1n[nH]c2cc3ncnc(Nc4cccc(Br)c4)c3cc12 | 8.764188307157719 | 0.5626417282039111 | 0.0 | pass | 10.304077265744088 |
| C=CC(=O)NC1=CC=C2N=CN=C(Nc3ccc(C)c(C(F)(F)F)c3)C21 | 8.674570657445464 | 0.8096611197289264 | 0.0 | pass | 10.470874383471523 |
| [2H]C([2H])([2H])Oc1c(OC)cccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.228592024598935 | 0.4424444747416858 | 0.0 | pass | 10.331384358963206 |

## Market-Comparable Novel Shortlist
| smiles | predicted_pIC50 | QED | max_market_similarity | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.162262388817652 | 0.453837758950271 | 0.2111111111111111 | pass | 10.492968111024924 |
| COc1cc(F)c(F)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.08324074548758 | 0.4579005357474424 | 0.2857142857142857 | pass | 10.404739207842402 |
| COc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.113341225066414 | 0.4560887691348292 | 0.2441860465116279 | pass | 10.47254657367218 |
| [2H]C([2H])([2H])Oc1c(OC)cccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.228592024598935 | 0.4424444747416858 | 0.2307692307692307 | pass | 10.302451731868452 |
| C#Cc1ccc(OC)c(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c1 | 9.159680697835066 | 0.4410336463311468 | 0.2371134020618556 | pass | 10.421623256190951 |

## Feasibility-Supported Optimized Candidates
| smiles | predicted_pIC50 | feasibility_score | vina_affinity_kcal | feasibility_status | max_active_similarity |
| --- | --- | --- | --- | --- | --- |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCC(C)(C)F)[nH]c2c1C(=O)NCC2 | 9.312988403947166 | 0.9397260121658272 | -9.096 | pass | 0.855072463768116 |
| C#Cc1c(F)ccc(Nc2c(-c3ccncc3OCC(C)(C)OC)[nH]c3c2C(=O)NCC3)c1C | 9.262148061408196 | 0.9311306960407896 | -8.98 | pass | 0.7763157894736842 |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.162262388817652 | 0.9192835144416824 | -9.079 | pass | 0.8103448275862069 |
| COc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.113341225066414 | 0.9492089433520629 | -8.862 | pass | 0.8035714285714286 |
| COCCOc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1F)C(=O)NCC2 | 9.195620914997406 | 0.9652467337375004 | -8.638 | pass | 0.8636363636363636 |

## Experimental-Readiness Snapshot
| smiles | predicted_pIC50 | experimental_readiness_score | experimental_readiness_status | experimental_track | docking_rescore |
| --- | --- | --- | --- | --- | --- |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.162262388817652 | 0.8321981770453616 | ready | benchmark_ready | 0.7687956704951397 |
| COc1cc(F)c(F)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.08324074548758 | 0.8324876541997084 | ready | benchmark_ready | 0.7689637474630646 |
| COc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.113341225066414 | 0.8243058872714294 | ready | benchmark_ready | 0.7464322632994712 |
| COc1cc(C)c(C)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.113341225066412 | 0.8178676007814992 | ready | benchmark_ready | 0.7210563604847683 |
| [2H]C([2H])([2H])Oc1c(OC)cccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.228592024598935 | 0.8301105110796562 | ready | benchmark_ready | 0.7468516382613155 |

## Cross-Database Validation Snapshot
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | cross_database_independent_support_count | cross_database_status | experimental_readiness_score |
| --- | --- | --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.97262998622889 | 0.6892018730622121 | 0.7303277343237046 | 3 | strong | 0.8227878921894052 |
| COC(CO)CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.834301012116741 | 0.698154859936329 | 0.7542332781656754 | 3 | strong | 0.8509445325132541 |
| COc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.113341225066414 | 0.557338070414523 | 0.564530592899725 | 2 | strong | 0.8243058872714294 |
| COc1cc(F)c(F)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.08324074548758 | 0.5581964178545092 | 0.5679420147937321 | 2 | strong | 0.8324876541997084 |
| [2H]C([2H])([2H])Oc1cc(C)c(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.162262388817652 | 0.5380376899411612 | 0.5408354238275989 | 2 | strong | 0.8321981770453616 |

## Prospective Validation Batch
| prospective_batch_rank | candidate_source | predicted_pIC50 | experimental_readiness_score | prospective_acquisition_score | experimental_readiness_status |
| --- | --- | --- | --- | --- | --- |
| 1 | diverse | 8.645927367368346 | 0.8773393855572236 | 0.9927051305858234 | ready |
| 2 | diverse | 8.764188307157719 | 0.8446851413872266 | 0.9801701893683148 | ready |
| 3 | diverse | 8.570758754753866 | 0.899526984310922 | 0.980106875287414 | ready |
| 4 | diverse | 8.823788860318864 | 0.8656285476089107 | 0.9762812691282232 | ready |
| 5 | shortlist | 9.08324074548758 | 0.8331898715990441 | 0.9684926809968016 | ready |

## Verifiable RL Candidates
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | experimental_readiness_score | rl_priority_score |
| --- | --- | --- | --- | --- | --- |
| CCOc1cc(C(N)=O)ccc1-c1cc2c(N[C@H](C)c3ccccc3)ncnc2[nH]1 | 9.281720220568843 | 0.5383178899799252 | 0.5255296785415469 | 0.7369360631632614 | 11.682551136055492 |
| COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC1CCC(NS(C)(=O)=O)CC1 | 9.355638389729048 | 0.5726818783669506 | 0.6049033910733639 | 0.7396218029514866 | 11.443991549636578 |
| CCOc1cc(CO)ccc1-c1cc2c(NC(C)c3ccccc3)ncnc2[nH]1 | 9.366026856646968 | 0.5632923998162127 | 0.5766702304711735 | 0.7148375310778763 | 11.714828614556426 |
| CCOc1cc(OC)ccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.372811294999842 | 0.5371638558895002 | 0.5173912395499684 | 0.6925619584639161 | 11.613285658555718 |
| CCOc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.6213947221666 | 0.5902305767946632 | 0.5823146763774594 | 0.6738161153651983 | 11.589320387016228 |

## GPU DQN RL Candidates
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | evidence_arbiter_support | gpu_rl_priority_score |
| --- | --- | --- | --- | --- | --- |
| C/C=C/C(=O)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.246218617897418 | 0.7553457112034389 | 0.7722966446694595 | 0.6805804982430917 | 11.870882373627229 |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.972629986228892 | 0.6892018730622121 | 0.7303277343237046 | 0.6483261942656477 | 11.69067440256797 |
| CC(C)Oc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.30296226947643 | 0.5558513736681169 | 0.5636026816263443 | 0.6323939815417547 | 11.663863025862373 |
| CCOc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.621394722166604 | 0.5902305767946632 | 0.5823146763774594 | 0.5813125417580992 | 11.593586692599972 |
| CCOc1cnc(C)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.250489524945468 | 0.4571788327099894 | 0.4143047570287392 | 0.6348668055072944 | 11.570937201880357 |

## Main Artifacts
- `reports/model_performance_summary.json`
- `reports/model_robustness_summary.csv`
- `reports/gpu_gnn_benchmark.csv`
- `reports/gpu_gnn_performance_summary.json`
- `reports/ranked_egfr_dataset.csv`
- `reports/marketed_egfr_scored.csv`
- `reports/marketed_egfr_structural_benchmark.csv`
- `reports/generated_analogs_ranked.csv`
- `reports/generated_analogs_ranked.summary.json`
- `reports/ai_guided_analogs.summary.json`
- `reports/iterative_ai_optimized_candidates.summary.json`
- `reports/generated_analogs_ranked_structural_crossdb.csv`
- `reports/iterative_ai_optimized_candidates.csv`
- `reports/iterative_ai_optimized_candidates_feasibility.csv`
- `reports/iterative_ai_optimized_candidates_readiness.csv`
- `reports/iterative_ai_optimized_candidates_crossdb.csv`
- `reports/iterative_ai_optimized_candidates_crossdb.summary.json`
- `data/processed/pubchem_egfr_reference.csv`
- `data/processed/papyrus_egfr_reference.csv`
- `data/processed/papyrus_egfr_reference.summary.json`
- `data/processed/excape_egfr_reference.csv`
- `data/processed/excape_egfr_reference.summary.json`
- `data/processed/pubchem_egfr_reference.summary.json`
- `data/processed/pubchem_egfr_assay_catalog.csv`
- `reports/final_diverse_candidates.csv`
- `reports/market_comparable_novel_shortlist.csv`
- `reports/prospective_validation_batch.csv`
- `reports/prospective_validation_batch.summary.json`
- `reports/rl_verifiable/rl_top_candidates.csv`
- `reports/rl_verifiable/rl_top_candidates_crossdb.csv`
- `reports/rl_verifiable/rl_training_summary.json`
- `reports/rl_gpu_dqn/gpu_rl_top_candidates.csv`
- `reports/rl_gpu_dqn/gpu_rl_training_summary.json`
- `reports/reward_hacking_challenge/reward_hacking_challenge_summary.csv`
- `reports/source_holdout_benchmark.csv`
- `reports/source_holdout_benchmark.json`
- `reports/rediscovery_benchmark/rediscovery_panel.csv`
- `reports/rediscovery_benchmark/rediscovery_summary.json`
- `reports/technical_notebook/technical_notebook_summary.md`
- `reports/technical_notebook/technical_notebook_metrics.json`
- `reports/technical_notebook_history/context_memory.md`
- `reports/technical_notebook_quick/technical_notebook_summary.md`
- `reports/technical_notebook_quick/technical_notebook_metrics.json`