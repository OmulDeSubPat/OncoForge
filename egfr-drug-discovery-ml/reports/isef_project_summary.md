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
- Scaffold split RMSE / R2: `0.7387377174247358` / `0.7164146865622143`
- Temporal split RMSE / R2: `1.696926725066737` / `-1.0931397302759307`

## Audit Diagnostics
- Audit pass rate: `0.13754416413562265`
- Audit review rate: `0.6471827930329139`
- Audit fail rate: `0.21527304283146345`
- Median reward hacking risk: `0.2`
- Mean audit demotion: `1045.7554081695903` positions
- Feasibility pass rate on optimized candidates: `1.0`
- Best Vina affinity on optimized candidates: `-9.405`
- Mean interaction support on optimized candidates: `0.7742222222222223`
- Mean experimental readiness: `0.7569876111549493`
- Cross-database mean consensus: `0.46347767146734126`
- Cross-database strong rate: `0.21333333333333335`
- External evidence mean support: `0.46723946551026746`
- External evidence pass rate: `0.6933333333333334`
- Evidence arbiter mean support: `0.6489876090586252`
- Evidence arbiter pass rate: `0.8`
- Papyrus molecules / mean support: `7323` / `0.4440460194190368`
- ExCAPE molecules / mean support: `5009` / `0.42960830412890716`
- PubChem mean enriched evidence: `0.25079777818118626`
- PubChem strong evidence rate: `0.015457788347205707`
- PubChem virtual/proxy exposure rate: `0.1759809750297265`
- Prospective validation batch size: `18`
- Prospective mean acquisition score: `1.05659231908917`
- Prospective mean structure-evidence support: `0.694305511826595`
- Broad analog benchmark count / mean generator priority: `1841` / `0.8613714843168803`
- Broad analog mean adaptive prior: `0.6965573179190007`
- AI-guided benchmark count / mean generator priority: `1436` / `0.857666061293726`
- AI-guided mean adaptive prior: `0.6028420528232306`
- Iterative benchmark count / top mean final score: `1725` / `10.389418239102628`
- Iterative mean adaptive prior: `0.3856983394335749`
- Generator suite artifact present: `True`
- Ablation suite artifact present: `True`
- RL mean cross-database consensus: `0.4632687118503663`
- RL mean external evidence support: `0.4532101163652832`
- RL mean structure-evidence support: `0.6318537518612249`
- RL ready rate: `0.45`
- GPU GNN scaffold snapshot: `multiview_reference | RMSE 0.739`
- GPU RL mean external evidence support: `0.46774799840493514`
- GPU RL mean structure-evidence support: `0.6036954004260144`
- GPU RL best episode return: `9.363332943920927`
- GPU actor-critic mean external evidence support: `0.5050015843423694`
- GPU actor-critic mean structure-evidence support: `0.5865416083934687`
- GPU actor-critic best episode return: `9.009732373428726`
- Best repeated-seed scaffold model: `multiview_ensemble | RMSE 0.739 +/- 0.000`
- Reward-hacking challenge snapshot: `trusted pass 1.000, proxy demoted>=20 1.000, proxy review/fail 0.580`
- Source holdout snapshot: `best excape_chembl20 RMSE 0.828, mean recall@20% 0.723`
- Rediscovery benchmark snapshot: `protected top10 0.368, naive top10 0.000, protected median rank 130.0`

## Verifiable RL Diagnostics
- Best episode return: `7.8617046853957255`
- Mean episode return: `5.034713648709077`

## GPU RL Diagnostics
- Best episode return: `9.363332943920927`
- Mean episode return: `5.889560339185893`

## Top Ranked Training-Space Molecules
| smiles | predicted_pIC50 | QED | reward_hacking_risk | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| C/C=C/C(=O)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.24621861789742 | 0.6685952655904649 | 0.0 | pass | 10.749370548939227 |
| COC(C)(C)COc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1C)C(=O)NCC2 | 9.64492553317416 | 0.5102437252597349 | 0.0 | pass | 10.698916496302727 |
| CCOc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.6213947221666 | 0.4521900852216164 | 0.0 | pass | 10.64509070761073 |
| [2H]C([2H])([2H])Oc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.386549939172664 | 0.4855433472748625 | 0.0 | pass | 10.549651298502416 |
| CNc1ncc2ncnc(Nc3cccc(Br)c3)c2n1 | 8.963399705573321 | 0.7684706048512752 | 0.0 | pass | 10.547735227339848 |

## Scored Marketed EGFR Drugs
| name | predicted_pIC50 | vina_affinity_kcal | interaction_support_score | docking_rescore | final_score |
| --- | --- | --- | --- | --- | --- |
| Sunvozertinib | 8.46419220283475 | -8.383 | 0.76 | 0.7835104889027267 | 9.52021018485538 |
| Osimertinib | 8.454906685137393 | -8.79 | 0.8133333333333334 | 0.8377011954524191 | 9.064268118923406 |
| Gefitinib | 7.636176314075532 | -8.852 | 0.9333333333333332 | 0.8481458156764541 | 7.87364504984742 |
| Dacomitinib | 7.646033286866971 | -8.664 | 0.8133333333333334 | 0.8233868445370008 | 7.68615424526805 |
| Erlotinib | 7.668325671346465 | -8.175 | 0.96 | 0.7600710128516652 | 5.36156209009101 |

## Diverse Generated Candidates
| smiles | predicted_pIC50 | QED | reward_hacking_risk | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| CC(Nc1ncnc2[nH]c(-c3ccc(CO)cc3O)cc12)c1ccccc1 | 9.16692333454908 | 0.4315961143710816 | 0.0 | pass | 10.416755097041785 |
| [2H]C([2H])([2H])Oc1c(OC)cccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.228592024598932 | 0.4424444747416858 | 0.0 | pass | 10.359611863631136 |
| COc1cc(-c2cc3c(NC(C)c4ccccc4)ncnc3[nH]2)c(OC)cc1CO | 9.283308630803573 | 0.4251416065263941 | 0.0 | pass | 10.402258024033754 |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.972629986228894 | 0.6673221117389069 | 0.0 | pass | 10.620566501041786 |
| CN(COCO)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.902024069699786 | 0.6265265096325219 | 0.0 | pass | 10.634297772505784 |

## Market-Comparable Novel Shortlist
| smiles | predicted_pIC50 | QED | max_market_similarity | audit_status | final_score |
| --- | --- | --- | --- | --- | --- |
| COc1cc(F)c(F)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.08324074548758 | 0.4579005357474424 | 0.2857142857142857 | pass | 10.414110228595526 |
| CC(Nc1ncnc2[nH]c(-c3ccc(CO)cc3O)cc12)c1ccccc1 | 9.16692333454908 | 0.4315961143710816 | 0.2025316455696202 | pass | 10.372426290233893 |
| C#COCC(Nc1ncnc2[nH]c(-c3ccccc3OC)cc12)c1ccccc1 | 9.080887848582057 | 0.4623943998062069 | 0.231578947368421 | pass | 10.510157806727875 |
| C#COCC(Nc1ncnc2[nH]c(-c3ccccc3OCC)cc12)c1ccccc1 | 9.331308546958905 | 0.4178865792937883 | 0.2023809523809523 | pass | 10.547083422027365 |
| COc1cc(C#N)ccc1-c1cc2c(NC(CF)c3ccccc3)ncnc2[nH]1 | 9.015212815563302 | 0.5037673692390708 | 0.2307692307692307 | pass | 10.356342269652972 |

## Feasibility-Supported Optimized Candidates
| smiles | predicted_pIC50 | feasibility_score | vina_affinity_kcal | feasibility_status | max_active_similarity |
| --- | --- | --- | --- | --- | --- |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCCN(C)C)[nH]c2c1C(=O)NCC2 | 9.292313608973728 | 0.8982251177230288 | -8.94 | pass | 0.8405797101449275 |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCCOCF)[nH]c2c1C(=O)NCC2 | 9.453609949023404 | 0.8347266458587146 | -8.309 | pass | 0.8529411764705882 |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.972629986228892 | 0.8691313859424343 | -8.37 | pass | 0.8333333333333334 |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCCOC(F)F)[nH]c2c1C(=O)NCC2 | 9.415455874446565 | 0.8664823428874844 | -8.616 | pass | 0.8405797101449275 |
| Cc1c(F)cccc1Nc1c(-c2ccncc2OCC(C)(C)F)[nH]c2c1C(=O)NCC2 | 9.312988403947166 | 0.837226012165827 | -9.096 | pass | 0.855072463768116 |

## Experimental-Readiness Snapshot
| smiles | predicted_pIC50 | experimental_readiness_score | experimental_readiness_status | experimental_track | docking_rescore |
| --- | --- | --- | --- | --- | --- |
| COc1cc(F)c(F)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.08324074548758 | 0.8392290946124742 | ready | benchmark_ready | 0.7689637474630646 |
| COc1cc(C)c(C)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.113341225066414 | 0.8245175874618648 | ready | benchmark_ready | 0.7210563604847683 |
| CC(Nc1ncnc2[nH]c(-c3ccc(CO)cc3O)cc12)c1ccccc1 | 9.16692333454908 | 0.8287663105711461 | ready | benchmark_ready | 0.7607830846831743 |
| COc1cc(-c2cc3c(NC(C)c4ccccc4)ncnc3[nH]2)c(OC)cc1CO | 9.283308630803573 | 0.830604499323361 | ready | benchmark_ready | 0.709907429995418 |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.97262998622889 | 0.829635124878633 | ready | benchmark_ready | 0.6886850760219971 |

## Cross-Database Validation Snapshot
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | cross_database_independent_support_count | cross_database_status | experimental_readiness_score |
| --- | --- | --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.972629986228892 | 0.6756233784385562 | 0.6995199526859875 | 3 | strong | 0.5528947235689337 |
| CN(C)c1ncc2ncnc(Nc3cccc(Br)c3)c2n1 | 8.823788860318864 | 0.7768718149277987 | 0.7416470656194251 | 2 | strong | 0.602394370498466 |
| O=C(/C=C\Br)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.941211608052592 | 0.683597795032502 | 0.7059879723945909 | 3 | strong | 0.5492641682044882 |
| COc1ccccc1-c1cc2c(N[C@H](CF)c3ccccc3)ncnc2[nH]1 | 9.071690957623556 | 0.5439598672623045 | 0.567146373204751 | 2 | strong | 0.5404405222852615 |
| CC(=O)N(C)c1ncc2ncnc(Nc3cccc(Br)c3)c2n1 | 8.419067927143379 | 0.6499667846943299 | 0.6551517646545211 | 2 | strong | 0.5896072035426717 |

## Prospective Validation Batch
| prospective_batch_rank | candidate_source | predicted_pIC50 | experimental_readiness_score | prospective_acquisition_score | experimental_readiness_status |
| --- | --- | --- | --- | --- | --- |
| 1 | shortlist | 9.08324074548758 | 0.8392034867735595 | 1.1725305798669483 | ready |
| 2 | shortlist | 9.16692333454908 | 0.8287368609493163 | 1.1485440109278968 | ready |
| 3 | diverse | 8.902024069699786 | 0.8330357212655548 | 1.1330954983176122 | ready |
| 4 | diverse | 8.941917847337407 | 0.8010778050315156 | 1.1114634421692977 | ready |
| 5 | diverse | 9.228592024598932 | 0.8348273284381087 | 1.1078663574718968 | ready |

## Verifiable RL Candidates
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | experimental_readiness_score | rl_priority_score |
| --- | --- | --- | --- | --- | --- |
| C#Cc1ccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c(OCC)c1 | 9.426633674968434 | 0.5357600946037698 | 0.5437538002803827 | 0.7932390892992975 | 12.256336296829415 |
| CCOc1cc(F)c(OC)cc1-c1cc2c(NC(CO)c3ccccc3)ncnc2[nH]1 | 9.268580967720943 | 0.5418417509972525 | 0.5456824567050277 | 0.7611212542396901 | 11.93540276939071 |
| CCOc1cc(F)c(OC)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.268580967720943 | 0.5418417509972525 | 0.5456824567050277 | 0.7610524050644931 | 11.93452564974498 |
| CCOc1cc(C#N)c(OC)cc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.177113686402464 | 0.5180265881897491 | 0.520708582972952 | 0.7845906465180059 | 11.914687438607924 |
| COc1cc(-c2cc3c(N[C@H](C)c4ccccc4)ncnc3[nH]2)c(OC)cc1C(N)=O | 9.034397654999644 | 0.5545453853111878 | 0.5438110648870471 | 0.8126594827383532 | 11.807680191197472 |

## GPU DQN RL Candidates
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | evidence_arbiter_support | gpu_rl_priority_score |
| --- | --- | --- | --- | --- | --- |
| CN(CO)CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.886919612836042 | 0.7014110353976795 | 0.7571787859500023 | 0.7864466945377588 | 12.349805016137092 |
| CCOc1cc(CO)ccc1-c1cc2c(N[C@H](C)c3ccccc3)ncnc2[nH]1 | 9.366026856646968 | 0.5632923998162127 | 0.5766702304711735 | 0.7249626241386451 | 12.181857959882995 |
| CCOc1cccc(-c2cc3c(N[C@H](CO)c4ccccc4)ncnc3[nH]2)c1OC | 9.38283067919044 | 0.5503410755706755 | 0.5501287988258726 | 0.694111683349421 | 12.13872346488448 |
| CN(COCO)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.902024069699786 | 0.695140871417389 | 0.7533011483229155 | 0.652058486619714 | 12.09414067454941 |
| CCOc1ccccc1-c1cc2c(N[C@H](CO)c3ccccc3)ncnc2[nH]1 | 9.621394722166604 | 0.5902305767946632 | 0.5823146763774594 | 0.6014897261470167 | 12.05003425399052 |

## GPU Actor-Critic Candidates
| smiles | predicted_pIC50 | cross_database_consensus_score | external_evidence_support | experimental_readiness_score | actor_critic_priority_score |
| --- | --- | --- | --- | --- | --- |
| O=C(/C=C\F)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.972629986228892 | 0.6892018730622121 | 0.7303277343237046 | 0.8281137421600975 | 12.179749124176736 |
| O=C(/C=C\Cl)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 9.045329269414507 | 0.7528100574422054 | 0.7699592026825351 | 0.8684480850366674 | 12.047076312154667 |
| C=CNCc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.533679113475976 | 0.6646588940938674 | 0.7325442258309528 | 0.8338937490146051 | 11.859274659171676 |
| O=C(/C=C\Br)Cc1cc2c(Nc3cccc(Br)c3)ncnc2cn1 | 8.94121160805259 | 0.6975172506464827 | 0.7378096511259074 | 0.8276201910398203 | 11.970640908739943 |
| COCCOc1cnccc1-c1[nH]c2c(c1Nc1cccc(F)c1F)C(=O)NCC2 | 9.195620914997408 | 0.38244683133467 | 0.3402651932958304 | 0.772271767825734 | 12.042617180845031 |

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
- `reports/generation_benchmark_suite.csv`
- `reports/studii_ablatie/studii_ablatie.csv`
- `reports/studii_ablatie/rezumat_studii_ablatie.md`
- `reports/generated_analogs_ranked_structural_crossdb.csv`
- `reports/ai_guided_analogs_structural_crossdb.csv`
- `reports/iterative_ai_optimized_candidates.csv`
- `reports/iterative_ai_optimized_candidates_structural_feasibility.csv`
- `reports/iterative_ai_optimized_candidates_structural_crossdb.csv`
- `reports/iterative_ai_optimized_candidates_structural_crossdb.summary.json`
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
- `reports/rl_gpu_actor_critic/gpu_actor_critic_top_candidates.csv`
- `reports/rl_gpu_actor_critic/gpu_actor_critic_summary.json`
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