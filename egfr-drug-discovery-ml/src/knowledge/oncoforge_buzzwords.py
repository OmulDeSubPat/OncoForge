from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuzzwordEntry:
    term: str
    category: str
    used_in_project: bool
    short_definition: str
    detailed_explanation: str
    why_it_matters: str
    oncoforge_usage: str
    common_pitfall: str
    related_terms: tuple[str, ...] = ()


E = BuzzwordEntry


BUZZWORD_ENTRIES: tuple[BuzzwordEntry, ...] = (
    E(
        term="Multi-agent system",
        category="Project Architecture",
        used_in_project=True,
        short_definition="A design where multiple specialized scorers judge the same molecule from different angles.",
        detailed_explanation=(
            "A multi-agent system splits decision-making into roles. One component can focus on potency, another on chemistry quality, "
            "another on safety, and another on whether the molecule is even inside the evidence envelope of the data."
        ),
        why_it_matters="Drug discovery is naturally multi-objective, so one score is rarely enough.",
        oncoforge_usage="OncoForge uses separate potency, chemistry, safety, domain, and audit logic before final ranking.",
        common_pitfall="It is not truly multi-agent if the extra columns never change the final decision.",
        related_terms=("Audit agent", "Verified reward", "Reward hacking"),
    ),
    E(
        term="Verifiable reward",
        category="Project Architecture",
        used_in_project=True,
        short_definition="A reward made from components that can be traced and checked after the run.",
        detailed_explanation=(
            "In reinforcement learning, reward tells the agent what is good. A verifiable reward is not a mystery number. "
            "It is built from explicit pieces such as potency, feasibility, structural support, novelty, and penalties."
        ),
        why_it_matters="If reward cannot be explained afterward, the optimizer can exploit loopholes without anyone noticing.",
        oncoforge_usage="OncoForge stores reward pieces and compares naive reward with protected reward.",
        common_pitfall="A reward is not verifiable if the final score cannot be decomposed afterward.",
        related_terms=("Reward shaping", "Reward hacking", "Verified reward"),
    ),
    E(
        term="Reward hacking",
        category="Project Architecture",
        used_in_project=True,
        short_definition="When an optimizer finds ways to score highly without solving the real scientific goal.",
        detailed_explanation=(
            "In molecular design, reward hacking can mean exploiting a QSAR model, drifting outside the training distribution, "
            "or proposing reactive chemistry that looks numerically strong but is not a realistic lead."
        ),
        why_it_matters="This is one of the main reasons generated molecules can look impressive but fail under scrutiny.",
        oncoforge_usage="OncoForge computes reward-hacking risk and uses audit penalties to demote suspicious molecules.",
        common_pitfall="It is not enough to mention reward hacking in the report; the code must actually detect and penalize it.",
        related_terms=("Audit agent", "Applicability domain", "Naive reward"),
    ),
    E(
        term="Audit agent",
        category="Project Architecture",
        used_in_project=True,
        short_definition="The logic that asks whether a molecule looks good for the right reasons.",
        detailed_explanation=(
            "An audit layer checks whether strong scores are supported by safety, domain, uncertainty, and agreement signals. "
            "It acts like a control system for the rest of the pipeline."
        ),
        why_it_matters="Without audit logic, proxy scores can quietly dominate selection.",
        oncoforge_usage="OncoForge uses pass, review, and fail audit states plus anti-hacking penalties.",
        common_pitfall="An audit that only logs warnings but never changes ranking is much weaker than one that changes selection.",
        related_terms=("Multi-agent system", "Verified reward", "Reward hacking"),
    ),
    E(
        term="Verified reward",
        category="Project Architecture",
        used_in_project=True,
        short_definition="The final protected reward after penalties and support checks are applied.",
        detailed_explanation=(
            "A naive reward shows what the optimizer would do if it mostly chased potency. A verified reward is the corrected version "
            "after audit logic, uncertainty, domain support, and chemistry guardrails are applied."
        ),
        why_it_matters="This is closer to the score you would actually trust when choosing candidates.",
        oncoforge_usage="OncoForge stores verified reward separately from naive reward and uses it in RL and ranking.",
        common_pitfall="If verified reward is almost identical to naive reward for every molecule, the protection layer is probably weak.",
        related_terms=("Naive reward", "Verifiable reward", "Reward hacking"),
    ),
    E(
        term="Naive reward",
        category="Project Architecture",
        used_in_project=True,
        short_definition="The raw score before stronger protective checks are applied.",
        detailed_explanation=(
            "Naive reward is a baseline. It shows what the system would choose if it optimized surface-level metrics without strong anti-hacking controls."
        ),
        why_it_matters="Comparing naive reward with protected reward is a direct way to prove the audit layer matters.",
        oncoforge_usage="OncoForge stores naive score and naive rank, then compares them against protected ranking.",
        common_pitfall="If a project only shows the final protected score, it cannot prove that protection changed anything.",
        related_terms=("Verified reward", "Audit demotion", "Reward hacking"),
    ),
    E(
        term="Applicability domain",
        category="Machine Learning",
        used_in_project=True,
        short_definition="The region of chemistry where a predictive model has enough similar examples to be trusted.",
        detailed_explanation=(
            "A model is strongest when a new molecule resembles data it has seen before. Applicability domain measures whether a candidate "
            "is close enough to the training distribution to deserve confidence."
        ),
        why_it_matters="A potent prediction outside the applicability domain is much easier to over-trust than it should be.",
        oncoforge_usage="OncoForge computes train-set similarity and uses applicability in both scoring and anti-hacking logic.",
        common_pitfall="Novelty is not automatically good if the model has no evidence there.",
        related_terms=("Novelty", "Uncertainty estimation", "Tanimoto similarity"),
    ),
    E(
        term="Novelty",
        category="Machine Learning",
        used_in_project=True,
        short_definition="How different a molecule is from the training set or from marketed drugs.",
        detailed_explanation=(
            "Novelty matters because you want genuinely new candidates, not just copies of known chemistry. But useful novelty still needs evidence and plausibility."
        ),
        why_it_matters="The best molecules are often new enough to matter but not so strange that the model becomes unreliable.",
        oncoforge_usage="OncoForge measures novelty against training molecules and marketed EGFR drugs.",
        common_pitfall="A novelty metric can reward weirdness if it is not balanced with applicability and feasibility.",
        related_terms=("Applicability domain", "Diversity", "Market novelty score"),
    ),
    E(
        term="Diversity",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A way to keep the final shortlist from collapsing into many near-duplicate molecules.",
        detailed_explanation=(
            "A pipeline can produce many molecules that are almost the same. Diversity control keeps a broader portfolio of ideas by filtering overly similar candidates."
        ),
        why_it_matters="A diverse shortlist is more useful experimentally and scientifically than one chemistry repeated twenty times.",
        oncoforge_usage="OncoForge applies similarity filtering to create the final diverse candidate set.",
        common_pitfall="Too much diversity pressure can remove excellent chemistry if it is applied before quality filters.",
        related_terms=("Tanimoto similarity", "Scaffold", "Final diverse candidates"),
    ),
    E(
        term="Feature engineering",
        category="Machine Learning",
        used_in_project=True,
        short_definition="Turning raw molecules into numerical signals that models can learn from.",
        detailed_explanation=(
            "Models cannot directly read a molecule the way a chemist does. Feature engineering converts molecular structure into fingerprints, descriptors, and hybrid vectors."
        ),
        why_it_matters="Good features often matter as much as model choice on medicinal-chemistry datasets.",
        oncoforge_usage="OncoForge uses fingerprints, descriptors, and hybrid features in the multiview ensemble.",
        common_pitfall="Adding more features is not automatically better if they are noisy or redundant.",
        related_terms=("Descriptor", "ECFP", "Multiview ensemble"),
    ),
    E(
        term="Descriptor",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A handcrafted numerical property of a molecule, such as molecular weight or polar surface area.",
        detailed_explanation=(
            "Descriptors summarize chemistry in ways medicinal chemists recognize: size, lipophilicity, hydrogen-bonding capacity, ring counts, and more."
        ),
        why_it_matters="Descriptors help the model understand overall drug-like character, not just local fragments.",
        oncoforge_usage="OncoForge uses descriptor vectors alongside fingerprints in model training and chemistry scoring.",
        common_pitfall="Descriptors are useful summaries, but they are not direct biology.",
        related_terms=("Feature engineering", "QED", "Lipinski rule of five"),
    ),
    E(
        term="ECFP / Morgan fingerprint",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A circular fingerprint that encodes local atom neighborhoods into a fixed-length vector.",
        detailed_explanation=(
            "Morgan fingerprints start at each atom, look outward in layers, and convert those local environments into a machine-friendly representation."
        ),
        why_it_matters="This is one of the most widely used molecular representations for QSAR, similarity, and diversity.",
        oncoforge_usage="OncoForge uses ECFP-like fingerprints for modeling, novelty, similarity, and ranking.",
        common_pitfall="Fingerprints are powerful, but they do not directly capture full 3D binding geometry.",
        related_terms=("Tanimoto similarity", "QSAR", "Feature engineering"),
    ),
    E(
        term="Ensemble learning",
        category="Machine Learning",
        used_in_project=True,
        short_definition="Combining multiple models so the final prediction is more stable than any one model alone.",
        detailed_explanation=(
            "Different models make different mistakes. An ensemble averages or combines them so that one model's weakness is partly canceled by another."
        ),
        why_it_matters="Ensembles are often more robust than single models on noisy chemistry datasets.",
        oncoforge_usage="OncoForge uses a multiview ensemble built from tree-based model families.",
        common_pitfall="An ensemble of many nearly identical models may not help as much as a more diverse one.",
        related_terms=("Random Forest", "Extra Trees", "Uncertainty estimation"),
    ),
    E(
        term="Random Forest",
        category="Machine Learning",
        used_in_project=True,
        short_definition="An ensemble of decision trees trained on random subsets of the data and features.",
        detailed_explanation=(
            "Each tree sees a different view of the problem, and the forest averages them to reduce overfitting compared with a single tree."
        ),
        why_it_matters="Random Forest remains a strong baseline for many QSAR problems.",
        oncoforge_usage="OncoForge used Random Forest in earlier stages and still benchmarks it among model families.",
        common_pitfall="People sometimes dismiss tree models too quickly because deep learning exists.",
        related_terms=("Extra Trees", "Ensemble learning", "QSAR"),
    ),
    E(
        term="Extra Trees",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A tree ensemble similar to Random Forest, but with more randomness in how splits are chosen.",
        detailed_explanation=(
            "Extra Trees increase randomness beyond Random Forest, which can improve robustness and reduce variance on some tasks."
        ),
        why_it_matters="It is a strong and complementary model family for molecular fingerprints.",
        oncoforge_usage="The current multiview ensemble in OncoForge includes Extra Trees as a core component.",
        common_pitfall="More randomness is not automatically better; it must still improve generalization.",
        related_terms=("Random Forest", "Scaffold split", "Ensemble learning"),
    ),
    E(
        term="HistGradientBoosting",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A boosting model that builds trees sequentially to fix previous prediction errors.",
        detailed_explanation=(
            "Unlike bagging methods that average independent trees, gradient boosting adds trees one after another so each new tree learns from the current residual error."
        ),
        why_it_matters="It gives the ensemble a different model family and improves diversity.",
        oncoforge_usage="OncoForge includes HistGradientBoosting in the multiview ensemble benchmark and final model set.",
        common_pitfall="Boosting can overfit if regularization is not handled carefully.",
        related_terms=("Ensemble learning", "Extra Trees", "Model family benchmark"),
    ),
    E(
        term="Uncertainty estimation",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A measure of how confident the model is about a prediction.",
        detailed_explanation=(
            "Two molecules can receive the same predicted potency while having very different levels of support. Uncertainty helps separate stronger inferences from weaker ones."
        ),
        why_it_matters="In discovery pipelines, confidence often matters almost as much as the prediction itself.",
        oncoforge_usage="OncoForge uses ensemble disagreement as uncertainty and penalizes risky predictions.",
        common_pitfall="A model can output uncertainty numbers that still do not match real error rates.",
        related_terms=("Calibration", "Ensemble learning", "Applicability domain"),
    ),
    E(
        term="Calibration",
        category="Machine Learning",
        used_in_project=True,
        short_definition="Adjusting uncertainty estimates so they better match real-world prediction error.",
        detailed_explanation=(
            "A model can say it is uncertain, but that uncertainty may still be too small or too large. Calibration rescales it so the uncertainty becomes more honest."
        ),
        why_it_matters="Without calibration, uncertainty can look scientific while still being misleading.",
        oncoforge_usage="OncoForge stores uncertainty scaling and reports calibration over multiple split regimes.",
        common_pitfall="A raw ensemble standard deviation is not automatically calibrated.",
        related_terms=("Uncertainty estimation", "Temporal split", "Scaffold split"),
    ),
    E(
        term="Scaffold split",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A train-test split where molecules are separated by scaffold instead of at random.",
        detailed_explanation=(
            "Scaffold split is harder than random split because the model must generalize to new chemotypes rather than to close analogs."
        ),
        why_it_matters="It is much closer to the real challenge of proposing chemistry that the model has not memorized.",
        oncoforge_usage="OncoForge treats scaffold split as one of its main credibility metrics.",
        common_pitfall="Projects often overstate performance by reporting only random split results.",
        related_terms=("Murcko scaffold", "Random split", "Temporal split"),
    ),
    E(
        term="Temporal split",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A train-test split based on time, where older data trains the model and newer data tests it.",
        detailed_explanation=(
            "Temporal split mimics prospective use more closely because the model is tested on chemistry that became available later."
        ),
        why_it_matters="It is often harsher than random split and therefore more realistic.",
        oncoforge_usage="OncoForge reports temporal split metrics when dated assay information is available.",
        common_pitfall="Weak temporal performance often reveals reality rather than failure.",
        related_terms=("Random split", "Scaffold split", "Calibration"),
    ),
    E(
        term="Source holdout benchmark",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A validation setup where one public source is left out of training and used only for testing.",
        detailed_explanation=(
            "A source holdout benchmark checks whether a model trained on the rest of the evidence can still generalize to chemistry that is concentrated in a different database distribution."
        ),
        why_it_matters="It is a more realistic external-validation stress test than reshuffling one merged dataset.",
        oncoforge_usage="OncoForge now reports leave-one-source-out performance on exclusive ChEMBL, BindingDB, Papyrus, and ExCAPE subsets.",
        common_pitfall="If the same molecules leak into both train and test, the holdout is much less convincing.",
        related_terms=("External validation", "Scaffold split", "Generalization"),
    ),
    E(
        term="Rediscovery benchmark",
        category="Project Architecture",
        used_in_project=True,
        short_definition="A benchmark that asks whether known strong molecules are recovered inside a hard panel.",
        detailed_explanation=(
            "Rediscovery is a sanity check. The system is challenged to surface marketed or literature-supported EGFR positives even when they compete against strong generated or ranked challengers."
        ),
        why_it_matters="It helps show that the ranking system does not lose contact with known medicinal-chemistry reality.",
        oncoforge_usage="OncoForge now compares naive and protected recovery of marketed and IUPHAR EGFR positives inside a rediscovery panel.",
        common_pitfall="If novelty penalties are left untouched, rediscovery can look artificially weak because known controls are punished for being known.",
        related_terms=("Protected ranking", "External evidence", "Benchmark panel"),
    ),
    E(
        term="QSAR",
        category="Machine Learning",
        used_in_project=True,
        short_definition="Quantitative Structure-Activity Relationship: predicting biological activity from molecular structure.",
        detailed_explanation=(
            "QSAR models learn statistical links between what a molecule looks like and how active it tends to be in experiments."
        ),
        why_it_matters="It is still one of the fastest ways to prioritize molecules before wet-lab work.",
        oncoforge_usage="The core predictor in OncoForge is a QSAR stack for EGFR potency.",
        common_pitfall="QSAR predicts correlations, not full mechanism.",
        related_terms=("pIC50", "Regression", "Descriptors"),
    ),
    E(
        term="Reinforcement learning",
        category="Reinforcement Learning",
        used_in_project=True,
        short_definition="A learning setup where an agent improves by taking actions and receiving reward signals.",
        detailed_explanation=(
            "Instead of learning from fixed correct answers, the agent tries actions, receives feedback, and updates its strategy so future choices improve."
        ),
        why_it_matters="It fits molecule editing naturally because generation can be framed as a sequence of decisions.",
        oncoforge_usage="OncoForge includes a verifiable-reward RL loop over medicinal-chemistry actions.",
        common_pitfall="Not every iterative search is true RL; the system needs state, action, reward, and learning updates.",
        related_terms=("Q-learning", "Policy", "Reward shaping"),
    ),
    E(
        term="Q-learning",
        category="Reinforcement Learning",
        used_in_project=True,
        short_definition="An RL algorithm that learns how valuable each action is in each state.",
        detailed_explanation=(
            "Q-learning stores values for state-action pairs and updates them using observed rewards and future value estimates."
        ),
        why_it_matters="It is simple, interpretable, and a good first RL baseline.",
        oncoforge_usage="OncoForge uses tabular Q-learning with replay and action priors.",
        common_pitfall="Q-learning becomes hard when the state or action space grows too large.",
        related_terms=("State", "Policy", "Replay buffer"),
    ),
    E(
        term="Replay buffer",
        category="Reinforcement Learning",
        used_in_project=True,
        short_definition="A memory of earlier RL experiences that can be replayed for more stable learning.",
        detailed_explanation=(
            "Instead of learning only from the most recent step, the agent stores useful transitions and revisits them. "
            "This improves sample efficiency and stability."
        ),
        why_it_matters="Replay makes better use of expensive molecular evaluations.",
        oncoforge_usage="OncoForge uses replay inside the RL training loop.",
        common_pitfall="Replay helps only if the experiences stored are relevant and diverse enough.",
        related_terms=("Q-learning", "Reward shaping", "Action priors"),
    ),
    E(
        term="Reward shaping",
        category="Reinforcement Learning",
        used_in_project=True,
        short_definition="Designing reward components so the RL agent learns faster and more safely.",
        detailed_explanation=(
            "Reward shaping adds intermediate guidance, such as potency improvement, structural support, or penalties, instead of waiting for a single final signal."
        ),
        why_it_matters="It can make RL usable on real chemistry problems instead of purely toy examples.",
        oncoforge_usage="OncoForge rewards potency, feasibility, structural support, and interaction support while penalizing risky behavior.",
        common_pitfall="Poor reward shaping can itself create loopholes and cause reward hacking.",
        related_terms=("Verifiable reward", "Reward hacking", "Q-learning"),
    ),
    E(
        term="SMILES",
        category="Cheminformatics",
        used_in_project=True,
        short_definition="A text notation that represents a molecule as a string.",
        detailed_explanation=(
            "SMILES lets chemistry be stored and processed in ordinary files. Atoms, bonds, branches, rings, and stereochemistry can all be encoded as text."
        ),
        why_it_matters="Most cheminformatics workflows start from SMILES because it is compact and easy to move between tools.",
        oncoforge_usage="OncoForge reads, cleans, mutates, scores, and ranks molecules mainly through SMILES-based workflows.",
        common_pitfall="Different SMILES strings can represent the same molecule, so standardization matters.",
        related_terms=("Canonical SMILES", "RDKit", "Fingerprints"),
    ),
    E(
        term="Canonical SMILES",
        category="Cheminformatics",
        used_in_project=True,
        short_definition="A standardized SMILES form used so one molecule maps to one consistent string.",
        detailed_explanation=(
            "Canonicalization removes many string-level differences that would otherwise make identical molecules look different in files."
        ),
        why_it_matters="Without canonical SMILES, duplicates can contaminate datasets and confuse ranking.",
        oncoforge_usage="OncoForge canonicalizes imported and generated molecules before deduplication and scoring.",
        common_pitfall="Canonicalization standardizes valid structures; it does not repair invalid chemistry.",
        related_terms=("SMILES", "Data cleaning", "Deduplication"),
    ),
    E(
        term="Scaffold / Murcko scaffold",
        category="Cheminformatics",
        used_in_project=True,
        short_definition="The core molecular backbone obtained after stripping away many side chains.",
        detailed_explanation=(
            "A scaffold captures the core chemotype of a molecule and is often used to group related compounds."
        ),
        why_it_matters="Scaffolds help answer whether the model generalizes beyond close analogs.",
        oncoforge_usage="OncoForge uses scaffold logic for validation splits, feasibility, and chemistry comparisons.",
        common_pitfall="Molecules with the same scaffold can still behave very differently because side chains matter.",
        related_terms=("Scaffold split", "Chemotype", "Murcko scaffold"),
    ),
    E(
        term="Matched molecular pair",
        category="Medicinal Chemistry",
        used_in_project=True,
        short_definition="A pair of molecules that differ by one well-defined chemical change.",
        detailed_explanation=(
            "Matched molecular pairs let chemists reason about specific edits, such as bromine to chlorine or methoxy to ethoxy."
        ),
        why_it_matters="This is far closer to real medicinal chemistry than unconstrained random generation.",
        oncoforge_usage="OncoForge uses matched transformations and related med-chem rules to generate controllable analogs.",
        common_pitfall="A useful transformation on one scaffold does not automatically transfer to every scaffold.",
        related_terms=("Lead optimization", "Reaction-aware generation", "Medicinal chemistry transformation"),
    ),
    E(
        term="BRICS fragments",
        category="Cheminformatics",
        used_in_project=True,
        short_definition="A fragment decomposition scheme that breaks molecules into chemically meaningful pieces.",
        detailed_explanation=(
            "BRICS applies rule-based cuts so fragment frequencies and reuse can be tracked across large molecule collections."
        ),
        why_it_matters="Fragment support is useful evidence that a generated molecule still resembles real medicinal chemistry.",
        oncoforge_usage="OncoForge uses BRICS-derived fragment support inside feasibility scoring.",
        common_pitfall="Fragments are supportive evidence, not proof that a full molecule is practical or active.",
        related_terms=("Feasibility score", "Fragment support", "Scaffold"),
    ),
    E(
        term="Tanimoto similarity",
        category="Cheminformatics",
        used_in_project=True,
        short_definition="A common similarity measure for molecular fingerprints.",
        detailed_explanation=(
            "Tanimoto compares the overlap between two fingerprint vectors. Higher values usually mean more shared structural patterns."
        ),
        why_it_matters="It underlies novelty checks, diversity filtering, applicability estimation, and active-neighbor support.",
        oncoforge_usage="OncoForge uses Tanimoto similarity for train-set similarity, marketed-drug comparison, and shortlist diversity control.",
        common_pitfall="Similarity depends heavily on which fingerprint representation you chose.",
        related_terms=("ECFP", "Novelty", "Applicability domain"),
    ),
    E(
        term="QED",
        category="Medicinal Chemistry",
        used_in_project=True,
        short_definition="Quantitative Estimate of Drug-likeness, a summary score for drug-like property balance.",
        detailed_explanation=(
            "QED combines molecular size, lipophilicity, polarity, and hydrogen-bond features into a single drug-likeness-style score."
        ),
        why_it_matters="It is a convenient summary of whether a molecule looks somewhat drug-like in early prioritization.",
        oncoforge_usage="OncoForge uses QED in chemistry support, ranking, RL reporting, and technical summaries.",
        common_pitfall="QED is useful, but it is not a substitute for potency, safety, or selectivity.",
        related_terms=("Descriptors", "Lipinski rule of five", "Chemistry agent"),
    ),
    E(
        term="SA score",
        category="Medicinal Chemistry",
        used_in_project=True,
        short_definition="Synthetic Accessibility score, an estimate of how easy or hard a molecule may be to make.",
        detailed_explanation=(
            "SA scores usually combine fragment familiarity and structural complexity to estimate synthetic difficulty."
        ),
        why_it_matters="A molecule that looks powerful but is unrealistically hard to synthesize is much less actionable.",
        oncoforge_usage="OncoForge uses SA-like logic in chemistry scoring and feasibility assessment.",
        common_pitfall="SA score is a heuristic, not a full retrosynthesis engine.",
        related_terms=("Feasibility score", "Reaction-aware generation", "Synthetic tractability"),
    ),
    E(
        term="PAINS",
        category="Medicinal Chemistry",
        used_in_project=True,
        short_definition="Pan-Assay Interference Compounds, motifs that often create misleading assay readouts.",
        detailed_explanation=(
            "Some chemical motifs frequently look active for the wrong reasons, such as reactivity, aggregation, or assay interference."
        ),
        why_it_matters="It helps reduce the chance of chasing false positives.",
        oncoforge_usage="OncoForge checks PAINS-like alerts as part of safety and chemistry screening.",
        common_pitfall="PAINS filters are warnings, not automatic scientific truth.",
        related_terms=("Structural alerts", "Safety agent", "Reactive warhead"),
    ),
    E(
        term="Lipinski rule of five",
        category="Medicinal Chemistry",
        used_in_project=True,
        short_definition="A famous rule-of-thumb about property ranges that often support oral drug-likeness.",
        detailed_explanation=(
            "Lipinski-style rules look at molecular weight, lipophilicity, hydrogen-bond donors, and hydrogen-bond acceptors."
        ),
        why_it_matters="They help keep generated molecules in a more realistic medicinal-chemistry region.",
        oncoforge_usage="OncoForge tracks Lipinski pressure in chemistry scoring and feasibility logic.",
        common_pitfall="Many real drugs break one or more rules, so they are guidelines rather than absolute laws.",
        related_terms=("QED", "Descriptors", "Drug-likeness"),
    ),
    E(
        term="Docking",
        category="Structural Biology",
        used_in_project=True,
        short_definition="Computationally placing a molecule into a protein pocket to estimate fit and pose.",
        detailed_explanation=(
            "Docking searches for plausible binding orientations and reports a scoring estimate. It gives an orthogonal structural view beyond ligand-only models."
        ),
        why_it_matters="Docking helps answer whether a molecule at least looks structurally compatible with the target pocket.",
        oncoforge_usage="OncoForge docks top candidates and marketed comparators into the EGFR receptor workflow.",
        common_pitfall="Docking scores are not the same thing as experimental binding affinity.",
        related_terms=("AutoDock Vina", "Rescoring", "Pose"),
    ),
    E(
        term="AutoDock Vina",
        category="Structural Biology",
        used_in_project=True,
        short_definition="A popular molecular docking engine used to predict ligand poses and approximate binding scores.",
        detailed_explanation=(
            "Vina searches the target pocket for likely ligand placements and reports a score, usually in kcal/mol, for the best poses it finds."
        ),
        why_it_matters="It provides a standardized structural benchmark for comparing generated molecules against marketed drugs.",
        oncoforge_usage="OncoForge uses Vina in its structural consensus rescoring workflow and stores the resulting pose files.",
        common_pitfall="Vina scores should be interpreted comparatively, not as exact physical truth.",
        related_terms=("Docking", "Pose", "Vina affinity"),
    ),
    E(
        term="Rescoring",
        category="Structural Biology",
        used_in_project=True,
        short_definition="Re-evaluating molecules with an extra scoring layer after the initial ranking.",
        detailed_explanation=(
            "A molecule can first be ranked by QSAR and then re-checked structurally. This adds an orthogonal signal and often filters molecules that only looked good statistically."
        ),
        why_it_matters="Rescoring improves trust because final selection is supported by more than one kind of evidence.",
        oncoforge_usage="OncoForge rescoring combines Vina and reference-ligand alignment before final feasibility and shortlist selection.",
        common_pitfall="Rescoring is only valuable if it adds new information rather than repeating the same signal in a different form.",
        related_terms=("Docking", "Structural priority score", "Consensus score"),
    ),
    E(
        term="Hydrogen bond",
        category="Structural Biology",
        used_in_project=True,
        short_definition="A directional non-covalent interaction that often helps stabilize ligand binding.",
        detailed_explanation=(
            "Hydrogen bonds occur between donor and acceptor atoms and are one of the most interpretable contact types in ligand binding."
        ),
        why_it_matters="Named residue-level contacts are easier for jurors and scientists to interpret than raw docking scores alone.",
        oncoforge_usage="OncoForge counts hydrogen-bond-like contacts in docked poses as part of interaction support.",
        common_pitfall="Contact counts help, but geometry and context still matter.",
        related_terms=("Interaction support score", "Pose analysis", "Key residues"),
    ),
    E(
        term="Salt bridge",
        category="Structural Biology",
        used_in_project=True,
        short_definition="A strong electrostatic interaction between oppositely charged groups.",
        detailed_explanation=(
            "Salt bridges are often treated as especially important contacts because charge complementarity can strongly stabilize a pose."
        ),
        why_it_matters="They provide interpretable structural evidence when present in plausible poses.",
        oncoforge_usage="OncoForge includes salt-bridge-like contacts in the interaction analysis summary.",
        common_pitfall="A predicted salt bridge is meaningful only if the atoms and geometry are chemically reasonable.",
        related_terms=("Hydrogen bond", "Interaction support score", "Pose"),
    ),
    E(
        term="Interaction support score",
        category="Project Architecture",
        used_in_project=True,
        short_definition="A summary score describing how well a docked pose matches useful residue-level contacts.",
        detailed_explanation=(
            "Instead of trusting one docking number, interaction support asks whether the pose also makes sensible contacts with important residues in the target pocket."
        ),
        why_it_matters="This makes structural screening easier to explain and less dependent on one docking score.",
        oncoforge_usage="OncoForge computes interaction support from hydrogen bonds, hydrophobic contacts, salt bridges, and key EGFR residues.",
        common_pitfall="A single aggregate interaction score is useful, but the residue list still matters.",
        related_terms=("Docking", "Pose", "Structural priority score"),
    ),
    E(
        term="Interaction-aware ranking",
        category="Project Architecture",
        used_in_project=True,
        short_definition="A ranking layer that explicitly rewards molecules whose docked poses show stronger residue-level support.",
        detailed_explanation=(
            "Interaction-aware ranking does not stop at QSAR or even at docking score. It asks whether residue-level contacts, "
            "key-pocket interactions, and structural guardrails support the candidate strongly enough to affect the final order."
        ),
        why_it_matters="This makes the multi-agent story more convincing because structure becomes a real agent, not just a post-hoc note.",
        oncoforge_usage="OncoForge now computes structure-agent support and structure-augmented ranking after docking and interaction analysis.",
        common_pitfall="If structural evidence is calculated but never changes selection, it is not really an interaction-aware ranker.",
        related_terms=("Interaction support score", "Docking", "Structural priority score"),
    ),
    E(
        term="Feasibility score",
        category="Project Architecture",
        used_in_project=True,
        short_definition="An evidence-based estimate of whether a generated molecule looks realistic enough to deserve follow-up.",
        detailed_explanation=(
            "Feasibility in OncoForge blends active-neighbor support, scaffold support, fragment support, traceability of generation, synthetic accessibility, and structural evidence."
        ),
        why_it_matters="It is one of the strongest defenses against the criticism that generated molecules are purely fictional.",
        oncoforge_usage="OncoForge computes feasibility for optimized, generated, and RL-derived candidates and uses it in final ranking.",
        common_pitfall="Feasibility is computational support, not experimental proof.",
        related_terms=("SA score", "Docking", "Fragment support", "Wet-lab validation"),
    ),
    E(
        term="Experimental readiness",
        category="Project Architecture",
        used_in_project=True,
        short_definition="A score describing how ready a computationally generated molecule is for serious downstream validation.",
        detailed_explanation=(
            "Experimental readiness combines feasibility, structural support, multisource neighbor evidence, chemistry quality, and low proxy-risk behavior "
            "into one rubric for deciding which molecules deserve the next validation step."
        ),
        why_it_matters="It gives a more juror-friendly answer to the question 'why should anyone believe this molecule is worth testing?'",
        oncoforge_usage="OncoForge computes experimental-readiness scores and labels candidates as ready, supporting, or hold.",
        common_pitfall="Readiness is still not the same as wet-lab proof; it is a triage and prioritization framework.",
        related_terms=("Feasibility score", "Prospective validation batch", "Verifiable reward"),
    ),
    E(
        term="Prospective validation batch",
        category="Project Architecture",
        used_in_project=True,
        short_definition="A small, high-priority set of molecules selected for the next round of serious evaluation.",
        detailed_explanation=(
            "Instead of keeping only one top-ranked list, a prospective batch balances exploitation, novelty, uncertainty, and diversity "
            "so the next validation step is informative as well as promising."
        ),
        why_it_matters="This is much closer to how real discovery teams decide what to test next.",
        oncoforge_usage="OncoForge builds an active-learning style prospective validation batch from optimized, diverse, shortlisted, and RL candidates.",
        common_pitfall="A prospective batch should not be a simple copy of the top-N by one score.",
        related_terms=("Active learning", "Experimental readiness", "Diversity"),
    ),
    E(
        term="Active learning",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A strategy that chooses which new examples would be most useful to evaluate next.",
        detailed_explanation=(
            "In active learning, the system does not only ask which molecules look best. It also asks which molecules would teach the model the most "
            "if they were validated, often balancing uncertainty, novelty, and expected value."
        ),
        why_it_matters="It turns ranking into a smarter experimental-planning problem.",
        oncoforge_usage="OncoForge now uses an active-learning style acquisition score to build a prospective validation batch.",
        common_pitfall="Active learning is not just picking the most uncertain molecules; it should balance informativeness and quality.",
        related_terms=("Prospective validation batch", "Uncertainty estimation", "Exploration"),
    ),
    E(
        term="Reaction-aware generation",
        category="Medicinal Chemistry",
        used_in_project=True,
        short_definition="Generating molecules through edits that resemble plausible medicinal-chemistry transformations.",
        detailed_explanation=(
            "Reaction-aware generation favors substitutions and derivatizations that look like recognizable chemistry operations, "
            "such as SNAr, alkylation, acylation, carbamate formation, or ether extension."
        ),
        why_it_matters="This makes generated molecules feel less arbitrary and more grounded in how chemists actually optimize leads.",
        oncoforge_usage="OncoForge now includes a larger library of reaction-style scaffold-preserving transformations for analog generation.",
        common_pitfall="A reaction-aware edit is still only a heuristic if no full synthesis route is checked.",
        related_terms=("Matched molecular pair", "Lead optimization", "Synthetic accessibility"),
    ),
    E(
        term="Multi-source support",
        category="Machine Learning",
        used_in_project=True,
        short_definition="Evidence that similar molecules or assay support appear across more than one public data source.",
        detailed_explanation=(
            "When similar chemistry is supported by multiple curated databases, the signal is often more trustworthy than a single isolated observation."
        ),
        why_it_matters="It strengthens the case for a candidate even before any new wet-lab experiment is run.",
        oncoforge_usage="OncoForge uses multisource cleaned EGFR data and now tracks source-backed neighbor support as part of readiness and feasibility.",
        common_pitfall="Multiple databases do not eliminate assay bias automatically; they just improve the evidence base.",
        related_terms=("BindingDB", "ChEMBL", "Experimental readiness"),
    ),
    E(
        term="Cross-database validation",
        category="Project Architecture",
        used_in_project=True,
        short_definition="Checking whether a candidate is supported by more than one independent public bioactivity source.",
        detailed_explanation=(
            "Cross-database validation asks whether similar chemistry and active patterns show up consistently across curated resources such as "
            "ChEMBL, BindingDB, IUPHAR, and a merged multisource consensus set. It treats agreement across sources as stronger evidence than a signal from one table alone."
        ),
        why_it_matters="It gives a much stronger answer to the question of whether a generated molecule is grounded in known biology without needing a new wet-lab assay.",
        oncoforge_usage="OncoForge computes cross-database consensus, agreement counts, supporting-source labels, and strong/moderate/weak support states for prioritized candidates.",
        common_pitfall="Cross-database support is still computational evidence; it strengthens plausibility but does not replace new experiments.",
        related_terms=("Multi-source support", "Experimental readiness", "BindingDB", "ChEMBL"),
    ),
    E(
        term="External evidence agent",
        category="Project Architecture",
        used_in_project=True,
        short_definition="A specialized agent that scores how strongly a molecule is supported by independent public assay evidence.",
        detailed_explanation=(
            "An external evidence agent turns support from public pharmacology and assay databases into an explicit ranking signal. "
            "Instead of leaving those sources as passive annotations, it treats them as a decision-making component that can promote or demote candidates."
        ),
        why_it_matters="This makes the project more defensible because external biological evidence becomes part of the selection logic itself.",
        oncoforge_usage="OncoForge now computes external-evidence support from cross-database agreement plus PubChem, IUPHAR, BindingDB, and multisource neighborhood signals.",
        common_pitfall="It is not really an agent if the evidence is calculated but never affects ranking or prioritization.",
        related_terms=("Cross-database validation", "Experimental readiness", "Verifiable reward"),
    ),
    E(
        term="IUPHAR / Guide to Pharmacology",
        category="Cancer Biology",
        used_in_project=True,
        short_definition="A curated pharmacology knowledgebase that links ligands, targets, and interaction evidence.",
        detailed_explanation=(
            "The Guide to Pharmacology database, maintained through IUPHAR and collaborators, contains curated target-ligand relationships and "
            "pharmacological annotations. It is useful as an external reference source because it is independently curated from resources like ChEMBL and BindingDB."
        ),
        why_it_matters="Using IUPHAR as an additional reference strengthens the case that top candidates align with chemistry already supported in independent pharmacology resources.",
        oncoforge_usage="OncoForge downloads EGFR interaction records from Guide to Pharmacology and uses them inside cross-database validation.",
        common_pitfall="IUPHAR is a high-value reference source, but it is not a full replacement for larger bioactivity corpora or experimental confirmation.",
        related_terms=("Cross-database validation", "BindingDB", "ChEMBL", "EGFR"),
    ),
    E(
        term="PubChem BioAssay",
        category="Cancer Biology",
        used_in_project=True,
        short_definition="A public resource of assay outcomes and activity measurements linked to compounds and biological targets.",
        detailed_explanation=(
            "PubChem BioAssay stores large numbers of screening and confirmatory assay results. For a target like EGFR, it can provide an additional line of evidence "
            "about whether related chemistry has shown active or inactive behavior in public experiments."
        ),
        why_it_matters="It gives OncoForge another independent evidence source that is different from ChEMBL, BindingDB, and IUPHAR.",
        oncoforge_usage="OncoForge now builds a filtered EGFR PubChem reference set and uses it in cross-database validation plus the external evidence agent.",
        common_pitfall="PubChem assay data are rich but noisy; they need target filtering, aggregation, and careful interpretation.",
        related_terms=("External evidence agent", "Cross-database validation", "Experimental readiness", "EGFR"),
    ),
    E(
        term="Lead optimization",
        category="Medicinal Chemistry",
        used_in_project=True,
        short_definition="Improving a promising chemical starting point instead of searching blindly from scratch.",
        detailed_explanation=(
            "Lead optimization is the stage where chemists modify scaffolds and substituents to improve potency, selectivity, safety, and developability."
        ),
        why_it_matters="Most serious medicinal-chemistry projects live here rather than in unconstrained random generation.",
        oncoforge_usage="OncoForge is fundamentally a lead-optimization system around EGFR-active seeds and analog generation.",
        common_pitfall="A project that really does lead optimization should say so clearly instead of calling everything de novo design.",
        related_terms=("Matched molecular pair", "Scaffold", "Medicinal chemistry transformation"),
    ),
    E(
        term="pIC50",
        category="Cancer Biology",
        used_in_project=True,
        short_definition="The negative logarithm of IC50, used to put potency on a cleaner and more model-friendly scale.",
        detailed_explanation=(
            "Because IC50 spans wide concentration ranges, pIC50 compresses the scale and flips the direction so larger values mean stronger potency."
        ),
        why_it_matters="This is the main potency endpoint predicted and optimized by OncoForge.",
        oncoforge_usage="Model training, ranking, RL reward, and summaries all rely on predicted pIC50.",
        common_pitfall="A one-unit difference in pIC50 is large; it means a tenfold change in IC50.",
        related_terms=("IC50", "QSAR", "Regression"),
    ),
    E(
        term="EGFR",
        category="Cancer Biology",
        used_in_project=True,
        short_definition="Epidermal Growth Factor Receptor, an important kinase target in cancer.",
        detailed_explanation=(
            "EGFR is a receptor tyrosine kinase involved in growth signaling. In several cancers, abnormal EGFR signaling drives disease."
        ),
        why_it_matters="It is the primary biological target used as the case study for OncoForge.",
        oncoforge_usage="The dataset, marketed benchmark, docking workflow, and generated candidates in OncoForge are centered on EGFR inhibition.",
        common_pitfall="Strong EGFR inhibition alone does not guarantee a molecule is a real drug.",
        related_terms=("Kinase inhibitor", "Binding pocket", "Marketed EGFR drugs"),
    ),
    E(
        term="Wet-lab validation",
        category="Cancer Biology",
        used_in_project=False,
        short_definition="Real experimental testing of generated molecules in biochemical or cellular assays.",
        detailed_explanation=(
            "Computational evidence can prioritize molecules, but only laboratory experiments can prove whether they truly work and remain acceptable in real systems."
        ),
        why_it_matters="This is the final standard of proof in drug discovery.",
        oncoforge_usage="OncoForge does not perform wet-lab validation yet, so it uses feasibility, docking, and benchmarking as honest pre-experimental support.",
        common_pitfall="Strong computational evidence should never be presented as if it were already experimental confirmation.",
        related_terms=("Feasibility score", "Docking", "QSAR"),
    ),
    E(
        term="Graph neural network (GNN)",
        category="Machine Learning",
        used_in_project=True,
        short_definition="A neural network that learns directly from atoms and bonds instead of only fixed fingerprints.",
        detailed_explanation=(
            "A GNN treats a molecule as a graph. Atoms exchange information with neighboring atoms through message-passing layers, "
            "so the model can learn structural patterns from the connectivity itself."
        ),
        why_it_matters="It gives OncoForge a modern learned representation to compare against the classical ensemble.",
        oncoforge_usage="OncoForge benchmarks a GPU graph regressor against the multiview ensemble and a consensus blend.",
        common_pitfall="A GNN is not automatically better than tree models on medium-sized medicinal-chemistry datasets.",
        related_terms=("Message passing", "ECFP / Morgan fingerprint", "Consensus blend"),
    ),
    E(
        term="Deep Q-Network (DQN)",
        category="Reinforcement Learning",
        used_in_project=True,
        short_definition="A neural RL method that learns Q-values with a neural network instead of a lookup table.",
        detailed_explanation=(
            "Q-learning estimates how good an action is from a given state. A DQN replaces the table of Q-values with a neural network, "
            "which lets it generalize across many related states and actions."
        ),
        why_it_matters="It is a stronger RL baseline than a purely tabular policy when the action space becomes richer.",
        oncoforge_usage="OncoForge uses a GPU DQN to rank medicinal-chemistry actions with verifiable reward components.",
        common_pitfall="If reward is badly designed, a stronger RL agent can exploit it even more efficiently.",
        related_terms=("Verifiable reward", "Replay buffer", "Target network"),
    ),
    E(
        term="DirectML",
        category="ML Infrastructure",
        used_in_project=True,
        short_definition="A Windows GPU backend that lets PyTorch run on supported hardware without the full CUDA stack.",
        detailed_explanation=(
            "DirectML is a Microsoft backend for accelerated tensor operations on Windows. It is especially useful when a project "
            "cannot easily install large CUDA-specific builds or when the GPU environment is constrained."
        ),
        why_it_matters="It lets OncoForge run GPU neural models in a separate environment without disrupting the main CPU pipeline.",
        oncoforge_usage="OncoForge uses DirectML for the graph regressor and GPU DQN stages in the Python 3.12 GPU environment.",
        common_pitfall="DirectML acceleration does not guarantee the same performance characteristics as CUDA.",
        related_terms=("GPU acceleration", "PyTorch", "Graph neural network (GNN)"),
    ),
    E(
        term="Papyrus",
        category="Bioactivity Databases",
        used_in_project=True,
        short_definition="A large standardized bioactivity dataset assembled from multiple public sources for machine learning.",
        detailed_explanation=(
            "Papyrus aggregates and standardizes medicinal-chemistry measurements from public databases so researchers can train models "
            "on more consistent data than a single source alone often provides."
        ),
        why_it_matters="It adds orthogonal public evidence beyond ChEMBL and BindingDB.",
        oncoforge_usage="OncoForge uses Papyrus as an external validation source in cross-database consensus and evidence scoring.",
        common_pitfall="Papyrus improves evidence coverage, but it still inherits some noise from source databases.",
        related_terms=("Cross-database validation", "BindingDB", "ExCAPE-DB"),
    ),
    E(
        term="ExCAPE-DB",
        category="Bioactivity Databases",
        used_in_project=True,
        short_definition="A chemogenomics database combining large-scale ChEMBL and PubChem activity data.",
        detailed_explanation=(
            "ExCAPE-DB was built to support machine learning by collecting large numbers of compound-target annotations and harmonizing them "
            "into a format more suitable for benchmarking and large-scale predictive modeling."
        ),
        why_it_matters="It gives OncoForge another independent source of EGFR-related evidence patterns.",
        oncoforge_usage="OncoForge uses ExCAPE-DB support scores in its external evidence layer and cross-database validator.",
        common_pitfall="Large data volume does not automatically mean higher confidence than smaller curated datasets.",
        related_terms=("Cross-database validation", "Papyrus", "PubChem BioAssay"),
    ),
)
