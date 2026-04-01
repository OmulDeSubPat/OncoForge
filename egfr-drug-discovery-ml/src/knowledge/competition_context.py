from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiteratureEntry:
    key: str
    title: str
    citation: str
    url: str
    category: str
    why_it_matters: str
    short_quote: str
    comparison_axis: str | None = None
    comparison_label: str | None = None
    comparison_value: float | None = None
    comparison_unit: str | None = None
    comparison_note: str | None = None


@dataclass(frozen=True)
class ProjectPhase:
    phase_id: str
    title: str
    date_label: str
    commit: str
    focus: str
    upgrades: tuple[str, ...]


L = LiteratureEntry
P = ProjectPhase


COMPETITION_LITERATURE: tuple[LiteratureEntry, ...] = (
    L(
        key="qed_2012",
        title="Quantifying the chemical beauty of drugs",
        citation="Bickerton et al., Nature Chemistry, 2012.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC3524573/",
        category="Drug-likeness and prioritization",
        why_it_matters="Supports the QED term used throughout candidate ranking and explains why one scalar property is not enough.",
        short_quote="We propose a measure of druglikeness based on the concept of desirability called Quantitative Estimate of Druglikeness (QED).",
    ),
    L(
        key="sascore_2009",
        title="Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions",
        citation="Ertl and Schuffenhauer, Journal of Cheminformatics, 2009.",
        url="https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8",
        category="Medicinal chemistry realism",
        why_it_matters="Explains the synthetic accessibility score that helps prevent ranking unrealistic molecules too highly.",
        short_quote="A novel method to estimate synthetic accessibility of molecules has been developed.",
    ),
    L(
        key="pains_2010",
        title="New substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries and for their exclusion in bioassays",
        citation="Baell and Holloway, Journal of Medicinal Chemistry, 2010.",
        url="https://pubmed.ncbi.nlm.nih.gov/20131845/",
        category="Assay artifacts and safety filters",
        why_it_matters="Justifies the structural alert logic and caution around compounds that can game early screening signals.",
        short_quote="New substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries and for their exclusion in bioassays.",
    ),
    L(
        key="vina_2010",
        title="AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading",
        citation="Trott and Olson, Journal of Computational Chemistry, 2010.",
        url="https://pubmed.ncbi.nlm.nih.gov/19499576/",
        category="Structure-based scoring",
        why_it_matters="Supports the docking stage used to rescore top-ranked candidates against EGFR.",
        short_quote="AutoDock Vina achieves an approximately two orders of magnitude speed-up compared with AutoDock 4.",
    ),
    L(
        key="bindingdb_2015",
        title="BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology",
        citation="Gilson et al., Nucleic Acids Research, 2016.",
        url="https://academic.oup.com/nar/article/44/D1/D1045/2502601",
        category="Reference databases",
        why_it_matters="Validates BindingDB as an independent source for cross-database evidence and source holdout experiments.",
        short_quote="BindingDB, www.bindingdb.org, is a publicly accessible database of experimental protein-small molecule interaction data.",
    ),
    L(
        key="iuphar_2018",
        title="Accessing Expert-Curated Pharmacological Data in the IUPHAR/BPS Guide to PHARMACOLOGY",
        citation="Sharman et al., Current Protocols in Bioinformatics, 2018.",
        url="https://pubmed.ncbi.nlm.nih.gov/30040201/",
        category="Reference databases",
        why_it_matters="Supports the use of Guide to Pharmacology as an expert-curated external validation source.",
        short_quote="The IUPHAR/BPS Guide to PHARMACOLOGY is an expert-curated, open-access database.",
    ),
    L(
        key="excape_2017",
        title="ExCAPE-DB: an integrated large scale dataset facilitating Big Data analysis in chemogenomics",
        citation="Sun et al., Journal of Cheminformatics, 2017.",
        url="https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0203-5",
        category="Reference databases",
        why_it_matters="Supports the ExCAPE-DB branch of the external evidence system and contextualizes the scale of the evidence pool.",
        short_quote="This dataset comprises over 70 million SAR data points from publicly available databases.",
    ),
    L(
        key="papyrus_2022",
        title="Papyrus: a large-scale curated dataset aimed at bioactivity predictions",
        citation="Bequignon et al., Journal of Cheminformatics, 2022.",
        url="https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00672-x",
        category="Reference databases",
        why_it_matters="Supports the use of Papyrus as a large curated benchmark-like source for public bioactivity modeling.",
        short_quote="One of the areas with accelerated developments is the prediction of bioactivity, specifically the prediction of ligand-protein affinity.",
    ),
    L(
        key="chembl_2023",
        title="ChEMBL database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods",
        citation="Mendez et al., Nucleic Acids Research, 2024.",
        url="https://academic.oup.com/nar/article/52/D1/D1180/7337608",
        category="Reference databases",
        why_it_matters="Anchors the project's main medicinal-chemistry data source in a standard public drug-discovery platform.",
        short_quote="A drug discovery platform spanning multiple bioactivity data types and time periods.",
    ),
    L(
        key="pubchem_2023",
        title="PubChem 2023 update",
        citation="Kim et al., Nucleic Acids Research, 2023.",
        url="https://academic.oup.com/nar/article/51/D1/D1373/6777787",
        category="Reference databases",
        why_it_matters="Supports the use of PubChem BioAssay evidence and explains why PubChem expands assay breadth beyond medicinal-chemistry papers.",
        short_quote="PubChem is a data aggregator, which collects chemical information from hundreds of data sources.",
    ),
    L(
        key="moldqn_2019",
        title="Optimization of Molecules via Deep Reinforcement Learning",
        citation="Zhou et al., Scientific Reports, 2019.",
        url="https://www.nature.com/articles/s41598-019-47148-x",
        category="Generative molecular optimization",
        why_it_matters="Provides baseline RL framing for molecule optimization and motivates chemically valid action spaces.",
        short_quote="By only allowing chemically valid actions, we ensure that all the molecules generated are valid.",
    ),
    L(
        key="reinvent4_2024",
        title="Reinvent 4: Modern AI-driven generative molecule design",
        citation="Loeffler et al., Journal of Cheminformatics, 2024.",
        url="https://link.springer.com/article/10.1186/s13321-024-00812-5",
        category="Generative molecular optimization",
        why_it_matters="Offers an industry-oriented reference point for AI molecule design systems and supports the design-make-test-analyze framing.",
        short_quote="REINVENT 4 is a modern open-source generative AI framework for the design of small molecules.",
    ),
    L(
        key="egfr_ml_2023",
        title="Machine Learning-Based Approach to Developing Potent EGFR Inhibitors for Breast Cancer-Design, Synthesis, and In Vitro Evaluation",
        citation="Nada et al., ACS Omega, 2023.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10483653/",
        category="EGFR study comparisons",
        why_it_matters="Provides a target-matched EGFR ML study with both computational metrics and wet-lab follow-up.",
        short_quote="Random Forest had the highest mean validation R-squared score of 0.717.",
        comparison_axis="regression_r2",
        comparison_label="Nada 2023 RF validation R2",
        comparison_value=0.717,
        comparison_unit="R2",
        comparison_note="Five-fold cross-validation on a curated EGFR dataset of about 9,000 compounds.",
    ),
    L(
        key="egfr_ml_scaffold_2023",
        title="Machine Learning-Based Approach to Developing Potent EGFR Inhibitors for Breast Cancer-Design, Synthesis, and In Vitro Evaluation",
        citation="Nada et al., ACS Omega, 2023.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10483653/",
        category="EGFR study comparisons",
        why_it_matters="Adds a scaffold-family-specific EGFR reference point for a chemically narrower subset.",
        short_quote="The top-performing ML model was evaluated on N-substituted quinazolin-4-amine-based compounds, yielding an R-squared score of 0.86.",
        comparison_axis="regression_r2",
        comparison_label="Nada 2023 focused-scaffold R2",
        comparison_value=0.86,
        comparison_unit="R2",
        comparison_note="Specialized subset and not directly comparable to broader multisource scaffold splits.",
    ),
    L(
        key="deepegfr_2025",
        title="DeepEGFR: a graph neural network for bioactivity classification of EGFR inhibitors",
        citation="Scientific Reports, 2025.",
        url="https://www.nature.com/articles/s41598-025-22126-8",
        category="EGFR study comparisons",
        why_it_matters="Provides a recent EGFR-specific deep learning comparison point that uses scaffold-based splitting and emphasizes interpretable feature analysis.",
        short_quote="DeepEGFR demonstrated exceptional performance, achieving approximately 94% F1-score across training and testing datasets.",
        comparison_axis="classification_f1",
        comparison_label="DeepEGFR 2025 test F1",
        comparison_value=0.94,
        comparison_unit="F1",
        comparison_note="Three-class EGFR activity classification task on 8,263 compounds with scaffold-based splitting.",
    ),
    L(
        key="deep_egfr_dataset_2025",
        title="DeepEGFR: a graph neural network for bioactivity classification of EGFR inhibitors",
        citation="Scientific Reports, 2025.",
        url="https://www.nature.com/articles/s41598-025-22126-8",
        category="EGFR study comparisons",
        why_it_matters="Provides a directly visible EGFR study scale for side-by-side dataset context.",
        short_quote="Following data curation and filtering, the final dataset comprised 8,263 compounds.",
    ),
)


PROJECT_PHASES: tuple[ProjectPhase, ...] = (
    P(
        phase_id="V0",
        title="Repository setup",
        date_label="2026-03-12",
        commit="711848e7",
        focus="Clean repository initialization",
        upgrades=(
            "Project scaffold, data folders, model folders, and baseline scripts were organized.",
            "The codebase was prepared for reproducible command-line execution.",
        ),
    ),
    P(
        phase_id="V1",
        title="Baseline QSAR and analog generation",
        date_label="2026-03-12",
        commit="b5ec2804",
        focus="Single-source EGFR prediction plus heuristic generation",
        upgrades=(
            "ChEMBL-centered EGFR data cleaning and random/scaffold RF models were in place.",
            "Ranking, broad analog generation, AI-guided analogs, iterative optimization, and market comparison already existed.",
            "This is the snapshot mirrored on the Desktop and is the clearest starting point for the report narrative.",
        ),
    ),
    P(
        phase_id="V2",
        title="Multisource evidence and multi-agent audit",
        date_label="2026-03-14",
        commit="2961f364",
        focus="Upgrade from baseline ranking to guarded lead prioritization",
        upgrades=(
            "BindingDB and multisource merging were added to widen evidence coverage.",
            "The ranking system gained multi-agent scoring, audit demotions, calibration, and technical notebook assets.",
            "Structural rescoring and stronger benchmarking became first-class outputs.",
        ),
    ),
    P(
        phase_id="V3",
        title="Cross-database validation and prospective triage",
        date_label="2026-03-16",
        commit="514d5fb0",
        focus="Independent evidence checks and stronger selection logic",
        upgrades=(
            "Papyrus, ExCAPE-DB, IUPHAR, and PubChem evidence were integrated.",
            "Source holdout, rediscovery, reward-hacking challenge, feasibility, and experimental readiness were added.",
            "A prospective validation batch replaced simple top-N selection as the main shortlist.",
        ),
    ),
    P(
        phase_id="V4",
        title="GNN, RL, and ISEF automation",
        date_label="2026-03-16 onward",
        commit="514d5fb0+",
        focus="Neural baselines, RL variants, and full report automation",
        upgrades=(
            "GPU GNN, verifiable RL, GPU DQN, and GPU actor-critic branches were added.",
            "The pipeline gained automated project summary, Word glossary, and technical notebook generation.",
            "The final project story shifted from raw generation to audited, evidence-aware lead prioritization.",
        ),
    ),
)
