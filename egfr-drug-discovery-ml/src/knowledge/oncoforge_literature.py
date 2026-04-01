from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkMetric:
    name: str
    value: float
    split: str = ""
    notes: str = ""


@dataclass(frozen=True)
class LiteratureEntry:
    key: str
    short_label: str
    citation: str
    title: str
    category: str
    year: int
    source_url: str
    quote: str
    why_it_matters: str
    dataset_size: int | None = None
    metrics: tuple[BenchmarkMetric, ...] = ()


M = BenchmarkMetric
L = LiteratureEntry


LITERATURE_ENTRIES: tuple[LiteratureEntry, ...] = (
    L(
        key="yarden_2001_erbb",
        short_label="Yarden 2001",
        citation="Yarden Y, Sliwkowski MX. Untangling the ErbB signalling network. Nat Rev Mol Cell Biol. 2001.",
        title="Untangling the ErbB signalling network",
        category="EGFR Biology",
        year=2001,
        source_url="https://pubmed.ncbi.nlm.nih.gov/11252954/",
        quote="The network is often dysregulated in cancer.",
        why_it_matters="Explains why EGFR remains a strong oncology target and why ranking EGFR-focused candidates is clinically meaningful.",
    ),
    L(
        key="attwood_2021_kinase_trends",
        short_label="Attwood 2021",
        citation="Attwood MM et al. Trends in kinase drug discovery: targets, indications and inhibitor design. Nat Rev Drug Discov. 2021.",
        title="Trends in kinase drug discovery: targets, indications and inhibitor design",
        category="Kinase Drug Discovery",
        year=2021,
        source_url="https://pubmed.ncbi.nlm.nih.gov/34354255/",
        quote="Oncology is still the predominant area for their application.",
        why_it_matters="Frames EGFR inhibitor design inside the broader kinase-drug landscape used by industry and academia.",
    ),
    L(
        key="chembl_2023",
        short_label="ChEMBL 2023",
        citation="Zdrazil B et al. The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic Acids Res. 2023.",
        title="The ChEMBL Database in 2023",
        category="Data Resource",
        year=2023,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10767899/",
        quote="ChEMBL is a manually curated, high-quality, large-scale, open, FAIR resource.",
        why_it_matters="Justifies the project's use of curated medicinal-chemistry data as a backbone for QSAR and ranking.",
    ),
    L(
        key="bindingdb_2024",
        short_label="BindingDB 2024",
        citation="BindingDB in 2024: a FAIR knowledgebase of protein-small molecule binding data. Nucleic Acids Res. 2024.",
        title="BindingDB in 2024: a FAIR knowledgebase of protein-small molecule binding data",
        category="Data Resource",
        year=2024,
        source_url="https://pubmed.ncbi.nlm.nih.gov/39574417/",
        quote="supports diverse applications including medicinal chemistry ... training of artificial intelligence models",
        why_it_matters="Supports the project's decision to add BindingDB-style measured affinity data to the evidence stack.",
    ),
    L(
        key="pubchem_2023",
        short_label="PubChem 2023",
        citation="Kim S et al. PubChem 2023 update. Nucleic Acids Res. 2023.",
        title="PubChem 2023 update",
        category="Data Resource",
        year=2023,
        source_url="https://pubmed.ncbi.nlm.nih.gov/36305812/",
        quote="Data from more than 120 data sources was added to PubChem.",
        why_it_matters="Explains why PubChem is useful as an external evidence layer instead of just a raw compound catalog.",
    ),
    L(
        key="iuphar_2024",
        short_label="IUPHAR 2024",
        citation="Harding SD et al. The IUPHAR/BPS Guide to PHARMACOLOGY in 2024. Nucleic Acids Res. 2024.",
        title="The IUPHAR/BPS Guide to PHARMACOLOGY in 2024",
        category="Data Resource",
        year=2024,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10767925/",
        quote="open-access, expert-curated, online database",
        why_it_matters="IUPHAR gives a higher-confidence pharmacology reference layer that is useful for sanity checks and rediscovery controls.",
    ),
    L(
        key="excape_2017",
        short_label="ExCAPE-DB 2017",
        citation="Sun J et al. ExCAPE-DB: an integrated large scale dataset facilitating Big Data analysis in chemogenomics. J Cheminform. 2017.",
        title="ExCAPE-DB: an integrated large scale dataset facilitating Big Data analysis in chemogenomics",
        category="Data Resource",
        year=2017,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5340785/",
        quote="This dataset comprises over 70 million SAR data points.",
        why_it_matters="Shows why adding ExCAPE-like large-scale chemogenomics data can widen chemical space coverage.",
        dataset_size=998131,
    ),
    L(
        key="consensus_dataset_2022",
        short_label="Consensus Dataset 2022",
        citation="Svensson F et al. A Consensus Compound/Bioactivity Dataset for Data-Driven Drug Design and Chemogenomics. Molecules. 2022.",
        title="A Consensus Compound/Bioactivity Dataset for Data-Driven Drug Design and Chemogenomics",
        category="Data Curation",
        year=2022,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9028877/",
        quote="differences in compound and target coverage advocating the combined use of data from multiple sources",
        why_it_matters="Directly supports the multi-source design choice in OncoForge instead of relying on a single database.",
        dataset_size=1100000,
    ),
    L(
        key="moleculenet_2018",
        short_label="MoleculeNet 2018",
        citation="Wu Z et al. MoleculeNet: a benchmark for molecular machine learning. Chem Sci. 2018.",
        title="MoleculeNet: a benchmark for molecular machine learning",
        category="Benchmarking",
        year=2018,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/",
        quote="Random splitting ... is often not correct for chemical data.",
        why_it_matters="Supports reporting scaffold-aware and out-of-sample benchmarks instead of only easier random splits.",
        dataset_size=700000,
    ),
    L(
        key="vina_2010",
        short_label="Vina 2010",
        citation="Trott O, Olson AJ. AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. J Comput Chem. 2010.",
        title="AutoDock Vina: improving the speed and accuracy of docking",
        category="Docking",
        year=2010,
        source_url="https://pubmed.ncbi.nlm.nih.gov/19499576/",
        quote="improving the speed and accuracy of docking",
        why_it_matters="Provides the structural-docking backbone for comparing generated molecules with marketed EGFR inhibitors.",
    ),
    L(
        key="olivecrona_2017",
        short_label="Olivecrona 2017",
        citation="Olivecrona M et al. Molecular de-novo design through deep reinforcement learning. J Cheminform. 2017.",
        title="Molecular de-novo design through deep reinforcement learning",
        category="Generative Modeling",
        year=2017,
        source_url="https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x",
        quote="generate structures with certain specified desirable properties",
        why_it_matters="A foundational reference for sequence-based RL in molecular generation and lead optimization.",
    ),
    L(
        key="moldqn_2019",
        short_label="MolDQN 2019",
        citation="Zhou Z et al. Optimization of Molecules via Deep Reinforcement Learning. Sci Rep. 2019.",
        title="Optimization of Molecules via Deep Reinforcement Learning",
        category="Generative Modeling",
        year=2019,
        source_url="https://www.nature.com/articles/s41598-019-47148-x",
        quote="ensuring 100% chemical validity",
        why_it_matters="Relevant for the project's action-based generator and verifiable RL framing.",
    ),
    L(
        key="guacamol_2019",
        short_label="GuacaMol 2019",
        citation="Brown N et al. GuacaMol: Benchmarking Models for De Novo Molecular Design. J Chem Inf Model. 2019.",
        title="GuacaMol: Benchmarking Models for De Novo Molecular Design",
        category="Benchmarking",
        year=2019,
        source_url="https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839",
        quote="To standardize the assessment ... we propose an evaluation framework, GuacaMol",
        why_it_matters="Connects OncoForge's rediscovery and optimization narrative to a widely used generative-design benchmark culture.",
    ),
    L(
        key="merk_2018",
        short_label="Merk 2018",
        citation="Merk D et al. De Novo Design of Bioactive Small Molecules by Artificial Intelligence. Mol Inform. 2018.",
        title="De Novo Design of Bioactive Small Molecules by Artificial Intelligence",
        category="Generative Modeling",
        year=2018,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5838524/",
        quote="experimentally validate the applicability of generative AI to prospective de novo molecule design",
        why_it_matters="Useful as a prospective-design reference showing why ranking should lead toward experimentally testable compounds.",
    ),
    L(
        key="singh_2015",
        short_label="Singh 2015",
        citation="Singh H et al. QSAR based model for discriminating EGFR inhibitors and non-inhibitors using Random forest. Biol Direct. 2015.",
        title="QSAR based model for discriminating EGFR inhibitors and non-inhibitors using Random forest",
        category="EGFR QSAR",
        year=2015,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4372225/",
        quote="achieved accuracy 84.95% with MCC 0.49",
        why_it_matters="Provides a target-specific EGFR baseline from the earlier classification era.",
        dataset_size=3528,
        metrics=(
            M(name="accuracy_pct", value=84.95, split="fivefold_cv", notes="EGFR10 classification set"),
            M(name="mcc", value=0.49, split="fivefold_cv", notes="Random forest classifier"),
        ),
    ),
    L(
        key="chang_2024",
        short_label="Chang 2024",
        citation="Chang H et al. Machine Learning-Based Virtual Screening and Identification of the Fourth-Generation EGFR Inhibitors. ACS Omega. 2024.",
        title="Machine Learning-Based Virtual Screening and Identification of the Fourth-Generation EGFR Inhibitors",
        category="EGFR QSAR",
        year=2024,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10795152/",
        quote="R2 of 0.745 and an MSE of 0.255 were obtained for test set.",
        why_it_matters="Provides a modern EGFR regression comparator on a curated target-specific dataset.",
        dataset_size=221,
        metrics=(
            M(name="r2", value=0.745, split="test", notes="SVR on triple-mutant EGFR inhibitors"),
            M(name="mse", value=0.255, split="test", notes="reported directly in paper"),
            M(name="rmse", value=0.505, split="test", notes="sqrt(MSE), derived for plotting"),
        ),
    ),
    L(
        key="moshawih_2024",
        short_label="Moshawih 2024",
        citation="Moshawih S et al. Consensus holistic virtual screening for drug discovery: a novel machine learning model approach. Sci Rep. 2024.",
        title="Consensus holistic virtual screening for drug discovery: a novel machine learning model approach",
        category="Virtual Screening",
        year=2024,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11134635/",
        quote="High R2 values for AA2AR (0.891) and EGFR (0.797)",
        why_it_matters="Adds an EGFR external-validation comparator from a consensus screening workflow.",
        metrics=(
            M(name="r2", value=0.797, split="external_validation", notes="EGFR target, external validation"),
        ),
    ),
    L(
        key="egfrap_2025",
        short_label="EGFRAP 2025",
        citation="Gupta A et al. EGFRAP: a predictive machine learning model for assessing small molecule activity against the epidermal growth factor receptor. RSC Med Chem. 2025.",
        title="EGFRAP: a predictive machine learning model for assessing small molecule activity against EGFR",
        category="EGFR QSAR",
        year=2025,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12288231/",
        quote="R2 value of 0.67, an RMSE of 0.89 and an MAE of 0.61",
        why_it_matters="A recent EGFR-focused regression reference with an explicit external test split.",
        dataset_size=8102,
        metrics=(
            M(name="r2", value=0.67, split="test", notes="external test set"),
            M(name="rmse", value=0.89, split="test", notes="external test set"),
            M(name="mae", value=0.61, split="test", notes="external test set"),
            M(name="r2", value=0.63, split="tenfold_cv", notes="cross-validation"),
            M(name="rmse", value=0.94, split="tenfold_cv", notes="cross-validation"),
        ),
    ),
)

