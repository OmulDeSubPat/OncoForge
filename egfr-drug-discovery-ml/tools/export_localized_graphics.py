from __future__ import annotations

import importlib
import shutil
from contextlib import contextmanager
from pathlib import Path
import sys

import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT


PROJECT_ROOT = CONFIG_PROJECT_ROOT
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_DIR = PROJECT_ROOT / "grafice 30-3-2026"

TRANSLATIONS = [
    ("Cross-Database Support vs Experimental Readiness", "Sustinere intre baze de date vs pregatire experimentala"),
    ("Cross-database consensus score", "Scor de consens intre baze de date"),
    ("Cross-Database Validation Strength", "Intensitatea validarii intre baze de date"),
    ("External Evidence Agent Support", "Sustinere din dovezi externe"),
    ("External evidence support", "Sustinere din dovezi externe"),
    ("Experimental readiness score", "Scor de pregatire experimentala"),
    ("Prospective Validation Batch: Exploration vs Readiness", "Lot prospectiv: explorare vs pregatire"),
    ("Prospective acquisition score", "Scor de selectie prospectiva"),
    ("Selected prospective batch", "Lot prospectiv selectat"),
    ("Reward-Hacking Challenge: Does Protected Ranking Push Risky Molecules Down?", "Challenge reward hacking: clasarea protejata impinge jos moleculele riscante?"),
    ("Reward-Hacking Challenge: Audit Outcomes by Cohort", "Challenge reward hacking: rezultatele auditului pe cohorte"),
    ("Known EGFR Reference Recovery in a Hard Candidate Panel", "Recuperarea referintelor EGFR intr-un panel dificil"),
    ("Protected Ranking Shift for Reference EGFR Positives", "Schimbarea clasarii protejate pentru referintele EGFR pozitive"),
    ("Repeated-Seed Robustness of Model Families", "Robustetea familiilor de modele la seminte repetate"),
    ("Single-Agent Proxy Ranking vs Protected Multi-Agent Selection", "Clasare proxy single-agent vs selectie multi-agent protejata"),
    ("Audit Risk Distribution Across Multi-Agent Decisions", "Distributia riscului in deciziile multi-agent"),
    ("Naive Reward vs Protected Multi-Agent Ranking", "Scor naiv vs clasare multi-agent protejata"),
    ("Candidates Demoted by the Multi-Agent Audit", "Candidati retrogradati de auditul multi-agent"),
    ("Multi-Agent Support Profile for Top Leads", "Profilul de suport multi-agent pentru candidatii de top"),
    ("Novelty Must Stay Inside the Evidence Envelope", "Noutatea trebuie sa ramana in zona sustinuta de dovezi"),
    ("Marketed Drugs vs Generated Candidate Quality", "Medicamente aprobate vs calitatea candidatilor generati"),
    ("Generalization of the EGFR potency model", "Generalizarea modelului de potenta EGFR"),
    ("Validation outside the training sources", "Validare pe surse tinute in afara antrenarii"),
    ("Predicted vs true", "Valori prezise vs valori reale"),
    ("Chemical Space Projection (PCA of Morgan Fingerprints)", "Proiectie a spatiului chimic (PCA pe amprente Morgan)"),
    ("Distribution of Predicted EGFR Potency", "Distributia potentei EGFR estimate"),
    ("Distribution of Drug-Likeness (QED)", "Distributia asemanarii cu medicamentele (QED)"),
    ("Potency vs Drug-Likeness", "Potenta estimata vs asemanare cu medicamentele"),
    ("Average Candidate Score Across Optimization Rounds", "Scorul mediu al candidatilor pe runde de optimizare"),
    ("Best Candidate Score Across Optimization Rounds", "Cel mai bun scor al candidatilor pe runde de optimizare"),
    ("Candidate Feasibility vs Predicted Potency", "Fezabilitatea candidatilor vs potenta estimata"),
    ("Verifiable RL Training Curve", "Curba de antrenare RL verificabil"),
    ("Average Verifiable Reward Components", "Componentele medii ale recompensei verificabile"),
    ("RL Candidates: Public Evidence vs Final RL Priority", "Candidati RL: dovezi publice vs prioritate finala RL"),
    ("GPU DQN Verifiable RL", "RL verificabil GPU DQN"),
    ("Protected multi-agent", "Multi-agent protejat"),
    ("Single-agent proxy", "Proxy single-agent"),
    ("Random", "Aleator"),
    ("Scaffold", "Schelete"),
    ("Temporal", "Temporal"),
    ("Protected", "Protejat"),
    ("Naive", "Naiv"),
    ("Marketed EGFR", "EGFR aprobat"),
    ("Novel shortlist", "Shortlist nou"),
    ("Top protected candidates", "Candidatii protejati de top"),
    ("Potency", "Potenta"),
    ("Chemistry", "Chimie"),
    ("Safety", "Siguranta"),
    ("Domain", "Domeniu"),
    ("Mean predicted pIC50", "Media pIC50 estimat"),
    ("Mean reward-hacking risk", "Riscul mediu de reward hacking"),
    ("Review or fail rate", "Rata de revizuire sau respingere"),
    ("Audit pass rate", "Rata de trecere a auditului"),
    ("Top-k shortlist size", "Dimensiunea shortlist-ului Top-k"),
    ("Reward hacking risk", "Risc de reward hacking"),
    ("Support score", "Scor de suport"),
    ("Novelty score", "Scor de noutate"),
    ("Applicability score", "Scor de aplicabilitate"),
    ("Naive score", "Scor naiv"),
    ("Protected final score", "Scor final protejat"),
    ("Positions lost after anti-hacking audit", "Pozitii pierdute dupa auditul anti-hacking"),
    ("Predicted pIC50", "pIC50 estimat"),
    ("Strong-active recall", "Recall pentru molecule puternic active"),
    ("Leave-One-Source-Out Generalization", "Generalizare leave-one-source-out"),
    ("Source Holdout Recovery of Strong EGFR Actives", "Recuperarea moleculelor EGFR puternice in validarea pe surse"),
    ("Rediscovery recall", "Recall de redescoperire"),
    ("Naive rank - protected rank", "Rang naiv - rang protejat"),
    ("Episode return", "Retur pe episod"),
    ("Mean contribution", "Contributie medie"),
    ("Scaffold RMSE", "RMSE pe split de schelete"),
    ("Training molecules", "Molecule de antrenare"),
    ("Generated candidates", "Candidati generati"),
    ("Optimization Round", "Runda de optimizare"),
    ("Average Final Score", "Scor final mediu"),
    ("Best Final Score", "Cel mai bun scor final"),
    ("Candidate count", "Numar de candidati"),
    ("Molecule count", "Numar de molecule"),
    ("Count", "Numar"),
    ("Episode", "Episod"),
    ("Value", "Valoare"),
    ("Rate", "Rata"),
    ("QED", "QED"),
    ("pass", "trece"),
    ("review", "revizuire"),
    ("fail", "respins"),
    ("strong", "puternic"),
    ("moderate", "moderat"),
    ("weak", "slab"),
]

LEGEND_TEXT = """# Legenda grafice

Acest fisier explica pe scurt ce reprezinta graficele exportate in folderul `grafice 30-3-2026`.
Textele au fost localizate in romana pentru prezentare si pentru citire mai usoara.

## pIC50
Reprezinta o masura logaritmica a potentei estimate.
In general, valori mai mari indica o activitate estimata mai buna impotriva tintei EGFR.

## QED
Este un scor de asemanare cu proprietatile tipice ale medicamentelor.
Valori mai mari sunt, de regula, mai favorabile, dar nu garanteaza singure utilitatea moleculei.

## RMSE
Este eroarea medie patratica exprimata pe aceeasi scara cu variabila prezisa.
Valori mai mici sunt mai bune pentru ca indica abateri mai mici intre predictii si valori reale.

## MAE
Este eroarea medie absoluta.
Valori mai mici inseamna predictii mai apropiate de datele reale.

## R2
Este coeficientul de determinare.
Valori mai apropiate de 1 indica faptul ca modelul explica mai bine variatia datelor.

## Incertitudine predictiva
Arata cat de nesigur este modelul in privinta unei predictii.
Valori mai mari inseamna ca estimarea trebuie interpretata cu mai multa prudenta.

## Scor de fezabilitate
Rezuma cat de realist pare un candidat din punct de vedere chimic si practic.
Valori mai mari sunt, in general, mai bune.

## Scor de pregatire experimentala
Indica daca un candidat este mai aproape de a merita testare ulterioara.
Valori mai mari sugereaza prioritate mai buna pentru validare.

## Scor de consens intre baze de date
Arata cat de bine este sustinut un candidat de surse publice independente.
Valori mai mari indica dovezi externe mai consistente.

## Scor de noutate
Masoara cat de diferit este candidatul fata de moleculele deja cunoscute.
Valori mari pot fi utile, dar trebuie echilibrate cu fezabilitatea si dovezile existente.

## Scor de aplicabilitate
Arata cat de aproape este un candidat de zona de date in care modelul invata bine.
Valori mici inseamna risc mai mare ca predictia sa fie nesigura.

## Risc de reward hacking
Sugereaza cat de probabil este ca o molecula sa para buna numeric fara sa fie convingatoare chimic.
Valori mai mici sunt preferabile.

## Curbe de antrenare RL
Arata cum evolueaza recompensa obtinuta de agent pe parcursul episoadelor.
O tendinta ascendenta poate fi favorabila, dar trebuie interpretata impreuna cu graficele de audit, fezabilitate si dovezi externe.

## Heatmap de suport multi-agent
Fiecare coloana este un candidat, iar fiecare linie este o sursa de suport.
Valorile mai mari inseamna un acord mai puternic intre agentii sau filtrele specializate.
"""


def _translate_text(value):
    if not isinstance(value, str):
        return value
    translated = value
    for source, target in TRANSLATIONS:
        translated = translated.replace(source, target)
    return translated


@contextmanager
def _localized_matplotlib():
    import matplotlib.colorbar as colorbar_module
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    original_text_set_text = Text.set_text
    original_figure_savefig = Figure.savefig
    original_pyplot_savefig = plt.savefig
    original_colorbar_set_label = colorbar_module.Colorbar.set_label

    def patched_set_text(self, text):
        return original_text_set_text(self, _translate_text(text))

    def _remap_path(path_like):
        if path_like is None:
            return path_like
        path = Path(path_like)
        if not path.is_absolute():
            return path_like
        try:
            relative = path.relative_to(REPORTS_DIR)
        except ValueError:
            return path_like
        destination = TARGET_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def patched_figure_savefig(self, fname, *args, **kwargs):
        return original_figure_savefig(self, _remap_path(fname), *args, **kwargs)

    def patched_pyplot_savefig(fname, *args, **kwargs):
        return original_pyplot_savefig(_remap_path(fname), *args, **kwargs)

    def patched_colorbar_set_label(self, label, *args, **kwargs):
        return original_colorbar_set_label(self, _translate_text(label), *args, **kwargs)

    Text.set_text = patched_set_text
    Figure.savefig = patched_figure_savefig
    plt.savefig = patched_pyplot_savefig
    colorbar_module.Colorbar.set_label = patched_colorbar_set_label
    try:
        yield
    finally:
        Text.set_text = original_text_set_text
        Figure.savefig = original_figure_savefig
        plt.savefig = original_pyplot_savefig
        colorbar_module.Colorbar.set_label = original_colorbar_set_label


def _import(module_name: str):
    return importlib.import_module(module_name)


def _ensure_clean_target() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)


def _set_output_dir(module_name: str) -> object:
    module = _import(module_name)
    if hasattr(module, "OUTPUT_DIR"):
        original = Path(module.OUTPUT_DIR)
        try:
            relative = original.relative_to(REPORTS_DIR)
        except ValueError:
            relative = Path(original.name)
        module.OUTPUT_DIR = TARGET_DIR / relative
        module.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return module


def _copy_tree_images(source_dir: Path, destination_dir: Path) -> int:
    if not source_dir.exists():
        return 0
    copied = 0
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".bmp"}:
            continue
        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _generate_root_report_plots() -> None:
    _import("src.visualization.report_plots").main()
    _import("src.visualization.chemical_space_pca").main()


def _generate_top_molecules_grid() -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw

    input_path = REPORTS_DIR / "iterative_ai_optimized_candidates.csv"
    if not input_path.exists():
        input_path = REPORTS_DIR / "final_diverse_candidates.csv"
    if not input_path.exists():
        return

    df = pd.read_csv(input_path, low_memory=False).head(12)
    molecules = []
    legends = []
    for _, row in df.iterrows():
        molecule = Chem.MolFromSmiles(str(row["smiles"]))
        if molecule is None:
            continue
        molecules.append(molecule)
        legends.append(
            "Scor={score:.2f}\npIC50 estimat={pic50:.2f}\nQED={qed:.2f}".format(
                score=float(row.get("final_score", 0.0)),
                pic50=float(row.get("predicted_pIC50", 0.0)),
                qed=float(row.get("QED", 0.0)),
            )
        )

    if not molecules:
        return

    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=3,
        subImgSize=(350, 300),
        legends=legends,
        useSVG=False,
    )
    out_path = TARGET_DIR / "top_molecules_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out_path))


def _generate_crossdb_and_feasibility() -> None:
    crossdb_module = _import("src.evaluation.run_cross_database_validation")
    feasibility_module = _import("src.feasibility.assess_candidates")

    crossdb_candidates = [
        REPORTS_DIR / "iterative_ai_optimized_candidates_crossdb.csv",
        REPORTS_DIR / "iterative_ai_optimized_candidates_structural_crossdb.csv",
    ]
    crossdb_df = None
    for path in crossdb_candidates:
        if path.exists():
            crossdb_df = pd.read_csv(path, low_memory=False)
            break
    if crossdb_df is not None and not crossdb_df.empty:
        crossdb_module._plot_consensus_vs_readiness(crossdb_df, TARGET_DIR / "cross_database_consensus_vs_readiness.png")
        crossdb_module._plot_status_counts(crossdb_df, TARGET_DIR / "cross_database_status_counts.png")
        crossdb_module._plot_external_evidence(crossdb_df, TARGET_DIR / "external_evidence_support_vs_potency.png")

    feasibility_specs = [
        (
            REPORTS_DIR / "iterative_ai_optimized_candidates_structural_feasibility.csv",
            TARGET_DIR / "iterative_ai_optimized_candidates_structural_feasibility_feasibility_vs_potency.png",
        ),
        (
            REPORTS_DIR / "generated_analogs_ranked_structural_feasibility.csv",
            TARGET_DIR / "generated_analogs_ranked_structural_feasibility_feasibility_vs_potency.png",
        ),
        (
            REPORTS_DIR / "ai_guided_analogs_structural_feasibility.csv",
            TARGET_DIR / "ai_guided_analogs_structural_feasibility_feasibility_vs_potency.png",
        ),
    ]

    for csv_path, png_path in feasibility_specs:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        feasibility_module._plot_feasibility(df, png_path.parent)
        generic = png_path.parent / "feasibility_vs_potency.png"
        if generic.exists():
            shutil.copy2(generic, png_path)


def _generate_benchmark_and_rl_plots() -> None:
    source_holdout_module = _import("src.evaluation.run_source_holdout_benchmark")
    reward_module = _import("src.evaluation.run_reward_hacking_challenge")
    rediscovery_module = _import("src.evaluation.run_rediscovery_benchmark")
    robustness_module = _import("src.models.run_model_robustness_benchmark")
    prospective_module = _import("src.evaluation.select_prospective_validation_batch")
    rl_module = _import("src.rl.train_verifiable_rl")
    gpu_dqn_module = _import("src.rl.train_gpu_dqn")
    actor_critic_module = _import("src.rl.train_gpu_actor_critic")

    summary_path = REPORTS_DIR / "source_holdout_benchmark.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path, low_memory=False)
        source_holdout_module._plot_holdout_rmse(summary_df, TARGET_DIR / "source_holdout_rmse.png")
        source_holdout_module._plot_holdout_recall(summary_df, TARGET_DIR / "source_holdout_recall.png")

    reward_summary_path = REPORTS_DIR / "reward_hacking_challenge" / "reward_hacking_challenge_summary.csv"
    if reward_summary_path.exists():
        reward_df = pd.read_csv(reward_summary_path, low_memory=False)
        reward_out_dir = TARGET_DIR / "reward_hacking_challenge"
        reward_out_dir.mkdir(parents=True, exist_ok=True)
        reward_module._plot_rank_shift(reward_df, reward_out_dir)
        reward_module._plot_status_rates(reward_df, reward_out_dir)
        _copy_if_exists(
            REPORTS_DIR / "reward_hacking_challenge" / "reward_hacking_challenge_summary.csv",
            reward_out_dir / "reward_hacking_challenge_summary.csv",
        )
        _copy_if_exists(
            REPORTS_DIR / "reward_hacking_challenge" / "reward_hacking_challenge_examples.csv",
            reward_out_dir / "reward_hacking_challenge_examples.csv",
        )

    rediscovery_curves = REPORTS_DIR / "rediscovery_benchmark" / "rediscovery_recall_at_k.csv"
    rediscovery_panel = REPORTS_DIR / "rediscovery_benchmark" / "rediscovery_panel.csv"
    rediscovery_out_dir = TARGET_DIR / "rediscovery_benchmark"
    rediscovery_out_dir.mkdir(parents=True, exist_ok=True)
    if rediscovery_curves.exists():
        curves_df = pd.read_csv(rediscovery_curves, low_memory=False)
        rediscovery_module._plot_recall(curves_df, rediscovery_out_dir / "rediscovery_recall_at_k.png")
        _copy_if_exists(rediscovery_curves, rediscovery_out_dir / "rediscovery_recall_at_k.csv")
    if rediscovery_panel.exists():
        panel_df = pd.read_csv(rediscovery_panel, low_memory=False)
        rediscovery_module._plot_rank_shift(panel_df, rediscovery_out_dir / "rediscovery_rank_shift.png")
        _copy_if_exists(rediscovery_panel, rediscovery_out_dir / "rediscovery_panel.csv")

    robustness_summary = REPORTS_DIR / "model_robustness_summary.csv"
    if robustness_summary.exists():
        robustness_df = pd.read_csv(robustness_summary, low_memory=False)
        robustness_module._plot_robustness(robustness_df, TARGET_DIR)

    prospective_csv = REPORTS_DIR / "prospective_validation_batch.csv"
    if prospective_csv.exists():
        selected_df = pd.read_csv(prospective_csv, low_memory=False)
        all_candidates = selected_df.copy()
        prospective_module._plot_prospective_batch(
            all_candidates=all_candidates,
            selected=selected_df,
            out_dir=TARGET_DIR,
        )

    rl_dir = TARGET_DIR / "rl_verifiable"
    rl_dir.mkdir(parents=True, exist_ok=True)
    rl_episode_path = REPORTS_DIR / "rl_verifiable" / "rl_episode_summary.csv"
    rl_step_path = REPORTS_DIR / "rl_verifiable" / "rl_step_ledger.csv"
    rl_top_path = REPORTS_DIR / "rl_verifiable" / "rl_top_candidates.csv"
    if rl_episode_path.exists():
        rl_episode_df = pd.read_csv(rl_episode_path, low_memory=False)
        rl_module._plot_training_curve(rl_episode_df, rl_dir)
        _copy_if_exists(rl_episode_path, rl_dir / "rl_episode_summary.csv")
    if rl_step_path.exists():
        rl_step_df = pd.read_csv(rl_step_path, low_memory=False)
        rl_module._plot_reward_breakdown(rl_step_df, rl_dir)
        _copy_if_exists(rl_step_path, rl_dir / "rl_step_ledger.csv")
    if rl_top_path.exists():
        rl_top_df = pd.read_csv(rl_top_path, low_memory=False)
        rl_module._plot_external_evidence_priority(rl_top_df, rl_dir)
        _copy_if_exists(rl_top_path, rl_dir / "rl_top_candidates.csv")
    _copy_if_exists(REPORTS_DIR / "rl_verifiable" / "rl_training_summary.json", rl_dir / "rl_training_summary.json")
    _copy_if_exists(REPORTS_DIR / "rl_verifiable" / "rl_vs_baselines.csv", rl_dir / "rl_vs_baselines.csv")

    gpu_dqn_dir = TARGET_DIR / "rl_gpu_dqn"
    gpu_dqn_dir.mkdir(parents=True, exist_ok=True)
    gpu_dqn_episode_path = REPORTS_DIR / "rl_gpu_dqn" / "gpu_rl_episode_summary.csv"
    if gpu_dqn_episode_path.exists():
        gpu_dqn_episode_df = pd.read_csv(gpu_dqn_episode_path, low_memory=False)
        gpu_dqn_module._plot_curve(gpu_dqn_episode_df, gpu_dqn_dir)
        _copy_if_exists(gpu_dqn_episode_path, gpu_dqn_dir / "gpu_rl_episode_summary.csv")
    _copy_if_exists(REPORTS_DIR / "rl_gpu_dqn" / "gpu_rl_top_candidates.csv", gpu_dqn_dir / "gpu_rl_top_candidates.csv")
    _copy_if_exists(REPORTS_DIR / "rl_gpu_dqn" / "gpu_rl_training_summary.json", gpu_dqn_dir / "gpu_rl_training_summary.json")

    actor_critic_dir = TARGET_DIR / "rl_gpu_actor_critic"
    actor_critic_dir.mkdir(parents=True, exist_ok=True)
    actor_critic_episode_path = REPORTS_DIR / "rl_gpu_actor_critic" / "gpu_actor_critic_episode_summary.csv"
    if actor_critic_episode_path.exists():
        actor_critic_episode_df = pd.read_csv(actor_critic_episode_path, low_memory=False)
        actor_critic_module._plot_curve(actor_critic_episode_df.rename(columns={"loss": "mean_loss"}), actor_critic_dir)
        _copy_if_exists(actor_critic_episode_path, actor_critic_dir / "gpu_actor_critic_episode_summary.csv")
    _copy_if_exists(REPORTS_DIR / "rl_gpu_actor_critic" / "gpu_actor_critic_top_candidates.csv", actor_critic_dir / "gpu_actor_critic_top_candidates.csv")
    _copy_if_exists(REPORTS_DIR / "rl_gpu_actor_critic" / "gpu_actor_critic_summary.json", actor_critic_dir / "gpu_actor_critic_summary.json")


def _generate_visual_modules() -> None:
    visual_modules = [
        "src.visualization.build_public_presentation_visuals",
        "src.visualization.build_davinci_presentation_visuals",
        "src.visualization.build_key_presentation_visuals",
        "src.visualization.build_ml_chapter_figures",
        "src.visualization.build_polished_presentation_visuals",
        "src.visualization.build_davinci_competition_graphics",
    ]
    for module_name in visual_modules:
        _set_output_dir(module_name).main()

    technical_notebook_module = _import("src.visualization.technical_notebook_plots")
    technical_notebook_module.build_assets(out_dir=TARGET_DIR / "technical_notebook")


def _generate_presentation_visuals_aliases() -> None:
    destination_dir = TARGET_DIR / "presentation_visuals"
    destination_dir.mkdir(parents=True, exist_ok=True)
    alias_pairs = [
        (TARGET_DIR / "technical_notebook" / "pipeline_flowchart.png", destination_dir / "01_pipeline_flowchart.png"),
        (TARGET_DIR / "ml_chapter_figures" / "ml_01_split_performance.png", destination_dir / "02_model_validation.png"),
        (TARGET_DIR / "technical_notebook" / "single_agent_vs_multi_agent.png", destination_dir / "03_single_vs_multi_agent.png"),
        (TARGET_DIR / "technical_notebook" / "technical_notebook_chemical_space.png", destination_dir / "04_chemical_space.png"),
        (TARGET_DIR / "prospective_batch_readiness_vs_novelty.png", destination_dir / "05_prospective_batch.png"),
        (TARGET_DIR / "technical_notebook" / "top_leads_agent_support_heatmap.png", destination_dir / "06_agent_support_heatmap.png"),
        (TARGET_DIR / "top_molecules_grid.png", destination_dir / "07_top_candidates_grid.png"),
    ]
    for source, destination in alias_pairs:
        _copy_if_exists(source, destination)


def _generate_quick_notebook_aliases() -> None:
    source_dir = TARGET_DIR / "technical_notebook"
    destination_dir = TARGET_DIR / "technical_notebook_quick"
    destination_dir.mkdir(parents=True, exist_ok=True)
    quick_names = [
        "audit_rank_demotions.png",
        "marketed_vs_generated_boxplots.png",
        "naive_vs_protected_scores.png",
        "novelty_vs_applicability.png",
        "risk_distribution_by_audit_status.png",
        "technical_notebook_chemical_space.png",
        "top_leads_agent_support_heatmap.png",
    ]
    for name in quick_names:
        _copy_if_exists(source_dir / name, destination_dir / name)


def _copy_existing_localized_sets() -> None:
    _copy_tree_images(REPORTS_DIR / "presentation_visuals_davinci_v2", TARGET_DIR / "presentation_visuals_davinci_v2")


def _write_legends() -> None:
    for destination in [PROJECT_ROOT / "legenda_grafice.md", TARGET_DIR / "legenda_grafice.md"]:
        destination.write_text(LEGEND_TEXT, encoding="utf-8")


def _write_inventory() -> None:
    image_paths = sorted(
        path.relative_to(TARGET_DIR).as_posix()
        for path in TARGET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".bmp"}
    )
    inventory_path = TARGET_DIR / "inventar_grafice.txt"
    inventory_path.write_text("\n".join(image_paths), encoding="utf-8")


def main() -> None:
    _ensure_clean_target()
    with _localized_matplotlib():
        _generate_root_report_plots()
        _generate_top_molecules_grid()
        _generate_crossdb_and_feasibility()
        _generate_benchmark_and_rl_plots()
        _generate_visual_modules()
    _generate_presentation_visuals_aliases()
    _generate_quick_notebook_aliases()
    _copy_existing_localized_sets()
    _write_legends()
    _write_inventory()
    print(f"[OK] Grafice exportate in: {TARGET_DIR}")


if __name__ == "__main__":
    main()
