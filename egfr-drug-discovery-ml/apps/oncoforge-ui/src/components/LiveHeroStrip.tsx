import { memo, useMemo } from "react";

import { InteractiveBarChart, InteractiveLineChart } from "@/components/InteractiveCharts";
import { SectionCard } from "@/components/SectionCard";
import { StatusPill } from "@/components/StatusPill";
import type { LibraryRow, OverviewPayload, SelectedMolecule, TimelinePayload } from "@/types";

interface LiveHeroStripProps {
  overview: OverviewPayload;
  selected: SelectedMolecule | null;
  library: LibraryRow[];
  timeline: TimelinePayload;
  uiMode: "basic" | "expert";
  onOpenMolecule: () => void;
  onOpenLibrary: () => void;
  onSelectCandidate: (smiles: string) => void;
}

function quickVerdict(selected: SelectedMolecule | null) {
  if (!selected) {
    return "Alege un candidat pentru analiza";
  }
  if (selected.score >= 2.5 && selected.deltaPic50 >= 0 && selected.marketSimilarity <= 0.78) {
    return "Candidat puternic pentru lista prioritara";
  }
  if (selected.score >= 1.5) {
    return "Candidat bun, dar necesita validare";
  }
  return "Candidat secundar";
}

function quickAction(selected: SelectedMolecule | null) {
  if (!selected) {
    return "Deschide biblioteca sau sectiunea de triere pentru a selecta prima molecula.";
  }
  if (selected.score >= 2.5) {
    return "Compara candidatul cu piata si pregateste lista prioritara pentru export.";
  }
  if (selected.score >= 1.5) {
    return "Verifica pragurile si riscul, apoi decide daca intra in setul de comparatie.";
  }
  return "Pastreaza molecula ca referinta si continua generarea.";
}

export const LiveHeroStrip = memo(function LiveHeroStrip({
  overview,
  selected,
  library,
  timeline,
  uiMode,
  onOpenMolecule,
  onOpenLibrary,
  onSelectCandidate,
}: LiveHeroStripProps) {
  const topLeads = useMemo(() => library.slice(0, uiMode === "basic" ? 3 : 4), [library, uiMode]);
  const lineCategories = timeline.generations.slice(-6).map((frame) => `R${frame.round}.${frame.seedStep}`);
  const bestScoreSeries = timeline.generations.slice(-6).map((frame) => frame.bestScore);
  const promotedSeries = timeline.generations.slice(-6).map((frame) => frame.promotedCandidates);
  const verdict = quickVerdict(selected);
  const selectedPic50 =
    typeof selected?.metrics.primary.find((item) => item.label === "pIC50")?.value === "number"
      ? (selected?.metrics.primary.find((item) => item.label === "pIC50")?.value as number)
      : selected?.score ?? 0;

  return (
    <div className="mt-3">
      <SectionCard
        eyebrow="Rezumat live"
        title="Rezumat executiv pentru molecula activa"
        subtitle="Primul ecran prioritizat pentru laborator: candidatul activ, verdictul rapid, topul candidatilor si doua grafice live."
        action={
          <div className="flex flex-wrap gap-2">
            <button className="control-button px-3 py-1.5 text-xs" onClick={onOpenMolecule}>
              Deschide molecula
            </button>
            <button className="control-button px-3 py-1.5 text-xs" onClick={onOpenLibrary}>
              Deschide biblioteca
            </button>
          </div>
        }
        className="overflow-hidden"
      >
        <div className="grid gap-4 2xl:grid-cols-[minmax(340px,0.95fr)_minmax(260px,0.7fr)_minmax(0,1.15fr)]">
          <div className="relative overflow-hidden rounded-[28px] border border-white/6 bg-[linear-gradient(145deg,rgba(9,18,33,0.96),rgba(7,15,28,0.84))] p-4 shadow-glow">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(36,214,234,0.16),transparent_40%),radial-gradient(circle_at_bottom_right,rgba(64,217,143,0.08),transparent_36%)]" />
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="ui-kicker">Molecula activa</p>
                <h3 className="mt-2 text-xl font-semibold text-white">
                  {selected ? `Candidat #${selected.rank}` : "Fara selectie activa"}
                </h3>
                <p className="mt-2 text-sm leading-7 text-slate-300">
                  {selected
                    ? `${selected.action || "Mutatie noua"} | Comparator ${selected.marketReference || "-"}`
                    : "Selecteaza un candidat din biblioteca sau din triere pentru a porni analiza."}
                </p>
              </div>
              {selected ? <StatusPill status={selected.status} label={selected.status.toUpperCase()} /> : null}
            </div>

            <div className="relative mt-4 grid gap-4 lg:grid-cols-[180px_minmax(0,1fr)]">
              <div className="flex min-h-[180px] items-center justify-center rounded-[24px] border border-white/6 bg-slate-950/90 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                {selected?.view.svg2d ? (
                  <img src={selected.view.svg2d} alt={`Molecula ${selected.rank}`} className="max-h-[150px] w-full object-contain motion-safe:animate-float-gentle" />
                ) : (
                  <p className="text-center text-sm leading-7 text-slate-400">Structura 2D apare aici dupa selectia unei molecule.</p>
                )}
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-[22px] border border-white/6 bg-white/5 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                  <p className="ui-kicker">Verdict rapid</p>
                  <p className="mt-2 text-lg font-semibold text-white">{verdict}</p>
                </div>
                <div className="rounded-[22px] border border-white/6 bg-white/5 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                  <p className="ui-kicker">pIC50 / scor</p>
                  <p className="mt-2 text-lg font-semibold tabular-nums text-white">
                    {selected ? `${selectedPic50.toFixed(2)} / ${selected.score.toFixed(2)}` : "--"}
                  </p>
                </div>
                <div className="rounded-[22px] border border-white/6 bg-white/5 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] sm:col-span-2">
                  <p className="ui-kicker">Actiune recomandata</p>
                  <p className="mt-2 text-sm leading-7 text-slate-300">{quickAction(selected)}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-[26px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="ui-kicker">Top candidati</p>
                <p className="mt-1 text-sm text-slate-300">Cele mai bune intrari din biblioteca curenta.</p>
              </div>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200">
                {library.length} intrari
              </span>
            </div>
            <div className="mt-3 space-y-2">
              {topLeads.length ? (
                topLeads.map((entry) => (
                  <button
                    key={entry.id}
                    className={`flex w-full items-center justify-between rounded-[22px] border px-4 py-3 text-left transition duration-300 ${
                      entry.smiles === selected?.smiles
                        ? "border-forge-cyan/45 bg-[linear-gradient(135deg,rgba(36,214,234,0.14),rgba(12,20,36,0.9))]"
                        : "border-white/8 bg-white/5 hover:bg-white/10"
                    }`}
                    onClick={() => onSelectCandidate(entry.smiles)}
                  >
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-white">Candidat #{entry.rank}</p>
                      <p className="mt-1 truncate text-xs text-slate-400">
                        {entry.pic50.toFixed(2)} pIC50 | scor {entry.score.toFixed(2)}
                      </p>
                    </div>
                    <span className="text-xs text-slate-400">R{entry.round}</span>
                  </button>
                ))
              ) : (
                <p className="text-sm leading-7 text-slate-300">Candidatii vor aparea aici dupa prima generatie.</p>
              )}
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="rounded-[22px] border border-white/6 bg-white/5 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Status sesiune</p>
                <p className="mt-2 text-lg font-semibold text-white">{overview.statusLabel}</p>
              </div>
              <div className="rounded-[22px] border border-white/6 bg-white/5 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Promovate</p>
                <p className="mt-2 text-lg font-semibold tabular-nums text-white">{overview.summary.promotedCount}</p>
              </div>
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <div className="rounded-[26px] border border-white/6 bg-slate-950/70 p-4">
              <p className="ui-kicker">Trend scor maxim</p>
              <InteractiveLineChart
                categories={lineCategories.length ? lineCategories : ["R0"]}
                xLabel="Etapa"
                yLabel="Scor"
                className="mt-4 h-[230px]"
                formatValue={(value) => value.toFixed(2)}
                series={[
                  {
                    label: "Scor maxim",
                    color: "#24d6ea",
                    data: bestScoreSeries.length ? bestScoreSeries : [0],
                  },
                ]}
              />
            </div>

            <div className="rounded-[26px] border border-white/6 bg-slate-950/70 p-4">
              <p className="ui-kicker">Promovari recente</p>
              <InteractiveBarChart
                categories={lineCategories.length ? lineCategories : ["R0"]}
                xLabel="Etapa"
                yLabel="Promovate"
                className="mt-4 h-[230px]"
                formatValue={(value) => value.toFixed(0)}
                series={[
                  {
                    label: "Promovate",
                    color: "#40d98f",
                    data: promotedSeries.length ? promotedSeries : [0],
                  },
                ]}
              />
            </div>
          </div>
        </div>
      </SectionCard>
    </div>
  );
});
