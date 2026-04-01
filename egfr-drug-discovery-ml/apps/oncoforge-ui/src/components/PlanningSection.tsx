import { useMemo, useState } from "react";

import { InteractiveBarChart, InteractiveLineChart } from "@/components/InteractiveCharts";
import { SectionCard } from "@/components/SectionCard";
import type { ChemistNotebookEntry, ExperimentalPlanEntry, SessionCompareItem } from "@/types";

interface PlanningSectionProps {
  planner: ExperimentalPlanEntry[];
  sessionCompare: SessionCompareItem[];
  notebookEntries: ChemistNotebookEntry[];
}

export function PlanningSection({
  planner,
  sessionCompare,
  notebookEntries,
}: PlanningSectionProps) {
  const [tagFilter, setTagFilter] = useState("toate");
  const sessionLabels = sessionCompare.map((entry) => entry.sessionName);
  const filteredNotes = useMemo(() => {
    if (tagFilter === "toate") {
      return notebookEntries;
    }
    return notebookEntries.filter((entry) => entry.tags.includes(tagFilter) || entry.verdict === tagFilter);
  }, [notebookEntries, tagFilter]);

  const availableFilters = useMemo(
    () =>
      ["toate", ...new Set(notebookEntries.flatMap((entry) => [entry.verdict, ...entry.tags]))].filter(Boolean),
    [notebookEntries],
  );

  return (
    <div className="space-y-4">
      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_380px]">
        <SectionCard
          eyebrow="Plan experimental"
          title="Top candidati pentru urmatorul pas"
          subtitle="Ce merita testat, cu ce control pozitiv si cata materie ar fi utila pentru primul screening."
        >
          <div className="space-y-3">
            {planner.map((entry) => (
              <div key={entry.smiles} className="rounded-3xl border border-white/6 bg-slate-950/70 p-4">
                <div className="grid gap-4 xl:grid-cols-[160px_minmax(0,1fr)_170px]">
                  <div>
                    <p className="ui-kicker">Prioritate</p>
                    <p className="mt-2 text-xl font-semibold text-white">{entry.priority}</p>
                    <p className="mt-2 text-sm text-slate-300">#{entry.rank}</p>
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white">{entry.name}</p>
                    <p className="mt-2 break-all font-mono text-xs text-slate-400">{entry.smiles}</p>
                    <div className="mt-3 space-y-2 text-sm text-slate-300">
                      <p>Assay: <span className="text-white">{entry.assay}</span></p>
                      <p>Control: <span className="text-white">{entry.control}</span></p>
                      <p>Ruta: <span className="text-white">{entry.route || "-"}</span></p>
                    </div>
                  </div>
                  <div>
                    <div className="rounded-2xl border border-white/6 bg-white/5 p-3">
                      <p className="ui-kicker">Material necesar</p>
                      <p className="mt-2 text-sm leading-7 text-white">{entry.materialPlan}</p>
                    </div>
                    <div className="mt-3 rounded-2xl border border-white/6 bg-white/5 p-3">
                      <p className="ui-kicker">Cost estimat</p>
                      <p className="mt-2 text-xl font-semibold text-white">${entry.estimatedCost.toFixed(2)}</p>
                    </div>
                  </div>
                </div>
                <div className="mt-4 rounded-2xl border border-white/6 bg-white/5 p-3 text-sm leading-7 text-slate-300">
                  {entry.rationale}
                </div>
              </div>
            ))}
          </div>
        </SectionCard>

        <SectionCard
          eyebrow="Carnet global"
          title="Notele chimistului"
          subtitle="Filtreaza rapid moleculele marcate cu verdict sau tag."
        >
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              {availableFilters.map((item) => (
                <button
                  key={item}
                  className={`rounded-full border px-3 py-1.5 text-xs ${
                    tagFilter === item ? "border-forge-cyan/60 bg-forge-cyan/15 text-white" : "border-white/10 bg-white/5 text-slate-300"
                  }`}
                  onClick={() => setTagFilter(item)}
                >
                  {item}
                </button>
              ))}
            </div>
            <div className="space-y-3">
              {filteredNotes.length ? (
                filteredNotes.map((entry) => (
                  <div key={`${entry.smiles}-${entry.updatedAt}`} className="rounded-2xl border border-white/6 bg-white/5 p-4">
                    <p className="text-sm font-semibold text-white">{entry.verdict}</p>
                    <p className="mt-2 break-all font-mono text-xs text-slate-400">{entry.smiles}</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {entry.tags.map((tag) => (
                        <span key={tag} className="rounded-full border border-white/10 bg-slate-950/70 px-3 py-1 text-xs text-slate-200">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <p className="mt-3 text-sm leading-7 text-slate-300">{entry.note || "Fara nota text."}</p>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-white/6 bg-white/5 p-4 text-sm text-slate-300">
                  Nu exista inca note care sa corespunda filtrului ales.
                </div>
              )}
            </div>
          </div>
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Comparatie intre sesiuni"
          title="Cum se compara sesiunile OncoSynth"
          subtitle="Poti vedea repede ce mod a mers mai bine si unde s-a blocat fiecare sesiune."
        >
          <InteractiveBarChart
            categories={sessionLabels}
            xLabel="Sesiune"
            yLabel="Valoare"
            className="h-[360px]"
            formatValue={(value) => value.toFixed(1)}
            series={[
              { label: "Promovate", color: "#40d98f", data: sessionCompare.map((entry) => entry.promotedCount) },
              { label: "Cel mai bun pIC50", color: "#24d6ea", data: sessionCompare.map((entry) => entry.bestPic50) },
              { label: "Cost mediu 10 mg", color: "#f59e0b", data: sessionCompare.map((entry) => entry.meanCost10mg) },
            ]}
          />
        </SectionCard>

        <SectionCard
          eyebrow="Probleme dominante"
          title="Unde s-a blocat fiecare sesiune"
          subtitle="Aceasta cronologie ajuta sa vezi daca problema este costul, incertitudinea sau trierea prea stricta."
        >
          <InteractiveLineChart
            categories={sessionLabels}
            xLabel="Sesiune"
            yLabel="Indicator"
            className="h-[360px]"
            formatValue={(value) => value.toFixed(2)}
            series={[
              { label: "Cel mai bun scor", color: "#73a6ff", data: sessionCompare.map((entry) => entry.bestScore) },
              { label: "Incertitudine medie", color: "#fb7185", data: sessionCompare.map((entry) => entry.meanUncertainty) },
              { label: "Cost mediu", color: "#f59e0b", data: sessionCompare.map((entry) => entry.meanCost10mg) },
            ]}
          />
        </SectionCard>
      </div>

      <SectionCard
        eyebrow="Sinteza rapida"
        title="Rezumat textual pe sesiuni"
        subtitle="Constrangerea dominanta este formulata direct pentru consultare rapida, fara accesarea logurilor."
      >
        <div className="grid gap-3 xl:grid-cols-2">
          {sessionCompare.map((entry) => (
            <div key={entry.sessionName} className={`rounded-2xl border p-4 ${entry.isCurrent ? "border-forge-cyan/40 bg-forge-cyan/10" : "border-white/8 bg-white/5"}`}>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-white">{entry.sessionName}</p>
                  <p className="mt-1 text-xs text-slate-400">{entry.modeLabel} | {entry.statusLabel}</p>
                </div>
                <span className="text-xs text-slate-400">{entry.updatedAt ? new Date(entry.updatedAt).toLocaleString("ro-RO") : "-"}</span>
              </div>
              <p className="mt-3 text-sm leading-7 text-slate-300">{entry.bottleneck}</p>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
