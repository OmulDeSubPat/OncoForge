import { memo } from "react";

import { SectionCard } from "@/components/SectionCard";
import { InteractiveBarChart, InteractiveLineChart } from "@/components/Charts";
import { StatusPill } from "@/components/StatusPill";
import type { LibraryRow, TimelinePayload } from "@/types";

interface TimelineRailProps {
  timeline: TimelinePayload;
  library: LibraryRow[];
  selectedRound: number;
  onJump: (round: number, candidateSmiles?: string) => void;
}

export const TimelineRail = memo(function TimelineRail({ timeline, library, selectedRound, onJump }: TimelineRailProps) {
  const bestScoreSeries = [
    {
      id: "best-score",
      label: "Scor maxim",
      color: "#24d6ea",
      data: timeline.generations.map((frame) => ({
        x: frame.round,
        y: frame.bestScore,
        label: `Runda ${frame.round}`,
      })),
    },
    {
      id: "cost-mediu",
      label: "Cost mediu 10 mg",
      color: "#73a6ff",
      data: timeline.generations.map((frame) => ({
        x: frame.round,
        y: frame.avgCost10mg,
        label: `Runda ${frame.round}`,
      })),
    },
  ];
  const countBars = timeline.generations.map((frame) => ({
    label: `R${frame.round}`,
    value: frame.promotedCandidates,
    color: "#40d98f",
  }));

  if (!timeline.generations.length) {
    return (
      <SectionCard
        eyebrow="Evolutie iterativa"
        title="Cronologie pe generatii"
        subtitle="Aici va aparea istoricul rundelor, impreuna cu relatiile parinte-copil."
        className="h-full"
      >
        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-6 text-sm leading-8 text-slate-300">
          Timeline-ul este gol pana cand worker-ul proceseaza prima runda. Dupa aceea, fiecare etapa va afisa numarul de candidati noi, promovarile si scorul maxim atins.
        </div>
      </SectionCard>
    );
  }

  return (
    <SectionCard
      eyebrow="Evolutie iterativa"
      title="Cronologie pe generatii"
      subtitle="Relatiile parinte-copil, imbunatatirile si densitatea fiecarei runde raman vizibile."
      className="h-full"
    >
      <div className="space-y-4">
        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Trenduri pe runde</p>
              <p className="mt-1 text-sm text-slate-300">Graficul arata cum evolueaza scorul maxim si costul mediu al loturilor succesive.</p>
            </div>
            <StatusPill status="active" label={`Runda ${selectedRound}`} />
          </div>
          <InteractiveLineChart
            series={bestScoreSeries}
            xLabel="Runda"
            yLabel="Valoare"
            className="mt-4"
            valueFormatter={(value) => value.toFixed(2)}
          />
          <InteractiveBarChart
            data={countBars}
            xLabel="Runda"
            yLabel="Molecule promovate"
            className="mt-4"
            valueFormatter={(value) => value.toFixed(0)}
          />
        </div>

        <div className="overflow-x-auto">
          <div className="flex min-w-max gap-3 pb-1">
            {timeline.generations.map((frame) => {
              const candidate = library.find((item) => item.round === frame.round);
              const active = frame.round === selectedRound;
              return (
                <button
                  key={`${frame.round}-${frame.seedStep}`}
                  className={`w-72 rounded-3xl border p-4 text-left transition ${
                    active
                      ? "border-forge-cyan/60 bg-forge-cyan/12 shadow-glow"
                      : "border-white/8 bg-white/5 hover:bg-white/10"
                  }`}
                  onClick={() => onJump(frame.round, candidate?.smiles)}
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Runda {frame.round}</p>
                    {candidate ? <StatusPill status={candidate.status} label={candidate.statusLabel} /> : null}
                  </div>
                  <p className="mt-2 text-lg font-semibold text-white">{candidate?.action || "Etapa de optimizare"}</p>
                  <p className="mt-2 text-sm text-slate-300">{candidate?.route || "Mutatie generata din worker"}</p>
                  <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-slate-300">
                    <span>Noi: {frame.newCandidates}</span>
                    <span>Promovate: {frame.promotedCandidates}</span>
                    <span>Scor: {frame.bestScore.toFixed(2)}</span>
                    <span>Cost mediu: ${frame.avgCost10mg.toFixed(2)}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </SectionCard>
  );
});
