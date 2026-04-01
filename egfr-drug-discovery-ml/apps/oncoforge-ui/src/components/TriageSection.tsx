import { useMemo, useState } from "react";

import { InteractiveBarChart, InteractiveLineChart, InteractiveScatterChart } from "@/components/InteractiveCharts";
import { SectionCard } from "@/components/SectionCard";
import { defaultWhatIfWeights, normalizeWeights, rankLibraryWithWhatIf, type WhatIfWeights } from "@/lib/ranking";
import type { DashboardAnalyticsPayload, LibraryRow, RLMonitorPayload } from "@/types";

interface TriageSectionProps {
  library: LibraryRow[];
  monitor: RLMonitorPayload;
  analytics: DashboardAnalyticsPayload;
  selectedSmiles: string | null;
  onSelectCandidate: (smiles: string) => void;
}

type HeatmapMetric = "score" | "pic50" | "cost" | "risk";

function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function metricValue(row: LibraryRow, metric: HeatmapMetric) {
  if (metric === "score") {
    return row.score;
  }
  if (metric === "pic50") {
    return row.pic50;
  }
  if (metric === "cost") {
    return row.cost10mg;
  }
  return row.risk ?? 0;
}

function metricColor(row: LibraryRow, metric: HeatmapMetric) {
  const value = metricValue(row, metric);
  if (metric === "score") {
    const alpha = clamp(value / 14, 0.12, 1);
    return `rgba(36,214,234,${alpha})`;
  }
  if (metric === "pic50") {
    const alpha = clamp(value / 10, 0.12, 1);
    return `rgba(64,217,143,${alpha})`;
  }
  if (metric === "cost") {
    const alpha = clamp(value / 80, 0.12, 1);
    return `rgba(245,158,11,${alpha})`;
  }
  const alpha = clamp(value / 0.35, 0.12, 1);
  return `rgba(251,113,133,${alpha})`;
}

function SliderField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="rounded-2xl border border-white/6 bg-slate-950/70 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-white">{label}</span>
        <span className="text-xs text-slate-300">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={0.05}
        max={0.6}
        step={0.01}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-3 h-2 w-full accent-cyan-300"
      />
    </label>
  );
}

export function TriageSection({
  library,
  monitor,
  analytics,
  selectedSmiles,
  onSelectCandidate,
}: TriageSectionProps) {
  const [weights, setWeights] = useState<WhatIfWeights>(defaultWhatIfWeights);
  const [heatmapMetric, setHeatmapMetric] = useState<HeatmapMetric>("score");
  const normalizedWeights = useMemo(() => normalizeWeights(weights), [weights]);

  const rankedLibrary = useMemo(() => rankLibraryWithWhatIf(library, weights), [library, weights]);
  const topWhatIf = rankedLibrary.slice(0, 8);
  const rounds = useMemo(() => Array.from(new Set(library.map((row) => row.round))).sort((a, b) => a - b), [library]);
  const heatmapRows = useMemo(
    () =>
      rounds.map((round) =>
        [...library]
          .filter((row) => row.round === round)
          .sort((left, right) => right.score - left.score)
          .slice(0, 8),
      ),
    [library, rounds],
  );

  const penaltyMeans = useMemo(() => {
    if (!monitor.penaltySeries.length) {
      return { toxicity: 0, invalid: 0, uncertainty: 0, reward: 0 };
    }
    const count = monitor.penaltySeries.length;
    return {
      toxicity: monitor.penaltySeries.reduce((sum, entry) => sum + entry.toxicityPenalty, 0) / count,
      invalid: monitor.penaltySeries.reduce((sum, entry) => sum + entry.invalidPenalty, 0) / count,
      uncertainty: monitor.penaltySeries.reduce((sum, entry) => sum + entry.uncertaintyPenalty, 0) / count,
      reward: monitor.penaltySeries.reduce((sum, entry) => sum + entry.rewardRiskPenalty, 0) / count,
    };
  }, [monitor.penaltySeries]);

  const uncertaintyScatter = library.slice(0, 60).map((row) => ({
    label: `#${row.rank}`,
    x: row.uncertainty,
    y: row.score,
    color: row.smiles === selectedSmiles ? "#73a6ff" : row.status === "promovata" ? "#40d98f" : "#24d6ea",
    size: row.smiles === selectedSmiles ? 9 : 6,
    meta: `${row.action || "Mutatie"} | risc ${(row.risk ?? 0).toFixed(2)}`,
  }));

  const roundLabels = analytics.agentSeries.map((entry) => `R${entry.round}`);
  const stabilityLabels = analytics.rankingStability.map((entry) => `R${entry.round}`);
  const etichetePonderi: Record<keyof WhatIfWeights, string> = {
    potency: "Potenta",
    toxicity: "Siguranta",
    synthesizability: "Sintetizabilitate",
    cost: "Cost",
    novelty: "Noutate",
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-4 2xl:grid-cols-[420px_minmax(0,1fr)]">
        <SectionCard
          eyebrow="Reponderare interactiva"
          title="Simuleaza o alta ierarhizare"
          subtitle="Muta ponderile obiectivelor si vezi instant ce molecule urca sau coboara in triere."
        >
          <div className="space-y-3">
            <SliderField label="Potenta" value={weights.potency} onChange={(value) => setWeights((current) => ({ ...current, potency: value }))} />
            <SliderField label="Siguranta / toxicitate" value={weights.toxicity} onChange={(value) => setWeights((current) => ({ ...current, toxicity: value }))} />
            <SliderField label="Sintetizabilitate" value={weights.synthesizability} onChange={(value) => setWeights((current) => ({ ...current, synthesizability: value }))} />
            <SliderField label="Cost" value={weights.cost} onChange={(value) => setWeights((current) => ({ ...current, cost: value }))} />
            <SliderField label="Noutate" value={weights.novelty} onChange={(value) => setWeights((current) => ({ ...current, novelty: value }))} />

            <div className="grid gap-3 sm:grid-cols-2">
              {Object.entries(normalizedWeights).map(([key, value]) => (
                <div key={key} className="rounded-2xl border border-white/6 bg-white/5 p-3">
                  <p className="ui-kicker">{etichetePonderi[key as keyof WhatIfWeights]}</p>
                  <p className="mt-2 text-lg font-semibold text-white">{(value * 100).toFixed(0)}%</p>
                </div>
              ))}
            </div>
          </div>
        </SectionCard>

        <SectionCard
          eyebrow="Top recalculat"
          title="Clasament live dupa noile ponderi"
          subtitle="Aceasta lista nu modifica executia curenta. Este o simulare locala de prioritizare pentru chimist."
        >
          <div className="space-y-3">
            {topWhatIf.map((entry) => (
              <button
                key={entry.row.id}
                className={`grid w-full gap-3 rounded-2xl border px-4 py-4 text-left transition lg:grid-cols-[72px_minmax(0,1fr)_120px_120px] ${
                  entry.row.smiles === selectedSmiles
                    ? "border-forge-cyan/50 bg-forge-cyan/12"
                    : "border-white/8 bg-white/5 hover:bg-white/10"
                }`}
                onClick={() => onSelectCandidate(entry.row.smiles)}
              >
                <div>
                  <p className="ui-kicker">Rang nou</p>
                  <p className="mt-2 text-xl font-semibold text-white">#{entry.altRank}</p>
                </div>
                <div>
                  <p className="ui-kicker">Molecula</p>
                  <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-200">{entry.row.smiles}</p>
                </div>
                <div>
                  <p className="ui-kicker">Scor recalculat</p>
                  <p className="mt-2 text-lg font-semibold text-white">{entry.altScore.toFixed(3)}</p>
                </div>
                <div>
                  <p className="ui-kicker">Delta vs scor curent</p>
                  <p className={`mt-2 text-lg font-semibold ${entry.scoreDelta >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                    {entry.scoreDelta >= 0 ? "+" : ""}
                    {entry.scoreDelta.toFixed(3)}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Heatmap pe generatii"
          title="Harta rapida a loturilor"
          subtitle="Pe OY sunt rundele, pe OX primele molecule din fiecare runda. Culoarea urmareste metrica aleasa."
          action={
            <div className="flex flex-wrap gap-2">
              {[
                { key: "score", label: "Scor" },
                { key: "pic50", label: "pIC50" },
                { key: "cost", label: "Cost" },
                { key: "risk", label: "Risc" },
              ].map((item) => (
                <button
                  key={item.key}
                  className={`rounded-full border px-3 py-1.5 text-xs ${heatmapMetric === item.key ? "border-forge-cyan/60 bg-forge-cyan/15 text-white" : "border-white/10 bg-white/5 text-slate-300"}`}
                  onClick={() => setHeatmapMetric(item.key as HeatmapMetric)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          }
        >
          <div className="space-y-3">
            {heatmapRows.length ? (
              heatmapRows.map((roundRows, roundIndex) => (
                <div key={`heat-${rounds[roundIndex]}`} className="grid gap-2" style={{ gridTemplateColumns: "96px repeat(8, minmax(0, 1fr))" }}>
                  <div className="rounded-2xl border border-white/6 bg-white/5 px-3 py-2 text-sm font-semibold text-white">
                    Runda {rounds[roundIndex]}
                  </div>
                  {Array.from({ length: 8 }).map((_, cellIndex) => {
                    const row = roundRows[cellIndex];
                    return row ? (
                      <button
                        key={row.id}
                        className="rounded-2xl border border-white/5 px-2 py-3 text-left text-xs text-white transition hover:scale-[1.02]"
                        style={{ backgroundColor: metricColor(row, heatmapMetric) }}
                        onClick={() => onSelectCandidate(row.smiles)}
                      >
                        <span className="block font-semibold">#{row.rank}</span>
                        <span className="mt-1 block text-[11px] text-slate-100/90">{metricValue(row, heatmapMetric).toFixed(heatmapMetric === "cost" ? 2 : 2)}</span>
                      </button>
                    ) : (
                      <div key={`empty-${roundIndex}-${cellIndex}`} className="rounded-2xl border border-white/5 bg-white/[0.03] px-2 py-3 text-xs text-slate-500" />
                    );
                  })}
                </div>
              ))
            ) : (
              <div className="rounded-2xl border border-white/6 bg-slate-950/70 p-4 text-sm text-slate-300">
                Heatmap-ul va aparea dupa prima generatie cu mai multe runde.
              </div>
            )}
          </div>
        </SectionCard>

        <SectionCard
          eyebrow="Incredere si calibrare"
          title="Cum arata increderea lotului"
          subtitle="Axe clasice pentru relatia dintre scor, incertitudine, penalizari si stabilitatea ierarhiei."
        >
          <div className="grid gap-4">
            <InteractiveScatterChart
              points={uncertaintyScatter}
              xLabel="Incertitudine"
              yLabel="Scor final"
              className="h-[300px]"
              formatX={(value) => value.toFixed(3)}
              formatY={(value) => value.toFixed(2)}
            />
            <InteractiveBarChart
              categories={["Toxicitate", "Invaliditate", "Incertitudine", "Recompensa"]}
              xLabel="Tip penalizare"
              yLabel="Medie lot"
              className="h-[280px]"
              formatValue={(value) => value.toFixed(2)}
              series={[
                {
                  label: "Penalizare medie",
                  color: "#73a6ff",
                  data: [penaltyMeans.toxicity, penaltyMeans.invalid, penaltyMeans.uncertainty, penaltyMeans.reward],
                },
              ]}
            />
          </div>
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Agenti in timp"
          title="Contributia agentilor pe runda"
          subtitle="Arata cine a tras lotul in sus pe parcursul optimizarii."
        >
          <InteractiveLineChart
            categories={roundLabels}
            xLabel="Runda"
            yLabel="Contributie medie"
            className="h-[340px]"
            formatValue={(value) => value.toFixed(2)}
            series={[
              { label: "Generator", color: "#24d6ea", data: analytics.agentSeries.map((entry) => entry.generator) },
              { label: "Toxicitate", color: "#fb7185", data: analytics.agentSeries.map((entry) => entry.toxicity) },
              { label: "Validator", color: "#40d98f", data: analytics.agentSeries.map((entry) => entry.validator) },
              { label: "Optimizator", color: "#73a6ff", data: analytics.agentSeries.map((entry) => entry.optimizer) },
            ]}
          />
        </SectionCard>

        <SectionCard
          eyebrow="Stabilitate ierarhie"
          title="Scor maxim, medie si dispersie"
          subtitle="Daca dispersia creste, lotul se separa mai clar. Daca rata de promovare urca, lista prioritara devine mai utila."
        >
          <InteractiveLineChart
            categories={stabilityLabels}
            xLabel="Runda"
            yLabel="Scor / rata"
            className="h-[340px]"
            formatValue={(value) => value.toFixed(2)}
            series={[
              { label: "Scor maxim", color: "#24d6ea", data: analytics.rankingStability.map((entry) => entry.topScore) },
              { label: "Scor mediu", color: "#40d98f", data: analytics.rankingStability.map((entry) => entry.meanScore) },
              { label: "Rata promovare", color: "#f59e0b", data: analytics.rankingStability.map((entry) => entry.promotedRate) },
              { label: "Dispersie", color: "#73a6ff", data: analytics.rankingStability.map((entry) => entry.scoreSpread) },
            ]}
          />
        </SectionCard>
      </div>
    </div>
  );
}
