import { memo } from "react";

import { SectionCard } from "@/components/SectionCard";
import { InteractiveBarChart, ProgressRing, RadarChart } from "@/components/Charts";
import type { BreakdownItem, SelectedMolecule } from "@/types";

interface MetricsPanelProps {
  selected: SelectedMolecule | null;
}

function toneClass(tone: string) {
  switch (tone) {
    case "primary":
      return "text-cyan-100";
    case "warning":
      return "text-amber-300";
    case "success":
      return "text-emerald-300";
    case "info":
      return "text-blue-300";
    default:
      return "text-slate-200";
  }
}

function BreakdownRow({ item }: { item: BreakdownItem }) {
  const positive = item.tone !== "negative";
  const width = Math.min(100, Math.max(4, Math.abs(item.value) * 10));
  return (
    <div className="space-y-2 rounded-2xl border border-white/5 bg-white/5 p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm text-slate-300">{item.label}</span>
        <span className={`font-mono text-sm font-semibold ${positive ? "text-emerald-300" : "text-rose-300"}`}>
          {positive ? "+" : "-"}
          {Math.abs(item.value).toFixed(3)}
          {item.unit ? ` ${item.unit}` : ""}
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-900">
        <div
          className={`h-2 rounded-full ${positive ? "bg-gradient-to-r from-forge-green to-forge-cyan" : "bg-gradient-to-r from-forge-red to-forge-amber"}`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

export const MetricsPanel = memo(function MetricsPanel({ selected }: MetricsPanelProps) {
  if (!selected) {
    return (
      <SectionCard
        eyebrow="Dashboard stiintific"
        title="Metrici si explicatii"
        subtitle="Selecteaza o molecula pentru a vedea scorurile, radarul multi-obiectiv si breakdown-ul de cost."
        className="h-full"
      >
        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-6 text-sm leading-8 text-slate-300">
          Metricile vor aparea aici dupa ce OncoSynth are o molecula selectata. Panoul va include pIC50, incertitudine, QED, SA score, comparatia fata de piata si componentele scorului final.
        </div>
      </SectionCard>
    );
  }

  const metrics = selected?.metrics;
  const radarValues = metrics?.radar.map((item) => item.value) ?? [];
  const primaryPic50 = metrics?.primary.find((item) => item.label === "pIC50");
  const uncertainty = metrics?.primary.find((item) => item.label === "Incertitudine");
  const rankingChartData = (selected?.rankingBreakdown ?? []).map((item) => ({
    label: item.label
      .replace("Baza ", "")
      .replace("Noutate fata de ", "")
      .replace("Penalizare ", ""),
    value: item.tone === "negative" ? -Math.abs(item.value) : item.value,
    color: item.tone === "negative" ? "#f87171" : "#24d6ea",
  }));
  const costChartData = (selected?.costBreakdown ?? []).map((item) => ({
    label: item.label.replace("Cost ", "").replace("Complexitate ", ""),
    value: item.value,
    color: "#73a6ff",
  }));
  const profileChartData = [
    {
      label: "Potenta",
      value: Math.min(10, typeof primaryPic50?.value === "number" ? primaryPic50.value : 0),
      color: "#24d6ea",
    },
    {
      label: "QED",
      value: Number(metrics?.primary.find((item) => item.label === "QED")?.value ?? 0) * 10,
      color: "#40d98f",
    },
    {
      label: "Certitudine",
      value: Math.max(0, 1 - Number(typeof uncertainty?.value === "number" ? uncertainty.value : 0)) * 10,
      color: "#73a6ff",
    },
    {
      label: "Noutate",
      value: (metrics?.comparison?.novelty ?? 0) * 10,
      color: "#f59e0b",
    },
  ];
  const decisionSummary =
    selected.score >= 2.5 && (metrics?.comparison?.novelty ?? 0) >= 0.2
      ? "Candidat bun pentru shortlist. Are scor puternic si suficienta noutate fata de piata."
      : selected.score >= 1.5
        ? "Candidat interesant, dar trebuie verificat atent in raport cu costul, noutatea sau alertele."
        : "Candidat mai slab pentru prioritate imediata. Merita pastrat in biblioteca, nu in prima linie.";

  return (
    <SectionCard
      eyebrow="Dashboard stiintific"
      title="Metrici si explicatii"
      subtitle="Semnalele moleculare, scoring-ul si costul estimat raman vizibile pentru fiecare candidat selectat."
      className="h-full"
    >
      <div className="space-y-4">
        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
          <p className="ui-kicker">Concluzie rapida</p>
          <p className="mt-3 text-base font-semibold text-white">
            {selected.score >= 2.5 ? "Bun de urmarit" : selected.score >= 1.5 ? "Necesita verificare" : "Prioritate redusa"}
          </p>
          <p className="mt-2 text-sm leading-7 text-slate-300">{decisionSummary}</p>
        </div>

        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Semnal principal</p>
              <p className="mt-2 text-4xl font-semibold text-white">{typeof primaryPic50?.value === "number" ? primaryPic50.value.toFixed(2) : "--"}</p>
              <p className="mt-2 text-sm text-slate-300">pIC50 prezis pentru molecula selectata</p>
            </div>
            <ProgressRing
              value={1 - Number(typeof uncertainty?.value === "number" ? uncertainty.value : 0)}
              label="Certitudine"
              subtitle="Mai mare inseamna predictie mai stabila."
              color="#40d98f"
            />
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {(metrics?.primary ?? []).map((item) => (
              <div key={item.label} className="rounded-2xl border border-white/5 bg-white/5 p-3">
                <p className="text-[11px] uppercase tracking-[0.24em] text-slate-400">{item.label}</p>
                <p className={`mt-2 text-lg font-semibold ${toneClass(item.tone)}`}>{item.value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-3 xl:grid-cols-2">
          <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
            <p className="ui-kicker">Radar multi-obiectiv</p>
            <RadarChart labels={(metrics?.radar ?? []).map((item) => item.axis)} values={radarValues} className="mt-4 h-64 w-full" />
          </div>

          <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
            <p className="ui-kicker">Alerte si comparator</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {(metrics?.riskFlags ?? []).map((flag) => (
                <span
                  key={flag.label}
                  className={`rounded-full border px-3 py-1 text-xs ${
                    flag.tone === "success"
                      ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-100"
                      : flag.tone === "warning"
                        ? "border-amber-400/30 bg-amber-400/10 text-amber-100"
                        : "border-rose-400/30 bg-rose-400/10 text-rose-100"
                  }`}
                >
                  {flag.label}
                </span>
              ))}
            </div>

            {metrics?.comparison ? (
              <div className="mt-4 space-y-3 rounded-2xl border border-white/5 bg-white/5 p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-300">Comparator piata</span>
                  <span className="text-sm font-semibold text-white">{metrics.comparison.referenceName}</span>
                </div>
                <div className="flex items-center justify-between text-sm text-slate-300">
                  <span>Similaritate</span>
                  <span className="font-mono text-white">{metrics.comparison.similarity.toFixed(3)}</span>
                </div>
                <div className="flex items-center justify-between text-sm text-slate-300">
                  <span>Noutate</span>
                  <span className="font-mono text-white">{metrics.comparison.novelty.toFixed(3)}</span>
                </div>
                <div className="flex items-center justify-between text-sm text-slate-300">
                  <span>Suport de piata</span>
                  <span className="font-mono text-white">{metrics.comparison.marketSupport.toFixed(3)}</span>
                </div>
              </div>
            ) : null}
          </div>
        </div>

        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
          <p className="ui-kicker">Profil rapid al moleculei</p>
          <InteractiveBarChart
            data={profileChartData}
            xLabel="Criteriu"
            yLabel="Scor relativ"
            className="mt-4"
            valueFormatter={(value) => value.toFixed(1)}
          />
        </div>

        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
          <p className="ui-kicker">Componentele scorului</p>
          <InteractiveBarChart
            data={rankingChartData}
            xLabel="Componente de ranking"
            yLabel="Contributie"
            className="mt-4"
            valueFormatter={(value) => value.toFixed(2)}
          />
          <div className="mt-4 space-y-3">
            {(selected?.rankingBreakdown ?? []).map((item) => (
              <BreakdownRow key={item.label} item={item} />
            ))}
          </div>
        </div>

        <details className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-white">Detaliu cost estimat</summary>
          <InteractiveBarChart
            data={costChartData}
            xLabel="Indicator cost"
            yLabel="Valoare"
            className="mt-4"
            valueFormatter={(value) => value.toFixed(2)}
          />
          <div className="mt-4 space-y-3">
            {(selected?.costBreakdown ?? []).map((item) => (
              <BreakdownRow key={item.label} item={item} />
            ))}
          </div>
        </details>
      </div>
    </SectionCard>
  );
});
