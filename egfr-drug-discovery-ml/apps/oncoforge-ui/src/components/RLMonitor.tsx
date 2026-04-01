import { memo } from "react";

import { InteractiveBarChart, InteractiveLineChart, ProgressRing } from "@/components/Charts";
import { SectionCard } from "@/components/SectionCard";
import type { RLMonitorPayload, SelectedMolecule } from "@/types";

interface RLMonitorProps {
  selected: SelectedMolecule | null;
  monitor: RLMonitorPayload;
}

function PenaltyRow({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-300">{label}</span>
        <span className={`text-sm font-semibold ${tone}`}>{value.toFixed(3)}</span>
      </div>
      <div className="mt-2 h-2 rounded-full bg-slate-900">
        <div
          className="h-2 rounded-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green"
          style={{ width: `${Math.min(100, Math.max(2, value * 100))}%` }}
        />
      </div>
    </div>
  );
}

export const RLMonitor = memo(function RLMonitor({ selected, monitor }: RLMonitorProps) {
  const latestPenalty = monitor.penaltySeries[monitor.penaltySeries.length - 1];
  const verifiedReward = latestPenalty?.verifiedReward ?? 0;
  const rewardChartSeries = [
    {
      id: "best-score",
      label: "Scor maxim",
      color: "#24d6ea",
      data: monitor.rewardSeries.map((item) => ({
        x: item.round,
        y: item.bestScore,
        label: `Runda ${item.round}`,
      })),
    },
    {
      id: "verified-reward",
      label: "Recompensa verificata",
      color: "#40d98f",
      data: monitor.penaltySeries.map((item) => ({
        x: item.round,
        y: item.verifiedReward,
        label: `Runda ${item.round}`,
      })),
    },
  ];
  const policyChartSeries = [
    {
      id: "exploration",
      label: "Explorare",
      color: "#73a6ff",
      data: monitor.penaltySeries.map((item) => ({
        x: item.round,
        y: item.exploration,
        label: `Runda ${item.round}`,
      })),
    },
    {
      id: "exploitation",
      label: "Exploatare",
      color: "#fbbf24",
      data: monitor.penaltySeries.map((item) => ({
        x: item.round,
        y: item.exploitation,
        label: `Runda ${item.round}`,
      })),
    },
  ];
  const penaltyBars = latestPenalty
    ? [
        { label: "Toxicitate", value: latestPenalty.toxicityPenalty, color: "#f87171" },
        { label: "Invaliditate", value: latestPenalty.invalidPenalty, color: "#fb923c" },
        { label: "Incertitudine", value: latestPenalty.uncertaintyPenalty, color: "#fbbf24" },
        { label: "Recompensa", value: latestPenalty.rewardRiskPenalty, color: "#ef4444" },
      ]
    : [];

  if (!monitor.rewardSeries.length && !monitor.penaltySeries.length) {
    return (
      <SectionCard
        eyebrow="Monitor RLVR"
        title="Recompensa verificabila"
        subtitle="Curbele RL vor aparea dupa prima runda procesata de worker."
        className="h-full"
      >
        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-6 text-sm leading-8 text-slate-300">
          Panoul RL va afisa evolutia recompensei, penalizarile si balanta explorare versus exploatare dupa ce OncoSynth genereaza primele rezultate.
        </div>
      </SectionCard>
    );
  }

  return (
    <SectionCard
      eyebrow="Monitor RLVR"
      title="Recompensa verificabila"
      subtitle="Recompensa nu este opaca: penalizarile, explorarea si promotiile raman explicite in timp."
      className="h-full"
    >
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-[auto_1fr]">
          <ProgressRing
            value={Math.min(1, Math.max(0, verifiedReward / 10))}
            label="Recompensa verificata"
            subtitle="Medie pe ultima runda procesata."
            color="#40d98f"
          />

          <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
            <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Molecula curenta</p>
            <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-300">{selected?.smiles ?? "Nu exista selectie activa."}</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                <p className="text-xs text-slate-400">Explorare</p>
                <p className="mt-1 text-2xl font-semibold text-white">{latestPenalty?.exploration?.toFixed(2) ?? "--"}</p>
              </div>
              <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                <p className="text-xs text-slate-400">Exploatare</p>
                <p className="mt-1 text-2xl font-semibold text-white">{latestPenalty?.exploitation?.toFixed(2) ?? "--"}</p>
              </div>
            </div>
          </div>
        </div>

        <InteractiveLineChart
          series={rewardChartSeries}
          xLabel="Runda"
          yLabel="Scor / recompensa"
          valueFormatter={(value) => value.toFixed(2)}
        />

        <InteractiveLineChart
          series={policyChartSeries}
          xLabel="Runda"
          yLabel="Balanta politica"
          valueFormatter={(value) => value.toFixed(2)}
        />

        {latestPenalty ? (
          <>
            <InteractiveBarChart
              data={penaltyBars}
              xLabel="Tip penalizare"
              yLabel="Magnitudine"
              valueFormatter={(value) => value.toFixed(3)}
            />
            <div className="grid gap-3 md:grid-cols-2">
              <PenaltyRow label="Penalizare toxicitate" value={latestPenalty.toxicityPenalty} tone="text-rose-300" />
              <PenaltyRow label="Penalizare invaliditate" value={latestPenalty.invalidPenalty} tone="text-amber-300" />
              <PenaltyRow label="Penalizare incertitudine" value={latestPenalty.uncertaintyPenalty} tone="text-amber-300" />
              <PenaltyRow label="Penalizare risc recompensa" value={latestPenalty.rewardRiskPenalty} tone="text-rose-300" />
            </div>
          </>
        ) : (
          <div className="rounded-2xl border border-white/5 bg-white/5 p-4 text-sm text-slate-300">
            Penalizarile vor aparea dupa prima runda procesata.
          </div>
        )}

        <div className="rounded-3xl border border-forge-green/20 bg-forge-green/10 p-4 text-sm leading-6 text-green-50">
          {monitor.verifiableNotes.map((note, index) => (
            <p key={index} className={index === 0 ? "" : "mt-2"}>
              {note}
            </p>
          ))}
        </div>
      </div>
    </SectionCard>
  );
});
