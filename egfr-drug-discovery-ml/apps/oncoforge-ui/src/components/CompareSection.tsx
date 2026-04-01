import { useEffect, useMemo, useState } from "react";

import { fetchMoleculeDetail } from "@/api";
import { InteractiveBarChart } from "@/components/InteractiveCharts";
import { SectionCard } from "@/components/SectionCard";
import type { LibraryRow, MarketComparePayload, SelectedMolecule } from "@/types";

interface CompareSectionProps {
  sessionName: string;
  selected: SelectedMolecule | null;
  marketCompare: MarketComparePayload;
  library: LibraryRow[];
  compareSmiles: string[];
  onSelectCandidate: (smiles: string) => void;
  onCompareSmilesChange: (smiles: string[]) => void;
}

function deltaTone(delta: number) {
  if (delta > 0) {
    return "text-emerald-300";
  }
  if (delta < 0) {
    return "text-rose-300";
  }
  return "text-slate-300";
}

export function CompareSection({
  sessionName,
  selected,
  marketCompare,
  library,
  compareSmiles,
  onSelectCandidate,
  onCompareSmilesChange,
}: CompareSectionProps) {
  const [batchDetails, setBatchDetails] = useState<Record<string, SelectedMolecule>>({});
  const topSelectable = useMemo(() => library.slice(0, 16), [library]);

  useEffect(() => {
    if (!selected) {
      return;
    }
    const next = Array.from(new Set([selected.smiles, ...compareSmiles])).slice(0, 4);
    if (next.join("|") !== compareSmiles.join("|")) {
      onCompareSmilesChange(next);
    }
  }, [compareSmiles, onCompareSmilesChange, selected?.smiles]);

  useEffect(() => {
    let cancelled = false;
    async function loadBatch() {
      const missingSmiles = compareSmiles.filter((smiles) => !batchDetails[smiles]);
      if (!missingSmiles.length) {
        return;
      }
      const entries = await Promise.all(
        missingSmiles.map(async (smiles) => {
          const payload = await fetchMoleculeDetail({ sessionName, smiles });
          return [smiles, payload.selected] as const;
        }),
      );
      if (cancelled) {
        return;
      }
      setBatchDetails((current) => {
        const next = { ...current };
        for (const [smiles, detail] of entries) {
          if (detail) {
            next[smiles] = detail;
          }
        }
        return next;
      });
    }

    if (compareSmiles.length) {
      void loadBatch();
    }

    return () => {
      cancelled = true;
    };
  }, [batchDetails, compareSmiles, sessionName]);

  const compareEntries = marketCompare.entries;
  const batchCards = compareSmiles.map((smiles) => batchDetails[smiles]).filter(Boolean);

  return (
    <div className="space-y-4">
      <SectionCard
        eyebrow="Comparator direct"
        title="Candidat versus molecule de pe piata"
        subtitle="Aceleasi axe pentru candidatul selectat si comparatorii principali. Barele sunt normalizate pentru lectura rapida."
      >
        <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_360px]">
          <InteractiveBarChart
            categories={marketCompare.axes}
            xLabel="Axa comparativa"
            yLabel="Scor normalizat"
            className="h-[380px]"
            formatValue={(value) => value.toFixed(2)}
            series={compareEntries.map((entry, index) => ({
              label: entry.name,
              color: ["#24d6ea", "#40d98f", "#73a6ff", "#f59e0b"][index % 4],
              data: [
                entry.normalized.potency,
                entry.normalized.qed,
                entry.normalized.sa,
                entry.normalized.cost,
                entry.normalized.novelty,
                entry.normalized.risk,
              ],
            }))}
          />

          <div className="space-y-3">
            {compareEntries.map((entry) => (
              <div key={entry.id} className="rounded-2xl border border-white/6 bg-slate-950/70 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-white">{entry.name}</p>
                    <p className="mt-1 text-xs text-slate-400">{entry.referenceClass}</p>
                  </div>
                  <span className={`rounded-full border px-3 py-1 text-xs ${entry.kind === "selectata" ? "border-cyan-400/40 bg-cyan-400/10 text-cyan-50" : "border-white/10 bg-white/5 text-slate-200"}`}>
                    {entry.kind === "selectata" ? "Candidat" : "Piata"}
                  </span>
                </div>
                <div className="mt-3 grid gap-2 text-sm text-slate-300 sm:grid-cols-2">
                  <p>pIC50: <span className="font-semibold text-white">{entry.raw.potency.toFixed(2)}</span></p>
                  <p>QED: <span className="font-semibold text-white">{entry.raw.qed.toFixed(2)}</span></p>
                  <p>SA: <span className="font-semibold text-white">{entry.raw.sa.toFixed(2)}</span></p>
                  <p>Cost 10 mg: <span className="font-semibold text-white">${entry.raw.cost.toFixed(2)}</span></p>
                  <p>Noutate: <span className="font-semibold text-white">{entry.raw.novelty.toFixed(2)}</span></p>
                  <p>Risc: <span className="font-semibold text-white">{entry.raw.risk.toFixed(2)}</span></p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </SectionCard>

      <SectionCard
        eyebrow="Comparatie multipla"
        title="Compara 2-4 molecule in paralel"
        subtitle="Selecteaza candidati din lotul curent. Vei vedea structura 2D, proprietatile cheie si de ce una este peste alta."
      >
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {topSelectable.map((item) => {
              const checked = compareSmiles.includes(item.smiles);
              return (
                <button
                  key={item.id}
                  className={`rounded-full border px-3 py-1.5 text-xs transition ${
                    checked
                      ? "border-forge-cyan/60 bg-forge-cyan/15 text-white"
                      : "border-white/10 bg-white/5 text-slate-300"
                  }`}
                  onClick={() =>
                    onCompareSmilesChange(((current) => {
                      if (current.includes(item.smiles)) {
                        return current.filter((smiles) => smiles !== item.smiles);
                      }
                      if (current.length >= 4) {
                        return [...current.slice(1), item.smiles];
                      }
                      return [...current, item.smiles];
                    })(compareSmiles))
                  }
                >
                  #{item.rank} | R{item.round}
                </button>
              );
            })}
          </div>

          <div className="grid gap-4 xl:grid-cols-2 2xl:grid-cols-4">
            {batchCards.map((detail) => {
              const detailPic50 = Number(detail.metrics.primary.find((item) => item.label === "pIC50")?.value ?? 0);
              const deltaVsMarket = selected ? detail.score - selected.score : 0;
              const reason =
                detailPic50 > (((selected?.metrics.primary.find((item) => item.label === "pIC50" && typeof item.value === "number")?.value as number | undefined) ?? 0))
                  ? "Mai puternica pe potenta."
                  : detail.cost10mg < (selected?.cost10mg ?? Number.POSITIVE_INFINITY)
                    ? "Mai ieftina la screening."
                    : (detail.rankingBreakdown.find((item) => item.label.includes("Noutate"))?.value ?? 0) > 0.15
                      ? "Mai diferita fata de piata."
                      : "Pare apropiata de candidatul principal.";
              return (
                <div key={detail.smiles} className="rounded-3xl border border-white/6 bg-slate-950/70 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-white">#{detail.rank}</p>
                      <p className="mt-1 text-xs text-slate-400">{detail.action || "Mutatie"}</p>
                    </div>
                    <button className="control-button px-3 py-1.5 text-xs" onClick={() => onSelectCandidate(detail.smiles)}>
                      Deschide
                    </button>
                  </div>
                  <div className="mt-4 flex h-[220px] items-center justify-center rounded-2xl border border-white/6 bg-slate-950/80 p-3">
                    {detail.view.svg2d ? (
                      <img src={detail.view.svg2d} alt={`Molecula ${detail.rank}`} className="max-h-[190px] w-full object-contain" />
                    ) : (
                      <p className="text-sm text-slate-400">Structura 2D nu este disponibila.</p>
                    )}
                  </div>
                  <div className="mt-4 space-y-2 text-sm text-slate-300">
                    <p>Scor: <span className="font-semibold text-white">{detail.score.toFixed(2)}</span></p>
                    <p>pIC50: <span className="font-semibold text-white">{detail.metrics.primary.find((item) => item.label === "pIC50")?.value}</span></p>
                    <p>QED: <span className="font-semibold text-white">{detail.metrics.primary.find((item) => item.label === "QED")?.value}</span></p>
                    <p>Cost 10 mg: <span className="font-semibold text-white">${detail.cost10mg.toFixed(2)}</span></p>
                    <p>Comparator: <span className="font-semibold text-white">{detail.marketReference || "-"}</span></p>
                    <p className={deltaTone(deltaVsMarket)}>
                      Delta vs molecula activa: {deltaVsMarket >= 0 ? "+" : ""}
                      {deltaVsMarket.toFixed(2)}
                    </p>
                  </div>
                  <div className="mt-4 rounded-2xl border border-white/6 bg-white/5 p-3 text-sm leading-7 text-slate-300">
                    {reason}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
