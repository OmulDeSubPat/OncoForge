import { memo, useEffect, useMemo, useState } from "react";

import { InteractiveBarChart } from "@/components/InteractiveCharts";
import { SectionCard } from "@/components/SectionCard";
import { StatusPill } from "@/components/StatusPill";
import { ThreeDMoleculeViewer } from "@/components/ThreeDMoleculeViewer";
import type { AgentCard, FlowEdge, LibraryRow, SelectedMolecule } from "@/types";

interface DecisionWorkbenchProps {
  selected: SelectedMolecule | null;
  library: LibraryRow[];
  agents: AgentCard[];
  flows: FlowEdge[];
  compareSmiles: string[];
  uiMode: "basic" | "expert";
  onSelectCandidate: (smiles: string) => void;
  onToggleCompare: (smiles: string) => void;
  onOpenCompare: () => void;
}

type ViewerMode = "2D" | "3D";
type RenderStyle = "ball-stick" | "line";

function metricValue(selected: SelectedMolecule, label: string) {
  return selected.metrics.primary.find((item) => item.label === label)?.value;
}

function asNumber(value: number | string | undefined) {
  return typeof value === "number" ? value : Number(value ?? 0);
}

function quickVerdict(selected: SelectedMolecule) {
  if (selected.score >= 2.5 && selected.deltaPic50 >= 0 && selected.marketSimilarity <= 0.78) {
    return "Candidat puternic pentru lista prioritara";
  }
  if (selected.score >= 1.5) {
    return "Candidat bun, dar necesita validare";
  }
  return "Candidat secundar";
}

function quickNextStep(selected: SelectedMolecule) {
  if (selected.score >= 2.5) {
    return "Comparatie cu piata, verificarea semnalelor de risc si pregatire pentru exportul listei prioritare.";
  }
  if (selected.score >= 1.5) {
    return "Verifica pragurile picate si vede daca poate fi depasit de o varianta mai ieftina sau mai noua.";
  }
  return "Pastreaza molecula in biblioteca si foloseste-o ca punct de referinta pentru iteratia urmatoare.";
}

export const DecisionWorkbench = memo(function DecisionWorkbench({
  selected,
  library,
  agents,
  flows,
  compareSmiles,
  uiMode,
  onSelectCandidate,
  onToggleCompare,
  onOpenCompare,
}: DecisionWorkbenchProps) {
  const [viewerMode, setViewerMode] = useState<ViewerMode>(uiMode === "expert" ? "3D" : "2D");
  const [renderStyle, setRenderStyle] = useState<RenderStyle>("ball-stick");

  useEffect(() => {
    if (uiMode === "basic" && viewerMode === "3D") {
      setViewerMode("2D");
    }
  }, [uiMode, viewerMode]);

  const topCandidates = useMemo(() => library.slice(0, 6), [library]);
  const compareRows = useMemo(
    () =>
      compareSmiles
        .map((smiles) => library.find((item) => item.smiles === smiles))
        .filter((entry): entry is LibraryRow => Boolean(entry)),
    [compareSmiles, library],
  );

  if (!selected) {
    return (
      <SectionCard
        eyebrow="Molecula activa"
        title="Fisa unica de decizie"
        subtitle="Aceasta pagina reuneste vizualizarea, scorurile, explicatiile, comparatorul si agentii intr-un singur ecran."
      >
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_360px]">
          <div className="relative overflow-hidden rounded-[30px] border border-white/6 bg-slate-950/78 p-6">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(36,214,234,0.18),transparent_50%),radial-gradient(circle_at_bottom,rgba(64,217,143,0.08),transparent_52%)]" />
            <div className="relative flex min-h-[560px] flex-col items-center justify-center text-center">
              <div className="flex h-28 w-28 items-center justify-center rounded-full border border-forge-cyan/35 bg-forge-cyan/10 text-4xl text-cyan-100 shadow-[0_0_30px_rgba(36,214,234,0.2)]">
                AI
              </div>
              <h3 className="mt-6 text-3xl font-semibold text-white">Selecteaza o molecula din biblioteca</h3>
              <p className="mt-4 max-w-2xl text-base leading-8 text-slate-300">
                Dupa prima generatie, pagina moleculei va arata intr-un singur loc structura 2D sau 3D, verdictul rapid,
                explicatiile pro si contra, contributia agentilor si comparatorii de piata.
              </p>

              <div className="mt-8 grid w-full max-w-3xl gap-3 md:grid-cols-3">
                {[
                  "Porneste sesiunea sau actualizeaza backend-ul.",
                  "Selecteaza un candidat din topul bibliotecii sau din cautare.",
                  "Deschide comparatia pentru a-l evalua fata de moleculele de piata.",
                ].map((step, index) => (
                  <div key={step} className="rounded-2xl border border-white/6 bg-white/5 p-4 text-left">
                    <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Pas {index + 1}</p>
                    <p className="mt-2 text-sm leading-7 text-slate-300">{step}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-[28px] border border-white/6 bg-slate-950/70 p-4">
              <p className="ui-kicker">Top candidati disponibili</p>
              <div className="mt-3 space-y-2">
                {topCandidates.length ? (
                  topCandidates.map((entry) => (
                    <button
                      key={entry.id}
                      className="flex w-full items-center justify-between rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-left transition hover:bg-white/10"
                      onClick={() => onSelectCandidate(entry.smiles)}
                    >
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-white">Candidat #{entry.rank}</p>
                        <p className="mt-1 truncate text-xs text-slate-400">{entry.action || "Mutatie noua"}</p>
                      </div>
                      <span className="text-xs text-slate-400">R{entry.round}</span>
                    </button>
                  ))
                ) : (
                  <p className="text-sm leading-7 text-slate-300">Biblioteca este inca goala. Candidatii vor aparea aici dupa prima runda.</p>
                )}
              </div>
            </div>

            <div className="rounded-[28px] border border-white/6 bg-slate-950/70 p-4">
              <p className="ui-kicker">Functionalitati disponibile</p>
              <div className="mt-3 space-y-3 text-sm leading-7 text-slate-300">
                <p>Vizualizare 2D si 3D cu comutare rapida.</p>
                <p>Metrici cheie si componentele scorului, fara sa cauti in mai multe taburi.</p>
                <p>Set de comparatie persistent pentru comparatia multipla si comparatorii de piata.</p>
              </div>
            </div>
          </div>
        </div>
      </SectionCard>
    );
  }

  const selectedPic50 = asNumber(metricValue(selected, "pIC50"));
  const selectedQed = asNumber(metricValue(selected, "QED"));
  const selectedUncertainty = asNumber(metricValue(selected, "Incertitudine"));
  const selectedSa = asNumber(metricValue(selected, "SA score"));
  const noveltyBreakdown = selected.rankingBreakdown.find((item) => item.label.toLowerCase().includes("noutate"))?.value ?? 0;
  const selectedRisk = selected.admet.reactivityRisk;
  const marketNovelty = selected.metrics.comparison?.novelty ?? Math.max(0, noveltyBreakdown);
  const marketSupport = selected.metrics.comparison?.marketSupport ?? 0;
  const chartSeries = [
    {
      label: "Candidat activ",
      color: "#24d6ea",
      data: [selectedPic50, selectedQed * 10, Math.max(0, 10 - selectedSa), Math.max(0, 10 - selected.cost10mg / 40), Math.max(0, noveltyBreakdown * 20), Math.max(0, 10 - selectedRisk * 10)],
    },
  ];
  const keyMetrics = [
    { label: "pIC50", value: selectedPic50.toFixed(2), hint: "potenta" },
    { label: "Scor final", value: selected.score.toFixed(2), hint: "ranking" },
    { label: "QED", value: selectedQed.toFixed(2), hint: "calitate" },
    { label: "SA", value: selectedSa.toFixed(2), hint: "sinteza" },
    { label: "Cost 10 mg", value: `$${selected.cost10mg.toFixed(2)}`, hint: "screening" },
    { label: "Incertitudine", value: selectedUncertainty.toFixed(2), hint: "predictie" },
    { label: "Noutate", value: noveltyBreakdown.toFixed(2), hint: "piata" },
    { label: "Risc", value: selectedRisk.toFixed(2), hint: "semnal" },
  ];

  const topFlows = [...flows].sort((left, right) => right.weight - left.weight).slice(0, 4);
  const agentNames = new Map(agents.map((agent) => [agent.id, agent.name]));

  return (
    <div className="space-y-4">
      <SectionCard
        eyebrow="Molecula activa"
        title="Fisa unica de decizie"
        subtitle="Vizualizarea, explicatiile, comparatorul si agentii sunt aduse impreuna pentru decizie rapida."
        action={<StatusPill status={selected.status} label={selected.status.toUpperCase()} />}
      >
        <div className="space-y-4">
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.18fr)_380px]">
            <div className="relative overflow-hidden rounded-[30px] border border-white/6 bg-slate-950/74 p-4">
              <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(36,214,234,0.12),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(115,166,255,0.08),transparent_32%)]" />
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="ui-kicker">Candidat #{selected.rank}</p>
                  <h3 className="mt-2 text-2xl font-semibold text-white">{selected.action || "Molecula selectata"}</h3>
                  <p className="mt-2 text-sm leading-7 text-slate-300">
                    Ruta: {selected.route || "-"} | Parinte: {selected.parent || "-"} | Comparator: {selected.marketReference || "-"}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    className={`rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                      viewerMode === "2D" ? "border-forge-cyan/60 bg-forge-cyan/15 text-white" : "border-white/10 bg-white/5 text-slate-300"
                    }`}
                    onClick={() => setViewerMode("2D")}
                  >
                    2D
                  </button>
                  <button
                    className={`rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                      viewerMode === "3D" ? "border-forge-cyan/60 bg-forge-cyan/15 text-white" : "border-white/10 bg-white/5 text-slate-300"
                    }`}
                    onClick={() => setViewerMode("3D")}
                    disabled={!selected.view.molBlock}
                  >
                    3D
                  </button>
                  {uiMode === "expert" ? (
                    <>
                      <button
                        className={`rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                          renderStyle === "ball-stick" ? "border-forge-cyan/60 bg-forge-cyan/15 text-white" : "border-white/10 bg-white/5 text-slate-300"
                        }`}
                        onClick={() => setRenderStyle("ball-stick")}
                      >
                        Bile si legaturi
                      </button>
                      <button
                        className={`rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                          renderStyle === "line" ? "border-forge-cyan/60 bg-forge-cyan/15 text-white" : "border-white/10 bg-white/5 text-slate-300"
                        }`}
                        onClick={() => setRenderStyle("line")}
                      >
                        Linie
                      </button>
                    </>
                  ) : null}
                </div>
              </div>

              <div className="relative mt-4 rounded-[28px] border border-white/6 bg-slate-950/82 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                {viewerMode === "3D" ? (
                  <ThreeDMoleculeViewer
                    molBlock={selected.view.molBlock}
                    renderStyle={renderStyle}
                    heightClassName={uiMode === "basic" ? "h-[360px]" : "h-[420px]"}
                    emptyMessage="Vizualizarea 3D nu este disponibila pentru molecula selectata."
                    caption="Roteste si mareste direct in vizualizare. In modul simplificat, 2D ramane calea cea mai fluida pentru selectie rapida."
                  />
                ) : (
                  <div className={`flex ${uiMode === "basic" ? "min-h-[360px]" : "min-h-[420px]"} items-center justify-center rounded-[24px] border border-white/5 bg-slate-950/90 p-4`}>
                    {selected.view.svg2d ? (
                      <img src={selected.view.svg2d} alt={`Molecula ${selected.rank}`} className={`${uiMode === "basic" ? "max-h-[300px]" : "max-h-[360px]"} w-full object-contain motion-safe:animate-float-gentle`} />
                    ) : (
                      <p className="text-sm text-slate-400">Vizualizarea 2D nu este disponibila.</p>
                    )}
                  </div>
                )}
              </div>

              <div className="mt-4 grid gap-3 lg:grid-cols-2">
                <div className="rounded-[22px] border border-white/6 bg-white/5 p-4">
                  <p className="ui-kicker">SMILES</p>
                  <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-100">{selected.smiles}</p>
                </div>
                <div className="rounded-[22px] border border-white/6 bg-white/5 p-4">
                  <p className="ui-kicker">Linie evolutiva</p>
                  <p className="mt-2 break-all text-sm leading-7 text-slate-300">{selected.lineagePath || "-"}</p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-[30px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Verdict rapid</p>
                <p className="mt-2 text-2xl font-semibold text-white">{quickVerdict(selected)}</p>
                <p className="mt-3 text-sm leading-7 text-slate-300">{quickNextStep(selected)}</p>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  <div className="rounded-[22px] border border-white/6 bg-white/5 p-3">
                    <p className="text-xs text-slate-400">Delta pIC50</p>
                    <p className="mt-1 text-lg font-semibold tabular-nums text-white">{selected.deltaPic50 >= 0 ? "+" : ""}{selected.deltaPic50.toFixed(2)}</p>
                  </div>
                  <div className="rounded-[22px] border border-white/6 bg-white/5 p-3">
                    <p className="text-xs text-slate-400">Delta scor</p>
                    <p className="mt-1 text-lg font-semibold tabular-nums text-white">{selected.deltaScore >= 0 ? "+" : ""}{selected.deltaScore.toFixed(2)}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-[30px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="ui-kicker">Comparator de piata</p>
                    <p className="mt-1 text-sm text-slate-300">Context rapid fata de molecula de referinta.</p>
                  </div>
                  <button className="control-button px-3 py-1.5 text-xs" onClick={onOpenCompare}>
                    Deschide comparatia
                  </button>
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <div className="rounded-[22px] border border-white/8 bg-white/5 p-3">
                    <p className="ui-kicker">Comparator</p>
                    <p className="mt-2 text-lg font-semibold text-white">{selected.marketReference || "-"}</p>
                  </div>
                  <div className="rounded-[22px] border border-white/8 bg-white/5 p-3">
                    <p className="ui-kicker">Noutate fata de piata</p>
                    <p className="mt-2 text-lg font-semibold tabular-nums text-white">{marketNovelty.toFixed(2)}</p>
                  </div>
                  <div className="rounded-[22px] border border-white/8 bg-white/5 p-3">
                    <p className="ui-kicker">Suport comparativ</p>
                    <p className="mt-2 text-lg font-semibold tabular-nums text-white">{marketSupport.toFixed(2)}</p>
                  </div>
                  <div className="rounded-[22px] border border-white/8 bg-white/5 p-3">
                    <p className="ui-kicker">Set curent de comparatie</p>
                    <p className="mt-2 text-lg font-semibold text-white">{compareRows.length}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-[30px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Schimba candidatul</p>
                <div className="mt-3 space-y-2">
                  {topCandidates.map((entry) => {
                    const active = entry.smiles === selected.smiles;
                    const compared = compareSmiles.includes(entry.smiles);
                    return (
                      <div
                        key={entry.id}
                        className={`rounded-[22px] border px-4 py-3 ${active ? "border-forge-cyan/45 bg-[linear-gradient(135deg,rgba(36,214,234,0.14),rgba(12,20,36,0.9))]" : "border-white/8 bg-white/5"}`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <button className="min-w-0 flex-1 text-left" onClick={() => onSelectCandidate(entry.smiles)}>
                            <p className="text-sm font-semibold text-white">Candidat #{entry.rank}</p>
                            <p className="mt-1 break-words text-xs leading-6 text-slate-400">{entry.action || "Mutatie noua"}</p>
                          </button>
                          <button
                            className={`rounded-full border px-3 py-1 text-[11px] transition ${
                              compared ? "border-forge-cyan/45 bg-forge-cyan/10 text-white" : "border-white/10 bg-slate-950/80 text-slate-300"
                            }`}
                            onClick={() => onToggleCompare(entry.smiles)}
                          >
                            {compared ? "In set" : "Compara"}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_420px]">
            <div className="space-y-4">
              <div className="rounded-[30px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Metrici cheie</p>
                <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  {keyMetrics.map((item) => (
                    <div key={item.label} className="rounded-[22px] border border-white/6 bg-white/5 p-3">
                      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-400">{item.label}</p>
                      <p className="mt-2 text-lg font-semibold tabular-nums text-white">{item.value}</p>
                      <p className="mt-1 text-xs text-slate-500">{item.hint}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[30px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Profil relativ al candidatului</p>
                <InteractiveBarChart
                  categories={["Potenta", "QED", "Sinteza", "Cost", "Noutate", "Risc invers"]}
                  xLabel="Criteriu"
                  yLabel="Scor relativ"
                  className="mt-4 h-[360px]"
                  formatValue={(value) => value.toFixed(1)}
                  series={chartSeries}
                />
              </div>

              <div className="grid gap-4 xl:grid-cols-2">
                <div className="rounded-[30px] border border-emerald-400/18 bg-emerald-400/8 p-4">
                  <p className="ui-kicker text-emerald-100/90">Semnale pro</p>
                  <div className="mt-3 space-y-3">
                    {selected.explainability.pros.slice(0, 3).map((item) => (
                      <div key={item} className="rounded-2xl border border-emerald-300/20 bg-black/10 p-3 text-sm leading-7 text-emerald-50">
                        {item}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-[30px] border border-amber-400/18 bg-amber-400/8 p-4">
                  <p className="ui-kicker text-amber-100/90">Semnale contra</p>
                  <div className="mt-3 space-y-3">
                    {selected.explainability.cons.slice(0, 3).map((item) => (
                      <div key={item} className="rounded-2xl border border-amber-300/20 bg-black/10 p-3 text-sm leading-7 text-amber-50">
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-[30px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Praguri trecute si picate</p>
                <div className="mt-3 space-y-2">
                  {selected.explainability.thresholds.map((item) => (
                    <div
                      key={item.label}
                      className={`rounded-2xl border px-4 py-3 ${
                        item.passed ? "border-emerald-400/18 bg-emerald-400/8" : "border-amber-400/18 bg-amber-400/8"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-semibold text-white">{item.label}</p>
                        <span className="text-xs text-slate-100">{item.passed ? "Trecut" : "Picat"}</span>
                      </div>
                      <p className="mt-2 text-xs leading-6 text-slate-300">
                        Valoare: {item.value} | Referinta: {item.reference}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[30px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Agenti si contributie</p>
                <div className="mt-3 space-y-3">
                  {selected.agentContributions.map((agent) => (
                    <div key={agent.id} className="rounded-2xl border border-white/6 bg-white/5 p-3">
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm font-semibold text-white">{agent.name}</span>
                        <StatusPill status={agent.status} label={`${Math.round(agent.contribution * 100)}%`} />
                      </div>
                      <p className="mt-2 text-xs leading-6 text-slate-400">{agent.lastAction}</p>
                      <div className="mt-3 h-2 rounded-full bg-slate-950/90">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green"
                          style={{ width: `${Math.max(8, agent.contribution * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[30px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <p className="ui-kicker">Flux dominant intre agenti</p>
                <div className="mt-3 space-y-3">
                  {topFlows.map((flow) => (
                    <div key={`${flow.source}-${flow.target}`} className="rounded-2xl border border-white/6 bg-white/5 p-3">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-semibold text-white">
                          {agentNames.get(flow.source) ?? flow.source}
                          {" -> "}
                          {agentNames.get(flow.target) ?? flow.target}
                        </p>
                        <span className="text-xs text-slate-400">{Math.round(flow.weight * 100)}%</span>
                      </div>
                      <div className="mt-3 h-2 rounded-full bg-slate-950/90">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green"
                          style={{ width: `${Math.max(8, flow.weight * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </SectionCard>
    </div>
  );
});
