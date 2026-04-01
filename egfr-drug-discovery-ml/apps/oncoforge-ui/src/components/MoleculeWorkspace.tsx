import { memo, useEffect, useRef, useState } from "react";

import { SectionCard } from "@/components/SectionCard";
import { StatusPill } from "@/components/StatusPill";
import type { LibraryRow, SelectedMolecule, ViewMode } from "@/types";

interface MoleculeWorkspaceProps {
  selected: SelectedMolecule | null;
  library: LibraryRow[];
  onSelectCandidate: (smiles: string) => void;
}

type RenderStyle = "ball-stick" | "line";

export const MoleculeWorkspace = memo(function MoleculeWorkspace({ selected, library, onSelectCandidate }: MoleculeWorkspaceProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("3D");
  const [renderStyle, setRenderStyle] = useState<RenderStyle>("ball-stick");
  const viewerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (viewMode !== "3D" || !selected?.view.molBlock || !viewerRef.current) {
      return;
    }

    let cancelled = false;
    const mount = viewerRef.current;
    const activeSelected = selected;

    async function render3D() {
      const $3Dmol = await import("3dmol");
      if (cancelled || !mount) {
        return;
      }

      mount.innerHTML = "";
      const viewer = $3Dmol.createViewer(mount, { backgroundColor: "#06111d" });
      viewer.clear();
      viewer.addModel(activeSelected.view.molBlock, "mol");
      if (renderStyle === "ball-stick") {
        viewer.setStyle({}, { stick: { radius: 0.18, colorscheme: "cyanCarbon" }, sphere: { scale: 0.28 } });
      } else {
        viewer.setStyle({}, { line: { colorscheme: "cyanCarbon", linewidth: 2.5 } });
      }
      viewer.zoomTo();
      viewer.render();
    }

    void render3D();

    return () => {
      cancelled = true;
      if (mount) {
        mount.innerHTML = "";
      }
    };
  }, [renderStyle, selected?.smiles, selected?.view.molBlock, viewMode]);

  const topCandidates = library.slice(0, 5);
  const decisionLabel = !selected
    ? "Asteapta selectie"
    : selected.score >= 2.5 && selected.deltaPic50 >= 0
      ? "Bun de urmarit"
      : selected.score >= 1.5
        ? "Necesita verificare"
        : "Prioritate mica";
  const mainRisk = !selected
    ? "-"
    : selected.marketSimilarity >= 0.8
      ? "Foarte apropiata de piata"
      : selected.cost100mg >= 1000
        ? "Cost de sinteza ridicat"
        : selected.deltaPic50 < 0
          ? "Nu depaseste clar parintele"
          : "Fara risc dominant";
  const nextStep = !selected
    ? "-"
    : decisionLabel === "Bun de urmarit"
      ? "Trimite molecula in shortlist si export."
      : decisionLabel === "Necesita verificare"
        ? "Verifica metricile si compara cu alte 2-3 lead-uri."
        : "Tine-o in biblioteca, dar nu o prioritiza acum.";

  if (!selected) {
    return (
      <SectionCard
        eyebrow="Workspace molecular"
        title="Vizualizare si selectie"
        subtitle="Cand nu exista inca o molecula selectata, workspace-ul ramane curat si orientat spre pasul urmator."
        className="h-full"
      >
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_360px]">
          <div className="relative overflow-hidden rounded-3xl border border-white/6 bg-slate-950/80 p-6 shadow-glow">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(36,214,234,0.16),transparent_50%),radial-gradient(circle_at_bottom,rgba(64,217,143,0.08),transparent_52%)]" />
            <div className="relative flex min-h-[520px] flex-col items-center justify-center text-center">
              <div className="flex h-28 w-28 items-center justify-center rounded-full border border-forge-cyan/30 bg-forge-cyan/10 text-4xl text-cyan-100 shadow-[0_0_30px_rgba(36,214,234,0.2)]">
                3D
              </div>
              <h3 className="mt-6 text-3xl font-semibold text-white">Initializati o sesiune sau selectati o molecula</h3>
              <p className="mt-4 max-w-2xl text-base leading-8 text-slate-300">
                OncoSynth va afisa aici vizualizarea 2D si 3D, contextul de optimizare, comparatorul de piata si contributiile agentilor pentru molecula selectata.
              </p>

              <div className="mt-8 grid w-full max-w-3xl gap-3 md:grid-cols-3">
                <div className="rounded-2xl border border-white/6 bg-white/5 p-4 text-left">
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-400">1. Pornire</p>
                  <p className="mt-2 text-sm leading-7 text-slate-300">Apasa `Porneste generarea` in bara de control de sus.</p>
                </div>
                <div className="rounded-2xl border border-white/6 bg-white/5 p-4 text-left">
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-400">2. Selectie</p>
                  <p className="mt-2 text-sm leading-7 text-slate-300">Selectati un candidat din biblioteca dupa generarea primului lot.</p>
                </div>
                <div className="rounded-2xl border border-white/6 bg-white/5 p-4 text-left">
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-400">3. Analiza</p>
                  <p className="mt-2 text-sm leading-7 text-slate-300">Exploreaza structura, scorul, costul si traiectoria sa in timp.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Top candidati disponibili</p>
              {topCandidates.length ? (
                <div className="mt-3 space-y-2">
                  {topCandidates.map((entry) => (
                    <button
                      key={entry.id}
                      className="flex w-full items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-left text-slate-200 transition hover:bg-white/10"
                      onClick={() => onSelectCandidate(entry.smiles)}
                    >
                      <span>
                        <span className="block text-sm font-semibold">Rang #{entry.rank}</span>
                        <span className="mt-1 block text-xs text-slate-400">{entry.action || "Molecula noua"}</span>
                      </span>
                      <span className="text-xs text-slate-400">R{entry.round}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-sm leading-7 text-slate-300">Biblioteca este inca goala. Dupa prima generatie, candidatii vor aparea automat aici.</p>
              )}
            </div>

            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Ce va aparea aici</p>
              <div className="mt-3 space-y-3 text-sm text-slate-300">
                <p>Viewer 3D interactiv cu rotire si zoom.</p>
                <p>Structura 2D si context decizional pentru ranking.</p>
                <p>Contributii explicite ale agentilor si comparatie cu piata.</p>
              </div>
            </div>
          </div>
        </div>
      </SectionCard>
    );
  }

  return (
    <SectionCard
      eyebrow="Workspace molecular"
      title="Vizualizare si selectie"
      subtitle="Molecula selectata poate fi inspectata in 3D sau 2D, iar contextul de optimizare ramane langa ea."
      action={selected ? <StatusPill status={selected.status} label={selected.status.toUpperCase()} /> : undefined}
      className="h-full"
    >
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          {(["3D", "2D", "compare"] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              className={`rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                viewMode === mode
                  ? "border-forge-cyan/60 bg-forge-cyan/15 text-white"
                  : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
              }`}
              onClick={() => setViewMode(mode)}
            >
              {mode}
            </button>
          ))}

          <div className="ml-auto flex flex-wrap items-center gap-2">
            <button
              className={`rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                renderStyle === "ball-stick"
                  ? "border-forge-cyan/60 bg-forge-cyan/15 text-white"
                  : "border-white/10 bg-white/5 text-slate-300"
              }`}
              onClick={() => setRenderStyle("ball-stick")}
            >
              Ball-stick
            </button>
            <button
              className={`rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                renderStyle === "line"
                  ? "border-forge-cyan/60 bg-forge-cyan/15 text-white"
                  : "border-white/10 bg-white/5 text-slate-300"
              }`}
              onClick={() => setRenderStyle("line")}
            >
              Linie
            </button>
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_360px]">
          <div className="relative overflow-hidden rounded-3xl border border-white/6 bg-slate-950/80 p-4 shadow-glow">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(36,214,234,0.16),transparent_50%),radial-gradient(circle_at_bottom,rgba(64,217,143,0.08),transparent_52%)]" />
            <div className="absolute inset-0 bg-grid-fine opacity-15" />

            <div className="relative">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="ui-kicker">Molecula activa</p>
                  <h3 className="mt-2 text-2xl font-semibold text-white">{selected ? `Molecula #${selected.rank}` : "Nicio molecula selectata"}</h3>
                  <p className="mt-2 max-w-2xl text-sm leading-7 text-slate-300">
                    {selected ? `Transformare: ${selected.action || "-"} | Ruta: ${selected.route || "-"}` : "Initializati o sesiune sau selectati o molecula din biblioteca."}
                  </p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-white/6 px-4 py-3 text-right">
                  <p className="ui-kicker">Comparator piata</p>
                  <p className="mt-1 text-sm font-semibold text-white">{selected?.marketReference || "-"}</p>
                  <p className="mt-1 text-xs text-slate-400">Cost 10 mg: ${selected?.cost10mg?.toFixed(2) ?? "--"}</p>
                </div>
              </div>

              <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_240px]">
                <div className="relative rounded-3xl border border-white/6 bg-slate-950/50 p-4">
                  {viewMode === "3D" ? (
                    <div className="space-y-3">
                      <div ref={viewerRef} className="h-[410px] w-full rounded-2xl border border-white/5 bg-slate-950/80" />
                      <p className="text-xs leading-6 text-slate-400">Roteste si da zoom direct in viewer. Randarea este generata din MolBlock-ul calculat in backend.</p>
                    </div>
                  ) : viewMode === "2D" ? (
                    <div className="flex min-h-[410px] items-center justify-center rounded-2xl border border-white/5 bg-slate-950/80 p-4">
                      {selected?.view.svg2d ? <img src={selected.view.svg2d} alt="Molecula 2D" className="max-h-[360px] w-full object-contain" /> : <p className="text-slate-400">Vizualizarea 2D va aparea aici.</p>}
                    </div>
                  ) : (
                    <div className="grid min-h-[410px] gap-4 md:grid-cols-2">
                      <div className="rounded-2xl border border-white/5 bg-slate-950/80 p-4">
                        <p className="ui-kicker">Structura 2D</p>
                        <div className="mt-4 flex h-[320px] items-center justify-center">
                          {selected?.view.svg2d ? <img src={selected.view.svg2d} alt="Comparatie 2D" className="max-h-[300px] w-full object-contain" /> : null}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-white/5 bg-slate-950/80 p-4">
                        <p className="ui-kicker">Context decizional</p>
                        <div className="mt-4 space-y-3 text-sm text-slate-300">
                          <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                            <span className="text-slate-400">Linie evolutiva</span>
                            <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-200">{selected?.lineagePath || "-"}</p>
                          </div>
                          <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                            <span className="text-slate-400">Formula / atomi</span>
                            <p className="mt-2 text-white">{selected?.view.formula || "-"} | {selected?.view.atomCount ?? 0} atomi</p>
                          </div>
                          <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                            <span className="text-slate-400">Delta fata de parinte</span>
                            <p className="mt-2 text-white">pIC50 {selected?.deltaPic50?.toFixed(2) ?? "--"} | Scor {selected?.deltaScore?.toFixed(2) ?? "--"}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-3">
                  <div className="rounded-2xl border border-white/5 bg-white/5 p-4">
                    <p className="ui-kicker">SMILES</p>
                    <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-100">{selected?.smiles || "-"}</p>
                  </div>
                  <div className="rounded-2xl border border-white/5 bg-white/5 p-4">
                    <p className="ui-kicker">Parinte</p>
                    <p className="mt-2 break-all text-sm leading-6 text-slate-300">{selected?.parent || "-"}</p>
                  </div>
                  <div className="rounded-2xl border border-white/5 bg-white/5 p-4">
                    <p className="ui-kicker">Contributii agenti</p>
                    <div className="mt-3 space-y-2">
                      {(selected?.agentContributions ?? []).map((agent) => (
                        <div key={agent.id}>
                          <div className="flex items-center justify-between text-xs text-slate-300">
                            <span>{agent.name}</span>
                            <span>{Math.round(agent.contribution * 100)}%</span>
                          </div>
                          <div className="mt-1 h-2 rounded-full bg-slate-900">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green"
                              style={{ width: `${Math.max(8, agent.contribution * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="ui-kicker">Rezumat pentru chimist</p>
              <div className="mt-3 space-y-3">
                <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                  <span className="text-sm text-slate-300">Verdict rapid</span>
                  <p className="mt-2 text-lg font-semibold text-white">{decisionLabel}</p>
                </div>
                <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                  <span className="text-sm text-slate-300">Risc principal</span>
                  <p className="mt-2 text-sm leading-7 text-white">{mainRisk}</p>
                </div>
                <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                  <span className="text-sm text-slate-300">Pasul urmator</span>
                  <p className="mt-2 text-sm leading-7 text-white">{nextStep}</p>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="ui-kicker">Schimba molecula</p>
              <div className="mt-3 space-y-2">
                {topCandidates.map((entry) => (
                  <button
                    key={entry.id}
                    className={`flex w-full items-center justify-between rounded-2xl border px-3 py-3 text-left transition ${
                      entry.smiles === selected?.smiles
                        ? "border-forge-cyan/50 bg-forge-cyan/12 text-white"
                        : "border-white/10 bg-white/5 text-slate-200 hover:bg-white/10"
                    }`}
                    onClick={() => onSelectCandidate(entry.smiles)}
                  >
                    <span>
                      <span className="block text-sm font-semibold">Rang #{entry.rank}</span>
                      <span className="mt-1 block break-words text-xs leading-6 text-slate-400">{entry.action || "Mutatie noua"}</span>
                    </span>
                    <span className="text-right text-xs text-slate-400">R{entry.round}<br />pIC50 {entry.pic50.toFixed(2)}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="ui-kicker">Indicatori rapizi</p>
              <div className="mt-3 space-y-3">
                <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-200">Scor final</span>
                    <span className="text-sm font-semibold text-white">{selected?.score.toFixed(2) ?? "--"}</span>
                  </div>
                </div>
                <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-200">Cost 100 mg</span>
                    <span className="text-sm font-semibold text-white">${selected?.cost100mg.toFixed(2) ?? "--"}</span>
                  </div>
                </div>
                <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-200">Similaritate piata</span>
                    <span className="text-sm font-semibold text-white">{selected?.marketSimilarity.toFixed(3) ?? "--"}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </SectionCard>
  );
});
