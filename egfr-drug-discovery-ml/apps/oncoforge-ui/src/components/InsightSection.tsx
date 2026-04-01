import type { ChangeEvent } from "react";

import { SectionCard } from "@/components/SectionCard";
import type { ChemistNotebookEntry, DecisionEvent, SelectedMolecule } from "@/types";

interface InsightSectionProps {
  selected: SelectedMolecule | null;
  notebookEntry: ChemistNotebookEntry | null;
  localHistory: DecisionEvent[];
  onNotebookChange: (entry: ChemistNotebookEntry) => void;
}

const verdictOptions: ChemistNotebookEntry["verdict"][] = [
  "merita sinteza",
  "de revazut",
  "prea scumpa",
  "schelet interesant",
];

const quickTags = ["legatura hinge buna", "cost mic", "risc toxicitate", "noutate", "candidat principal", "rezerva"];
const reviewWorkflowTags = [
  { label: "Fixeaza", tag: "pinat" },
  { label: "Aproba", tag: "aprobat" },
  { label: "Respinge", tag: "respins_local" },
  { label: "Retesteaza", tag: "retestare" },
] as const;

export function InsightSection({
  selected,
  notebookEntry,
  localHistory,
  onNotebookChange,
}: InsightSectionProps) {
  if (!selected) {
    return (
      <SectionCard
        eyebrow="Explicatii si risc"
        title="Fundamentarea prioritizarii"
        subtitle="Selecteaza mai intai o molecula pentru a vedea argumentele pro si contra, semnalele de risc si istoricul de decizie."
      >
        <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-6 text-sm leading-8 text-slate-300">
          Sectiunea aceasta este gandita pentru lectura rapida de chimist: trei motive pro, trei motive contra, praguri trecute sau picate, semnale de risc si note de laborator.
        </div>
      </SectionCard>
    );
  }

  const activeSelected = selected;
  const mergedHistory = [...selected.decisionHistory, ...localHistory].sort((left, right) => {
    const leftRound = left.round ?? 0;
    const rightRound = right.round ?? 0;
    return leftRound - rightRound;
  });
  const noteState =
    notebookEntry ??
    ({
      smiles: selected.smiles,
      verdict: "de revazut",
      tags: [],
      note: "",
      updatedAt: "",
    } satisfies ChemistNotebookEntry);

  function updateNote(patch: Partial<ChemistNotebookEntry>) {
    onNotebookChange({
      ...noteState,
      ...patch,
      smiles: activeSelected.smiles,
      updatedAt: new Date().toISOString(),
    });
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_420px]">
        <SectionCard
          eyebrow="Explicabilitate"
          title="De ce a urcat aceasta molecula"
          subtitle="Motive pro/contra, agentul dominant si penalizarile explicite din scor."
        >
          <div className="grid gap-4 xl:grid-cols-2">
            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="ui-kicker">3 motive pro</p>
              <div className="mt-3 space-y-3">
                {selected.explainability.pros.map((item) => (
                  <div key={item} className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-3 text-sm leading-7 text-emerald-50">
                    {item}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="ui-kicker">3 motive contra</p>
              <div className="mt-3 space-y-3">
                {selected.explainability.cons.map((item) => (
                  <div key={item} className="rounded-2xl border border-amber-400/20 bg-amber-400/10 p-3 text-sm leading-7 text-amber-50">
                    {item}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-4 xl:grid-cols-[280px_minmax(0,1fr)]">
            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="ui-kicker">Agent dominant</p>
              <p className="mt-2 text-xl font-semibold text-white">{selected.explainability.dominantAgent}</p>
              <p className="mt-3 text-sm leading-7 text-slate-300">{selected.explainability.summary}</p>
            </div>

            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="ui-kicker">Penalizari aplicate</p>
              {selected.explainability.penalties.length ? (
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  {selected.explainability.penalties.map((item) => (
                    <div key={item.label} className="rounded-2xl border border-rose-400/20 bg-rose-400/10 p-3">
                      <p className="text-sm text-rose-50">{item.label}</p>
                      <p className="mt-2 text-lg font-semibold text-white">-{Math.abs(item.value).toFixed(3)}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-300">Nu exista penalizari importante pentru molecula selectata.</p>
              )}
            </div>
          </div>
        </SectionCard>

        <SectionCard
          eyebrow="Carnetul chimistului"
          title="Note rapide pe molecula"
          subtitle="Salveaza verdictul, tag-urile si observatiile de laborator pentru lista prioritara."
        >
          <div className="space-y-4">
            <label className="field-card">
              <span>Verdict</span>
              <select
                value={noteState.verdict}
                onChange={(event) => updateNote({ verdict: event.target.value as ChemistNotebookEntry["verdict"] })}
                className="rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-white outline-none"
              >
                {verdictOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>

            <div className="rounded-2xl border border-white/6 bg-slate-950/70 p-4">
              <p className="ui-kicker">Flux de revizie</p>
              <p className="mt-2 text-sm leading-7 text-slate-300">
                Marcheaza rapid moleculele care trebuie fixate, aprobate, respinse sau retrimise la retestare.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {reviewWorkflowTags.map((item) => {
                  const active = noteState.tags.includes(item.tag);
                  return (
                    <button
                      key={item.tag}
                      className={`rounded-full border px-3 py-1.5 text-xs transition ${
                        active ? "border-forge-cyan/60 bg-forge-cyan/15 text-white" : "border-white/10 bg-white/5 text-slate-300"
                      }`}
                      onClick={() =>
                        updateNote({
                          tags: active ? noteState.tags.filter((entry) => entry !== item.tag) : [...noteState.tags, item.tag],
                        })
                      }
                    >
                      {item.label}
                    </button>
                  );
                })}
              </div>
              <div className="mt-3 rounded-2xl border border-white/6 bg-white/5 p-3 text-sm leading-7 text-slate-300">
                {reviewWorkflowTags
                  .filter((item) => noteState.tags.includes(item.tag))
                  .map((item) => item.label)
                  .join(", ") || "Nicio actiune de revizie nu este setata momentan."}
              </div>
            </div>

            <div className="rounded-2xl border border-white/6 bg-slate-950/70 p-4">
              <p className="ui-kicker">Tag-uri rapide</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {quickTags.map((tag) => {
                  const active = noteState.tags.includes(tag);
                  return (
                    <button
                      key={tag}
                      className={`rounded-full border px-3 py-1.5 text-xs transition ${
                        active ? "border-forge-cyan/60 bg-forge-cyan/15 text-white" : "border-white/10 bg-white/5 text-slate-300"
                      }`}
                      onClick={() =>
                        updateNote({
                          tags: active ? noteState.tags.filter((entry) => entry !== tag) : [...noteState.tags, tag],
                        })
                      }
                    >
                      {tag}
                    </button>
                  );
                })}
              </div>
            </div>

            <label className="field-card">
              <span>Observatii</span>
              <textarea
                value={noteState.note}
                onChange={(event: ChangeEvent<HTMLTextAreaElement>) => updateNote({ note: event.target.value })}
                rows={7}
                className="rounded-2xl border border-white/10 bg-slate-950/80 px-3 py-3 text-sm leading-7 text-white outline-none"
                placeholder="Ex: merita pusa in primul lot de screening; are cost bun, dar as verifica presiunea pe forma nativa."
              />
            </label>

            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="ui-kicker">Ultima actualizare</p>
              <p className="mt-2 text-sm text-slate-300">
                {noteState.updatedAt ? new Date(noteState.updatedAt).toLocaleString("ro-RO") : "Nota nu a fost salvata inca."}
              </p>
            </div>
          </div>
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_360px]">
        <SectionCard
          eyebrow="ADMET / semnale de risc"
          title="Panou euristic de risc"
          subtitle="Semnale rapide pentru PAINS, alerte structurale, masa, polaritate, grupari reactive si indicatorul pentru forma nativa."
        >
          <div className="grid gap-3 md:grid-cols-2">
            {selected.admet.liabilities.map((item) => (
              <div
                key={item.label}
                className={`rounded-2xl border p-4 ${
                  item.tone === "success"
                    ? "border-emerald-400/20 bg-emerald-400/10"
                    : item.tone === "warning"
                      ? "border-amber-400/20 bg-amber-400/10"
                      : "border-rose-400/20 bg-rose-400/10"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-semibold text-white">{item.label}</p>
                  <span className="text-xs text-slate-100">{item.value}</span>
                </div>
                <p className="mt-2 text-sm leading-7 text-slate-100">{item.note}</p>
              </div>
            ))}
          </div>
        </SectionCard>

        <SectionCard
          eyebrow="Semnal final"
          title="Rezumat risc"
          subtitle="Doi indicatori rapizi pentru citire la prima vedere."
        >
          <div className="space-y-3">
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="ui-kicker">Indicator forma nativa</p>
              <p className="mt-2 text-2xl font-semibold text-white">{selected.admet.wildTypeProxy.toFixed(2)}</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="ui-kicker">Reactivitate</p>
              <p className="mt-2 text-2xl font-semibold text-white">{selected.admet.reactivityRisk.toFixed(2)}</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4 text-sm leading-7 text-slate-300">
              {selected.admet.summary}
            </div>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        eyebrow="Istoric de decizii"
        title="Cum a evoluat aceasta molecula"
        subtitle="Generator, parinte, mutatie, audit, comparator si notele salvate local sunt aduse intr-un singur flux."
      >
        <div className="space-y-3">
          {mergedHistory.map((item) => (
            <div
              key={item.id}
              className={`rounded-2xl border p-4 ${
                item.tone === "success"
                  ? "border-emerald-400/20 bg-emerald-400/10"
                  : item.tone === "warning"
                    ? "border-amber-400/20 bg-amber-400/10"
                    : "border-white/10 bg-white/5"
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-white">{item.title}</p>
                  <p className="mt-1 text-xs text-slate-400">
                    {item.round ? `Runda ${item.round}` : "Nota locala"} | {item.category}
                  </p>
                </div>
                {item.timestamp ? <span className="text-xs text-slate-400">{item.timestamp}</span> : null}
              </div>
              <p className="mt-3 text-sm leading-7 text-slate-100">{item.detail}</p>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
