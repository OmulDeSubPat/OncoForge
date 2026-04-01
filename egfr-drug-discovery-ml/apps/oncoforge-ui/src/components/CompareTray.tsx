import { SectionCard } from "@/components/SectionCard";
import type { LibraryRow, SelectedMolecule } from "@/types";

interface CompareTrayProps {
  selected: SelectedMolecule | null;
  library: LibraryRow[];
  compareSmiles: string[];
  uiMode: "basic" | "expert";
  expanded: boolean;
  onToggleCompare: (smiles: string) => void;
  onOpenMolecule: (smiles: string) => void;
  onOpenCompare: () => void;
  onClearCompare: () => void;
  onToggleExpanded: () => void;
}

export function CompareTray({
  selected,
  library,
  compareSmiles,
  uiMode,
  expanded,
  onToggleCompare,
  onOpenMolecule,
  onOpenCompare,
  onClearCompare,
  onToggleExpanded,
}: CompareTrayProps) {
  const compareRows = compareSmiles
    .map((smiles) => library.find((entry) => entry.smiles === smiles))
    .filter((entry): entry is LibraryRow => Boolean(entry));

  const quickSuggestions = library
    .filter((entry) => !compareSmiles.includes(entry.smiles))
    .slice(0, uiMode === "basic" ? 4 : 6);

  if (!selected && !compareRows.length && !library.length) {
    return null;
  }

  return (
    <div className="mt-4">
      <SectionCard
        eyebrow="Set de comparatie"
        title="Setul curent pentru analiza comparativa"
        subtitle="Selectiile raman vizibile intre sectiuni si pot fi transferate direct in ecranul de comparatie."
        action={
          <div className="flex flex-wrap gap-2">
            <button className="control-button px-3 py-1.5 text-xs" onClick={onToggleExpanded}>
              {expanded ? "Restrange" : "Extinde"}
            </button>
            <button className="control-button px-3 py-1.5 text-xs" onClick={onOpenCompare}>
              Deschide comparatia
            </button>
            <button className="control-button px-3 py-1.5 text-xs" onClick={onClearCompare} disabled={!compareRows.length}>
              Goleste setul
            </button>
          </div>
        }
      >
        {!expanded ? (
          <div className="rounded-[24px] border border-white/6 bg-slate-950/70 px-4 py-4 text-sm leading-7 text-slate-300">
            <div className="flex flex-wrap items-center gap-2">
              {compareRows.slice(0, 4).map((entry) => (
                <span key={entry.id} className="rounded-full border border-forge-cyan/20 bg-forge-cyan/10 px-3 py-1 text-xs text-cyan-50">
                  #{entry.rank} · R{entry.round}
                </span>
              ))}
              {!compareRows.length ? (
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                  Fara selectie activa
                </span>
              ) : null}
            </div>
            <p className="mt-3">
              {compareRows.length
                ? `Setul este restrans momentan. Sunt disponibile ${compareRows.length} molecule pentru comparatie.`
                : "Setul de comparatie este gol si ramane restrans pentru a economisi spatiu pe primul ecran."}
            </p>
          </div>
        ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
          <div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {compareRows.length ? (
                compareRows.map((entry) => (
                  <div key={entry.id} className="rounded-[24px] border border-white/6 bg-[linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0.03))] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                    <div className="flex items-start justify-between gap-3">
                      <button className="min-w-0 flex-1 text-left" onClick={() => onOpenMolecule(entry.smiles)}>
                        <p className="text-sm font-semibold text-white">Candidat #{entry.rank}</p>
                        <p className="mt-1 text-xs text-slate-400">R{entry.round} | scor {entry.score.toFixed(2)}</p>
                      </button>
                      <button className="text-xs text-slate-300 transition hover:text-white" onClick={() => onToggleCompare(entry.smiles)}>
                        Scoate
                      </button>
                    </div>
                    <p className="mt-3 break-words text-xs leading-6 text-slate-400">{entry.action || "Mutatie noua"}</p>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-300">
                      <span className="rounded-full border border-white/10 bg-slate-950/70 px-2 py-1">pIC50 {entry.pic50.toFixed(2)}</span>
                      <span className="rounded-full border border-white/10 bg-slate-950/70 px-2 py-1">QED {entry.qed.toFixed(2)}</span>
                      <span className="rounded-full border border-white/10 bg-slate-950/70 px-2 py-1">Cost ${entry.cost10mg.toFixed(2)}</span>
                      <span className="rounded-full border border-white/10 bg-slate-950/70 px-2 py-1">{entry.marketReference || "Fara comparator"}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-white/6 bg-slate-950/70 p-4 text-sm leading-7 text-slate-300 md:col-span-2 xl:col-span-4">
                  Nu exista inca molecule fixate pentru comparatie. Marcheaza candidatii relevanti din Molecula sau din Biblioteca.
                </div>
              )}
            </div>

            {selected ? (
              <div className="mt-3 rounded-[24px] border border-forge-cyan/20 bg-[linear-gradient(135deg,rgba(36,214,234,0.14),rgba(12,20,36,0.86))] p-4 text-sm leading-7 text-cyan-50">
                Molecula activa: #{selected.rank}. Foloseste `Compara` ca sa o fixezi in set, apoi adauga inca 1-3 candidati.
              </div>
            ) : null}
          </div>

          <div className="rounded-[28px] border border-white/6 bg-slate-950/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
            <p className="ui-kicker">Sugestii rapide</p>
            <p className="mt-2 text-sm leading-7 text-slate-300">
              Selectiile de mai jos sunt luate din topul curent al bibliotecii, ca sa formezi rapid o comparatie multipla coerenta.
            </p>
            <div className="mt-4 space-y-2">
              {quickSuggestions.length ? (
                quickSuggestions.map((entry) => (
                  <div key={entry.id} className="flex items-center justify-between gap-3 rounded-[22px] border border-white/6 bg-white/5 px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                    <button className="min-w-0 flex-1 text-left" onClick={() => onOpenMolecule(entry.smiles)}>
                      <p className="text-sm font-semibold text-white">Candidat #{entry.rank}</p>
                      <p className="mt-1 truncate text-xs text-slate-400">{entry.action || "Mutatie noua"}</p>
                    </button>
                    <button className="rounded-full border border-white/10 bg-slate-950/80 px-3 py-1 text-[11px] text-slate-200 transition hover:bg-white/10" onClick={() => onToggleCompare(entry.smiles)}>
                      Adauga
                    </button>
                  </div>
                ))
              ) : (
                <p className="text-sm leading-7 text-slate-300">Setul este complet sau biblioteca nu are inca suficienti candidati disponibili.</p>
              )}
            </div>
          </div>
        </div>
        )}
      </SectionCard>
    </div>
  );
}
