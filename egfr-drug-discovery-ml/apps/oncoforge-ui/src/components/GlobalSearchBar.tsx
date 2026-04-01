import { useDeferredValue } from "react";

import type { LabSectionKey } from "@/components/LabNavigation";
import { SectionCard } from "@/components/SectionCard";
import type { LibraryRow } from "@/types";

interface SectionSearchItem {
  id: LabSectionKey;
  label: string;
  subtitle: string;
}

interface GlobalSearchBarProps {
  query: string;
  expanded: boolean;
  onChange: (value: string) => void;
  onToggle: () => void;
  sections: SectionSearchItem[];
  library: LibraryRow[];
  onOpenSection: (section: LabSectionKey) => void;
  onOpenMolecule: (smiles: string) => void;
}

export function GlobalSearchBar({
  query,
  expanded,
  onChange,
  onToggle,
  sections,
  library,
  onOpenSection,
  onOpenMolecule,
}: GlobalSearchBarProps) {
  const deferredQuery = useDeferredValue(query);
  const normalized = deferredQuery.trim().toLowerCase();

  const sectionMatches = normalized
    ? sections.filter((section) => `${section.label} ${section.subtitle}`.toLowerCase().includes(normalized)).slice(0, 5)
    : [];

  const moleculeMatches = normalized
    ? library
        .filter((item) =>
          `${item.smiles} ${item.action} ${item.route} ${item.marketReference}`.toLowerCase().includes(normalized),
        )
        .slice(0, 6)
    : [];

  return (
    <SectionCard
      eyebrow="Cautare unificata"
      title="Navigare instant prin sectiuni si molecule"
      subtitle="Gaseste rapid o sectiune, un candidat sau o ruta sintetica fara sa schimbi contextul curent."
      className="mt-4"
      action={
        <button className="control-button px-3 py-1.5 text-xs" onClick={onToggle}>
          {expanded ? "Restrange" : "Extinde"}
        </button>
      }
    >
      <div className="grid gap-3 xl:grid-cols-[minmax(0,0.74fr)_minmax(360px,1.26fr)] xl:items-center">
        <div className="min-w-0 rounded-[24px] border border-white/6 bg-white/[0.03] px-4 py-3">
          <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Indexare live</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="rounded-full border border-white/10 bg-slate-950/80 px-3 py-1 text-xs text-slate-200">
              {sections.length} sectiuni
            </span>
            <span className="rounded-full border border-white/10 bg-slate-950/80 px-3 py-1 text-xs text-slate-200">
              {library.length} molecule
            </span>
            <span className="rounded-full border border-white/10 bg-slate-950/80 px-3 py-1 text-xs text-slate-200">
              Cautare dupa SMILES, ruta si comparator
            </span>
          </div>
        </div>

        <label className="relative w-full overflow-hidden rounded-[24px] border border-white/6 bg-slate-950/78 px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
          <div className="pointer-events-none absolute inset-y-0 right-0 w-32 bg-gradient-to-l from-forge-cyan/8 to-transparent" />
          <span className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Interogare</span>
          <input
            className="mt-2 w-full bg-transparent text-sm text-white outline-none placeholder:text-slate-500"
            placeholder="Cauta dupa sectiune, SMILES, mutatie, ruta sau comparator"
            value={query}
            onChange={(event) => onChange(event.target.value)}
          />
        </label>
      </div>

      {!expanded ? (
        <div className="mt-3 rounded-[24px] border border-white/6 bg-slate-950/65 px-4 py-3 text-sm leading-7 text-slate-300">
          {normalized
            ? `Rezultate gasite: ${sectionMatches.length} sectiuni si ${moleculeMatches.length} molecule.`
            : "Panoul de cautare este restrans. Il poti deschide doar cand ai nevoie de el."}
        </div>
      ) : normalized ? (
        <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(280px,0.72fr)_minmax(0,1.28fr)]">
          <div className="rounded-[28px] border border-white/6 bg-slate-950/65 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Sectiuni gasite</p>
            <div className="mt-3 space-y-2">
              {sectionMatches.length ? (
                sectionMatches.map((section) => (
                  <button
                    key={section.id}
                    className="w-full rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-left transition hover:bg-white/10"
                    onClick={() => onOpenSection(section.id)}
                  >
                    <span className="block text-sm font-semibold text-white">{section.label}</span>
                    <span className="mt-1 block text-xs leading-6 text-slate-400">{section.subtitle}</span>
                  </button>
                ))
              ) : (
                <p className="text-sm leading-7 text-slate-300">Nicio sectiune nu se potriveste cautarii curente.</p>
              )}
            </div>
          </div>

          <div className="rounded-[28px] border border-white/6 bg-slate-950/65 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Molecule gasite</p>
            <div className="mt-3 grid gap-3 xl:grid-cols-2">
              {moleculeMatches.length ? (
                moleculeMatches.map((item) => (
                  <button
                    key={item.id}
                    className="h-full rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-left transition hover:bg-white/10"
                    onClick={() => onOpenMolecule(item.smiles)}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm font-semibold text-white">Candidat #{item.rank}</span>
                      <span className="text-xs text-slate-400">R{item.round}</span>
                    </div>
                    <span className="mt-2 block break-all font-mono text-xs leading-6 text-slate-300">{item.smiles}</span>
                    <span className="mt-3 block text-xs leading-6 text-slate-400">{item.action || "Mutatie noua"}</span>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-slate-500">
                      <span>{item.route || "Ruta nedefinita"}</span>
                      <span>{item.marketReference || "Fara comparator"}</span>
                    </div>
                  </button>
                ))
              ) : (
                <p className="text-sm leading-7 text-slate-300 xl:col-span-2">Nicio molecula nu se potriveste cautarii curente.</p>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-4 rounded-[28px] border border-white/6 bg-slate-950/65 px-4 py-3 text-sm leading-7 text-slate-300">
          Cautarea globala devine activa imediat ce introduci un termen. Functioneaza bine pentru nume de sectiuni, SMILES, rute sintetice si comparatori de piata.
        </div>
      )}
    </SectionCard>
  );
}
