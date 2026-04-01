export type AppSectionId = "panou" | "statistici" | "molecula" | "ai" | "biblioteca" | "export" | "activitate";

interface SectionTabItem {
  id: AppSectionId;
  label: string;
  subtitle: string;
}

interface SectionTabsProps {
  activeSection: AppSectionId;
  onChange: (section: AppSectionId) => void;
}

const SECTIONS: SectionTabItem[] = [
  {
    id: "panou",
    label: "Panou general",
    subtitle: "Status sesiune, trenduri si sumar rapid",
  },
  {
    id: "statistici",
    label: "Statistici live",
    subtitle: "Distributii, comparatii si grafice suplimentare",
  },
  {
    id: "molecula",
    label: "Laborator molecular",
    subtitle: "Viewer 2D/3D si explicatia candidatului activ",
  },
  {
    id: "ai",
    label: "Agenti si RLVR",
    subtitle: "Contributii AI, reward si penalizari explicabile",
  },
  {
    id: "biblioteca",
    label: "Biblioteca si evolutie",
    subtitle: "Clasament, distributii si istoric pe runde",
  },
  {
    id: "export",
    label: "Export si autosave",
    subtitle: "Subseturi, import partial si draft-uri locale",
  },
  {
    id: "activitate",
    label: "Jurnal sesiune",
    subtitle: "Loguri, surse si context operational",
  },
];

export function SectionTabs({ activeSection, onChange }: SectionTabsProps) {
  return (
    <div className="mt-4 overflow-x-auto">
      <div className="flex min-w-max gap-3 pb-1">
        {SECTIONS.map((section) => {
          const active = section.id === activeSection;
          return (
            <button
              key={section.id}
              className={`min-w-[230px] rounded-3xl border px-4 py-4 text-left transition ${
                active
                  ? "border-forge-cyan/50 bg-forge-cyan/12 shadow-glow"
                  : "border-white/8 bg-white/5 hover:bg-white/10"
              }`}
              onClick={() => onChange(section.id)}
            >
              <p className="text-[11px] uppercase tracking-[0.24em] text-slate-400">Sectiune</p>
              <p className="mt-2 text-base font-semibold text-white">{section.label}</p>
              <p className="mt-2 text-sm leading-6 text-slate-300">{section.subtitle}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
