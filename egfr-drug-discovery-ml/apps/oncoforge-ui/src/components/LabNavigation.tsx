export type LabSectionKey =
  | "rezumat"
  | "triere"
  | "laborator"
  | "comparatie"
  | "insight"
  | "antrenare"
  | "planificare"
  | "biblioteca"
  | "export"
  | "activitate";

interface LabNavItem {
  key: LabSectionKey;
  label: string;
  description: string;
  badge?: string;
}

interface LabNavigationProps {
  items: LabNavItem[];
  activeSection: LabSectionKey;
  onChange: (section: LabSectionKey) => void;
}

export function LabNavigation({ items, activeSection, onChange }: LabNavigationProps) {
  const activeItem = items.find((item) => item.key === activeSection) ?? items[0];

  return (
    <nav className="section-enter mt-3 rounded-[28px] border border-white/8 bg-slate-950/72 p-3 shadow-soft backdrop-blur-xl">
      <div className="mb-3 flex items-center justify-between gap-3 px-1">
        <div>
          <p className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Navigatie laborator</p>
          <p className="mt-1 text-sm text-slate-300">Treci rapid intre etapele de selectie, comparatie si audit.</p>
        </div>
        <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200">
          {items.length} sectiuni
        </span>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-1">
        {items.map((item) => {
          const active = item.key === activeSection;
          return (
            <button
              key={item.key}
              className={`group min-w-fit rounded-[22px] border px-4 py-3 text-left transition duration-300 ${
                active
                  ? "border-forge-cyan/45 bg-[linear-gradient(135deg,rgba(36,214,234,0.16),rgba(12,20,36,0.92))] text-white shadow-[0_10px_30px_rgba(36,214,234,0.12)]"
                  : "border-white/10 bg-white/[0.03] text-slate-300 hover:bg-white/[0.06]"
              }`}
              onClick={() => onChange(item.key)}
            >
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold">{item.label}</span>
                {item.badge ? (
                  <span className={`rounded-full border px-2 py-0.5 text-[11px] ${active ? "border-forge-cyan/25 bg-slate-950/70 text-cyan-50" : "border-white/10 bg-slate-950/75 text-slate-200"}`}>
                    {item.badge}
                  </span>
                ) : null}
              </div>
              <div className={`mt-2 h-1.5 rounded-full transition ${active ? "bg-white/10" : "bg-transparent"}`}>
                <div className={`h-1.5 rounded-full transition-all duration-300 ${active ? "w-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green" : "w-0 bg-transparent"}`} />
              </div>
            </button>
          );
        })}
      </div>

      {activeItem ? (
        <div className="mt-3 rounded-[22px] border border-white/6 bg-white/5 px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-sm font-semibold text-white">{activeItem.label}</p>
            {activeItem.badge ? (
              <span className="rounded-full border border-white/10 bg-slate-950/75 px-2.5 py-1 text-[11px] text-slate-200">
                {activeItem.badge}
              </span>
            ) : null}
            <p className="text-sm leading-7 text-slate-300">{activeItem.description}</p>
          </div>
        </div>
      ) : null}
    </nav>
  );
}
