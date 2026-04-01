import type { ChangeEvent } from "react";

import { SectionCard } from "@/components/SectionCard";
import type { ControlForm, OverviewPayload } from "@/types";

interface SessionSetupPanelProps {
  overview: OverviewPayload;
  control: ControlForm;
  onFieldChange: <K extends keyof ControlForm>(key: K, value: ControlForm[K]) => void;
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field-card">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-white outline-none"
      />
    </label>
  );
}

export function SessionSetupPanel({ overview, control, onFieldChange }: SessionSetupPanelProps) {
  const modeOptions = [
    { value: "explorare", label: "Explorare rapida" },
    { value: "ghidat_ai", label: "Generare ghidata" },
    { value: "iterativ", label: "Optimizare iterativa" },
  ] as const;

  return (
    <SectionCard
      eyebrow="Configurare sesiune"
      title="Parametri experiment"
      subtitle="Setarile operationale sunt separate de restul analizelor, ca sa poata fi citite rapid inainte de rulare."
      className="h-full"
    >
      <div className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2">
          <label className="field-card">
            <span>Nume sesiune</span>
            <input
              type="text"
              value={control.sessionName}
              onChange={(event) => onFieldChange("sessionName", event.target.value)}
              className="rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-white outline-none"
            />
          </label>

          <label className="field-card">
            <span>Mod de generare</span>
            <select
              value={control.mode}
              onChange={(event: ChangeEvent<HTMLSelectElement>) => onFieldChange("mode", event.target.value as ControlForm["mode"])}
              className="rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-white outline-none"
            >
              {modeOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <NumberField label="Runde" value={control.rounds} min={1} max={12} onChange={(value) => onFieldChange("rounds", value)} />
          <NumberField label="Seminte" value={control.seedCount} min={1} max={20} onChange={(value) => onFieldChange("seedCount", value)} />
          <NumberField label="Variante / samanta" value={control.variantsPerSeed} min={1} max={64} onChange={(value) => onFieldChange("variantsPerSeed", value)} />
          <NumberField label="Beam width" value={control.beamWidth} min={1} max={32} onChange={(value) => onFieldChange("beamWidth", value)} />
        </div>

        <label className="flex items-center gap-3 rounded-2xl border border-white/6 bg-slate-950/60 px-4 py-3 text-sm text-slate-200">
          <input
            type="checkbox"
            checked={control.replaceExisting}
            onChange={(event) => onFieldChange("replaceExisting", event.target.checked)}
            className="h-4 w-4 rounded border-white/10 bg-slate-950/80"
          />
          <div>
            <p className="font-semibold text-white">Reseteaza sesiunea la urmatorul start</p>
            <p className="mt-1 text-xs leading-6 text-slate-400">Pastreaza activat pentru rulari curate. Dezactiveaza doar daca vrei sa continui peste artefactele existente.</p>
          </div>
        </label>

        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Stare</p>
            <p className="mt-2 text-lg font-semibold text-white">{overview.statusLabel}</p>
            <p className="mt-2 text-sm leading-6 text-slate-300">{overview.message}</p>
          </div>
          <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Mod activ</p>
            <p className="mt-2 text-lg font-semibold text-white">{overview.modeLabel}</p>
            <p className="mt-2 text-sm leading-6 text-slate-300">Worker-ul va folosi acest profil la urmatoarea lansare.</p>
          </div>
          <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Progres curent</p>
            <p className="mt-2 text-lg font-semibold text-white">{Math.round(overview.progress * 100)}%</p>
            <div className="mt-3 h-2 rounded-full bg-slate-900">
              <div
                className="h-2 rounded-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green"
                style={{ width: `${Math.max(4, Math.round(overview.progress * 100))}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </SectionCard>
  );
}
