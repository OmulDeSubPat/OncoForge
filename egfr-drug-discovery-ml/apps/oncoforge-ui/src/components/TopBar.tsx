import { StatusPill } from "@/components/StatusPill";
import type { OverviewPayload } from "@/types";

interface TopBarProps {
  overview: OverviewPayload;
  busyAction: string | null;
  uiMode: "basic" | "expert";
  compareCount: number;
  reviewCount: number;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  onRefresh: () => void;
  onExport: () => void;
  onImportClick: () => void;
  onUiModeChange: (mode: "basic" | "expert") => void;
}

export function TopBar({
  overview,
  busyAction,
  uiMode,
  compareCount,
  reviewCount,
  onStart,
  onStop,
  onReset,
  onRefresh,
  onExport,
  onImportClick,
  onUiModeChange,
}: TopBarProps) {
  const running = overview.running;
  const progressPercent = Math.max(0, Math.min(100, Math.round(overview.progress * 100)));

  return (
    <header className="glass-panel sticky top-3 z-30 overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(36,214,234,0.12),transparent_34%),radial-gradient(circle_at_top_right,rgba(115,166,255,0.12),transparent_28%)]" />
      <div className="pointer-events-none absolute -left-10 top-8 h-24 w-24 rounded-full bg-forge-cyan/12 blur-3xl motion-safe:animate-drift" />
      <div className="pointer-events-none absolute right-8 top-6 h-20 w-20 rounded-full bg-forge-blue/12 blur-3xl motion-safe:animate-drift" />
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-forge-cyan/90 to-transparent" />

      <div className="relative grid gap-4 px-4 py-4 sm:px-5 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-start">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-2 rounded-full border border-forge-cyan/20 bg-forge-cyan/8 px-3 py-1 text-[11px] uppercase tracking-[0.22em] text-forge-cyan/95">
              <span className={`h-2 w-2 rounded-full ${running ? "bg-forge-green motion-safe:animate-pulseSoft" : "bg-slate-500"}`} />
              OncoSynth
            </span>
            <StatusPill status={running ? "running" : "stopped"} label={running ? "In rulare" : "Oprit"} />
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
              {overview.modeLabel}
            </span>
            <span className="rounded-full border border-white/10 bg-slate-950/70 px-3 py-1 text-xs text-slate-200">
              {uiMode === "basic" ? "Interfata simplificata" : "Interfata avansata"}
            </span>
          </div>
          <div className="mt-3 flex flex-wrap items-end gap-x-4 gap-y-2">
            <h1 className="text-[2.15rem] font-semibold leading-none tracking-[-0.04em] text-white sm:text-[2.45rem]">
              {overview.sessionName}
            </h1>
            <div className="rounded-2xl border border-white/8 bg-white/[0.03] px-3 py-2 text-xs uppercase tracking-[0.16em] text-slate-300">
              Progres live: <span className="text-white">{progressPercent}%</span>
            </div>
          </div>
          <p className="mt-3 max-w-5xl text-sm leading-7 text-slate-300">
            Laborator IA pentru generare si optimizare EGFR, organizat pentru decizie rapida in modul simplificat si analiza detaliata in modul avansat.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 xl:justify-end">
          <button className="control-button control-button-primary disabled:opacity-60" onClick={onStart} disabled={busyAction !== null || running}>
            {busyAction === "start" ? "Se porneste..." : "Porneste generarea"}
          </button>
          <button className="control-button disabled:opacity-60" onClick={onStop} disabled={busyAction !== null || !running}>
            {busyAction === "stop" ? "Se opreste..." : "Stop generare"}
          </button>
          <button className="control-button disabled:opacity-60" onClick={onRefresh} disabled={busyAction !== null}>
            Actualizeaza
          </button>
          <button className="control-button disabled:opacity-60" onClick={onReset} disabled={busyAction !== null || running}>
            Reseteaza
          </button>
          <button className="control-button" onClick={onImportClick}>
            Import
          </button>
          <button className="control-button" onClick={onExport}>
            Export
          </button>
        </div>
      </div>

      <div className="relative border-t border-white/5 px-4 py-3 sm:px-5">
        <div className="mb-4 rounded-full border border-white/6 bg-slate-950/70 p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
          <div
            className={`h-2 rounded-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green transition-[width] duration-700 ${running ? "motion-safe:animate-pulseSoft" : ""}`}
            style={{ width: `${Math.max(progressPercent, running ? 10 : 0)}%` }}
          />
        </div>

        <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(330px,0.88fr)]">
          <div className="grid gap-3 lg:grid-cols-[repeat(3,minmax(0,160px))_minmax(280px,0.9fr)]">
            <CompactMetricCard label="Stare" value={overview.statusLabel} />
            <CompactMetricCard label="Progres" value={`${Math.round(overview.progress * 100)}%`} />
            <CompactMetricCard
              label="Ultima actualizare"
              value={overview.updatedAt ? new Date(overview.updatedAt).toLocaleTimeString("ro-RO") : "Asteapta"}
            />

            <div className="rounded-[24px] border border-white/8 bg-white/5 px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="ui-kicker">Flux UI</p>
                  <p className="mt-1 text-sm font-semibold text-white">
                    {uiMode === "basic" ? "Mod orientat pe decizie" : "Mod pentru analiza avansata"}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {([
                    { key: "basic", label: "Simplificat" },
                    { key: "expert", label: "Avansat" },
                  ] as const).map((option) => (
                    <button
                      key={option.key}
                      className={`rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] transition ${
                        uiMode === option.key
                          ? "border-forge-cyan/60 bg-forge-cyan/15 text-white"
                          : "border-white/10 bg-slate-950/70 text-slate-300 hover:bg-white/10"
                      }`}
                      onClick={() => onUiModeChange(option.key)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <span className="rounded-full border border-white/10 bg-slate-950/75 px-3 py-1 text-xs text-slate-200">
                  Set comparatie: {compareCount}
                </span>
                <span className="rounded-full border border-white/10 bg-slate-950/75 px-3 py-1 text-xs text-slate-200">
                  Molecule in revizie: {reviewCount}
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-[24px] border border-white/8 bg-white/5 px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
            <p className="ui-kicker">Flux operational recomandat</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {(uiMode === "basic"
                ? ["Porneste sesiunea", "Triere", "Molecula", "Comparatie", "Risc si decizie"]
                : ["Evaluare initiala", "Molecula", "Set comparatie", "Audit algoritmic", "Export si jurnal"]
              ).map((step, index) => (
                <span key={step} className="rounded-full border border-white/10 bg-slate-950/70 px-3 py-1.5 text-xs text-slate-200">
                  {index + 1}. {step}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function CompactMetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[24px] border border-white/8 bg-white/5 px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
      <p className="ui-kicker">{label}</p>
      <p className="mt-1 text-lg font-semibold tabular-nums text-white">{value}</p>
    </div>
  );
}
