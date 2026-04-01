import { SectionCard } from "@/components/SectionCard";
import type { OverviewPayload } from "@/types";

interface ActivitySectionProps {
  overview: OverviewPayload;
  logs: string;
  sources: string[];
}

export function ActivitySection({ overview, logs, sources }: ActivitySectionProps) {
  return (
    <div className="grid gap-4 2xl:grid-cols-[minmax(0,1.2fr)_380px]">
      <SectionCard
        eyebrow="Jurnal operational"
        title="Activitate worker"
        subtitle="Zona separata pentru diagnostic si trasabilitate, ca sa nu incarce vizual restul aplicatiei."
        className="h-full"
      >
        <div className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Sesiune</p>
              <p className="mt-2 text-lg font-semibold text-white">{overview.sessionName}</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Stare</p>
              <p className="mt-2 text-lg font-semibold text-white">{overview.statusLabel}</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Ultima actualizare</p>
              <p className="mt-2 text-lg font-semibold text-white">
                {overview.updatedAt ? new Date(overview.updatedAt).toLocaleTimeString("ro-RO") : "-"}
              </p>
            </div>
          </div>

          <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Mesaj sesiune</p>
            <p className="mt-3 text-sm leading-7 text-slate-300">{overview.message}</p>
          </div>

          <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Jurnal de executie</p>
            <pre className="mt-3 max-h-[560px] overflow-auto whitespace-pre-wrap rounded-2xl border border-white/5 bg-slate-950/85 p-3 font-mono text-xs leading-6 text-slate-300">
              {logs || "Nu exista inca evenimente in jurnal."}
            </pre>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        eyebrow="Trasabilitate"
        title="Surse si note"
        subtitle="Sursele utilizate de estimatorul de cost raman separate, pentru audit rapid."
        className="h-full"
      >
        <div className="space-y-3">
          {(sources ?? []).length ? (
            sources.map((source) => (
              <a
                key={source}
                href={source}
                target="_blank"
                rel="noreferrer"
                className="block rounded-2xl border border-white/6 bg-white/5 px-4 py-3 text-sm text-cyan-100 underline decoration-cyan-400/30 underline-offset-4"
              >
                {source}
              </a>
            ))
          ) : (
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4 text-sm leading-7 text-slate-300">
              Sursele vor aparea dupa prima rulare a worker-ului.
            </div>
          )}

          <div className="rounded-2xl border border-white/6 bg-white/5 p-4 text-sm leading-7 text-slate-300">
            Pastrez sursele si jurnalul intr-o sectiune separata pentru ca utilizatorii tehnici sa le poata verifica fara sa incarce vizual paginile de analiza.
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
