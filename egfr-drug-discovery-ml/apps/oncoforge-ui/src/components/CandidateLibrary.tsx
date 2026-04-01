import { memo, useDeferredValue, useEffect, useMemo, useState } from "react";

import { SectionCard } from "@/components/SectionCard";
import { StatusPill } from "@/components/StatusPill";
import type { CandidateStatus, LibraryRow } from "@/types";

interface CandidateLibraryProps {
  library: LibraryRow[];
  selectedSmiles: string | null;
  compareSmiles: string[];
  uiMode: "basic" | "expert";
  onSelectCandidate: (smiles: string) => void;
  onToggleCompare: (smiles: string) => void;
}

const statusFilters: Array<"all" | CandidateStatus> = ["all", "promovata", "revizie", "respinsa", "necunoscut"];

export const CandidateLibrary = memo(function CandidateLibrary({
  library,
  selectedSmiles,
  compareSmiles,
  uiMode,
  onSelectCandidate,
  onToggleCompare,
}: CandidateLibraryProps) {
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | CandidateStatus>("all");
  const [page, setPage] = useState(1);
  const deferredQuery = useDeferredValue(query);
  const pageSize = uiMode === "basic" ? 10 : 14;

  const visibleCandidates = useMemo(() => {
    return [...library]
      .filter((candidate) => {
        const queryText = `${candidate.smiles} ${candidate.parent} ${candidate.action} ${candidate.route} ${candidate.marketReference}`.toLowerCase();
        const matchesQuery = queryText.includes(deferredQuery.toLowerCase());
        const matchesStatus = statusFilter === "all" || candidate.status === statusFilter;
        return matchesQuery && matchesStatus;
      })
      .sort((left, right) => right.score - left.score);
  }, [deferredQuery, library, statusFilter]);

  useEffect(() => {
    setPage(1);
  }, [deferredQuery, statusFilter, uiMode]);

  const totalPages = Math.max(1, Math.ceil(visibleCandidates.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const pageCandidates = visibleCandidates.slice((safePage - 1) * pageSize, safePage * pageSize);
  const promotedVisible = visibleCandidates.filter((candidate) => candidate.status === "promovata").length;
  const meanVisibleCost = visibleCandidates.length
    ? visibleCandidates.reduce((sum, candidate) => sum + candidate.cost10mg, 0) / visibleCandidates.length
    : 0;

  return (
    <SectionCard
      eyebrow="Biblioteca de candidati"
      title="Molecule generate"
      subtitle="Lista este optimizata pentru selectie operationala: cautare, filtre, paginare si trimitere directa in setul de comparatie."
      className="h-full"
    >
      <div className="space-y-4">
        <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-end">
          <label className="rounded-2xl border border-white/6 bg-slate-950/70 px-4 py-3">
            <span className="text-xs uppercase tracking-[0.22em] text-slate-400">Cautare in biblioteca</span>
            <input
              className="mt-2 w-full bg-transparent text-sm text-white outline-none placeholder:text-slate-500"
              placeholder="Filtreaza dupa SMILES, ruta, mutatie sau comparator"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>

          <div className="flex flex-wrap gap-2 xl:justify-end">
            {statusFilters.map((status) => (
              <button
                key={status}
                className={`rounded-full border px-3 py-2 text-xs font-semibold uppercase tracking-[0.18em] transition ${
                  statusFilter === status
                    ? "border-forge-cyan/60 bg-forge-cyan/15 text-white"
                    : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
                onClick={() => setStatusFilter(status)}
              >
                {status === "all" ? "toate" : status}
              </button>
            ))}
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl border border-white/6 bg-white/5 p-3">
            <p className="ui-kicker">Molecule filtrate</p>
            <p className="mt-2 text-lg font-semibold text-white">{visibleCandidates.length}</p>
          </div>
          <div className="rounded-2xl border border-white/6 bg-white/5 p-3">
            <p className="ui-kicker">Promovate</p>
            <p className="mt-2 text-lg font-semibold text-white">{promotedVisible}</p>
          </div>
          <div className="rounded-2xl border border-white/6 bg-white/5 p-3">
            <p className="ui-kicker">Cost mediu 10 mg</p>
            <p className="mt-2 text-lg font-semibold text-white">${meanVisibleCost.toFixed(2)}</p>
          </div>
          <div className="rounded-2xl border border-white/6 bg-white/5 p-3">
            <p className="ui-kicker">Pagina curenta</p>
            <p className="mt-2 text-lg font-semibold text-white">
              {safePage} / {totalPages}
            </p>
          </div>
        </div>

        {uiMode === "basic" ? (
          <div className="grid gap-3 xl:grid-cols-2">
            {pageCandidates.map((candidate) => {
              const active = candidate.smiles === selectedSmiles;
              const inCompare = compareSmiles.includes(candidate.smiles);
              return (
                <div
                  key={candidate.id}
                  className={`rounded-[26px] border p-4 ${
                    active ? "border-forge-cyan/45 bg-forge-cyan/10" : "border-white/8 bg-slate-950/65"
                  }`}
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-white">Candidat #{candidate.rank}</p>
                      <p className="mt-1 text-xs text-slate-400">
                        R{candidate.round} | scor {candidate.score.toFixed(2)} | pIC50 {candidate.pic50.toFixed(2)}
                      </p>
                    </div>
                    <StatusPill status={candidate.status} label={candidate.statusLabel} />
                  </div>
                  <p className="mt-3 break-words text-sm leading-7 text-slate-300">{candidate.action || "Mutatie noua"}</p>
                  <div className="mt-3 grid gap-2 sm:grid-cols-2 text-xs text-slate-400">
                    <span>QED {candidate.qed.toFixed(2)}</span>
                    <span>Cost ${candidate.cost10mg.toFixed(2)}</span>
                    <span>{candidate.route || "Ruta nedefinita"}</span>
                    <span>{candidate.marketReference || "Fara comparator"}</span>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <button className="control-button px-3 py-1.5 text-xs" onClick={() => onSelectCandidate(candidate.smiles)}>
                      Deschide
                    </button>
                    <button
                      className={`rounded-full border px-3 py-1.5 text-xs transition ${
                        inCompare ? "border-forge-cyan/45 bg-forge-cyan/10 text-white" : "border-white/10 bg-white/5 text-slate-300"
                      }`}
                      onClick={() => onToggleCompare(candidate.smiles)}
                    >
                      {inCompare ? "Scoate din comparatie" : "Adauga in comparatie"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="overflow-hidden rounded-[28px] border border-white/6 bg-slate-950/65">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-white/5 text-left">
                <thead className="bg-white/5 text-xs uppercase tracking-[0.18em] text-slate-400">
                  <tr>
                    <th className="px-4 py-3">Rang</th>
                    <th className="px-4 py-3">Candidat</th>
                    <th className="px-4 py-3">Scor</th>
                    <th className="px-4 py-3">QED</th>
                    <th className="px-4 py-3">Cost 10 mg</th>
                    <th className="px-4 py-3">Runda</th>
                    <th className="px-4 py-3">Actiune si ruta</th>
                    <th className="px-4 py-3">Piata</th>
                    <th className="px-4 py-3">Stare</th>
                    <th className="px-4 py-3">Comparatie</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {pageCandidates.map((candidate) => {
                    const active = candidate.smiles === selectedSmiles;
                    const inCompare = compareSmiles.includes(candidate.smiles);
                    return (
                      <tr key={candidate.id} className={active ? "bg-forge-cyan/8" : "hover:bg-white/5"}>
                        <td className="px-4 py-4 font-semibold text-white">#{candidate.rank}</td>
                        <td className="max-w-[260px] px-4 py-4">
                          <button className="text-left" onClick={() => onSelectCandidate(candidate.smiles)}>
                            <div className="break-all font-mono text-xs leading-6 text-slate-300">{candidate.smiles}</div>
                          </button>
                        </td>
                        <td className="px-4 py-4">
                          <div className="text-sm font-semibold text-white">{candidate.score.toFixed(2)}</div>
                          <div className="text-xs text-slate-400">pIC50 {candidate.pic50.toFixed(2)}</div>
                        </td>
                        <td className="px-4 py-4 text-sm text-slate-300">{candidate.qed.toFixed(2)}</td>
                        <td className="px-4 py-4 text-sm text-slate-300">${candidate.cost10mg.toFixed(2)}</td>
                        <td className="px-4 py-4 text-sm text-slate-300">{candidate.round}</td>
                        <td className="max-w-[260px] px-4 py-4">
                          <div className="break-words text-sm text-slate-200">{candidate.action || "-"}</div>
                          <div className="break-words text-xs leading-6 text-slate-500">{candidate.route || "-"}</div>
                        </td>
                        <td className="px-4 py-4 text-sm text-slate-300">{candidate.marketReference || "-"}</td>
                        <td className="px-4 py-4">
                          <StatusPill status={candidate.status} label={candidate.statusLabel} />
                        </td>
                        <td className="px-4 py-4">
                          <button
                            className={`rounded-full border px-3 py-1 text-[11px] transition ${
                              inCompare ? "border-forge-cyan/45 bg-forge-cyan/10 text-white" : "border-white/10 bg-slate-950/80 text-slate-300"
                            }`}
                            onClick={() => onToggleCompare(candidate.smiles)}
                          >
                            {inCompare ? "In set" : "Compara"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!visibleCandidates.length ? (
          <div className="rounded-[26px] border border-white/6 bg-slate-950/65 p-4 text-sm leading-7 text-slate-300">
            Nu exista inca molecule care sa corespunda filtrelor curente. Porneste generarea sau relaxeaza cautarea.
          </div>
        ) : null}

        {visibleCandidates.length ? (
          <div className="flex flex-col gap-3 rounded-[26px] border border-white/6 bg-slate-950/65 p-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="text-sm leading-7 text-slate-300">
              Afisezi {pageCandidates.length} molecule din {visibleCandidates.length}. Paginarea reduce rerandarile si face biblioteca mai usor de scanat.
            </div>
            <div className="flex flex-wrap gap-2">
              <button className="control-button px-3 py-1.5 text-xs" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={safePage === 1}>
                Pagina anterioara
              </button>
              <button className="control-button px-3 py-1.5 text-xs" onClick={() => setPage((current) => Math.min(totalPages, current + 1))} disabled={safePage === totalPages}>
                Pagina urmatoare
              </button>
            </div>
          </div>
        ) : null}

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {visibleCandidates.slice(0, 4).map((candidate) => (
            <div key={candidate.id} className="rounded-2xl border border-white/5 bg-white/5 p-3">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Candidat #{candidate.rank}</p>
              <p className="mt-2 text-sm font-semibold text-white">{candidate.pic50.toFixed(2)} pIC50</p>
              <p className="mt-1 text-xs leading-6 text-slate-400">
                QED {candidate.qed.toFixed(2)} | Cost 10 mg ${candidate.cost10mg.toFixed(2)}
              </p>
            </div>
          ))}
        </div>
      </div>
    </SectionCard>
  );
});
