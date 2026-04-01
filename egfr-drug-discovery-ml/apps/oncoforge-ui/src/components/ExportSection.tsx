import { type ChangeEvent, useMemo, useRef, useState } from "react";

import { SectionCard } from "@/components/SectionCard";
import type { ChemistNotebookEntry, ControlForm, LibraryRow, SelectedMolecule } from "@/types";

type ExportScope = "selectata" | "top" | "promovate" | "review" | "runda" | "importate";
type ExportFormat = "json" | "csv";

function escapeQuotes(value: string) {
  return value.split(`"`).join(`""`);
}

interface ExportSectionProps {
  control: ControlForm;
  library: LibraryRow[];
  selected: SelectedMolecule | null;
  selectedRound: number;
  notebookEntries: ChemistNotebookEntry[];
  autosaveEnabled: boolean;
  autosaveSavedAt: string | null;
  onToggleAutosave: (value: boolean) => void;
  onRestoreAutosave: () => void;
  onClearAutosave: () => void;
}

function normalizeStatus(value: unknown): LibraryRow["status"] {
  if (value === "promovata" || value === "revizie" || value === "respinsa") {
    return value;
  }
  return "necunoscut";
}

function rowFromSelected(selected: SelectedMolecule): LibraryRow {
  const pic50Metric = selected.metrics.primary.find(
    (item) => item.label === "pIC50" && typeof item.value === "number",
  );
  const qedMetric = selected.metrics.primary.find(
    (item) => item.label === "QED" && typeof item.value === "number",
  );
  const uncertaintyMetric = selected.metrics.primary.find(
    (item) => item.label === "Incertitudine" && typeof item.value === "number",
  );

  return {
    id: `selected-${selected.rank}-${selected.smiles}`,
    rank: selected.rank,
    smiles: selected.smiles,
    parent: selected.parent,
    round: selected.round,
    status: selected.status,
    statusLabel: selected.status,
    score: selected.score,
    pic50: typeof pic50Metric?.value === "number" ? pic50Metric.value : 0,
    qed: typeof qedMetric?.value === "number" ? qedMetric.value : 0,
    uncertainty: typeof uncertaintyMetric?.value === "number" ? uncertaintyMetric.value : 0,
    cost10mg: selected.cost10mg,
    action: selected.action,
    route: selected.route,
    marketReference: selected.marketReference,
  };
}

function rowsToCsv(rows: LibraryRow[]) {
  const headers = [
    "rank",
    "smiles",
    "parent",
    "round",
    "status",
    "statusLabel",
    "score",
    "pic50",
    "qed",
    "uncertainty",
    "cost10mg",
    "action",
    "route",
    "marketReference",
  ];

  const escapeCell = (value: unknown) => `"${escapeQuotes(String(value ?? ""))}"`;
  const lines = rows.map((row) =>
    [
      row.rank,
      row.smiles,
      row.parent,
      row.round,
      row.status,
      row.statusLabel,
      row.score,
      row.pic50,
      row.qed,
      row.uncertainty,
      row.cost10mg,
      row.action,
      row.route,
      row.marketReference,
    ]
      .map(escapeCell)
      .join(","),
  );

  return [headers.join(","), ...lines].join("\n");
}

function parseCsv(text: string) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) {
    return [];
  }
  const headers = lines[0].split(",").map((item) => item.replace(/^"|"$/g, "").trim());
  return lines.slice(1).map((line, index) => {
    const rawValues = line.match(/(".*?"|[^",]+)(?=\s*,|\s*$)/g) ?? [];
    const record = Object.fromEntries(
      headers.map((header, headerIndex) => [
        header,
        rawValues[headerIndex]?.replace(/^"|"$/g, "").split(`""`).join(`"`).trim() ?? "",
      ]),
    );
    return {
      id: `import-${index}-${record.smiles ?? ""}`,
      rank: Number(record.rank ?? index + 1),
      smiles: String(record.smiles ?? ""),
      parent: String(record.parent ?? ""),
      round: Number(record.round ?? 0),
      status: normalizeStatus(record.status),
      statusLabel: String(record.statusLabel ?? record.status ?? "necunoscut"),
      score: Number(record.score ?? 0),
      pic50: Number(record.pic50 ?? 0),
      qed: Number(record.qed ?? 0),
      uncertainty: Number(record.uncertainty ?? 0),
      cost10mg: Number(record.cost10mg ?? 0),
      action: String(record.action ?? ""),
      route: String(record.route ?? ""),
      marketReference: String(record.marketReference ?? ""),
    } satisfies LibraryRow;
  });
}

function parseJsonRows(parsed: unknown): LibraryRow[] {
  const rawRows = Array.isArray(parsed)
    ? parsed
    : Array.isArray((parsed as { library?: unknown[] })?.library)
      ? (parsed as { library: unknown[] }).library
      : Array.isArray((parsed as { dashboard?: { library?: unknown[] } })?.dashboard?.library)
        ? (parsed as { dashboard: { library: unknown[] } }).dashboard.library
        : Array.isArray((parsed as { molecules?: unknown[] })?.molecules)
          ? (parsed as { molecules: unknown[] }).molecules
          : [];

  return rawRows.map((rawRow, index) => {
    const row = rawRow as Partial<LibraryRow>;
    return {
      id: String(row.id ?? `import-${index}-${row.smiles ?? ""}`),
      rank: Number(row.rank ?? index + 1),
      smiles: String(row.smiles ?? ""),
      parent: String(row.parent ?? ""),
      round: Number(row.round ?? 0),
      status: normalizeStatus(row.status),
      statusLabel: String(row.statusLabel ?? row.status ?? "necunoscut"),
      score: Number(row.score ?? 0),
      pic50: Number(row.pic50 ?? 0),
      qed: Number(row.qed ?? 0),
      uncertainty: Number(row.uncertainty ?? 0),
      cost10mg: Number(row.cost10mg ?? 0),
      action: String(row.action ?? ""),
      route: String(row.route ?? ""),
      marketReference: String(row.marketReference ?? ""),
    } satisfies LibraryRow;
  });
}

function saveBlob(filename: string, body: string, mime: string) {
  const blob = new Blob([body], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function ExportSection({
  control,
  library,
  selected,
  selectedRound,
  notebookEntries,
  autosaveEnabled,
  autosaveSavedAt,
  onToggleAutosave,
  onRestoreAutosave,
  onClearAutosave,
}: ExportSectionProps) {
  const importRef = useRef<HTMLInputElement | null>(null);
  const [exportScope, setExportScope] = useState<ExportScope>("top");
  const [exportFormat, setExportFormat] = useState<ExportFormat>("json");
  const [exportLimit, setExportLimit] = useState(12);
  const [importLimit, setImportLimit] = useState(12);
  const [importedRows, setImportedRows] = useState<LibraryRow[]>([]);
  const [importError, setImportError] = useState<string | null>(null);

  const exportRows = useMemo(() => {
    const reviewSmiles = new Set(
      notebookEntries
        .filter((entry) => entry.tags.some((tag) => ["pinat", "aprobat", "retestare"].includes(tag)))
        .map((entry) => entry.smiles),
    );
    switch (exportScope) {
      case "selectata":
        return selected ? [rowFromSelected(selected)] : [];
      case "promovate":
        return library.filter((item) => item.status === "promovata").slice(0, exportLimit);
      case "review":
        return library.filter((item) => reviewSmiles.has(item.smiles)).slice(0, exportLimit);
      case "runda":
        return library.filter((item) => item.round === selectedRound).slice(0, exportLimit);
      case "importate":
        return importedRows.slice(0, exportLimit);
      case "top":
      default:
        return library.slice(0, exportLimit);
    }
  }, [exportLimit, exportScope, importedRows, library, notebookEntries, selected, selectedRound]);

  async function handleImportFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const text = await file.text();
      const parsedRows = file.name.toLowerCase().endsWith(".csv")
        ? parseCsv(text)
        : parseJsonRows(JSON.parse(text));
      setImportedRows(parsedRows.slice(0, importLimit));
      setImportError(null);
    } catch (caught) {
      setImportError(caught instanceof Error ? caught.message : "Fisierul de import nu a putut fi citit.");
    } finally {
      event.target.value = "";
    }
  }

  function handleExportSubset() {
    if (!exportRows.length) {
      return;
    }
    const filename = `${control.sessionName}-${exportScope}-${exportRows.length}.${exportFormat}`;
    if (exportFormat === "json") {
      saveBlob(filename, JSON.stringify({ sessionName: control.sessionName, exportedAt: new Date().toISOString(), library: exportRows }, null, 2), "application/json");
      return;
    }
    saveBlob(filename, rowsToCsv(exportRows), "text/csv;charset=utf-8");
  }

  return (
    <div className="space-y-4">
      <input ref={importRef} type="file" accept=".json,.csv" className="hidden" onChange={handleImportFile} />

      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1.05fr)_420px]">
        <SectionCard
          eyebrow="Export inteligent"
          title="Subseturi de molecule"
          subtitle="Poti exporta doar moleculele care te intereseaza, fara sa descarci toata biblioteca."
        >
          <div className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                {[
                  { key: "selectata", label: "Selectata" },
                  { key: "top", label: "Top scor" },
                  { key: "promovate", label: "Promovate" },
                  { key: "review", label: "Revizie" },
                  { key: "runda", label: "Runda curenta" },
                  { key: "importate", label: "Importate" },
                ].map((option) => (
                <button
                  key={option.key}
                  className={`rounded-2xl border px-4 py-3 text-left transition ${
                    exportScope === option.key
                      ? "border-forge-cyan/50 bg-forge-cyan/12 text-white"
                      : "border-white/8 bg-white/5 text-slate-300 hover:bg-white/10"
                  }`}
                  onClick={() => setExportScope(option.key as ExportScope)}
                >
                  <span className="text-sm font-semibold">{option.label}</span>
                </button>
              ))}
            </div>

            <div className="grid gap-3 md:grid-cols-3">
              <label className="field-card">
                <span>Format export</span>
                <select
                  value={exportFormat}
                  onChange={(event) => setExportFormat(event.target.value as ExportFormat)}
                  className="rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-white outline-none"
                >
                  <option value="json">JSON</option>
                  <option value="csv">CSV</option>
                </select>
              </label>

              <label className="field-card">
                <span>Numar molecule</span>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={exportLimit}
                  onChange={(event) => setExportLimit(Math.max(1, Math.min(200, Number(event.target.value) || 1)))}
                  className="rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-white outline-none"
                />
              </label>

              <div className="flex items-end">
                <button
                  className="control-button control-button-primary w-full disabled:opacity-60"
                  onClick={handleExportSubset}
                  disabled={!exportRows.length}
                >
                  Exporta subsetul
                </button>
              </div>
            </div>

            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Previzualizare export</p>
              {exportRows.length ? (
                <div className="mt-3 space-y-2">
                  {exportRows.slice(0, 6).map((row) => (
                    <div key={row.id} className="rounded-2xl border border-white/6 bg-white/5 px-4 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm font-semibold text-white">#{row.rank}</span>
                        <span className="text-xs text-slate-400">Runda {row.round}</span>
                      </div>
                      <p className="mt-2 break-all font-mono text-xs text-slate-300">{row.smiles}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-300">Nu exista molecule in subsetul selectat.</p>
              )}
            </div>
          </div>
        </SectionCard>

        <SectionCard
          eyebrow="Salvare automata"
          title="Draft local al sesiunii"
          subtitle="Setarile interfetei si selectia curenta pot fi salvate automat in navigator."
          className="h-full"
        >
          <div className="space-y-4">
            <label className="flex items-center gap-3 rounded-2xl border border-white/6 bg-white/5 px-4 py-3 text-sm text-slate-200">
              <input
                type="checkbox"
                checked={autosaveEnabled}
                onChange={(event) => onToggleAutosave(event.target.checked)}
                className="h-4 w-4 rounded border-white/10 bg-slate-950/80"
              />
              <div>
                <p className="font-semibold text-white">Salvare automata activa</p>
                <p className="mt-1 text-xs leading-6 text-slate-400">
                  Salveaza local parametrii sesiunii, sectiunea deschisa si molecula selectata.
                </p>
              </div>
            </label>

            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Ultima salvare automata</p>
              <p className="mt-2 text-lg font-semibold text-white">
                {autosaveSavedAt ? new Date(autosaveSavedAt).toLocaleTimeString("ro-RO") : "Inca nu exista un draft local"}
              </p>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <button className="control-button" onClick={onRestoreAutosave}>
                Restaureaza draft
              </button>
              <button className="control-button" onClick={onClearAutosave}>
                Sterge draftul local
              </button>
            </div>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        eyebrow="Import partial"
        title="Incarca doar o parte din molecule"
        subtitle="Poti importa un snapshot JSON sau CSV si limita cate molecule intra in sesiunea de lucru curenta."
      >
        <div className="grid gap-4 xl:grid-cols-[280px_minmax(0,1fr)]">
          <div className="space-y-3">
            <label className="field-card">
              <span>Maxim molecule importate</span>
              <input
                type="number"
                min={1}
                max={200}
                value={importLimit}
                onChange={(event) => setImportLimit(Math.max(1, Math.min(200, Number(event.target.value) || 1)))}
                className="rounded-xl border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-white outline-none"
              />
            </label>

            <button className="control-button control-button-primary w-full" onClick={() => importRef.current?.click()}>
              Importa fisier subset
            </button>

            {importError ? (
              <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-100">
                {importError}
              </div>
            ) : null}
          </div>

          <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Previzualizare import</p>
            {importedRows.length ? (
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                {importedRows.map((row) => (
                  <div key={row.id} className="rounded-2xl border border-white/6 bg-white/5 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm font-semibold text-white">#{row.rank}</span>
                      <span className="text-xs text-slate-400">{row.statusLabel}</span>
                    </div>
                    <p className="mt-2 break-all font-mono text-xs text-slate-300">{row.smiles}</p>
                    <p className="mt-2 text-xs text-slate-400">
                      scor {row.score.toFixed(2)} | pIC50 {row.pic50.toFixed(2)} | cost ${row.cost10mg.toFixed(2)}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-3 text-sm text-slate-300">
                Dupa import, aici vezi doar primele molecule selectate conform limitei setate.
              </p>
            )}
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
