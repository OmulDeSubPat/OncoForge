import { type ChangeEvent, startTransition, useEffect, useMemo, useRef, useState } from "react";

import { fetchDashboard, resetSession, startSession, stopSession } from "@/api";
import { ActivitySection } from "@/components/ActivitySection";
import { CompareTray } from "@/components/CompareTray";
import { CompareSection } from "@/components/CompareSection";
import { DecisionWorkbench } from "@/components/DecisionWorkbench";
import { ExportSection } from "@/components/ExportSection";
import { GlobalSearchBar } from "@/components/GlobalSearchBar";
import { InsightSection } from "@/components/InsightSection";
import { LabNavigation, type LabSectionKey } from "@/components/LabNavigation";
import { LibrarySection } from "@/components/LibrarySection";
import { LiveHeroStrip } from "@/components/LiveHeroStrip";
import { OverviewSection } from "@/components/OverviewSection";
import { PlanningSection } from "@/components/PlanningSection";
import { SectionCard } from "@/components/SectionCard";
import { TopBar } from "@/components/TopBar";
import { TrainingSection } from "@/components/TrainingSection";
import { TriageSection } from "@/components/TriageSection";
import type { ChemistNotebookEntry, ControlForm, DashboardPayload, DecisionEvent, OverviewPayload } from "@/types";

function fallbackOverview(sessionName: string): OverviewPayload {
  return {
    sessionName,
    mode: "ghidat_ai",
    modeLabel: "Generare ghidata",
    status: "pregatire",
    statusLabel: "Pregatire",
    message: "Platforma este pregatita pentru prima sincronizare cu backend-ul.",
    updatedAt: "",
    running: false,
    progress: 0,
    summary: {
      moleculeCount: 0,
      promotedCount: 0,
      reviewCount: 0,
      rejectedCount: 0,
      bestPic50: 0,
      bestScore: 0,
      meanQed: 0,
    },
    bestMolecule: null,
    latestRound: null,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

const defaultControl: ControlForm = {
  sessionName: "sesiune_curenta",
  mode: "ghidat_ai",
  seedCount: 3,
  rounds: 3,
  variantsPerSeed: 12,
  beamWidth: 8,
  replaceExisting: true,
};

type UiMode = "basic" | "expert";

function notebookStorageKey(sessionName: string) {
  return `oncosynth.notebook.${sessionName}`;
}

function historyStorageKey(sessionName: string) {
  return `oncosynth.history.${sessionName}`;
}

export default function App() {
  const [control, setControl] = useState<ControlForm>(defaultControl);
  const [dashboard, setDashboard] = useState<DashboardPayload | null>(null);
  const [selectedSmiles, setSelectedSmiles] = useState<string | undefined>(undefined);
  const [selectedRound, setSelectedRound] = useState(1);
  const [activeSection, setActiveSection] = useState<LabSectionKey>("rezumat");
  const [uiMode, setUiMode] = useState<UiMode>("basic");
  const [compareSmiles, setCompareSmiles] = useState<string[]>([]);
  const [searchExpanded, setSearchExpanded] = useState(false);
  const [compareExpanded, setCompareExpanded] = useState(false);
  const [globalSearch, setGlobalSearch] = useState("");
  const [autosaveEnabled, setAutosaveEnabled] = useState(true);
  const [autosaveSavedAt, setAutosaveSavedAt] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [notesBySmiles, setNotesBySmiles] = useState<Record<string, ChemistNotebookEntry>>({});
  const [localHistory, setLocalHistory] = useState<Record<string, DecisionEvent[]>>({});
  const importRef = useRef<HTMLInputElement | null>(null);
  const selectedSmilesRef = useRef<string | undefined>(undefined);

  const overview = dashboard?.overview ?? fallbackOverview(control.sessionName);
  const selected = dashboard?.detail.selected ?? null;
  const library = dashboard?.library ?? [];
  const agents = dashboard?.agents ?? [];
  const flows = dashboard?.flows ?? [];
  const timeline = dashboard?.timeline ?? { generations: [], nodes: [], edges: [] };
  const rlMonitor = dashboard?.rlMonitor ?? { rewardSeries: [], penaltySeries: [], verifiableNotes: [] };
  const marketCompare = dashboard?.marketCompare ?? { candidateSmiles: "", axes: [], entries: [] };
  const sessionCompare = dashboard?.sessionCompare ?? [];
  const experimentalPlanner = dashboard?.experimentalPlanner ?? [];
  const analytics = {
    agentSeries: dashboard?.analytics?.agentSeries ?? [],
    rankingStability: dashboard?.analytics?.rankingStability ?? [],
    pipelineStages: dashboard?.analytics?.pipelineStages ?? [],
    pipelineProgress: dashboard?.analytics?.pipelineProgress ?? [],
    maturationSeries: dashboard?.analytics?.maturationSeries ?? [],
  };
  const notebookEntries = useMemo(() => Object.values(notesBySmiles), [notesBySmiles]);

  useEffect(() => {
    selectedSmilesRef.current = selectedSmiles;
  }, [selectedSmiles]);

  useEffect(() => {
    try {
      const rawDraft = window.localStorage.getItem(`oncosynth.autosave.${control.sessionName}`);
      if (!rawDraft) {
        setAutosaveSavedAt(null);
        return;
      }
      const parsed = JSON.parse(rawDraft) as { savedAt?: string };
      setAutosaveSavedAt(parsed.savedAt ?? null);
    } catch {
      setAutosaveSavedAt(null);
    }
  }, [control.sessionName]);

  useEffect(() => {
    if (!autosaveEnabled) {
      return;
    }

    const savedAt = new Date().toISOString();
    const payload = {
      savedAt,
      control,
      selectedSmiles,
      selectedRound,
      activeSection,
      uiMode,
      compareSmiles,
      searchExpanded,
      compareExpanded,
    };
    try {
      window.localStorage.setItem(`oncosynth.autosave.${control.sessionName}`, JSON.stringify(payload));
      setAutosaveSavedAt(savedAt);
    } catch {
      // Ignore local storage errors and keep the UI responsive.
    }
  }, [activeSection, autosaveEnabled, compareExpanded, compareSmiles, control, searchExpanded, selectedRound, selectedSmiles, uiMode]);

  useEffect(() => {
    try {
      const rawNotebook = window.localStorage.getItem(notebookStorageKey(control.sessionName));
      setNotesBySmiles(rawNotebook ? (JSON.parse(rawNotebook) as Record<string, ChemistNotebookEntry>) : {});
    } catch {
      setNotesBySmiles({});
    }

    try {
      const rawHistory = window.localStorage.getItem(historyStorageKey(control.sessionName));
      setLocalHistory(rawHistory ? (JSON.parse(rawHistory) as Record<string, DecisionEvent[]>) : {});
    } catch {
      setLocalHistory({});
    }
  }, [control.sessionName]);

  useEffect(() => {
    try {
      window.localStorage.setItem(notebookStorageKey(control.sessionName), JSON.stringify(notesBySmiles));
    } catch {
      // Ignore local storage failures and keep the UI responsive.
    }
  }, [control.sessionName, notesBySmiles]);

  useEffect(() => {
    try {
      window.localStorage.setItem(historyStorageKey(control.sessionName), JSON.stringify(localHistory));
    } catch {
      // Ignore local storage failures and keep the UI responsive.
    }
  }, [control.sessionName, localHistory]);

  async function loadDashboard(options?: { forceSmiles?: string; silent?: boolean }) {
    if (!options?.silent) {
      setLoading(true);
    }

    try {
      const activeSmiles = options?.forceSmiles ?? selectedSmilesRef.current;
      const payload = await fetchDashboard({
        sessionName: control.sessionName,
        limit: 120,
        smiles: activeSmiles,
      });

      setDashboard(payload);

      const nextSelected = activeSmiles ?? payload.detail.selected?.smiles;
      if (nextSelected && nextSelected !== selectedSmilesRef.current) {
        startTransition(() => setSelectedSmiles(nextSelected));
      }
      if (payload.detail.selected?.round) {
        setSelectedRound(payload.detail.selected.round);
      }
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Sincronizarea cu backend-ul a esuat.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadDashboard({ forceSmiles: selectedSmilesRef.current });
  }, [control.sessionName]);

  useEffect(() => {
    const intervalMs = dashboard?.overview.running
      ? ["rezumat", "triere", "laborator", "antrenare"].includes(activeSection)
        ? 7000
        : 12000
      : activeSection === "activitate"
        ? 18000
        : 30000;
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "visible" || busyAction) {
        return;
      }
      void loadDashboard({ forceSmiles: selectedSmilesRef.current, silent: true });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [activeSection, busyAction, control.sessionName, dashboard?.overview.running]);

  useEffect(() => {
    if (!selectedSmiles) {
      return;
    }
    if (dashboard?.detail.selected?.smiles === selectedSmiles) {
      return;
    }
    void loadDashboard({ forceSmiles: selectedSmiles, silent: true });
  }, [control.sessionName, dashboard?.detail.selected?.smiles, selectedSmiles]);

  useEffect(() => {
    if (globalSearch.trim()) {
      setSearchExpanded(true);
    }
  }, [globalSearch]);

  useEffect(() => {
    if (compareSmiles.length) {
      setCompareExpanded(true);
      return;
    }
    setCompareExpanded(false);
  }, [compareSmiles.length]);

  async function runAction(name: string, action: () => Promise<unknown>) {
    setBusyAction(name);
    try {
      await action();
      await loadDashboard({ forceSmiles: selectedSmilesRef.current });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : `Actiunea ${name} a esuat.`);
    } finally {
      setBusyAction(null);
    }
  }

  function handleFieldChange<K extends keyof ControlForm>(key: K, value: ControlForm[K]) {
    if (typeof value === "number") {
      const limits: Partial<Record<keyof ControlForm, [number, number]>> = {
        seedCount: [1, 20],
        rounds: [1, 12],
        variantsPerSeed: [1, 64],
        beamWidth: [1, 32],
      };
      const range = limits[key];
      if (range) {
        setControl((current) => ({ ...current, [key]: clamp(value, range[0], range[1]) as ControlForm[K] }));
        return;
      }
    }
    setControl((current) => ({ ...current, [key]: value }));
  }

  function appendLocalEvent(smiles: string, event: Omit<DecisionEvent, "id">) {
    setLocalHistory((current) => {
      const nextEntry: DecisionEvent = {
        ...event,
        id: `${event.category}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      };
      return {
        ...current,
        [smiles]: [...(current[smiles] ?? []), nextEntry],
      };
    });
  }

  function handleNotebookChange(entry: ChemistNotebookEntry) {
    setNotesBySmiles((current) => ({
      ...current,
      [entry.smiles]: entry,
    }));
    appendLocalEvent(entry.smiles, {
      title: "Nota chimistului",
      detail: `${entry.verdict}${entry.note ? ` | ${entry.note}` : ""}`,
      timestamp: new Date().toLocaleString("ro-RO"),
      category: "notebook",
      tone: "info",
      round: selected?.round,
    });
  }

  function handleExport() {
    const payload = JSON.stringify(
      {
        control,
        selectedSmiles: selectedSmilesRef.current,
        exportedAt: new Date().toISOString(),
        dashboard,
      },
      null,
      2,
    );
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${control.sessionName}-oncosynth-snapshot.json`;
    link.click();
    URL.revokeObjectURL(url);
    if (selectedSmilesRef.current) {
      appendLocalEvent(selectedSmilesRef.current, {
        title: "Export snapshot",
        detail: "Molecula curenta a fost inclusa intr-un export local al sesiunii.",
        timestamp: new Date().toLocaleString("ro-RO"),
        category: "export",
        tone: "success",
        round: selected?.round,
      });
    }
  }

  function handleImportClick() {
    importRef.current?.click();
  }

  async function handleImportFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Partial<{ control: Partial<ControlForm>; selectedSmiles: string }>;
      if (parsed.control) {
        setControl((current) => ({
          ...current,
          ...parsed.control,
        }));
      }
      if (parsed.selectedSmiles) {
        startTransition(() => setSelectedSmiles(parsed.selectedSmiles));
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Fisierul de configuratie nu a putut fi importat.");
    } finally {
      event.target.value = "";
    }
  }

  function handleRestoreAutosave() {
    try {
      const rawDraft = window.localStorage.getItem(`oncosynth.autosave.${control.sessionName}`);
      if (!rawDraft) {
        return;
      }
      const parsed = JSON.parse(rawDraft) as Partial<{
        control: ControlForm;
        selectedSmiles: string;
        selectedRound: number;
        activeSection: LabSectionKey;
        uiMode: UiMode;
        compareSmiles: string[];
        searchExpanded: boolean;
        compareExpanded: boolean;
        savedAt: string;
      }>;
      if (parsed.control) {
        setControl(parsed.control);
      }
      if (parsed.selectedSmiles) {
        startTransition(() => setSelectedSmiles(parsed.selectedSmiles));
      }
      if (parsed.selectedRound) {
        setSelectedRound(parsed.selectedRound);
      }
      if (parsed.activeSection) {
        setActiveSection(parsed.activeSection);
      }
      if (parsed.uiMode) {
        setUiMode(parsed.uiMode);
      }
      if (parsed.compareSmiles) {
        setCompareSmiles(parsed.compareSmiles);
      }
      if (typeof parsed.searchExpanded === "boolean") {
        setSearchExpanded(parsed.searchExpanded);
      }
      if (typeof parsed.compareExpanded === "boolean") {
        setCompareExpanded(parsed.compareExpanded);
      }
      setAutosaveSavedAt(parsed.savedAt ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Draftul local nu a putut fi restaurat.");
    }
  }

  function handleClearAutosave() {
    try {
      window.localStorage.removeItem(`oncosynth.autosave.${control.sessionName}`);
      setAutosaveSavedAt(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Draftul local nu a putut fi sters.");
    }
  }

  async function handleStart() {
    await runAction("start", async () => {
      await startSession(control);
      setSelectedSmiles(undefined);
      setCompareSmiles([]);
      setGlobalSearch("");
      setSearchExpanded(false);
      setActiveSection("rezumat");
    });
  }

  async function handleStop() {
    await runAction("stop", async () => {
      await stopSession(control.sessionName);
    });
  }

  async function handleReset() {
    await runAction("reset", async () => {
      await resetSession(control.sessionName);
      setSelectedSmiles(undefined);
      setCompareSmiles([]);
      setGlobalSearch("");
      setSearchExpanded(false);
      setActiveSection("rezumat");
    });
  }

  function handleSelectCandidate(smiles: string) {
    startTransition(() => setSelectedSmiles(smiles));
    setActiveSection("laborator");
    setGlobalSearch("");
    void loadDashboard({ forceSmiles: smiles, silent: true });
  }

  function handleToggleCompare(smiles: string) {
    setCompareSmiles((current) => {
      if (current.includes(smiles)) {
        return current.filter((entry) => entry !== smiles);
      }
      const next = Array.from(new Set([...(selectedSmilesRef.current ? [selectedSmilesRef.current] : []), ...current, smiles]));
      return next.slice(0, 4);
    });
  }

  function handleClearCompare() {
    setCompareSmiles(selectedSmilesRef.current ? [selectedSmilesRef.current] : []);
  }

  function handleOpenCompare() {
    if (selectedSmilesRef.current) {
      setCompareSmiles((current) => Array.from(new Set([selectedSmilesRef.current as string, ...current])).slice(0, 4));
    }
    setActiveSection("comparatie");
  }

  const showConnectionState = !dashboard && !!error;
  const showLoadingState = !dashboard && loading;
  const reviewCount = useMemo(
    () =>
      notebookEntries.filter((entry) =>
        entry.tags.some((tag) => ["pinat", "aprobat", "respins_local", "retestare"].includes(tag)),
      ).length,
    [notebookEntries],
  );

  const navigationItems = useMemo(
    () => [
      {
        key: "rezumat" as const,
        label: "Sesiune",
        description: "Pornire, parametri, status si rezumatul lotului curent.",
        badge: `${overview.summary.moleculeCount} molecule`,
      },
      {
        key: "triere" as const,
        label: "Triere",
        description: "Reponderare interactiva, harta pe generatii si grafice de incredere.",
        badge: `${analytics.rankingStability.length} runde`,
      },
      {
        key: "laborator" as const,
        label: "Molecula",
        description: "Vizualizare 2D/3D, fisa completa si schimbarea rapida intre candidati.",
        badge: selected ? `#${selected.rank}` : "fara selectie",
      },
      {
        key: "comparatie" as const,
        label: "Comparatie",
        description: "Candidat versus Osimertinib, Gefitinib, Erlotinib si comparatie multipla.",
        badge: `${marketCompare.entries.length} intrari`,
      },
      {
        key: "insight" as const,
        label: "Risc si explicatii",
        description: "De ce a fost promovata, semnale de risc, praguri si carnetul chimistului.",
        badge: selected ? "molecula activa" : "asteapta selectie",
      },
      {
        key: "antrenare" as const,
        label: "Audit IA",
        description: "RLVR, cronologie iterativa, agenti si convergenta recompensei.",
        badge: `${rlMonitor.penaltySeries.length} runde`,
      },
      {
        key: "planificare" as const,
        label: "Planificare",
        description: "Planner experimental, comparatie intre sesiuni si sinteza rapida.",
        badge: `${experimentalPlanner.length} candidati`,
      },
      {
        key: "biblioteca" as const,
        label: "Biblioteca",
        description: "Toate moleculele generate, cu filtre si selectie directa.",
        badge: `${library.length} intrari`,
      },
      {
        key: "export" as const,
        label: "Export",
        description: "Subseturi de molecule, import partial si autosave local.",
        badge: autosaveEnabled ? "salvare automata activa" : "salvare automata oprita",
      },
      {
        key: "activitate" as const,
        label: "Jurnal",
        description: "Loguri, surse si mesaje operationale pentru audit sau depanare.",
        badge: `${(dashboard?.sources ?? []).length} surse`,
      },
    ],
    [
      autosaveEnabled,
      analytics.rankingStability.length,
      dashboard?.sources,
      experimentalPlanner.length,
      library.length,
      marketCompare.entries.length,
      rlMonitor.penaltySeries.length,
      selected,
      overview.summary.moleculeCount,
    ],
  );
  const visibleSectionKeys = useMemo(
    () =>
      uiMode === "basic"
        ? new Set<LabSectionKey>(["rezumat", "triere", "laborator", "comparatie", "insight", "biblioteca"])
        : new Set<LabSectionKey>(["rezumat", "triere", "laborator", "comparatie", "insight", "antrenare", "planificare", "biblioteca", "export", "activitate"]),
    [uiMode],
  );
  const activeNavigationItems = useMemo(
    () => navigationItems.filter((item) => visibleSectionKeys.has(item.key)),
    [navigationItems, visibleSectionKeys],
  );

  useEffect(() => {
    if (!visibleSectionKeys.has(activeSection)) {
      setActiveSection("rezumat");
    }
  }, [activeSection, visibleSectionKeys]);

  function renderActiveSection() {
    if (showConnectionState || showLoadingState) {
      return null;
    }

    switch (activeSection) {
      case "rezumat":
        return (
          <OverviewSection
            overview={overview}
            control={control}
            library={library}
            timeline={timeline}
            analytics={analytics}
            onFieldChange={handleFieldChange}
          />
        );
      case "triere":
        return (
          <TriageSection
            library={library}
            monitor={rlMonitor}
            analytics={analytics}
            selectedSmiles={selectedSmiles ?? null}
            onSelectCandidate={handleSelectCandidate}
          />
        );
      case "laborator":
        return (
          <DecisionWorkbench
            selected={selected}
            library={library}
            agents={agents}
            flows={flows}
            compareSmiles={compareSmiles}
            uiMode={uiMode}
            onSelectCandidate={handleSelectCandidate}
            onToggleCompare={handleToggleCompare}
            onOpenCompare={handleOpenCompare}
          />
        );
      case "comparatie":
        return (
          <CompareSection
            sessionName={control.sessionName}
            selected={selected}
            marketCompare={marketCompare}
            library={library}
            compareSmiles={compareSmiles}
            onSelectCandidate={handleSelectCandidate}
            onCompareSmilesChange={setCompareSmiles}
          />
        );
      case "insight":
        return (
          <InsightSection
            selected={selected}
            notebookEntry={selected ? notesBySmiles[selected.smiles] ?? null : null}
            localHistory={selected ? localHistory[selected.smiles] ?? [] : []}
            onNotebookChange={handleNotebookChange}
          />
        );
      case "antrenare":
        return (
          <TrainingSection
            library={library}
            timeline={timeline}
            monitor={rlMonitor}
            selected={selected}
            selectedRound={selectedRound}
            onJump={(round, candidateSmiles) => {
              setSelectedRound(round);
              if (candidateSmiles) {
                handleSelectCandidate(candidateSmiles);
                return;
              }
              const fallbackCandidate = library.find((item) => item.round === round);
              if (fallbackCandidate) {
                handleSelectCandidate(fallbackCandidate.smiles);
              }
            }}
          />
        );
      case "planificare":
        return (
          <PlanningSection
            planner={experimentalPlanner}
            sessionCompare={sessionCompare}
            notebookEntries={notebookEntries}
          />
        );
      case "biblioteca":
        return (
          <LibrarySection
            library={library}
            selectedSmiles={selectedSmiles ?? null}
            uiMode={uiMode}
            compareSmiles={compareSmiles}
            onSelectCandidate={handleSelectCandidate}
            onToggleCompare={handleToggleCompare}
          />
        );
      case "export":
        return (
          <ExportSection
            control={control}
            library={library}
            selected={selected}
            selectedRound={selectedRound}
            notebookEntries={notebookEntries}
            autosaveEnabled={autosaveEnabled}
            autosaveSavedAt={autosaveSavedAt}
            onToggleAutosave={setAutosaveEnabled}
            onRestoreAutosave={handleRestoreAutosave}
            onClearAutosave={handleClearAutosave}
          />
        );
      case "activitate":
        return (
          <ActivitySection
            overview={overview}
            logs={dashboard?.logs ?? ""}
            sources={dashboard?.sources ?? []}
          />
        );
      default:
        return null;
    }
  }

  return (
    <div className="relative min-h-full overflow-hidden bg-forge-bg text-forge-text">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(36,214,234,0.18),transparent_26%),radial-gradient(circle_at_top_right,rgba(64,217,143,0.1),transparent_22%),radial-gradient(circle_at_bottom,rgba(115,166,255,0.12),transparent_30%),linear-gradient(180deg,rgba(3,8,18,0.18),transparent_35%)]" />
      <div className="pointer-events-none absolute inset-0 bg-grid-fine opacity-[0.09]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0,transparent_55%,rgba(2,6,23,0.42)_100%)]" />
      <div className="pointer-events-none absolute left-16 top-20 h-44 w-44 rounded-full bg-forge-cyan/10 blur-3xl motion-safe:animate-drift" />
      <div className="pointer-events-none absolute right-12 top-44 h-56 w-56 rounded-full bg-forge-green/10 blur-3xl motion-safe:animate-drift" />
      <div className="pointer-events-none absolute bottom-16 left-1/3 h-48 w-48 rounded-full bg-forge-blue/10 blur-3xl motion-safe:animate-drift" />

      <input ref={importRef} type="file" accept="application/json" className="hidden" onChange={handleImportFile} />

      <main className="relative mx-auto max-w-[1880px] px-4 py-4 sm:px-5 lg:px-6 lg:py-6">
        <TopBar
          overview={overview}
          busyAction={busyAction}
          uiMode={uiMode}
          compareCount={compareSmiles.length}
          reviewCount={reviewCount}
          onStart={handleStart}
          onStop={handleStop}
          onReset={handleReset}
          onRefresh={() => void loadDashboard({ forceSmiles: selectedSmilesRef.current })}
          onExport={handleExport}
          onImportClick={handleImportClick}
          onUiModeChange={setUiMode}
        />

        {error ? (
          <div className="mt-4 rounded-3xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            {error}
          </div>
        ) : null}

        {showLoadingState ? (
          <SectionCard
            eyebrow="Pornire platforma"
            title="Se sincronizeaza cu backend-ul"
            subtitle="OncoSynth incearca sa incarce sesiunea curenta si artefactele worker-ului."
            className="mt-4"
          >
            <div className="rounded-3xl border border-white/6 bg-slate-950/65 p-6 text-sm leading-7 text-slate-300">
              Asteapta cateva secunde. Daca starea ramane blocata, porneste backend-ul FastAPI si apasa `Actualizeaza`.
            </div>
          </SectionCard>
        ) : null}

        {showConnectionState ? (
          <SectionCard
            eyebrow="Conectare backend"
            title="OncoSynth nu poate ajunge inca la API"
            subtitle="Frontend-ul a pornit, dar backend-ul local nu raspunde. De aici vin mesajele pentru `/api/dashboard` si `/api/control/start`."
            className="mt-4"
          >
            <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
              <div className="rounded-3xl border border-white/6 bg-slate-950/70 p-5">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Pornire backend</p>
                <pre className="mt-3 overflow-auto rounded-2xl border border-white/5 bg-slate-950/90 p-4 font-mono text-xs leading-6 text-cyan-50">
{`cd "D:\\ONCS 2026\\egfr-drug-discovery-ml"
python -m uvicorn src.gui.oncoforge_api.app:app --host 127.0.0.1 --port 8000`}
                </pre>
                <p className="mt-4 text-sm leading-7 text-slate-300">
                  Dupa ce porneste serverul, apasa `Actualizeaza`. Frontend-ul foloseste direct `http://127.0.0.1:8000`.
                </p>
              </div>

              <div className="rounded-3xl border border-white/6 bg-slate-950/70 p-5">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Ce vei vedea dupa conectare</p>
                <div className="mt-4 space-y-3 text-sm text-slate-300">
                  <p>1. Rezumat separat pentru parametri si grafice de orientare.</p>
                  <p>2. Laborator molecular cu vizualizare 2D/3D si explicatii complete.</p>
                  <p>3. Monitor RLVR si biblioteca in pagini dedicate, mai usor de parcurs.</p>
                </div>
              </div>
            </div>
          </SectionCard>
        ) : null}

        {!showConnectionState && !showLoadingState ? (
          <>
            <LiveHeroStrip
              overview={overview}
              selected={selected}
              library={library}
              timeline={timeline}
              uiMode={uiMode}
              onOpenMolecule={() => setActiveSection("laborator")}
              onOpenLibrary={() => setActiveSection("biblioteca")}
              onSelectCandidate={handleSelectCandidate}
            />
            <LabNavigation items={activeNavigationItems} activeSection={activeSection} onChange={setActiveSection} />
            <GlobalSearchBar
              query={globalSearch}
              expanded={searchExpanded}
              onChange={setGlobalSearch}
              onToggle={() => setSearchExpanded((current) => !current)}
              sections={activeNavigationItems.map((item) => ({
                id: item.key,
                label: item.label,
                subtitle: item.description,
              }))}
              library={library}
              onOpenSection={(section) => {
                setActiveSection(section);
                setGlobalSearch("");
              }}
              onOpenMolecule={(smiles) => handleSelectCandidate(smiles)}
            />
            <CompareTray
              selected={selected}
              library={library}
              compareSmiles={compareSmiles}
              uiMode={uiMode}
              expanded={compareExpanded}
              onToggleCompare={handleToggleCompare}
              onOpenMolecule={handleSelectCandidate}
              onOpenCompare={handleOpenCompare}
              onClearCompare={handleClearCompare}
              onToggleExpanded={() => setCompareExpanded((current) => !current)}
            />
            <div className="mt-4">{renderActiveSection()}</div>
          </>
        ) : null}
      </main>
    </div>
  );
}
