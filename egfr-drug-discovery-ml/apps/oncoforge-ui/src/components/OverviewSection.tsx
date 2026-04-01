import { InteractiveBarChart, InteractiveLineChart, InteractiveScatterChart } from "@/components/InteractiveCharts";
import { SectionCard } from "@/components/SectionCard";
import { SessionSetupPanel } from "@/components/SessionSetupPanel";
import type { ControlForm, DashboardAnalyticsPayload, LibraryRow, OverviewPayload, TimelinePayload } from "@/types";

interface OverviewSectionProps {
  overview: OverviewPayload;
  control: ControlForm;
  library: LibraryRow[];
  timeline: TimelinePayload;
  analytics: DashboardAnalyticsPayload;
  onFieldChange: <K extends keyof ControlForm>(key: K, value: ControlForm[K]) => void;
}

function SummaryCard({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint: string;
  tone: string;
}) {
  return (
    <div className="glass-panel relative overflow-hidden p-4">
      <div className={`absolute inset-x-0 top-0 h-px bg-gradient-to-r ${tone}`} />
      <p className="ui-kicker">{label}</p>
      <p className="mt-3 text-2xl font-semibold text-white">{value}</p>
      <p className="mt-2 text-sm leading-7 text-slate-300">{hint}</p>
    </div>
  );
}

export function OverviewSection({ overview, control, library, timeline, analytics, onFieldChange }: OverviewSectionProps) {
  const rounds = timeline.generations.map((frame) => `R${frame.round}.${frame.seedStep}`);
  const roundSet = Array.from(new Set(library.map((item) => item.round))).sort((left, right) => left - right);
  const roundLabels = roundSet.map((round) => `R${round}`);
  const pipelineStageLabels = analytics.pipelineStages.map((entry) => entry.label);
  const pipelineRoundLabels = analytics.pipelineProgress.map((entry) => `R${entry.round}`);
  const maturationRoundLabels = analytics.maturationSeries.map((entry) => `R${entry.round}`);
  const statusCounts = {
    promovata: library.filter((item) => item.status === "promovata").length,
    revizie: library.filter((item) => item.status === "revizie").length,
    respinsa: library.filter((item) => item.status === "respinsa").length,
  };
  const meanPic50ByRound = roundSet.map((round) => {
    const items = library.filter((item) => item.round === round);
    return items.length ? items.reduce((sum, item) => sum + item.pic50, 0) / items.length : 0;
  });
  const meanQedByRound = roundSet.map((round) => {
    const items = library.filter((item) => item.round === round);
    return items.length ? items.reduce((sum, item) => sum + item.qed, 0) / items.length : 0;
  });
  const costAverageByRound = roundSet.map((round) => {
    const items = library.filter((item) => item.round === round);
    return items.length ? items.reduce((sum, item) => sum + item.cost10mg, 0) / items.length : 0;
  });
  const costMinimumByRound = roundSet.map((round) => {
    const items = library.filter((item) => item.round === round);
    return items.length ? Math.min(...items.map((item) => item.cost10mg)) : 0;
  });
  const topScatter = library.slice(0, 36).map((item) => ({
    label: `#${item.rank}`,
    x: item.cost10mg,
    y: item.pic50,
    color:
      item.status === "promovata" ? "#40d98f" : item.status === "revizie" ? "#f59e0b" : "#fb7185",
    size: item.smiles === overview.bestMolecule?.smiles ? 9 : 6,
    meta: `${item.marketReference || "Fara comparator"} | scor ${item.score.toFixed(2)}`,
  }));

  return (
    <div className="space-y-4">
      <SectionCard
        eyebrow="Prezentare operationala"
        title="Interpretarea sesiunii curente"
        subtitle="Panoul de rezumat este conceput pentru o evaluare structurata a sesiunii: status, trenduri si acces catre analiza moleculei active."
      >
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {[
            {
              title: "1. Verifica statusul",
              text: "Asigura-te ca sesiunea ruleaza si ca numarul de molecule creste de la o actualizare la alta.",
            },
            {
              title: "2. Urmareste trendurile",
              text: "Scorul, pIC50 si costul mediu iti spun daca biblioteca se imbunatateste sau nu.",
            },
            {
              title: "3. Deschide candidatul activ",
              text: "In Molecula activa vezi structura 2D/3D, proprietatile si motivele pentru selectie.",
            },
            {
              title: "4. Exporta lista prioritara",
              text: "Dupa triere, exporti doar subsetul care te intereseaza pentru analiza ulterioara.",
            },
          ].map((item) => (
            <div key={item.title} className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-sm font-semibold text-white">{item.title}</p>
              <p className="mt-2 text-sm leading-7 text-slate-300">{item.text}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryCard
          label="Status live"
          value={overview.running ? "Generare activa" : "Sesiune in asteptare"}
          hint="Pornire, stop si refresh raman in bara de control."
          tone="from-forge-cyan/80 via-forge-cyan/30 to-transparent"
        />
        <SummaryCard
          label="Best pIC50"
          value={overview.summary.bestPic50.toFixed(2)}
          hint="Cel mai puternic semnal curent din biblioteca."
          tone="from-forge-green/80 via-forge-green/30 to-transparent"
        />
        <SummaryCard
          label="Scor live maxim"
          value={overview.summary.bestScore.toFixed(2)}
          hint="Scor agregat dupa potenta, risc, cost si fezabilitate."
          tone="from-forge-blue/80 via-forge-blue/30 to-transparent"
        />
        <SummaryCard
          label="Molecule evaluate"
          value={String(overview.summary.moleculeCount)}
          hint="Numarul total de candidati disponibili pentru triere."
          tone="from-forge-amber/80 via-forge-amber/30 to-transparent"
        />
      </div>

      <div className="grid gap-4 2xl:grid-cols-[420px_minmax(0,1fr)]">
        <SessionSetupPanel overview={overview} control={control} onFieldChange={onFieldChange} />

        <SectionCard
          eyebrow="Rezumat operational"
          title="Indicatori prioritari ai sesiunii"
          subtitle="Panoul sintetizeaza reperele esentiale ale sesiunii curente pentru utilizarea in laborator."
          className="h-full"
        >
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="ui-kicker">Molecule promovate</p>
              <p className="mt-2 text-3xl font-semibold text-white">{overview.summary.promotedCount}</p>
              <p className="mt-2 text-sm leading-7 text-slate-300">Candidate trecute direct in lista prioritara.</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="ui-kicker">Necesita revizie</p>
              <p className="mt-2 text-3xl font-semibold text-white">{overview.summary.reviewCount}</p>
              <p className="mt-2 text-sm leading-7 text-slate-300">Cazuri bune, dar care mai cer verificare.</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="ui-kicker">Respinse</p>
              <p className="mt-2 text-3xl font-semibold text-white">{overview.summary.rejectedCount}</p>
              <p className="mt-2 text-sm leading-7 text-slate-300">Candidati eliminati pe criterii de risc sau calitate.</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="ui-kicker">QED mediu</p>
              <p className="mt-2 text-3xl font-semibold text-white">{overview.summary.meanQed.toFixed(2)}</p>
              <p className="mt-2 text-sm leading-7 text-slate-300">Calitatea medie a moleculelor din sesiunea curenta.</p>
            </div>
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Candidat curent</p>
              <p className="mt-2 text-lg font-semibold text-white">{overview.bestMolecule?.action || "Fara candidat selectat"}</p>
              <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-300">{overview.bestMolecule?.smiles || "Molecula prioritara va aparea aici dupa prima selectie."}</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-slate-950/65 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Ultima etapa</p>
              <p className="mt-2 text-lg font-semibold text-white">
                {overview.latestRound ? `Runda ${overview.latestRound.round} / seed ${overview.latestRound.seedStep}` : "Fara etape inregistrate"}
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {overview.latestRound
                  ? `${overview.latestRound.newCandidates} candidati noi, ${overview.latestRound.promotedCandidates} promovati.`
                  : "Cronologia va aparea imediat ce worker-ul proceseaza prima generatie."}
              </p>
            </div>
          </div>
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Evolutie"
          title="Scor maxim pe etape"
          subtitle="Axa OX reprezinta rundele si seed-urile procesate, iar OY arata scorul maxim atins in fiecare moment."
        >
          <InteractiveLineChart
            categories={rounds}
            xLabel="Etapa worker"
            yLabel="Scor live"
            series={[
              {
                label: "Scor maxim",
                color: "#24d6ea",
                data: timeline.generations.map((frame) => frame.bestScore),
              },
              {
                label: "Candidati promovati",
                color: "#40d98f",
                data: timeline.generations.map((frame) => frame.promotedCandidates),
              },
            ]}
            formatValue={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>

        <SectionCard
          eyebrow="Triere"
          title="Distributia pe status"
          subtitle="Grafic simplu pentru a intelege rapid cat de multe molecule sunt acceptate, puse in revizie sau respinse."
        >
          <InteractiveBarChart
            categories={["Promovate", "Revizie", "Respinse"]}
            xLabel="Status candidat"
            yLabel="Numar molecule"
            series={[
              {
                label: "Biblioteca curenta",
                color: "#24d6ea",
                data: [statusCounts.promovata, statusCounts.revizie, statusCounts.respinsa],
              },
            ]}
            formatValue={(value) => value.toFixed(0)}
            className="h-[360px]"
          />
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Dupa ranking"
          title="Funnel live al candidatilor"
          subtitle="Aceste etape sunt estimate live din semnalele worker-ului: audit, ghidaj structural, fezabilitate, diferentiere fata de piata si un proxy de readiness experimental."
        >
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_300px]">
            <InteractiveBarChart
              categories={pipelineStageLabels}
              xLabel="Etapa dupa ranking"
              yLabel="Numar molecule"
              series={[
                {
                  label: "Molecule active in funnel",
                  color: "#24d6ea",
                  data: analytics.pipelineStages.map((entry) => entry.count),
                },
              ]}
              formatValue={(value) => value.toFixed(0)}
              className="h-[360px]"
            />

            <div className="space-y-3">
              {analytics.pipelineStages.length ? (
                analytics.pipelineStages.map((entry, index) => (
                  <div key={`${entry.label}-${index}`} className="rounded-2xl border border-white/6 bg-white/5 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="ui-kicker">{entry.label}</p>
                        <p className="mt-2 text-lg font-semibold text-white">{entry.count} molecule</p>
                      </div>
                      <div className="rounded-full border border-forge-cyan/20 bg-forge-cyan/10 px-3 py-1 text-xs font-semibold text-cyan-100">
                        {(entry.share * 100).toFixed(0)}%
                      </div>
                    </div>
                    <p className="mt-3 text-sm leading-7 text-slate-300">{entry.note}</p>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-white/6 bg-slate-950/70 p-4 text-sm leading-7 text-slate-300">
                  Funnel-ul apare imediat ce backend-ul are molecule si semnale suficiente pentru etapele post-ranking.
                </div>
              )}
            </div>
          </div>
        </SectionCard>

        <SectionCard
          eyebrow="Maturizare live"
          title="Cum se imbunatateste cohorta de top"
          subtitle="Urmarim cohorta lider din fiecare runda, normalizata intre 0 si 1, ca sa vezi daca generarea produce candidati mai credibili pe masura ce ruleaza."
        >
          <InteractiveLineChart
            categories={maturationRoundLabels}
            xLabel="Runda"
            yLabel="Scor normalizat"
            className="h-[360px]"
            formatValue={(value) => value.toFixed(2)}
            series={[
              {
                label: "Reward verificat",
                color: "#24d6ea",
                data: analytics.maturationSeries.map((entry) => entry.verifiedRewardScore),
              },
              {
                label: "Fezabilitate",
                color: "#40d98f",
                data: analytics.maturationSeries.map((entry) => entry.feasibility),
              },
              {
                label: "Noutate",
                color: "#f59e0b",
                data: analytics.maturationSeries.map((entry) => entry.novelty),
              },
              {
                label: "Structura",
                color: "#73a6ff",
                data: analytics.maturationSeries.map((entry) => entry.structural),
              },
              {
                label: "Ready experimental",
                color: "#fb7185",
                data: analytics.maturationSeries.map((entry) => entry.experimentalReadyRate),
              },
            ]}
          />
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Pe runde"
          title="Cate molecule trec fiecare filtru"
          subtitle="Graficul se actualizeaza pe masura ce worker-ul avanseaza. Liniile arata unde se rarefiaza lotul dupa ranking si ce etapa franeaza cel mai mult promovarea."
        >
          <InteractiveLineChart
            categories={pipelineRoundLabels}
            xLabel="Runda"
            yLabel="Molecule care trec"
            className="h-[360px]"
            formatValue={(value) => value.toFixed(0)}
            series={[
              {
                label: "Audit",
                color: "#24d6ea",
                data: analytics.pipelineProgress.map((entry) => entry.auditPass),
              },
              {
                label: "Structura",
                color: "#40d98f",
                data: analytics.pipelineProgress.map((entry) => entry.structuralReady),
              },
              {
                label: "Fezabile",
                color: "#f59e0b",
                data: analytics.pipelineProgress.map((entry) => entry.feasible),
              },
              {
                label: "Piata",
                color: "#73a6ff",
                data: analytics.pipelineProgress.map((entry) => entry.marketReady),
              },
              {
                label: "Exp. proxy",
                color: "#fb7185",
                data: analytics.pipelineProgress.map((entry) => entry.experimentalProxy),
              },
              {
                label: "Promovate",
                color: "#c084fc",
                data: analytics.pipelineProgress.map((entry) => entry.promoted),
              },
            ]}
          />
        </SectionCard>

        <SectionCard
          eyebrow="Calitate pe runde"
          title="pIC50 mediu si QED mediu"
          subtitle="Acest grafic arata daca rundele noi aduc molecule mai puternice si mai curate din punct de vedere medicinal."
        >
          <InteractiveLineChart
            categories={roundLabels}
            xLabel="Runda"
            yLabel="Valoare medie"
            series={[
              {
                label: "pIC50 mediu",
                color: "#24d6ea",
                data: meanPic50ByRound,
              },
              {
                label: "QED mediu",
                color: "#40d98f",
                data: meanQedByRound,
              },
            ]}
            formatValue={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>

        <SectionCard
          eyebrow="Cost pe runde"
          title="Cost mediu si cost minim"
          subtitle="Daca aceste curbe cresc prea mult, biblioteca devine mai greu de sintetizat sau mai scump de testat."
        >
          <InteractiveLineChart
            categories={roundLabels}
            xLabel="Runda"
            yLabel="Cost estimat 10 mg"
            series={[
              {
                label: "Cost mediu",
                color: "#73a6ff",
                data: costAverageByRound,
              },
              {
                label: "Cost minim",
                color: "#f59e0b",
                data: costMinimumByRound,
              },
            ]}
            formatValue={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1.1fr)_360px]">
        <SectionCard
          eyebrow="Spatiu chimic util"
          title="Potenta versus cost"
          subtitle="Fiecare punct este o molecula. Pe OX este costul estimat pentru 10 mg, pe OY pIC50. Sus-stanga inseamna mai puternic la cost mai mic."
        >
          <InteractiveScatterChart
            points={topScatter}
            xLabel="Cost estimat 10 mg (USD)"
            yLabel="pIC50 prezis"
            formatX={(value) => value.toFixed(1)}
            formatY={(value) => value.toFixed(2)}
            className="h-[380px]"
          />
        </SectionCard>

        <SectionCard
          eyebrow="Comparator rapid"
          title="Candidat curent"
          subtitle="Rezumat pentru molecula lider a sesiunii."
          className="h-full"
        >
          <div className="space-y-3">
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Comparator piata</p>
              <p className="mt-2 text-lg font-semibold text-white">{overview.bestMolecule?.marketReference || "-"}</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Similaritate piata</p>
              <p className="mt-2 text-lg font-semibold text-white">{overview.bestMolecule?.marketSimilarity?.toFixed(3) ?? "-"}</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Cost estimat 10 mg</p>
              <p className="mt-2 text-lg font-semibold text-white">${overview.bestMolecule?.cost10mg?.toFixed(2) ?? "--"}</p>
            </div>
            <div className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-400">Incertitudine</p>
              <p className="mt-2 text-lg font-semibold text-white">{overview.bestMolecule?.uncertainty?.toFixed(3) ?? "--"}</p>
            </div>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
