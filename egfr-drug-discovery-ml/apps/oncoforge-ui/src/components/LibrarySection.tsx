import { CandidateLibrary } from "@/components/CandidateLibrary";
import { InteractiveBarChart, InteractiveScatterChart } from "@/components/InteractiveCharts";
import { SectionCard } from "@/components/SectionCard";
import type { LibraryRow } from "@/types";

interface LibrarySectionProps {
  library: LibraryRow[];
  selectedSmiles: string | null;
  compareSmiles: string[];
  uiMode: "basic" | "expert";
  onSelectCandidate: (smiles: string) => void;
  onToggleCompare: (smiles: string) => void;
}

export function LibrarySection({
  library,
  selectedSmiles,
  compareSmiles,
  uiMode,
  onSelectCandidate,
  onToggleCompare,
}: LibrarySectionProps) {
  const rounds = Array.from(new Set(library.map((item) => item.round))).sort((left, right) => left - right);
  const roundLabels = rounds.map((round) => `R${round}`);
  const scatterPoints = library.slice(0, 48).map((item) => ({
    label: `#${item.rank}`,
    x: item.qed,
    y: item.pic50,
    color:
      item.status === "promovata" ? "#40d98f" : item.status === "revizie" ? "#f59e0b" : "#fb7185",
    size: item.smiles === selectedSmiles ? 9 : 6,
    meta: `${item.action || "Mutatie"} | cost ${item.cost10mg.toFixed(2)} USD`,
  }));

  const roundCounts = rounds.map((round) => library.filter((item) => item.round === round).length);
  const promotedCounts = rounds.map((round) => library.filter((item) => item.round === round && item.status === "promovata").length);

  return (
    <div className="space-y-4">
      <SectionCard
        eyebrow="Analiza biblioteca"
        title="Biblioteca si selectie"
        subtitle="Biblioteca este analizata in trei pasi: harta chimica, randament pe runde si selectie din lista paginata."
      >
        <div className="grid gap-3 md:grid-cols-3">
          {[
            {
              title: "1. Harta QED versus pIC50",
              text: "Punctele din dreapta sus tind sa ofere cel mai bun compromis intre calitate medicinala si potenta.",
            },
            {
              title: "2. Volum pe runde",
              text: "Graficul pe runde arata daca generatiile noi aduc suficienti candidati utili sau doar maresc biblioteca.",
            },
            {
              title: "3. Lista paginata",
              text: "Din lista de jos deschizi candidatul sau il transferi direct in setul de comparatie, fara sa incarci intreaga masa de date odata.",
            },
          ].map((item) => (
            <div key={item.title} className="rounded-2xl border border-white/6 bg-white/5 p-4">
              <p className="text-sm font-semibold text-white">{item.title}</p>
              <p className="mt-2 text-sm leading-7 text-slate-300">{item.text}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Analiza biblioteca"
          title="Potenta versus QED"
          subtitle="Pe OX este QED, pe OY pIC50. Punctele din dreapta sus au cel mai bun compromis calitate-potenta."
        >
          <InteractiveScatterChart
            points={scatterPoints}
            xLabel="QED"
            yLabel="pIC50 prezis"
            formatX={(value) => value.toFixed(2)}
            formatY={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>

        <SectionCard
          eyebrow="Volum de selectie"
          title="Candidati pe runda"
          subtitle="Grafic comparativ intre totalul candidatilor si cate dintre ei au ajuns in starea promovata."
        >
          <InteractiveBarChart
            categories={roundLabels}
            xLabel="Runda"
            yLabel="Numar molecule"
            series={[
              {
                label: "Total",
                color: "#24d6ea",
                data: roundCounts,
              },
              {
                label: "Promovate",
                color: "#40d98f",
                data: promotedCounts,
              },
            ]}
            formatValue={(value) => value.toFixed(0)}
            className="h-[360px]"
          />
        </SectionCard>
      </div>

      <CandidateLibrary
        library={library}
        selectedSmiles={selectedSmiles}
        compareSmiles={compareSmiles}
        uiMode={uiMode}
        onSelectCandidate={onSelectCandidate}
        onToggleCompare={onToggleCompare}
      />
    </div>
  );
}
