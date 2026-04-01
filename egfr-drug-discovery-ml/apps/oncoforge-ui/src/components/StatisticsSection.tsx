import { InteractiveBarChart, InteractiveLineChart, InteractiveScatterChart } from "@/components/InteractiveCharts";
import { SectionCard } from "@/components/SectionCard";
import type { LibraryRow, TimelinePayload } from "@/types";

interface StatisticsSectionProps {
  library: LibraryRow[];
  timeline: TimelinePayload;
  selectedSmiles: string | null;
}

export function StatisticsSection({ library, timeline, selectedSmiles }: StatisticsSectionProps) {
  const roundSet = Array.from(new Set(timeline.generations.map((frame) => frame.round))).sort((left, right) => left - right);
  const roundLabels = roundSet.map((round) => `R${round}`);
  const totalPerRound = roundSet.map((round) => library.filter((item) => item.round === round).length);
  const promotedPerRound = roundSet.map((round) => library.filter((item) => item.round === round && item.status === "promovata").length);
  const meanPic50ByRound = roundSet.map((round) => {
    const items = library.filter((item) => item.round === round);
    return items.length ? items.reduce((sum, item) => sum + item.pic50, 0) / items.length : 0;
  });
  const meanQedByRound = roundSet.map((round) => {
    const items = library.filter((item) => item.round === round);
    return items.length ? items.reduce((sum, item) => sum + item.qed, 0) / items.length : 0;
  });
  const statusCounts = [
    library.filter((item) => item.status === "promovata").length,
    library.filter((item) => item.status === "revizie").length,
    library.filter((item) => item.status === "respinsa").length,
  ];
  const topScoreCategories = library.slice(0, 12).map((item) => `#${item.rank}`);
  const topScoreValues = library.slice(0, 12).map((item) => item.score);
  const scoreVsUncertainty = library.slice(0, 48).map((item) => ({
    label: `#${item.rank}`,
    x: item.uncertainty,
    y: item.score,
    color: item.smiles === selectedSmiles ? "#73a6ff" : item.status === "promovata" ? "#40d98f" : "#24d6ea",
    size: item.smiles === selectedSmiles ? 9 : 6,
    meta: `${item.action || "Mutatie"} | ${item.marketReference || "Fara comparator"}`,
  }));
  const costVsScore = library.slice(0, 48).map((item) => ({
    label: `#${item.rank}`,
    x: item.cost10mg,
    y: item.score,
    color: item.smiles === selectedSmiles ? "#73a6ff" : item.status === "promovata" ? "#40d98f" : "#24d6ea",
    size: item.smiles === selectedSmiles ? 9 : 6,
    meta: `pIC50 ${item.pic50.toFixed(2)} | QED ${item.qed.toFixed(2)}`,
  }));
  const potencyVsQed = library.slice(0, 48).map((item) => ({
    label: `#${item.rank}`,
    x: item.qed,
    y: item.pic50,
    color: item.smiles === selectedSmiles ? "#73a6ff" : item.status === "promovata" ? "#40d98f" : "#24d6ea",
    size: item.smiles === selectedSmiles ? 9 : 6,
    meta: `${item.action || "Mutatie"} | cost ${item.cost10mg.toFixed(2)} USD`,
  }));

  return (
    <div className="space-y-4">
      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Distributie live"
          title="Scor versus incertitudine"
          subtitle="OX arata incertitudinea, OY scorul final. Zona stanga-sus este cea mai buna: scor mare la incertitudine mica."
        >
          <InteractiveScatterChart
            points={scoreVsUncertainty}
            xLabel="Incertitudine"
            yLabel="Scor final"
            formatX={(value) => value.toFixed(2)}
            formatY={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>

        <SectionCard
          eyebrow="Eficienta selectie"
          title="Cost versus scor"
          subtitle="Grafic live pentru a vedea direct daca moleculele valoroase devin si prea costisitoare."
        >
          <InteractiveScatterChart
            points={costVsScore}
            xLabel="Cost estimat 10 mg (USD)"
            yLabel="Scor final"
            formatX={(value) => value.toFixed(2)}
            formatY={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Calitate lot"
          title="pIC50 versus QED"
          subtitle="Aici vezi direct daca moleculele puternice sunt si suficient de curate medicinal. Zona dreapta-sus este cea mai interesanta."
        >
          <InteractiveScatterChart
            points={potencyVsQed}
            xLabel="QED"
            yLabel="pIC50 prezis"
            formatX={(value) => value.toFixed(2)}
            formatY={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>

        <SectionCard
          eyebrow="Trend proprietati"
          title="Medii pe runda"
          subtitle="Urmarire simpla pentru a vedea daca rundele noi imbunatatesc puterea si calitatea medie a lotului."
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
      </div>

      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Volum pe runde"
          title="Total versus promovate"
          subtitle="Comparatie intre cate molecule au fost analizate in fiecare runda si cate au trecut in shortlist."
        >
          <InteractiveBarChart
            categories={roundLabels}
            xLabel="Runda"
            yLabel="Numar molecule"
            series={[
              {
                label: "Total",
                color: "#24d6ea",
                data: totalPerRound,
              },
              {
                label: "Promovate",
                color: "#40d98f",
                data: promotedPerRound,
              },
            ]}
            formatValue={(value) => value.toFixed(0)}
            className="h-[360px]"
          />
        </SectionCard>

        <SectionCard
          eyebrow="Status global"
          title="Biblioteca pe categorii"
        subtitle="Rezumat pentru starea generala a lotului curent."
        >
          <InteractiveBarChart
            categories={["Promovate", "Revizie", "Respinse"]}
            xLabel="Status"
            yLabel="Numar molecule"
            series={[
              {
                label: "Lot curent",
                color: "#73a6ff",
                data: statusCounts,
              },
            ]}
            formatValue={(value) => value.toFixed(0)}
            className="h-[360px]"
          />
        </SectionCard>
      </div>

      <SectionCard
        eyebrow="Leadboard extins"
        title="Top scoruri in biblioteca"
        subtitle="Cele mai bune molecule sunt afisate pe o axa simpla, usor de comparat vizual."
      >
        <InteractiveBarChart
          categories={topScoreCategories}
          xLabel="Rang"
          yLabel="Scor final"
          series={[
            {
              label: "Top 12",
              color: "#24d6ea",
              data: topScoreValues,
            },
          ]}
          formatValue={(value) => value.toFixed(2)}
          className="h-[380px]"
        />
      </SectionCard>
    </div>
  );
}
