import { InteractiveBarChart, InteractiveLineChart } from "@/components/InteractiveCharts";
import { RLMonitor } from "@/components/RLMonitor";
import { SectionCard } from "@/components/SectionCard";
import { TimelineRail } from "@/components/TimelineRail";
import type { LibraryRow, RLMonitorPayload, SelectedMolecule, TimelinePayload } from "@/types";

interface TrainingSectionProps {
  library: LibraryRow[];
  timeline: TimelinePayload;
  monitor: RLMonitorPayload;
  selected: SelectedMolecule | null;
  selectedRound: number;
  onJump: (round: number, candidateSmiles?: string) => void;
}

export function TrainingSection({
  library,
  timeline,
  monitor,
  selected,
  selectedRound,
  onJump,
}: TrainingSectionProps) {
  const roundLabels = monitor.penaltySeries.map((entry) => `R${entry.round}`);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 2xl:grid-cols-2">
        <SectionCard
          eyebrow="Monitor convergenta"
          title="Recompensa si explorare"
          subtitle="Grafic cu axe clasice: OX este runda, OY arata intensitatea recompensei verificate si balanta explorare versus exploatare."
        >
          <InteractiveLineChart
            categories={roundLabels}
            xLabel="Runda"
            yLabel="Scor / intensitate"
            series={[
              {
                label: "Recompensa verificata",
                color: "#24d6ea",
                data: monitor.penaltySeries.map((entry) => entry.verifiedReward),
              },
              {
                label: "Explorare",
                color: "#f59e0b",
                data: monitor.penaltySeries.map((entry) => entry.exploration),
              },
              {
                label: "Exploatare",
                color: "#40d98f",
                data: monitor.penaltySeries.map((entry) => entry.exploitation),
              },
            ]}
            formatValue={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>

        <SectionCard
          eyebrow="Monitor penalizari"
          title="Penalizari RLVR pe runda"
          subtitle="Comparatie directa intre toxicitate, invaliditate, incertitudine si risc de manipulare a recompensei."
        >
          <InteractiveBarChart
            categories={roundLabels}
            xLabel="Runda"
            yLabel="Penalizare medie"
            series={[
              {
                label: "Toxicitate",
                color: "#fb7185",
                data: monitor.penaltySeries.map((entry) => entry.toxicityPenalty),
              },
              {
                label: "Invaliditate",
                color: "#f59e0b",
                data: monitor.penaltySeries.map((entry) => entry.invalidPenalty),
              },
              {
                label: "Incertitudine",
                color: "#60a5fa",
                data: monitor.penaltySeries.map((entry) => entry.uncertaintyPenalty),
              },
              {
                label: "Risc recompensa",
                color: "#24d6ea",
                data: monitor.penaltySeries.map((entry) => entry.rewardRiskPenalty),
              },
            ]}
            formatValue={(value) => value.toFixed(2)}
            className="h-[360px]"
          />
        </SectionCard>
      </div>

      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1.2fr)_minmax(340px,0.8fr)]">
        <TimelineRail
          timeline={timeline}
          library={library}
          selectedRound={selectedRound}
          onJump={onJump}
        />
        <RLMonitor selected={selected} monitor={monitor} />
      </div>
    </div>
  );
}
