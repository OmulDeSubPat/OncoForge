import { memo } from "react";

import { SectionCard } from "@/components/SectionCard";
import { StatusPill } from "@/components/StatusPill";
import type { AgentCard, FlowEdge, SelectedMolecule } from "@/types";

interface AgentRailProps {
  agents: AgentCard[];
  flows: FlowEdge[];
  selected: SelectedMolecule | null;
}

function flowLabel(flow: FlowEdge, agents: AgentCard[]) {
  const source = agents.find((entry) => entry.id === flow.source)?.name ?? flow.source;
  const target = agents.find((entry) => entry.id === flow.target)?.name ?? flow.target;
  return `${source} -> ${target}`;
}

export const AgentRail = memo(function AgentRail({ agents, flows, selected }: AgentRailProps) {
  return (
    <SectionCard
      eyebrow="Orchestrare multi-agent"
      title="Panou agenti"
      subtitle="Fiecare agent contribuie explicit la scorul final, iar fluxul de date ramane vizibil."
      className="h-full"
    >
      <div className="space-y-4">
        <div className="rounded-2xl border border-white/5 bg-slate-950/55 p-4">
          <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Molecula focalizata</p>
          <p className="mt-2 text-lg font-semibold text-white">{selected?.smiles ? `Rang #${selected.rank}` : "Nicio molecula selectata"}</p>
          <p className="mt-2 break-all font-mono text-xs leading-6 text-slate-300">{selected?.smiles ?? "Astept primele molecule generate."}</p>
        </div>

        <div className="space-y-3">
          {agents.map((agent) => (
            <div key={agent.id} className="rounded-2xl border border-white/5 bg-slate-950/50 p-4 shadow-soft">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="font-semibold text-white">{agent.name}</h3>
                  <p className="mt-2 text-sm text-slate-300">{agent.headline}</p>
                  <p className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-500">{agent.lastAction}</p>
                </div>
                <StatusPill status={agent.status} label={agent.status.toUpperCase()} />
              </div>

              <div className="mt-4">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>Contributie in scorul final</span>
                  <span>{Math.round(agent.contribution * 100)}%</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-slate-900">
                  <div
                    className="h-2 rounded-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green"
                    style={{ width: `${Math.max(10, agent.contribution * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-2xl border border-forge-cyan/20 bg-forge-cyan/10 p-4">
          <p className="text-xs uppercase tracking-[0.24em] text-cyan-100">Data flow</p>
          <div className="mt-3 space-y-3">
            {flows.map((flow) => (
              <div key={`${flow.source}-${flow.target}`} className="space-y-1">
                <div className="flex items-center justify-between text-xs text-slate-300">
                  <span>{flowLabel(flow, agents)}</span>
                  <span>{Math.round(flow.weight * 100)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-900">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-forge-cyan via-forge-blue to-forge-green shadow-[0_0_14px_rgba(36,214,234,0.4)]"
                    style={{ width: `${Math.max(8, flow.weight * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </SectionCard>
  );
});
