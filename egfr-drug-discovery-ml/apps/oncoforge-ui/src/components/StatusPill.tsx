import type { AgentStatus, CandidateStatus } from "@/types";

type StatusValue = AgentStatus | CandidateStatus | "running" | "stopped" | "error";

const tones: Record<StatusValue, string> = {
  active: "border-cyan-400/40 bg-cyan-400/10 text-cyan-100",
  idle: "border-slate-500/30 bg-slate-600/15 text-slate-200",
  training: "border-blue-400/40 bg-blue-400/10 text-blue-100",
  monitoring: "border-violet-400/40 bg-violet-400/10 text-violet-100",
  promovata: "border-emerald-400/40 bg-emerald-400/10 text-emerald-100",
  revizie: "border-amber-400/40 bg-amber-400/10 text-amber-100",
  respinsa: "border-rose-400/40 bg-rose-400/10 text-rose-100",
  necunoscut: "border-slate-500/30 bg-slate-600/15 text-slate-200",
  running: "border-emerald-400/40 bg-emerald-400/10 text-emerald-100",
  stopped: "border-slate-500/30 bg-slate-600/15 text-slate-200",
  error: "border-rose-400/40 bg-rose-400/10 text-rose-100",
};

interface StatusPillProps {
  status: StatusValue;
  label: string;
}

export function StatusPill({ status, label }: StatusPillProps) {
  const tone = tones[status] ?? tones.necunoscut;
  const isDormant = status === "idle" || status === "stopped";
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] ${tone}`}>
      <span className="relative flex h-2.5 w-2.5 items-center justify-center">
        <span className={`absolute inset-0 rounded-full ${isDormant ? "bg-slate-400/35" : "bg-current/35 motion-safe:animate-pulseSoft"}`} />
        <span className={`relative h-2 w-2 rounded-full ${isDormant ? "bg-slate-400" : "bg-current"}`} />
      </span>
      {label}
    </span>
  );
}
