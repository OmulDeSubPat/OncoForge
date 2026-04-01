import type { ReactNode } from "react";

interface SectionCardProps {
  title: string;
  subtitle?: string;
  eyebrow?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function SectionCard({
  title,
  subtitle,
  eyebrow,
  action,
  children,
  className = "",
}: SectionCardProps) {
  return (
    <section className={`glass-panel section-enter relative overflow-visible ${className}`}>
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-forge-cyan/80 to-transparent" />
      <div className="pointer-events-none absolute right-6 top-3 h-24 w-24 rounded-full bg-forge-cyan/8 blur-3xl" />

      <div className="border-b border-white/6 px-4 py-4 sm:px-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            {eyebrow ? (
              <p className="inline-flex rounded-full border border-forge-cyan/20 bg-forge-cyan/8 px-2.5 py-1 text-[11px] tracking-[0.18em] text-forge-cyan/95">
                {eyebrow}
              </p>
            ) : null}
            <h2 className="mt-2 text-xl font-semibold tracking-[-0.02em] text-white sm:text-[1.35rem]">{title}</h2>
            {subtitle ? (
              <p className="mt-2 max-w-3xl text-sm leading-7 text-slate-300">{subtitle}</p>
            ) : null}
          </div>
          {action ? <div className="shrink-0 self-start rounded-2xl border border-white/6 bg-white/[0.03] p-1">{action}</div> : null}
        </div>
      </div>
      <div className="px-4 py-4 sm:px-5">{children}</div>
    </section>
  );
}
