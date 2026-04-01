import { memo, useId, useState } from "react";

interface SparklineProps {
  values: number[];
  stroke?: string;
  fill?: string;
  className?: string;
}

interface RadarChartProps {
  labels: string[];
  values: number[];
  className?: string;
}

interface ProgressRingProps {
  value: number;
  label: string;
  subtitle?: string;
  size?: number;
  color?: string;
}

export interface LineChartPoint {
  x: number | string;
  y: number;
  label?: string;
}

export interface LineChartSeries {
  id: string;
  label: string;
  color: string;
  data: LineChartPoint[];
}

interface InteractiveLineChartProps {
  series: LineChartSeries[];
  xLabel: string;
  yLabel: string;
  className?: string;
  valueFormatter?: (value: number) => string;
}

export interface BarChartItem {
  label: string;
  value: number;
  color?: string;
}

interface InteractiveBarChartProps {
  data: BarChartItem[];
  xLabel: string;
  yLabel: string;
  className?: string;
  valueFormatter?: (value: number) => string;
}

export interface ScatterPoint {
  x: number;
  y: number;
  label: string;
  color?: string;
  size?: number;
}

interface InteractiveScatterChartProps {
  data: ScatterPoint[];
  xLabel: string;
  yLabel: string;
  className?: string;
  xFormatter?: (value: number) => string;
  yFormatter?: (value: number) => string;
}

const CHART_WIDTH = 680;
const CHART_HEIGHT = 300;
const CHART_MARGIN = { top: 20, right: 28, bottom: 56, left: 62 };
const GRID_LINES = 4;

function formatAxisNumber(value: number) {
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}

function chartBounds() {
  return {
    left: CHART_MARGIN.left,
    right: CHART_WIDTH - CHART_MARGIN.right,
    top: CHART_MARGIN.top,
    bottom: CHART_HEIGHT - CHART_MARGIN.bottom,
  };
}

function safeDomain(values: number[]) {
  if (!values.length) {
    return { min: 0, max: 1 };
  }

  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  if (rawMin === rawMax) {
    return { min: rawMin - 1, max: rawMax + 1 };
  }

  const padding = (rawMax - rawMin) * 0.12;
  return { min: rawMin - padding, max: rawMax + padding };
}

function scaleY(value: number, min: number, max: number) {
  const bounds = chartBounds();
  const height = bounds.bottom - bounds.top;
  return bounds.bottom - ((value - min) / (max - min || 1)) * height;
}

function scaleLineX(index: number, total: number) {
  const bounds = chartBounds();
  const width = bounds.right - bounds.left;
  if (total <= 1) {
    return bounds.left + width / 2;
  }
  return bounds.left + (index / (total - 1)) * width;
}

function scaleScatterX(value: number, min: number, max: number) {
  const bounds = chartBounds();
  const width = bounds.right - bounds.left;
  return bounds.left + ((value - min) / (max - min || 1)) * width;
}

function axisTicks(min: number, max: number) {
  return Array.from({ length: GRID_LINES + 1 }, (_, index) => {
    const ratio = index / GRID_LINES;
    return max - (max - min) * ratio;
  });
}

export const Sparkline = memo(function Sparkline({
  values,
  stroke = "#24d6ea",
  fill = "rgba(36, 214, 234, 0.16)",
  className = "",
}: SparklineProps) {
  const gradientId = useId();

  if (!values.length) {
    return null;
  }

  const width = 320;
  const height = 120;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / Math.max(values.length - 1, 1);
  const points = values
    .map((value, index) => {
      const x = index * step;
      const y = height - ((value - min) / range) * (height - 20) - 10;
      return `${x},${y}`;
    })
    .join(" ");
  const area = `0,${height} ${points} ${width},${height}`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className={className}>
      <defs>
        <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={fill} />
          <stop offset="100%" stopColor="rgba(36, 214, 234, 0.01)" />
        </linearGradient>
      </defs>
      <polygon points={area} fill={`url(#${gradientId})`} />
      <polyline
        points={points}
        fill="none"
        stroke={stroke}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
});

export const RadarChart = memo(function RadarChart({ labels, values, className = "" }: RadarChartProps) {
  const size = 260;
  const center = size / 2;
  const radius = 92;
  const count = Math.max(labels.length, 3);
  const step = (Math.PI * 2) / count;

  const levels = [0.25, 0.5, 0.75, 1];
  const levelPolygons = levels.map((level) =>
    labels
      .map((_, index) => {
        const angle = step * index - Math.PI / 2;
        const x = center + Math.cos(angle) * radius * level;
        const y = center + Math.sin(angle) * radius * level;
        return `${x},${y}`;
      })
      .join(" "),
  );

  const dataPoints = labels
    .map((_, index) => {
      const angle = step * index - Math.PI / 2;
      const value = values[index] ?? 0;
      const x = center + Math.cos(angle) * radius * value;
      const y = center + Math.sin(angle) * radius * value;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className={className}>
      {levelPolygons.map((polygon, index) => (
        <polygon
          key={index}
          points={polygon}
          fill="none"
          stroke="rgba(148, 163, 184, 0.18)"
          strokeWidth="1"
        />
      ))}
      {labels.map((label, index) => {
        const angle = step * index - Math.PI / 2;
        const x = center + Math.cos(angle) * (radius + 18);
        const y = center + Math.sin(angle) * (radius + 18);
        return (
          <text
            key={label}
            x={x}
            y={y}
            fill="#cbd5e1"
            fontSize="10"
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {label}
          </text>
        );
      })}
      <polygon
        points={dataPoints}
        fill="rgba(36, 214, 234, 0.14)"
        stroke="rgba(36, 214, 234, 0.95)"
        strokeWidth="2.5"
      />
      {labels.map((_, index) => {
        const angle = step * index - Math.PI / 2;
        const value = values[index] ?? 0;
        const x = center + Math.cos(angle) * radius * value;
        const y = center + Math.sin(angle) * radius * value;
        return <circle key={index} cx={x} cy={y} r="3.6" fill="#d1fbff" />;
      })}
    </svg>
  );
});

export const ProgressRing = memo(function ProgressRing({
  value,
  label,
  subtitle,
  size = 96,
  color = "#24d6ea",
}: ProgressRingProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const radius = size / 2 - 8;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - clamped);

  return (
    <div className="flex items-center gap-4">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(148, 163, 184, 0.18)"
          strokeWidth="8"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div>
        <p className="text-xs uppercase tracking-[0.24em] text-slate-400">{label}</p>
        <p className="mt-1 text-2xl font-semibold text-white">{Math.round(clamped * 100)}%</p>
        {subtitle ? <p className="mt-1 text-sm text-slate-300">{subtitle}</p> : null}
      </div>
    </div>
  );
});

export const InteractiveLineChart = memo(function InteractiveLineChart({
  series,
  xLabel,
  yLabel,
  className = "",
  valueFormatter = formatAxisNumber,
}: InteractiveLineChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const bounds = chartBounds();
  const allValues = series.flatMap((entry) => entry.data.map((point) => point.y));
  const domain = safeDomain(allValues);
  const maxPoints = Math.max(...series.map((entry) => entry.data.length), 0);
  const activeIndex = hoveredIndex ?? Math.max(0, maxPoints - 1);
  const xTicks = series[0]?.data ?? [];
  const yTicks = axisTicks(domain.min, domain.max);

  if (!series.length || !maxPoints) {
    return (
      <div className={`rounded-3xl border border-white/6 bg-slate-950/65 p-4 text-sm text-slate-300 ${className}`}>
        Nu exista suficiente puncte pentru acest grafic.
      </div>
    );
  }

  return (
    <div className={`rounded-3xl border border-white/6 bg-slate-950/65 p-4 ${className}`}>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {series.map((entry) => (
          <div key={entry.id} className="flex items-center gap-2 text-xs text-slate-300">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
            <span>{entry.label}</span>
          </div>
        ))}
      </div>

      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="h-[280px] w-full">
        {yTicks.map((tick) => {
          const y = scaleY(tick, domain.min, domain.max);
          return (
            <g key={tick}>
              <line x1={bounds.left} y1={y} x2={bounds.right} y2={y} stroke="rgba(148,163,184,0.12)" strokeDasharray="4 6" />
              <text x={bounds.left - 10} y={y + 4} fill="#94a3b8" fontSize="11" textAnchor="end">
                {valueFormatter(tick)}
              </text>
            </g>
          );
        })}

        <line x1={bounds.left} y1={bounds.bottom} x2={bounds.right} y2={bounds.bottom} stroke="rgba(148,163,184,0.28)" />
        <line x1={bounds.left} y1={bounds.top} x2={bounds.left} y2={bounds.bottom} stroke="rgba(148,163,184,0.28)" />

        {xTicks.map((point, index) => {
          const x = scaleLineX(index, maxPoints);
          return (
            <g key={`${point.x}-${index}`}>
              <text x={x} y={bounds.bottom + 18} fill="#94a3b8" fontSize="11" textAnchor="middle">
                {point.label ?? point.x}
              </text>
              <rect
                x={x - Math.max(16, (bounds.right - bounds.left) / Math.max(maxPoints * 2, 2))}
                y={bounds.top}
                width={Math.max(32, (bounds.right - bounds.left) / Math.max(maxPoints, 1))}
                height={bounds.bottom - bounds.top}
                fill="transparent"
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
            </g>
          );
        })}

        {series.map((entry) => {
          const points = entry.data
            .map((point, index) => `${scaleLineX(index, maxPoints)},${scaleY(point.y, domain.min, domain.max)}`)
            .join(" ");
          return (
            <g key={entry.id}>
              <polyline
                points={points}
                fill="none"
                stroke={entry.color}
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              {entry.data.map((point, index) => {
                const active = index === activeIndex;
                const x = scaleLineX(index, maxPoints);
                const y = scaleY(point.y, domain.min, domain.max);
                return (
                  <circle
                    key={`${entry.id}-${index}`}
                    cx={x}
                    cy={y}
                    r={active ? 5 : 3}
                    fill={entry.color}
                    stroke={active ? "#f8fafc" : "transparent"}
                    strokeWidth={active ? 2 : 0}
                  />
                );
              })}
            </g>
          );
        })}

        <text x={(bounds.left + bounds.right) / 2} y={CHART_HEIGHT - 8} fill="#cbd5e1" fontSize="12" textAnchor="middle">
          {xLabel}
        </text>
        <text
          x={18}
          y={(bounds.top + bounds.bottom) / 2}
          fill="#cbd5e1"
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90 18 ${(bounds.top + bounds.bottom) / 2})`}
        >
          {yLabel}
        </text>

        {hoveredIndex !== null ? (
          <g>
            <rect x={bounds.right - 188} y={bounds.top + 12} width={172} height={22 + series.length * 18} rx="12" fill="rgba(2,6,23,0.92)" stroke="rgba(148,163,184,0.2)" />
            <text x={bounds.right - 174} y={bounds.top + 30} fill="#e2e8f0" fontSize="12">
              {xTicks[hoveredIndex]?.label ?? xTicks[hoveredIndex]?.x ?? "-"}
            </text>
            {series.map((entry, index) => {
              const point = entry.data[hoveredIndex];
              if (!point) {
                return null;
              }
              return (
                <g key={`${entry.id}-tooltip`}>
                  <circle cx={bounds.right - 173} cy={bounds.top + 44 + index * 18} r="3.5" fill={entry.color} />
                  <text x={bounds.right - 162} y={bounds.top + 48 + index * 18} fill="#cbd5e1" fontSize="11">
                    {entry.label}: {valueFormatter(point.y)}
                  </text>
                </g>
              );
            })}
          </g>
        ) : null}
      </svg>
    </div>
  );
});

export const InteractiveBarChart = memo(function InteractiveBarChart({
  data,
  xLabel,
  yLabel,
  className = "",
  valueFormatter = formatAxisNumber,
}: InteractiveBarChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const bounds = chartBounds();
  const values = data.map((item) => item.value);
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(0, ...values, 1);
  const domain = safeDomain([minValue, maxValue]);
  const yTicks = axisTicks(domain.min, domain.max);
  const width = bounds.right - bounds.left;
  const barWidth = data.length ? width / data.length : width;
  const baselineY = scaleY(0, domain.min, domain.max);

  if (!data.length) {
    return (
      <div className={`rounded-3xl border border-white/6 bg-slate-950/65 p-4 text-sm text-slate-300 ${className}`}>
        Nu exista bare de afisat inca.
      </div>
    );
  }

  return (
    <div className={`rounded-3xl border border-white/6 bg-slate-950/65 p-4 ${className}`}>
      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="h-[280px] w-full">
        {yTicks.map((tick) => {
          const y = scaleY(tick, domain.min, domain.max);
          return (
            <g key={tick}>
              <line x1={bounds.left} y1={y} x2={bounds.right} y2={y} stroke="rgba(148,163,184,0.12)" strokeDasharray="4 6" />
              <text x={bounds.left - 10} y={y + 4} fill="#94a3b8" fontSize="11" textAnchor="end">
                {valueFormatter(tick)}
              </text>
            </g>
          );
        })}

        <line x1={bounds.left} y1={baselineY} x2={bounds.right} y2={baselineY} stroke="rgba(148,163,184,0.28)" />
        <line x1={bounds.left} y1={bounds.top} x2={bounds.left} y2={bounds.bottom} stroke="rgba(148,163,184,0.28)" />

        {data.map((item, index) => {
          const x = bounds.left + index * barWidth + 10;
          const y = scaleY(Math.max(item.value, 0), domain.min, domain.max);
          const negativeY = scaleY(Math.min(item.value, 0), domain.min, domain.max);
          const height = Math.abs(negativeY - y) || 2;
          const top = item.value >= 0 ? y : baselineY;
          const isActive = index === hoveredIndex;
          return (
            <g key={`${item.label}-${index}`}>
              <rect
                x={x}
                y={top}
                width={Math.max(18, barWidth - 20)}
                height={height}
                rx="10"
                fill={item.color ?? "#24d6ea"}
                opacity={isActive ? 1 : 0.82}
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
              <text
                x={x + Math.max(18, barWidth - 20) / 2}
                y={bounds.bottom + 18}
                fill="#94a3b8"
                fontSize="11"
                textAnchor="middle"
              >
                {item.label}
              </text>
            </g>
          );
        })}

        <text x={(bounds.left + bounds.right) / 2} y={CHART_HEIGHT - 8} fill="#cbd5e1" fontSize="12" textAnchor="middle">
          {xLabel}
        </text>
        <text
          x={18}
          y={(bounds.top + bounds.bottom) / 2}
          fill="#cbd5e1"
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90 18 ${(bounds.top + bounds.bottom) / 2})`}
        >
          {yLabel}
        </text>

        {hoveredIndex !== null ? (
          <g>
            <rect x={bounds.right - 184} y={bounds.top + 12} width={168} height={50} rx="12" fill="rgba(2,6,23,0.92)" stroke="rgba(148,163,184,0.2)" />
            <text x={bounds.right - 170} y={bounds.top + 31} fill="#e2e8f0" fontSize="12">
              {data[hoveredIndex]?.label}
            </text>
            <text x={bounds.right - 170} y={bounds.top + 50} fill="#cbd5e1" fontSize="11">
              Valoare: {valueFormatter(data[hoveredIndex]?.value ?? 0)}
            </text>
          </g>
        ) : null}
      </svg>
    </div>
  );
});

export const InteractiveScatterChart = memo(function InteractiveScatterChart({
  data,
  xLabel,
  yLabel,
  className = "",
  xFormatter = formatAxisNumber,
  yFormatter = formatAxisNumber,
}: InteractiveScatterChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const bounds = chartBounds();
  const xDomain = safeDomain(data.map((point) => point.x));
  const yDomain = safeDomain(data.map((point) => point.y));
  const xTicks = axisTicks(xDomain.min, xDomain.max);
  const yTicks = axisTicks(yDomain.min, yDomain.max);

  if (!data.length) {
    return (
      <div className={`rounded-3xl border border-white/6 bg-slate-950/65 p-4 text-sm text-slate-300 ${className}`}>
        Nu exista puncte de comparatie disponibile.
      </div>
    );
  }

  return (
    <div className={`rounded-3xl border border-white/6 bg-slate-950/65 p-4 ${className}`}>
      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="h-[280px] w-full">
        {yTicks.map((tick) => {
          const y = scaleY(tick, yDomain.min, yDomain.max);
          return (
            <g key={`y-${tick}`}>
              <line x1={bounds.left} y1={y} x2={bounds.right} y2={y} stroke="rgba(148,163,184,0.12)" strokeDasharray="4 6" />
              <text x={bounds.left - 10} y={y + 4} fill="#94a3b8" fontSize="11" textAnchor="end">
                {yFormatter(tick)}
              </text>
            </g>
          );
        })}

        {xTicks.map((tick) => {
          const x = scaleScatterX(tick, xDomain.min, xDomain.max);
          return (
            <g key={`x-${tick}`}>
              <line x1={x} y1={bounds.top} x2={x} y2={bounds.bottom} stroke="rgba(148,163,184,0.06)" />
              <text x={x} y={bounds.bottom + 18} fill="#94a3b8" fontSize="11" textAnchor="middle">
                {xFormatter(tick)}
              </text>
            </g>
          );
        })}

        <line x1={bounds.left} y1={bounds.bottom} x2={bounds.right} y2={bounds.bottom} stroke="rgba(148,163,184,0.28)" />
        <line x1={bounds.left} y1={bounds.top} x2={bounds.left} y2={bounds.bottom} stroke="rgba(148,163,184,0.28)" />

        {data.map((point, index) => {
          const x = scaleScatterX(point.x, xDomain.min, xDomain.max);
          const y = scaleY(point.y, yDomain.min, yDomain.max);
          const active = hoveredIndex === index;
          return (
            <g key={`${point.label}-${index}`}>
              <circle
                cx={x}
                cy={y}
                r={active ? point.size ?? 7 : (point.size ?? 5)}
                fill={point.color ?? "#24d6ea"}
                fillOpacity={active ? 0.95 : 0.78}
                stroke={active ? "#f8fafc" : "transparent"}
                strokeWidth={active ? 2 : 0}
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
            </g>
          );
        })}

        <text x={(bounds.left + bounds.right) / 2} y={CHART_HEIGHT - 8} fill="#cbd5e1" fontSize="12" textAnchor="middle">
          {xLabel}
        </text>
        <text
          x={18}
          y={(bounds.top + bounds.bottom) / 2}
          fill="#cbd5e1"
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90 18 ${(bounds.top + bounds.bottom) / 2})`}
        >
          {yLabel}
        </text>

        {hoveredIndex !== null ? (
          <g>
            <rect x={bounds.right - 212} y={bounds.top + 12} width={196} height={68} rx="12" fill="rgba(2,6,23,0.92)" stroke="rgba(148,163,184,0.2)" />
            <text x={bounds.right - 198} y={bounds.top + 30} fill="#e2e8f0" fontSize="12">
              {data[hoveredIndex]?.label}
            </text>
            <text x={bounds.right - 198} y={bounds.top + 48} fill="#cbd5e1" fontSize="11">
              X: {xFormatter(data[hoveredIndex]?.x ?? 0)}
            </text>
            <text x={bounds.right - 198} y={bounds.top + 64} fill="#cbd5e1" fontSize="11">
              Y: {yFormatter(data[hoveredIndex]?.y ?? 0)}
            </text>
          </g>
        ) : null}
      </svg>
    </div>
  );
});
