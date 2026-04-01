import { useMemo, useState } from "react";

interface LineSeries {
  label: string;
  color: string;
  data: number[];
}

interface BarSeries {
  label: string;
  color: string;
  data: number[];
}

interface ScatterPoint {
  label: string;
  x: number;
  y: number;
  color?: string;
  size?: number;
  meta?: string;
}

interface SharedChartProps {
  className?: string;
  xLabel: string;
  yLabel: string;
  formatValue?: (value: number) => string;
}

interface InteractiveLineChartProps extends SharedChartProps {
  categories: string[];
  series: LineSeries[];
}

interface InteractiveBarChartProps extends SharedChartProps {
  categories: string[];
  series: BarSeries[];
}

interface InteractiveScatterChartProps extends SharedChartProps {
  points: ScatterPoint[];
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
}

const WIDTH = 820;
const HEIGHT = 340;
const MARGIN = { top: 18, right: 26, bottom: 58, left: 62 };

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function defaultFormatter(value: number) {
  if (Math.abs(value) >= 1000) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}

function buildNiceTicks(min: number, max: number, count = 5) {
  if (min === max) {
    return [min];
  }

  const span = Math.max(Math.abs(max - min), 1e-6);
  const rawStep = span / Math.max(count - 1, 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const candidates = [1, 2, 2.5, 5, 10];
  const step = candidates
    .map((candidate) => candidate * magnitude)
    .find((candidate) => candidate >= rawStep) ?? rawStep;

  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= end + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
}

function getInnerSize() {
  return {
    width: WIDTH - MARGIN.left - MARGIN.right,
    height: HEIGHT - MARGIN.top - MARGIN.bottom,
  };
}

function getPointX(index: number, total: number) {
  const inner = getInnerSize();
  if (total <= 1) {
    return MARGIN.left + inner.width / 2;
  }
  return MARGIN.left + (index / (total - 1)) * inner.width;
}

function buildLinePath(values: number[], min: number, max: number) {
  const inner = getInnerSize();
  const domain = max - min || 1;
  return values
    .map((value, index) => {
      const x = getPointX(index, values.length);
      const y = MARGIN.top + inner.height - ((value - min) / domain) * inner.height;
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
}

function valueToY(value: number, min: number, max: number) {
  const inner = getInnerSize();
  const domain = max - min || 1;
  return MARGIN.top + inner.height - ((value - min) / domain) * inner.height;
}

function axisLabel(label: string, x: number, y: number, rotate?: string) {
  return (
    <text
      x={x}
      y={y}
      fill="rgba(148,163,184,0.84)"
      fontSize="12"
      textAnchor="middle"
      transform={rotate}
    >
      {label}
    </text>
  );
}

export function InteractiveLineChart({
  categories,
  series,
  xLabel,
  yLabel,
  className = "",
  formatValue = defaultFormatter,
}: InteractiveLineChartProps) {
  const [hover, setHover] = useState<null | { leftPercent: number; topPercent: number; title: string; lines: string[] }>(null);
  const validSeries = series.filter((entry) => entry.data.length);
  const allValues = validSeries.flatMap((entry) => entry.data);

  const domain = useMemo(() => {
    if (!allValues.length) {
      return { min: 0, max: 1, ticks: [0, 0.5, 1] };
    }
    const rawMin = Math.min(...allValues);
    const rawMax = Math.max(...allValues);
    const padding = Math.max((rawMax - rawMin) * 0.12, 0.1);
    const min = rawMin - padding;
    const max = rawMax + padding;
    return { min, max, ticks: buildNiceTicks(min, max) };
  }, [allValues]);

  if (!validSeries.length || !categories.length) {
    return (
      <div className={`flex h-[320px] items-center justify-center rounded-3xl border border-white/6 bg-slate-950/72 text-sm text-slate-400 ${className}`}>
        Graficul liniar va aparea dupa prima serie de date.
      </div>
    );
  }

  return (
    <div className={`relative rounded-3xl border border-white/6 bg-slate-950/72 p-3 ${className}`}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-full w-full">
        <rect x={MARGIN.left} y={MARGIN.top} width={getInnerSize().width} height={getInnerSize().height} fill="rgba(5,12,25,0.82)" rx="18" />
        {domain.ticks.map((tick) => {
          const y = valueToY(tick, domain.min, domain.max);
          return (
            <g key={tick}>
              <line x1={MARGIN.left} y1={y} x2={WIDTH - MARGIN.right} y2={y} stroke="rgba(148,163,184,0.12)" strokeDasharray="4 8" />
              <text x={MARGIN.left - 12} y={y + 4} fill="rgba(148,163,184,0.76)" fontSize="11" textAnchor="end">
                {formatValue(tick)}
              </text>
            </g>
          );
        })}

        {categories.map((label, index) => {
          const x = getPointX(index, categories.length);
          return (
            <g key={label}>
              <line x1={x} y1={MARGIN.top} x2={x} y2={HEIGHT - MARGIN.bottom} stroke="rgba(148,163,184,0.08)" />
              <text x={x} y={HEIGHT - MARGIN.bottom + 20} fill="rgba(148,163,184,0.76)" fontSize="11" textAnchor="middle">
                {label}
              </text>
            </g>
          );
        })}

        <line x1={MARGIN.left} y1={HEIGHT - MARGIN.bottom} x2={WIDTH - MARGIN.right} y2={HEIGHT - MARGIN.bottom} stroke="rgba(255,255,255,0.12)" />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={HEIGHT - MARGIN.bottom} stroke="rgba(255,255,255,0.12)" />

        {validSeries.map((entry) => (
          <g key={entry.label}>
            <path
              d={buildLinePath(entry.data, domain.min, domain.max)}
              fill="none"
              stroke={entry.color}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="chart-series-line"
            />
            {entry.data.map((value, index) => {
              const x = getPointX(index, categories.length);
              const y = valueToY(value, domain.min, domain.max);
              return (
                <circle
                  key={`${entry.label}-${categories[index]}`}
                  cx={x}
                  cy={y}
                  r="5"
                  fill={entry.color}
                  stroke="#04111f"
                  strokeWidth="2"
                  className="chart-point"
                  onMouseEnter={() =>
                    setHover({
                      leftPercent: clamp((x / WIDTH) * 100, 12, 88),
                      topPercent: clamp((y / HEIGHT) * 100 - 8, 10, 78),
                      title: categories[index],
                      lines: validSeries.map((seriesEntry) => `${seriesEntry.label}: ${formatValue(seriesEntry.data[index] ?? 0)}`),
                    })
                  }
                  onMouseLeave={() => setHover(null)}
                />
              );
            })}
          </g>
        ))}

        {axisLabel(xLabel, WIDTH / 2, HEIGHT - 12)}
        {axisLabel(yLabel, 18, HEIGHT / 2, `rotate(-90 18 ${HEIGHT / 2})`)}
      </svg>

      <div className="mt-3 flex flex-wrap gap-3">
        {validSeries.map((entry) => (
          <div key={entry.label} className="flex items-center gap-2 rounded-full border border-white/8 bg-white/5 px-3 py-1.5 text-xs text-slate-200">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
            <span>{entry.label}</span>
          </div>
        ))}
      </div>

      {hover ? (
        <div
          className="pointer-events-none absolute z-10 min-w-[180px] rounded-2xl border border-white/10 bg-slate-950/95 px-3 py-2 text-xs text-slate-100 shadow-[0_10px_40px_rgba(2,6,23,0.6)]"
          style={{ left: `${hover.leftPercent}%`, top: `${hover.topPercent}%`, transform: "translate(-50%, -100%)" }}
        >
          <p className="font-semibold text-white">{hover.title}</p>
          {hover.lines.map((line) => (
            <p key={line} className="mt-1 text-slate-300">
              {line}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function InteractiveBarChart({
  categories,
  series,
  xLabel,
  yLabel,
  className = "",
  formatValue = defaultFormatter,
}: InteractiveBarChartProps) {
  const [hover, setHover] = useState<null | { leftPercent: number; topPercent: number; title: string; lines: string[] }>(null);
  const validSeries = series.filter((entry) => entry.data.length);
  const allValues = validSeries.flatMap((entry) => entry.data);
  const maxValue = Math.max(...allValues, 1);
  const ticks = buildNiceTicks(0, maxValue);
  const inner = getInnerSize();

  if (!validSeries.length || !categories.length) {
    return (
      <div className={`flex h-[320px] items-center justify-center rounded-3xl border border-white/6 bg-slate-950/72 text-sm text-slate-400 ${className}`}>
        Graficul de bare va aparea cand exista suficiente date.
      </div>
    );
  }

  const bandWidth = inner.width / Math.max(categories.length, 1);
  const groupedWidth = bandWidth * 0.72;
  const barWidth = groupedWidth / Math.max(validSeries.length, 1);

  return (
    <div className={`relative rounded-3xl border border-white/6 bg-slate-950/72 p-3 ${className}`}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-full w-full">
        <rect x={MARGIN.left} y={MARGIN.top} width={inner.width} height={inner.height} fill="rgba(5,12,25,0.82)" rx="18" />

        {ticks.map((tick) => {
          const y = valueToY(tick, 0, maxValue);
          return (
            <g key={tick}>
              <line x1={MARGIN.left} y1={y} x2={WIDTH - MARGIN.right} y2={y} stroke="rgba(148,163,184,0.12)" strokeDasharray="4 8" />
              <text x={MARGIN.left - 12} y={y + 4} fill="rgba(148,163,184,0.76)" fontSize="11" textAnchor="end">
                {formatValue(tick)}
              </text>
            </g>
          );
        })}

        {categories.map((label, index) => {
          const centerX = MARGIN.left + bandWidth * index + bandWidth / 2;
          return (
            <text key={label} x={centerX} y={HEIGHT - MARGIN.bottom + 20} fill="rgba(148,163,184,0.76)" fontSize="11" textAnchor="middle">
              {label}
            </text>
          );
        })}

        <line x1={MARGIN.left} y1={HEIGHT - MARGIN.bottom} x2={WIDTH - MARGIN.right} y2={HEIGHT - MARGIN.bottom} stroke="rgba(255,255,255,0.12)" />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={HEIGHT - MARGIN.bottom} stroke="rgba(255,255,255,0.12)" />

        {categories.map((label, categoryIndex) =>
          validSeries.map((entry, seriesIndex) => {
            const value = entry.data[categoryIndex] ?? 0;
            const x = MARGIN.left + bandWidth * categoryIndex + (bandWidth - groupedWidth) / 2 + barWidth * seriesIndex;
            const y = valueToY(value, 0, maxValue);
            const height = HEIGHT - MARGIN.bottom - y;
            return (
              <rect
                key={`${label}-${entry.label}`}
                x={x}
                y={y}
                width={Math.max(10, barWidth - 6)}
                height={Math.max(2, height)}
                rx="8"
                fill={entry.color}
                opacity="0.92"
                className="chart-bar"
                onMouseEnter={() =>
                  setHover({
                    leftPercent: clamp(((x + barWidth / 2) / WIDTH) * 100, 10, 90),
                    topPercent: clamp((y / HEIGHT) * 100 - 4, 12, 82),
                    title: label,
                    lines: validSeries.map((seriesEntry) => `${seriesEntry.label}: ${formatValue(seriesEntry.data[categoryIndex] ?? 0)}`),
                  })
                }
                onMouseLeave={() => setHover(null)}
              />
            );
          }),
        )}

        {axisLabel(xLabel, WIDTH / 2, HEIGHT - 12)}
        {axisLabel(yLabel, 18, HEIGHT / 2, `rotate(-90 18 ${HEIGHT / 2})`)}
      </svg>

      <div className="mt-3 flex flex-wrap gap-3">
        {validSeries.map((entry) => (
          <div key={entry.label} className="flex items-center gap-2 rounded-full border border-white/8 bg-white/5 px-3 py-1.5 text-xs text-slate-200">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
            <span>{entry.label}</span>
          </div>
        ))}
      </div>

      {hover ? (
        <div
          className="pointer-events-none absolute z-10 min-w-[170px] rounded-2xl border border-white/10 bg-slate-950/95 px-3 py-2 text-xs text-slate-100 shadow-[0_10px_40px_rgba(2,6,23,0.6)]"
          style={{ left: `${hover.leftPercent}%`, top: `${hover.topPercent}%`, transform: "translate(-50%, -100%)" }}
        >
          <p className="font-semibold text-white">{hover.title}</p>
          {hover.lines.map((line) => (
            <p key={line} className="mt-1 text-slate-300">
              {line}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function InteractiveScatterChart({
  points,
  xLabel,
  yLabel,
  className = "",
  formatValue = defaultFormatter,
  formatX,
  formatY,
}: InteractiveScatterChartProps) {
  const [hover, setHover] = useState<null | { leftPercent: number; topPercent: number; title: string; lines: string[] }>(null);
  const validPoints = points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const inner = getInnerSize();

  const domain = useMemo(() => {
    if (!validPoints.length) {
      return {
        xMin: 0,
        xMax: 1,
        yMin: 0,
        yMax: 1,
        xTicks: [0, 0.5, 1],
        yTicks: [0, 0.5, 1],
      };
    }

    const xValues = validPoints.map((point) => point.x);
    const yValues = validPoints.map((point) => point.y);
    const xRawMin = Math.min(...xValues);
    const xRawMax = Math.max(...xValues);
    const yRawMin = Math.min(...yValues);
    const yRawMax = Math.max(...yValues);
    const xPadding = Math.max((xRawMax - xRawMin) * 0.12, 0.1);
    const yPadding = Math.max((yRawMax - yRawMin) * 0.12, 0.1);
    const xMin = xRawMin - xPadding;
    const xMax = xRawMax + xPadding;
    const yMin = yRawMin - yPadding;
    const yMax = yRawMax + yPadding;
    return {
      xMin,
      xMax,
      yMin,
      yMax,
      xTicks: buildNiceTicks(xMin, xMax),
      yTicks: buildNiceTicks(yMin, yMax),
    };
  }, [validPoints]);

  if (!validPoints.length) {
    return (
      <div className={`flex h-[320px] items-center justify-center rounded-3xl border border-white/6 bg-slate-950/72 text-sm text-slate-400 ${className}`}>
        Graficul de dispersie va aparea dupa ce exista molecule in biblioteca.
      </div>
    );
  }

  const scaleX = (value: number) => MARGIN.left + ((value - domain.xMin) / (domain.xMax - domain.xMin || 1)) * inner.width;
  const scaleY = (value: number) => MARGIN.top + inner.height - ((value - domain.yMin) / (domain.yMax - domain.yMin || 1)) * inner.height;

  return (
    <div className={`relative rounded-3xl border border-white/6 bg-slate-950/72 p-3 ${className}`}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-full w-full">
        <rect x={MARGIN.left} y={MARGIN.top} width={inner.width} height={inner.height} fill="rgba(5,12,25,0.82)" rx="18" />

        {domain.yTicks.map((tick) => {
          const y = scaleY(tick);
          return (
            <g key={`y-${tick}`}>
              <line x1={MARGIN.left} y1={y} x2={WIDTH - MARGIN.right} y2={y} stroke="rgba(148,163,184,0.12)" strokeDasharray="4 8" />
              <text x={MARGIN.left - 12} y={y + 4} fill="rgba(148,163,184,0.76)" fontSize="11" textAnchor="end">
                {(formatY ?? formatValue)(tick)}
              </text>
            </g>
          );
        })}

        {domain.xTicks.map((tick) => {
          const x = scaleX(tick);
          return (
            <g key={`x-${tick}`}>
              <line x1={x} y1={MARGIN.top} x2={x} y2={HEIGHT - MARGIN.bottom} stroke="rgba(148,163,184,0.08)" />
              <text x={x} y={HEIGHT - MARGIN.bottom + 20} fill="rgba(148,163,184,0.76)" fontSize="11" textAnchor="middle">
                {(formatX ?? formatValue)(tick)}
              </text>
            </g>
          );
        })}

        <line x1={MARGIN.left} y1={HEIGHT - MARGIN.bottom} x2={WIDTH - MARGIN.right} y2={HEIGHT - MARGIN.bottom} stroke="rgba(255,255,255,0.12)" />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={HEIGHT - MARGIN.bottom} stroke="rgba(255,255,255,0.12)" />

        {validPoints.map((point) => {
          const x = scaleX(point.x);
          const y = scaleY(point.y);
          return (
            <circle
              key={`${point.label}-${point.x}-${point.y}`}
              cx={x}
              cy={y}
              r={point.size ?? 6}
              fill={point.color ?? "#24d6ea"}
              fillOpacity="0.88"
              stroke="#04111f"
              strokeWidth="2"
              className="chart-dot"
              onMouseEnter={() =>
                setHover({
                  leftPercent: clamp((x / WIDTH) * 100, 12, 88),
                  topPercent: clamp((y / HEIGHT) * 100 - 8, 10, 82),
                  title: point.label,
                  lines: [`X: ${(formatX ?? formatValue)(point.x)}`, `Y: ${(formatY ?? formatValue)(point.y)}`, ...(point.meta ? [point.meta] : [])],
                })
              }
              onMouseLeave={() => setHover(null)}
            />
          );
        })}

        {axisLabel(xLabel, WIDTH / 2, HEIGHT - 12)}
        {axisLabel(yLabel, 18, HEIGHT / 2, `rotate(-90 18 ${HEIGHT / 2})`)}
      </svg>

      {hover ? (
        <div
          className="pointer-events-none absolute z-10 min-w-[170px] rounded-2xl border border-white/10 bg-slate-950/95 px-3 py-2 text-xs text-slate-100 shadow-[0_10px_40px_rgba(2,6,23,0.6)]"
          style={{ left: `${hover.leftPercent}%`, top: `${hover.topPercent}%`, transform: "translate(-50%, -100%)" }}
        >
          <p className="font-semibold text-white">{hover.title}</p>
          {hover.lines.map((line) => (
            <p key={line} className="mt-1 text-slate-300">
              {line}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}
