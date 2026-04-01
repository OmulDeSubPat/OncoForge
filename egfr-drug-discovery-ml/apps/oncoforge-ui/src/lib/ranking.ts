import type { LibraryRow } from "@/types";

export interface WhatIfWeights {
  potency: number;
  toxicity: number;
  synthesizability: number;
  cost: number;
  novelty: number;
}

export interface RankedWhatIfRow {
  altRank: number;
  altScore: number;
  scoreDelta: number;
  row: LibraryRow;
}

export const defaultWhatIfWeights: WhatIfWeights = {
  potency: 0.34,
  toxicity: 0.2,
  synthesizability: 0.18,
  cost: 0.12,
  novelty: 0.16,
};

function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

export function normalizeWeights(weights: WhatIfWeights): WhatIfWeights {
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0) || 1;
  return {
    potency: weights.potency / total,
    toxicity: weights.toxicity / total,
    synthesizability: weights.synthesizability / total,
    cost: weights.cost / total,
    novelty: weights.novelty / total,
  };
}

export function computeWhatIfScore(row: LibraryRow, weights: WhatIfWeights): number {
  const normalized = normalizeWeights(weights);
  const potency = clamp(row.pic50 / 10);
  const safety = clamp(
    1 -
      (row.risk ?? 0) * 0.65 -
      Math.min(0.3, (row.structuralAlerts ?? 0) * 0.08) -
      ((row.pains ?? false) ? 0.22 : 0),
  );
  const synth = clamp(
    ((row.syntheticFeasibility ?? 0.5) * 0.65) +
      (1 - Math.max(0, (row.saScore ?? 4) - 1) / 6) * 0.35,
  );
  const cost = clamp(1 / (1 + row.cost10mg / 35));
  const novelty = clamp(row.novelty ?? Math.max(0, 1 - (row.marketSimilarity ?? 0)));
  const auditBonus = row.auditPass === false ? -0.06 : 0.03;

  return (
    potency * normalized.potency +
    safety * normalized.toxicity +
    synth * normalized.synthesizability +
    cost * normalized.cost +
    novelty * normalized.novelty +
    auditBonus
  );
}

export function rankLibraryWithWhatIf(library: LibraryRow[], weights: WhatIfWeights): RankedWhatIfRow[] {
  return [...library]
    .map((row) => {
      const altScore = computeWhatIfScore(row, weights);
      return {
        row,
        altScore,
        scoreDelta: altScore - row.score,
        altRank: 0,
      };
    })
    .sort((left, right) => right.altScore - left.altScore)
    .map((entry, index) => ({
      ...entry,
      altRank: index + 1,
    }));
}
