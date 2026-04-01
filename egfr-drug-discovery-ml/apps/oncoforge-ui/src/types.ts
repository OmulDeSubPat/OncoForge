export type AgentStatus = "active" | "idle" | "training" | "monitoring";
export type CandidateStatus = "promovata" | "revizie" | "respinsa" | "necunoscut";
export type ViewMode = "3D" | "2D" | "compare";

export interface ControlForm {
  sessionName: string;
  mode: "explorare" | "ghidat_ai" | "iterativ";
  seedCount: number;
  rounds: number;
  variantsPerSeed: number;
  beamWidth: number;
  replaceExisting: boolean;
}

export interface OverviewSummary {
  moleculeCount: number;
  promotedCount: number;
  reviewCount: number;
  rejectedCount: number;
  bestPic50: number;
  bestScore: number;
  meanQed: number;
}

export interface BestMoleculeSummary {
  smiles: string;
  status: CandidateStatus;
  action: string;
  score: number;
  pic50: number;
  uncertainty: number;
  qed: number;
  syntheticFeasibility: number;
  marketReference: string;
  marketSimilarity: number;
  cost10mg: number;
}

export interface LatestRoundSummary {
  round: number;
  seedStep: number;
  newCandidates: number;
  promotedCandidates: number;
  bestScore: number;
}

export interface OverviewPayload {
  sessionName: string;
  mode: string;
  modeLabel: string;
  status: string;
  statusLabel: string;
  message: string;
  updatedAt: string;
  running: boolean;
  progress: number;
  summary: OverviewSummary;
  bestMolecule: BestMoleculeSummary | null;
  latestRound: LatestRoundSummary | null;
}

export interface AgentCard {
  id: string;
  name: string;
  status: AgentStatus;
  contribution: number;
  headline: string;
  lastAction: string;
}

export interface FlowEdge {
  source: string;
  target: string;
  weight: number;
}

export interface MetricPrimary {
  label: string;
  value: number | string;
  tone: "primary" | "warning" | "success" | "info" | "neutral";
}

export interface RadarMetric {
  axis: string;
  value: number;
}

export interface RiskFlag {
  label: string;
  tone: "success" | "warning" | "danger";
}

export interface ComparisonInfo {
  referenceName: string;
  similarity: number;
  novelty: number;
  marketSupport: number;
}

export interface MoleculeMetrics {
  primary: MetricPrimary[];
  radar: RadarMetric[];
  riskFlags: RiskFlag[];
  comparison: ComparisonInfo | null;
}

export interface MoleculeView {
  smiles: string;
  molBlock: string;
  svg2d: string;
  atomCount: number;
  formula: string;
}

export interface BreakdownItem {
  label: string;
  value: number;
  tone?: "positive" | "negative";
  unit?: string;
}

export interface ExplainabilityThreshold {
  label: string;
  passed: boolean;
  value: string;
  reference: string;
}

export interface ExplainabilityPayload {
  pros: string[];
  cons: string[];
  dominantAgent: string;
  penalties: BreakdownItem[];
  thresholds: ExplainabilityThreshold[];
  summary: string;
}

export interface LiabilitySignal {
  label: string;
  tone: "success" | "warning" | "danger";
  value: string;
  note: string;
}

export interface AdmetPayload {
  summary: string;
  wildTypeProxy: number;
  reactivityRisk: number;
  liabilities: LiabilitySignal[];
}

export interface DecisionEvent {
  id: string;
  title: string;
  detail: string;
  timestamp: string;
  category: string;
  tone: "info" | "success" | "warning";
  round?: number;
}

export interface SelectedMolecule {
  rank: number;
  smiles: string;
  status: CandidateStatus;
  score: number;
  round: number;
  action: string;
  route: string;
  parent: string;
  lineagePath: string;
  deltaPic50: number;
  deltaQed: number;
  deltaScore: number;
  cost10mg: number;
  cost100mg: number;
  marketReference: string;
  marketSimilarity: number;
  view: MoleculeView;
  metrics: MoleculeMetrics;
  agentContributions: AgentCard[];
  rankingBreakdown: BreakdownItem[];
  costBreakdown: BreakdownItem[];
  explainability: ExplainabilityPayload;
  admet: AdmetPayload;
  decisionHistory: DecisionEvent[];
}

export interface DetailPayload {
  selected: SelectedMolecule | null;
}

export interface TimelineGeneration {
  round: number;
  seedStep: number;
  newCandidates: number;
  promotedCandidates: number;
  totalCandidates: number;
  bestScore: number;
  avgCost10mg: number;
  minCost10mg: number;
  timestamp: string;
}

export interface TimelineNode {
  id: string;
  parentId: string | null;
  label: string;
  status: CandidateStatus;
  round: number;
  rank: number;
  score: number;
  pic50: number;
  deltaPic50: number;
  deltaScore: number;
}

export interface TimelineEdge {
  source: string;
  target: string;
  label: string;
}

export interface TimelinePayload {
  generations: TimelineGeneration[];
  nodes: TimelineNode[];
  edges: TimelineEdge[];
}

export interface RewardPoint {
  round: number;
  bestScore: number;
  avgCost10mg: number;
  timestamp: string;
}

export interface PenaltyPoint {
  round: number;
  toxicityPenalty: number;
  invalidPenalty: number;
  uncertaintyPenalty: number;
  rewardRiskPenalty: number;
  verifiedReward: number;
  exploration: number;
  exploitation: number;
}

export interface RLMonitorPayload {
  rewardSeries: RewardPoint[];
  penaltySeries: PenaltyPoint[];
  verifiableNotes: string[];
}

export interface LibraryRow {
  id: string;
  rank: number;
  smiles: string;
  parent: string;
  round: number;
  status: CandidateStatus;
  statusLabel: string;
  score: number;
  pic50: number;
  qed: number;
  uncertainty: number;
  cost10mg: number;
  action: string;
  route: string;
  marketReference: string;
  saScore?: number;
  syntheticFeasibility?: number;
  marketSimilarity?: number;
  novelty?: number;
  risk?: number;
  pains?: boolean;
  structuralAlerts?: number;
  verifiedReward?: number;
  mw?: number;
  logP?: number;
  tpsa?: number;
  auditPass?: boolean;
  generatorPriority?: number;
}

export interface MarketCompareMetrics {
  potency: number;
  qed: number;
  sa: number;
  cost: number;
  novelty: number;
  risk: number;
}

export interface MarketCompareEntry {
  id: string;
  name: string;
  kind: "selectata" | "comparator";
  referenceClass: string;
  smiles: string;
  raw: MarketCompareMetrics;
  normalized: MarketCompareMetrics;
}

export interface MarketComparePayload {
  candidateSmiles: string;
  axes: string[];
  entries: MarketCompareEntry[];
}

export interface SessionCompareItem {
  sessionName: string;
  modeLabel: string;
  statusLabel: string;
  moleculeCount: number;
  promotedCount: number;
  bestPic50: number;
  bestScore: number;
  meanCost10mg: number;
  meanUncertainty: number;
  bottleneck: string;
  updatedAt: string;
  isCurrent: boolean;
}

export interface ExperimentalPlanEntry {
  smiles: string;
  rank: number;
  name: string;
  priority: string;
  assay: string;
  control: string;
  materialPlan: string;
  estimatedCost: number;
  rationale: string;
  route: string;
  status: string;
}

export interface AgentSeriesPoint {
  round: number;
  generator: number;
  toxicity: number;
  validator: number;
  optimizer: number;
}

export interface RankingStabilityPoint {
  round: number;
  topScore: number;
  meanScore: number;
  promotedRate: number;
  scoreSpread: number;
}

export interface PipelineStageSnapshot {
  label: string;
  count: number;
  share: number;
  note: string;
}

export interface PipelineProgressPoint {
  round: number;
  auditPass: number;
  structuralReady: number;
  feasible: number;
  marketReady: number;
  experimentalProxy: number;
  promoted: number;
}

export interface CandidateMaturationPoint {
  round: number;
  verifiedRewardScore: number;
  feasibility: number;
  novelty: number;
  structural: number;
  costScore: number;
  safety: number;
  experimentalReadyRate: number;
}

export interface DashboardAnalyticsPayload {
  agentSeries: AgentSeriesPoint[];
  rankingStability: RankingStabilityPoint[];
  pipelineStages: PipelineStageSnapshot[];
  pipelineProgress: PipelineProgressPoint[];
  maturationSeries: CandidateMaturationPoint[];
}

export interface ChemistNotebookEntry {
  smiles: string;
  verdict: "merita sinteza" | "de revazut" | "prea scumpa" | "schelet interesant";
  tags: string[];
  note: string;
  updatedAt: string;
}

export interface DashboardPayload {
  overview: OverviewPayload;
  agents: AgentCard[];
  flows: FlowEdge[];
  detail: DetailPayload;
  timeline: TimelinePayload;
  rlMonitor: RLMonitorPayload;
  library: LibraryRow[];
  marketCompare: MarketComparePayload;
  sessionCompare: SessionCompareItem[];
  experimentalPlanner: ExperimentalPlanEntry[];
  analytics: DashboardAnalyticsPayload;
  logs: string;
  sources: string[];
}

export interface AgentState {
  id: string;
  name: string;
  status: "active" | "idle" | "training" | "analysis" | "stable";
  score: number;
  lastAction: string;
  contribution: number;
  color: string;
}
export interface ObjectiveWeights {
  potency: number;
  toxicity: number;
  validity: number;
  synthesizability: number;
}

export interface Candidate {
  id: string;
  name: string;
  smiles: string;
  parent: string;
  round: number;
  score: number;
  status: string;
  reason: string;
  mutation: string;
  highlights: string[];
  metrics: Record<string, number>;
  agentContribution: Record<string, number>;
  graph: {
    nodes: Array<{ id: string; label: string; x: number; y: number; role: string }>;
    bonds: Array<{ from: string; to: string; order: number; kind?: string }>;
  };
}

export interface TimelineFrame {
  generation: number;
  candidateId: string;
  parentId: string;
  mutation: string;
  deltaPic50: number;
  deltaScore: number;
  reward: number;
  verifiedReward: number;
  toxicityPenalty: number;
  invalidPenalty: number;
  uncertaintyPenalty: number;
  exploration: number;
  exploitation: number;
}
