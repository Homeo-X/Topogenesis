import { EvidenceType, OutcomeType } from "./types";

export const outcomes: Array<{ id: OutcomeType; label: string }> = [
  { id: "COMPLETED_FULL", label: "Complete" },
  { id: "EXCEEDED_OBJECTIVE", label: "Exceeded" },
  { id: "COMPLETED_CREATIVELY", label: "Creative" },
  { id: "COMPLETED_WITH_COST", label: "Costly" },
  { id: "COMPLETED_LOW_QUALITY", label: "Low Quality" },
  { id: "REPLACED_WITH_EQUIVALENT", label: "Equivalent" },
  { id: "COMPLETED_PARTIAL", label: "Partial" },
  { id: "FAILED_ATTEMPTED", label: "Attempted" },
  { id: "FAILED_BLOCKED", label: "Blocked" },
  { id: "FAILED_AVOIDED", label: "Avoided" },
  { id: "SKIPPED_CONSCIOUSLY", label: "Skipped" }
];

export const evidenceTypes: Array<{ id: EvidenceType; label: string }> = [
  { id: "self_report", label: "Report" },
  { id: "reflection", label: "Reflect" },
  { id: "artifact", label: "Artifact" },
  { id: "timer", label: "Timer" }
];
