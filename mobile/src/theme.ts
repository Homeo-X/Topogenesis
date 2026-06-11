import { StyleSheet } from "react-native";

export const rankColors: Record<string, string> = {
  S: "#fbbf24", A: "#67e8f9", B: "#7dd3fc", C: "#94a3b8", D: "#64748b"
};

export const domainColors: Record<string, string> = {
  craft: "#f59e0b", creation: "#f472b6", mind: "#a78bfa", learning: "#818cf8",
  planning: "#34d399", order: "#2dd4bf", body: "#f87171", recovery: "#4ade80",
  courage: "#fb7185", social: "#fbbf24", exploration: "#38bdf8"
};

export type Tab = "system" | "checkin" | "quest" | "sheet" | "world";

export type IconName =
  | "timer"
  | "add"
  | "add-circle"
  | "bar-chart"
  | "book"
  | "checkmark-circle"
  | "close-circle"
  | "cloud-upload"
  | "compass"
  | "document-text"
  | "flag"
  | "map"
  | "moon"
  | "person"
  | "pulse"
  | "refresh"
  | "remove"
  | "scan"
  | "shield-checkmark"
  | "sparkles"
  | "terminal";

const dayStyles = StyleSheet.create({
  shell: {
    flex: 1,
    backgroundColor: "#bfefff"
  },
  header: {
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "rgba(14, 165, 233, 0.32)",
    backgroundColor: "rgba(207, 244, 255, 0.84)"
  },
  eyebrow: {
    color: "#0284c7",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0
  },
  title: {
    color: "#07324d",
    fontSize: 23,
    fontWeight: "800",
    letterSpacing: 0
  },
  levelBadge: {
    borderColor: "rgba(34, 211, 238, 0.62)",
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: "rgba(125, 211, 252, 0.48)"
  },
  levelText: {
    color: "#07324d",
    fontWeight: "800"
  },
  themeToggle: {
    width: 42,
    height: 42,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 116, 144, 0.32)",
    backgroundColor: "rgba(125, 211, 252, 0.48)",
    alignItems: "center",
    justifyContent: "center"
  },
  themeToggleText: {
    color: "#075985",
    fontSize: 20,
    fontWeight: "900"
  },
  content: {
    flex: 1
  },
  contentPad: {
    width: "100%",
    maxWidth: 920,
    alignSelf: "center",
    padding: 16,
    paddingBottom: 104
  },
  systemMessage: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.36)",
    backgroundColor: "rgba(224, 247, 255, 0.7)",
    padding: 14,
    marginBottom: 12
  },
  systemLabel: {
    color: "#0284c7",
    fontSize: 11,
    fontWeight: "800",
    marginBottom: 6
  },
  systemText: {
    color: "#123853",
    fontSize: 15,
    lineHeight: 22
  },
  panel: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.26)",
    backgroundColor: "rgba(255, 255, 255, 0.58)",
    padding: 16,
    marginBottom: 14
  },
  panelHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 12
  },
  panelTitle: {
    color: "#07324d",
    fontSize: 18,
    fontWeight: "800"
  },
  bodyText: {
    color: "#1d435f",
    fontSize: 15,
    lineHeight: 23
  },
  muted: {
    color: "#55758a",
    fontSize: 13
  },
  routeStatus: {
    color: "#0369a1",
    fontSize: 13,
    fontWeight: "800",
    marginTop: 8
  },
  input: {
    color: "#07324d",
    minHeight: 48,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.26)",
    backgroundColor: "rgba(255, 255, 255, 0.54)",
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginTop: 12,
    marginBottom: 12
  },
  arcBox: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.24)",
    backgroundColor: "rgba(224, 247, 255, 0.46)",
    padding: 12,
    marginTop: 12,
    marginBottom: 12
  },
  selectorBox: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.34)",
    backgroundColor: "rgba(186, 230, 253, 0.38)",
    padding: 12,
    marginTop: 12
  },
  assumptionCard: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.24)",
    backgroundColor: "rgba(224, 247, 255, 0.46)",
    padding: 12,
    marginBottom: 10
  },
  offerCard: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.34)",
    backgroundColor: "rgba(255, 255, 255, 0.62)",
    padding: 14,
    marginTop: 12
  },
  offerRank: {
    color: "#0284c7",
    fontSize: 13,
    fontWeight: "900",
    textTransform: "uppercase"
  },
  inlineActions: {
    gap: 8,
    marginTop: 10
  },
  arcTitle: {
    color: "#07324d",
    fontSize: 18,
    fontWeight: "900",
    marginBottom: 8
  },
  milestoneRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(56, 189, 248, 0.24)",
    paddingVertical: 10
  },
  milestoneDot: {
    width: 30,
    height: 30,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.24)",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255, 255, 255, 0.42)"
  },
  milestoneDotActive: {
    borderColor: "#22d3ee",
    backgroundColor: "rgba(125, 211, 252, 0.52)"
  },
  milestoneDotText: {
    color: "#075985",
    fontSize: 12,
    fontWeight: "900"
  },
  milestoneStatus: {
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: "800",
    textTransform: "uppercase"
  },
  segmentWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 12
  },
  segment: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.26)",
    paddingHorizontal: 10,
    paddingVertical: 8,
    backgroundColor: "rgba(255, 255, 255, 0.46)"
  },
  segmentActive: {
    backgroundColor: "rgba(125, 211, 252, 0.6)",
    borderColor: "#22d3ee"
  },
  segmentText: {
    color: "#55758a",
    fontSize: 13,
    fontWeight: "700"
  },
  segmentTextActive: {
    color: "#075985"
  },
  primaryButton: {
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: "#38bdf8",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
    paddingHorizontal: 14
  },
  primaryButtonText: {
    color: "#032332",
    fontWeight: "900",
    fontSize: 15
  },
  ghostButton: {
    minHeight: 46,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 116, 144, 0.32)",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
    paddingHorizontal: 14,
    marginTop: 10
  },
  ghostButtonText: {
    color: "#075985",
    fontWeight: "800",
    fontSize: 15
  },
  iconButton: {
    width: 38,
    height: 38,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.28)",
    alignItems: "center",
    justifyContent: "center"
  },
  pressed: {
    opacity: 0.72,
    transform: [{ scale: 0.98 }]
  },
  pressedStrong: {
    opacity: 0.86,
    transform: [{ scale: 0.98 }]
  },
  toggleRow: {
    minHeight: 48,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "rgba(56, 189, 248, 0.22)"
  },
  metric: {
    marginBottom: 14
  },
  metricHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 8
  },
  sectionLabel: {
    color: "#0284c7",
    fontSize: 12,
    fontWeight: "800",
    marginTop: 6,
    marginBottom: 6
  },
  stepRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10
  },
  progressHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 10,
    marginBottom: 8
  },
  progressTrack: {
    flex: 1,
    width: "100%",
    height: 10,
    borderRadius: 8,
    overflow: "hidden",
    backgroundColor: "rgba(14, 165, 233, 0.18)"
  },
  progressFill: {
    height: "100%",
    backgroundColor: "#38bdf8"
  },
  threatFill: {
    height: "100%",
    backgroundColor: "#f59e0b"
  },
  questMeta: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 10
  },
  chip: {
    borderRadius: 8,
    backgroundColor: "rgba(224, 247, 255, 0.54)",
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.26)",
    paddingHorizontal: 9,
    paddingVertical: 6
  },
  chipText: {
    color: "#1d435f",
    fontSize: 12,
    fontWeight: "800"
  },
  questText: {
    color: "#07324d",
    fontSize: 16,
    lineHeight: 23,
    fontWeight: "700"
  },
  planList: {
    gap: 10,
    marginTop: 8,
    marginBottom: 6
  },
  planStep: {
    flexDirection: "row",
    gap: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(14, 165, 233, 0.24)",
    backgroundColor: "rgba(255, 255, 255, 0.46)",
    padding: 12
  },
  planStepDone: {
    borderColor: "#22d3ee",
    backgroundColor: "rgba(186, 230, 253, 0.42)"
  },
  stepTime: {
    width: 50,
    minHeight: 56,
    borderRadius: 8,
    backgroundColor: "rgba(125, 211, 252, 0.58)",
    alignItems: "center",
    justifyContent: "center"
  },
  stepTimeText: {
    color: "#075985",
    fontSize: 17,
    fontWeight: "900"
  },
  stepTimeUnit: {
    color: "#0284c7",
    fontSize: 10,
    fontWeight: "800"
  },
  stepBody: {
    flex: 1
  },
  stepTitle: {
    color: "#07324d",
    fontSize: 15,
    fontWeight: "900",
    marginBottom: 4
  },
  stepOutput: {
    color: "#0284c7",
    fontSize: 13,
    lineHeight: 18,
    marginTop: 6,
    fontWeight: "700"
  },
  criteriaLine: {
    color: "#1d435f",
    fontSize: 14,
    lineHeight: 21,
    marginBottom: 3
  },
  antiAvoidance: {
    color: "#facc15",
    fontSize: 13,
    lineHeight: 19,
    marginTop: 8,
    fontWeight: "800"
  },
  rowGap: {
    marginTop: 12
  },
  bigNumber: {
    color: "#07324d",
    fontSize: 28,
    fontWeight: "900",
    marginBottom: 6
  },
  statRow: {
    minHeight: 38,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "rgba(56, 189, 248, 0.22)"
  },
  statName: {
    color: "#1d435f",
    fontSize: 15,
    fontWeight: "700"
  },
  statValue: {
    color: "#0284c7",
    fontSize: 16,
    fontWeight: "900"
  },
  logLine: {
    color: "#1d435f",
    fontSize: 14,
    lineHeight: 21,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(56, 189, 248, 0.22)"
  },
  nav: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    minHeight: 72,
    paddingTop: 8,
    paddingBottom: 10,
    paddingHorizontal: 8,
    backgroundColor: "rgba(207, 244, 255, 0.9)",
    borderTopWidth: 1,
    borderTopColor: "rgba(56, 189, 248, 0.34)",
    flexDirection: "row",
    justifyContent: "space-around"
  },
  navButton: {
    flex: 1,
    minWidth: 0,
    alignItems: "center",
    gap: 4,
    borderRadius: 8,
    paddingVertical: 5
  },
  navButtonActive: {
    backgroundColor: "rgba(125, 211, 252, 0.62)"
  },
  ceremonyBackdrop: {
    flex: 1,
    backgroundColor: "rgba(2, 12, 18, 0.88)",
    alignItems: "center",
    justifyContent: "center",
    padding: 28
  },
  ceremonyCard: {
    width: "100%",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(56, 189, 248, 0.6)",
    backgroundColor: "rgba(8, 27, 36, 0.97)",
    padding: 26,
    alignItems: "center",
    gap: 10
  },
  ceremonyTitle: {
    color: "#67e8f9",
    fontSize: 26,
    fontWeight: "900",
    letterSpacing: 4
  },
  ceremonyLine: {
    color: "#e2f6ff",
    fontSize: 15,
    textAlign: "center"
  },
  navText: {
    color: "#55758a",
    fontSize: 11,
    fontWeight: "800"
  },
  navTextActive: {
    color: "#075985"
  },
  menuRoot: {
    flex: 1,
    flexDirection: "row"
  },
  menuDrawer: {
    width: "86%",
    maxWidth: 360,
    height: "100%",
    backgroundColor: "#bfefff",
    borderRightWidth: 1,
    borderRightColor: "rgba(14, 165, 233, 0.32)"
  },
  menuBackdrop: {
    flex: 1,
    backgroundColor: "rgba(2, 16, 28, 0.45)"
  },
  menuHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingBottom: 12
  },
  menuTitle: {
    color: "#07324d",
    fontSize: 22,
    fontWeight: "800"
  },
  menuScroll: {
    flex: 1
  },
  menuContent: {
    padding: 16,
    paddingBottom: 48
  }
});

const nightStyles = StyleSheet.create({
  shell: { backgroundColor: "#03111f" },
  header: {
    borderBottomColor: "rgba(34, 211, 238, 0.34)",
    backgroundColor: "rgba(4, 26, 45, 0.9)"
  },
  eyebrow: { color: "#5ee7ff" },
  title: { color: "#e0fbff" },
  levelBadge: {
    borderColor: "rgba(34, 211, 238, 0.62)",
    backgroundColor: "rgba(8, 80, 120, 0.52)"
  },
  levelText: { color: "#d6fbff" },
  systemMessage: {
    borderColor: "rgba(34, 211, 238, 0.48)",
    backgroundColor: "rgba(5, 38, 66, 0.66)"
  },
  systemLabel: { color: "#22d3ee" },
  systemText: { color: "#d7f9ff" },
  panel: {
    borderColor: "rgba(56, 189, 248, 0.3)",
    backgroundColor: "rgba(6, 28, 50, 0.62)"
  },
  panelTitle: { color: "#e0fbff" },
  bodyText: { color: "#c9eef7" },
  muted: { color: "#83b8c8" },
  routeStatus: { color: "#8ff6ff" },
  input: {
    color: "#e0fbff",
    borderColor: "rgba(56, 189, 248, 0.32)",
    backgroundColor: "rgba(3, 18, 34, 0.68)"
  },
  arcBox: {
    borderColor: "rgba(56, 189, 248, 0.28)",
    backgroundColor: "rgba(3, 23, 42, 0.58)"
  },
  selectorBox: {
    borderColor: "rgba(34, 211, 238, 0.42)",
    backgroundColor: "rgba(6, 41, 68, 0.58)"
  },
  assumptionCard: {
    borderColor: "rgba(56, 189, 248, 0.28)",
    backgroundColor: "rgba(3, 23, 42, 0.58)"
  },
  offerCard: {
    borderColor: "rgba(34, 211, 238, 0.48)",
    backgroundColor: "rgba(4, 30, 55, 0.68)"
  },
  offerRank: { color: "#22d3ee" },
  arcTitle: { color: "#e0fbff" },
  milestoneRow: { borderBottomColor: "rgba(56, 189, 248, 0.24)" },
  milestoneDot: {
    borderColor: "rgba(56, 189, 248, 0.28)",
    backgroundColor: "rgba(3, 18, 34, 0.58)"
  },
  milestoneDotActive: {
    borderColor: "#22d3ee",
    backgroundColor: "rgba(8, 80, 120, 0.58)"
  },
  milestoneDotText: { color: "#cffafe" },
  segment: {
    borderColor: "rgba(56, 189, 248, 0.3)",
    backgroundColor: "rgba(3, 18, 34, 0.58)"
  },
  segmentActive: {
    backgroundColor: "rgba(14, 116, 144, 0.56)",
    borderColor: "#22d3ee"
  },
  segmentText: { color: "#9bd7e7" },
  segmentTextActive: { color: "#e0fbff" },
  primaryButton: { backgroundColor: "#22d3ee" },
  ghostButton: { borderColor: "rgba(34, 211, 238, 0.48)" },
  ghostButtonText: { color: "#d9fbff" },
  iconButton: { borderColor: "rgba(56, 189, 248, 0.34)" },
  toggleRow: { borderBottomColor: "rgba(56, 189, 248, 0.22)" },
  sectionLabel: { color: "#5ee7ff" },
  progressTrack: { backgroundColor: "rgba(56, 189, 248, 0.18)" },
  progressFill: { backgroundColor: "#22d3ee" },
  chip: {
    backgroundColor: "rgba(6, 41, 68, 0.6)",
    borderColor: "rgba(56, 189, 248, 0.32)"
  },
  chipText: { color: "#c9eef7" },
  questText: { color: "#e0fbff" },
  planStep: {
    borderColor: "rgba(56, 189, 248, 0.28)",
    backgroundColor: "rgba(3, 23, 42, 0.58)"
  },
  planStepDone: {
    borderColor: "#22d3ee",
    backgroundColor: "rgba(8, 80, 120, 0.46)"
  },
  stepTime: { backgroundColor: "rgba(8, 80, 120, 0.64)" },
  stepTimeText: { color: "#e0fbff" },
  stepTimeUnit: { color: "#67e8f9" },
  stepTitle: { color: "#e0fbff" },
  stepOutput: { color: "#67e8f9" },
  criteriaLine: { color: "#c9eef7" },
  antiAvoidance: { color: "#facc15" },
  bigNumber: { color: "#e0fbff" },
  statRow: { borderBottomColor: "rgba(56, 189, 248, 0.22)" },
  statName: { color: "#c9eef7" },
  statValue: { color: "#22d3ee" },
  logLine: {
    color: "#c9eef7",
    borderBottomColor: "rgba(56, 189, 248, 0.22)"
  },
  nav: {
    backgroundColor: "rgba(4, 26, 45, 0.92)",
    borderTopColor: "rgba(56, 189, 248, 0.34)"
  },
  navButtonActive: { backgroundColor: "rgba(14, 116, 144, 0.5)" },
  navText: { color: "#7fb4c4" },
  navTextActive: { color: "#e0fbff" },
  themeToggle: {
    borderColor: "rgba(34, 211, 238, 0.6)",
    backgroundColor: "rgba(8, 80, 120, 0.5)"
  },
  themeToggleText: { color: "#e0fbff" },
  menuDrawer: {
    backgroundColor: "#03111f",
    borderRightColor: "rgba(34, 211, 238, 0.34)"
  },
  menuTitle: { color: "#e0fbff" }
});

let activeNightMode = false;

export function setActiveNightMode(value: boolean) {
  activeNightMode = value;
}

export function isNightMode() {
  return activeNightMode;
}

export const styles = new Proxy(dayStyles, {
  get(target, property: keyof typeof dayStyles) {
    const base = target[property];
    const night = activeNightMode ? nightStyles[property as keyof typeof nightStyles] : undefined;
    return night ? [base, night] : base;
  }
}) as typeof dayStyles;
