import { cellOf, parseCell, CELL_SIZE_M } from "../location";
import {
  addGoal,
  makeInitialState,
  updateHost,
  generateQuestOffers,
  acceptQuest,
  submitOutcome,
  discoverCells,
  ensureStateShape
} from "../engine";

function ready(domain: Parameters<typeof addGoal>[2] = "exploration") {
  let s = addGoal(makeInitialState(), "Chart the city", domain);
  s = { ...s, profile: { ...s.profile, preferences: { ...s.profile.preferences, outdoorQuests: true, physicalQuests: true } } };
  s = updateHost(s, { ...s.host, energy: 0.6, focus: 0.55, stress: 0.3, timeAvailableMinutes: 40 });
  return s;
}

describe("cell quantization (pure)", () => {
  test("deterministic: same coordinate always yields the same cell", () => {
    expect(cellOf(-6.2088, 106.8456)).toBe(cellOf(-6.2088, 106.8456)); // Jakarta
    expect(cellOf(52.52, 13.405)).toBe(cellOf(52.52, 13.405)); // Berlin
  });

  test("points within a few meters share a cell; points ~1km apart do not", () => {
    const base = cellOf(-6.2088, 106.8456);
    expect(cellOf(-6.20881, 106.84561)).toBe(base); // ~1.5m away
    const oneKmNorth = cellOf(-6.2088 + 1000 / 111320, 106.8456);
    expect(oneKmNorth).not.toBe(base);
  });

  test("parseCell round-trips ids including negatives", () => {
    const id = cellOf(-6.2088, 106.8456);
    const parsed = parseCell(id)!;
    expect(parsed).toBeTruthy();
    expect(id).toBe(`c${parsed.row}_${parsed.col}`);
    expect(parseCell("not-a-cell")).toBeNull();
  });

  test("cell size constant is the documented ~400m", () => {
    expect(CELL_SIZE_M).toBe(400);
  });
});

describe("territory discovery (engine)", () => {
  test("new cells are recorded once; repeats dedupe; novelty flag tracks the latest sample", () => {
    let s = ready();
    s = discoverCells(s, ["c100_200"]);
    expect(s.world.discoveredCells).toEqual(["c100_200"]);
    expect(s.systemSignals.currentCellNovel).toBe(true);
    s = discoverCells(s, ["c100_200"]);
    expect(s.world.discoveredCells).toEqual(["c100_200"]); // no dupe
    expect(s.systemSignals.currentCellNovel).toBe(false); // known ground now
    s = discoverCells(s, ["c100_200", "c100_201"]);
    expect(s.world.discoveredCells).toEqual(["c100_200", "c100_201"]);
    expect(s.systemSignals.currentCellNovel).toBe(true);
  });

  test("discovery emits milestone ceremonies but never XP", () => {
    let s = ready();
    const xpBefore = s.progression.xp;
    s = discoverCells(s, ["c1_1"]);
    expect(s.pendingCeremony?.kind).toBe("discovery"); // first-territory milestone
    s = { ...s, pendingCeremony: null };
    s = discoverCells(s, ["c1_2", "c1_3", "c1_4", "c1_5"]); // crosses 5
    expect(s.pendingCeremony?.kind).toBe("discovery");
    expect(s.progression.xp).toBe(xpBefore); // the map is its own reward
  });

  test("empty input is a no-op", () => {
    const s = ready();
    expect(discoverCells(s, []).world.discoveredCells).toEqual([]);
  });
});

describe("Scout advisor", () => {
  test("novel ground surfaces a Scout note on exploration offers; known ground stays silent", () => {
    let s = ready("exploration");
    s = { ...s, systemSignals: { ...s.systemSignals, currentCellNovel: true } };
    const novel = generateQuestOffers(s);
    const scoutSpoke = novel.questOffers.some((o) => o.councilNote?.startsWith("Scout"));
    expect(scoutSpoke).toBe(true);
    s = { ...s, systemSignals: { ...s.systemSignals, currentCellNovel: false } };
    const known = generateQuestOffers(s);
    expect(known.questOffers.some((o) => o.councilNote?.startsWith("Scout"))).toBe(false);
  });
});

describe("location evidence", () => {
  test("a location item raises verification confidence and persists its cell id", () => {
    let s = ready("exploration");
    s = generateQuestOffers(s);
    s = acceptQuest(s, s.questOffers[0].id);
    const single = submitOutcome(s, "COMPLETED_FULL", "self_report", "went out");
    const withLocation = submitOutcome(s, "COMPLETED_FULL", "self_report", "went out", [
      { kind: "location", cellId: "c100_200", note: "Completed in newly charted territory." }
    ]);
    const outcome = withLocation.questHistory[0].outcome!;
    const locationItem = outcome.evidence!.find((e) => e.kind === "location")!;
    expect(locationItem.cellId).toBe("c100_200");
    expect(outcome.verificationConfidence!).toBeGreaterThan(single.questHistory[0].outcome!.verificationConfidence!);
    expect(withLocation.progression.xp).toBeGreaterThan(single.progression.xp);
  });
});

describe("migration", () => {
  test("old saves gain discoveredCells and currentCellNovel on load", () => {
    const old: any = ready();
    delete old.world.discoveredCells;
    delete old.systemSignals.currentCellNovel;
    const healed = ensureStateShape(JSON.parse(JSON.stringify(old)));
    expect(healed.world.discoveredCells).toEqual([]);
    expect(healed.systemSignals.currentCellNovel).toBe(false);
  });

  test("location preferences default OFF (strict opt-in)", () => {
    const prefs = makeInitialState().profile.preferences;
    expect(prefs.locationQuests).toBe(false);
  });
});
