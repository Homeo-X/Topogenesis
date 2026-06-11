import React, { useEffect, useState } from "react";
import { Text, View } from "react-native";

import { styles } from "../theme";
import { AnimatedMeter, GhostButton, Panel, PrimaryButton } from "../components/common";
import { FogMap } from "../components/FogMap";
import { useStore } from "../state/store";
import { locationEnabled, sampleCellOnce } from "../location";

export function WorldScreen() {
  const { state, getState, dispatch } = useStore();
  const threat = Math.round(state.world.threatLevel * 100);
  const regionPct = Math.round(state.world.regionProgress * 100);

  // On-demand scan: ONE position read when this tab opens (and on manual
  // re-scan). This is the only place the app reads location besides quest
  // completion — never continuous, never background.
  const [scanning, setScanning] = useState(false);
  const [currentCell, setCurrentCell] = useState<string | null>(null);
  const [scanFailed, setScanFailed] = useState(false);
  const [generating, setGenerating] = useState(false);

  async function scanHere() {
    if (!locationEnabled(getState()) || scanning) return;
    setScanning(true);
    setScanFailed(false);
    try {
      const cellId = await sampleCellOnce();
      if (!cellId) {
        setScanFailed(true);
        return;
      }
      setCurrentCell(cellId);
      await dispatch({ type: "DISCOVER_CELLS", cells: [cellId] });
    } finally {
      setScanning(false);
    }
  }

  useEffect(() => {
    void scanHere();
    // One scan per tab open, by design.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Quest-from-here: pin an exploration goal (create one if none) and generate.
  // The Scout advisor reads the novelty flag set by the scan above.
  async function questHere() {
    setGenerating(true);
    try {
      const fresh = getState();
      let goalId = fresh.profile.goals.find((g) => g.status === "active" && g.domain === "exploration")?.id;
      if (!goalId) {
        await dispatch({ type: "ADD_GOAL", text: "Explore the world around me", domain: "exploration" });
        goalId = getState().profile.goals[0]?.id;
      }
      await dispatch({ type: "GENERATE_QUEST", goalId });
    } finally {
      setGenerating(false);
    }
  }

  const novelGround = state.systemSignals.currentCellNovel && currentCell !== null;

  return (
    <View>
      <Panel title="World State" icon="map">
        <Text style={styles.bigNumber}>Chapter {state.world.chapter}</Text>
        <Text style={styles.muted}>Season {state.world.season}</Text>
        <Text style={styles.bodyText}>{state.world.currentRegion}</Text>
        <AnimatedMeter percent={regionPct} variant="normal" />
        <Text style={styles.muted}>Region progress {regionPct}%</Text>
        <AnimatedMeter percent={threat} variant="threat" />
        <Text style={styles.muted}>Threat pressure {threat}%</Text>
      </Panel>
      <Panel title="Charted Territory" icon="map">
        <FogMap cells={state.world.discoveredCells} currentCell={currentCell} />
        {locationEnabled(state) ? (
          <View>
            {scanning ? <Text style={styles.muted}>Reading your position…</Text> : null}
            {scanFailed ? (
              <Text style={styles.muted}>Couldn't get a position fix. Check location permission and try again.</Text>
            ) : null}
            {novelGround ? (
              <View>
                <Text style={styles.bodyText}>You're standing in uncharted territory.</Text>
                <PrimaryButton
                  icon="compass"
                  label={generating ? "Working..." : "Summon a quest here"}
                  onPress={() => void questHere()}
                />
                <Text style={styles.muted}>Offers land on the Quest Board.</Text>
              </View>
            ) : null}
            <GhostButton icon="scan" label={scanning ? "Scanning…" : "Re-scan position"} onPress={() => void scanHere()} />
          </View>
        ) : (
          <Text style={styles.muted}>Location quests are off. Enable them in Settings to chart territory here.</Text>
        )}
      </Panel>
      {state.world.unlockedLocations.length > 0 ? (
        <Panel title="Unlocked Regions" icon="map">
          {state.world.unlockedLocations.map((loc, index) => (
            <Text key={`${loc}_${index}`} style={styles.logLine}>
              {loc === state.world.currentRegion ? `> ${loc} (current)` : loc}
            </Text>
          ))}
        </Panel>
      ) : null}
      {state.world.activeMysteries.length > 0 ? (
        <Panel title="Active Mysteries" icon="sparkles">
          {state.world.activeMysteries.map((m, index) => (
            <Text key={`${m}_${index}`} style={styles.logLine}>{m}</Text>
          ))}
        </Panel>
      ) : null}
      {state.world.companions.length > 0 ? (
        <Panel title="Companions" icon="flag">
          {state.world.companions.map((c, index) => (
            <Text key={`${c}_${index}`} style={styles.logLine}>{c}</Text>
          ))}
        </Panel>
      ) : null}
      <Panel title="Chronicle" icon="book">
        {state.world.log.map((line, index) => (
          <Text key={`${line}_${index}`} style={styles.logLine}>{line}</Text>
        ))}
      </Panel>
    </View>
  );
}

