import React, { useState } from "react";
import { Text, TextInput, View } from "react-native";
import { questModes } from "../engine";
import { HostState, QuestMode } from "../types";
import { styles } from "../theme";
import { Metric, Panel, PrimaryButton, Segmented, SystemMessage } from "../components/common";
import { useStore } from "../state/store";
import { BodySection } from "./BodySection";

const MOODS = ["low", "flat", "steady", "good", "charged"];

export function CheckinScreen() {
  const { state, dispatch } = useStore();
  const [host, setHost] = useState<HostState>(state.host);

  const setNumber = (key: keyof HostState, value: number) => {
    setHost((current) => ({ ...current, [key]: value }));
  };

  function saveScan() {
    void dispatch({ type: "SAVE_SCAN", host });
  }

  return (
    <View>
      <SystemMessage text="Check in on how you're doing before getting a quest." />
      <Panel title="Quick Check-in" icon="pulse">
        <Metric label="Energy" value={host.energy} setValue={(value) => setNumber("energy", value)} />
        <Metric label="Focus" value={host.focus} setValue={(value) => setNumber("focus", value)} />
        <Metric label="Stress" value={host.stress} setValue={(value) => setNumber("stress", value)} />
        <Metric label="Need to Rest" value={host.recoveryNeed} setValue={(value) => setNumber("recoveryNeed", value)} />
        <Metric label="Up for a Challenge" value={host.challengeReadiness} setValue={(value) => setNumber("challengeReadiness", value)} />
        <Metric label="Body Feels" value={host.bodyStatus} setValue={(value) => setNumber("bodyStatus", value)} />
        <Metric label="Up for People" value={host.socialReadiness} setValue={(value) => setNumber("socialReadiness", value)} />
        <Metric label="Creative Spark" value={host.creativeReadiness} setValue={(value) => setNumber("creativeReadiness", value)} />
        <Text style={styles.muted}>Mood</Text>
        <Segmented
          options={MOODS.map((m) => ({ value: m, label: m }))}
          value={MOODS.includes(host.mood) ? host.mood : "steady"}
          onChange={(value) => setHost({ ...host, mood: value })}
        />
        <TextInput
          value={String(host.timeAvailableMinutes)}
          onChangeText={(value) => setNumber("timeAvailableMinutes", Number(value.replace(/[^0-9]/g, "")) || 5)}
          keyboardType="number-pad"
          style={styles.input}
        />
        <Segmented
          value={host.desiredMode}
          onChange={(value) => setHost((current) => ({ ...current, desiredMode: value as QuestMode }))}
          options={questModes.map((item) => ({ value: item.id, label: item.label }))}
        />
        <PrimaryButton icon="scan" label="Save Check-in" onPress={saveScan} />
      </Panel>

      <BodySection />
    </View>
  );
}
