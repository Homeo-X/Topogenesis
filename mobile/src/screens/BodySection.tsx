import React, { useState } from "react";
import { Text, TextInput, View } from "react-native";
import { styles } from "../theme";
import { Panel, PrimaryButton } from "../components/common";
import { useStore } from "../state/store";
import { BodyComposition } from "../types";
import { deriveBodySignals, latestBody } from "../body";

/** Fields offered in manual entry, shaped to a typical impedance scan. All
 *  optional; the user fills what their device reports. */
const FIELDS: Array<{ key: keyof BodyComposition; label: string }> = [
  { key: "weightKg", label: "Weight (kg)" },
  { key: "fatMassKg", label: "Fat mass (kg)" },
  { key: "muscleMassKg", label: "Muscle mass (kg)" },
  { key: "skeletalMuscleKg", label: "Skeletal muscle (kg)" },
  { key: "proteinMassKg", label: "Protein mass (kg)" },
  { key: "waterWeightKg", label: "Water weight (kg)" },
  { key: "boneMassKg", label: "Bone mass (kg)" },
  { key: "bmi", label: "BMI" },
  { key: "bodyFatPct", label: "Body fat (%)" },
  { key: "visceralFatGrade", label: "Visceral fat grade" },
  { key: "basalMetabolicRate", label: "BMR (kcal)" },
  { key: "smi", label: "SMI (kg/m²)" },
  { key: "bodyAge", label: "Body age" },
  { key: "segmentLeftArmPct", label: "Left arm (% std)" },
  { key: "segmentRightArmPct", label: "Right arm (% std)" },
  { key: "segmentTrunkPct", label: "Trunk (% std)" },
  { key: "segmentLeftLegPct", label: "Left leg (% std)" },
  { key: "segmentRightLegPct", label: "Right leg (% std)" }
];

export function BodySection() {
  const { state, dispatch } = useStore();
  const [draft, setDraft] = useState<Record<string, string>>({});
  const signals = deriveBodySignals(state);
  const latest = latestBody(state);

  function setField(key: string, raw: string) {
    setDraft((d) => ({ ...d, [key]: raw.replace(/[^0-9.]/g, "") }));
  }

  function saveScan() {
    const scan: BodyComposition = { recordedAt: Date.now() };
    for (const { key } of FIELDS) {
      const v = draft[key as string];
      if (v !== undefined && v !== "") {
        const n = Number(v);
        if (Number.isFinite(n)) (scan as unknown as Record<string, number>)[key as string] = n;
      }
    }
    void dispatch({ type: "ADD_BODY_SCAN", scan });
    setDraft({});
  }

  return (
    <View>
      <Text style={styles.muted}>Body data helps the System pick movement and recovery quests that fit you. It is never used to set weight or fat targets.</Text>

      {signals.hasData ? (
        <Panel title="What Your Body Data Shows" icon="pulse">
          {signals.muscleTrend ? (
            <Text style={styles.bodyText}>
              Muscle mass trend: {signals.muscleTrend === "up" ? "rising — your consistency is showing." : signals.muscleTrend === "down" ? "lower than last scan — worth a gentle, sustainable rebuild." : "holding steady."}
            </Text>
          ) : null}
          {signals.imbalanceNote ? <Text style={styles.bodyText}>{signals.imbalanceNote}</Text> : null}
          {signals.physicalReadiness !== null ? (
            <Text style={styles.muted}>Physical readiness (rough estimate): {Math.round(signals.physicalReadiness * 100)}%</Text>
          ) : null}
        </Panel>
      ) : null}

      <Panel title="Add Body Scan" icon="add-circle">
        <Text style={styles.muted}>Enter what your device reported. Leave the rest blank.</Text>
        {FIELDS.map(({ key, label }) => (
          <View key={key as string} style={styles.progressHeader}>
            <Text style={styles.bodyText}>{label}</Text>
            <TextInput
              value={draft[key as string] ?? ""}
              onChangeText={(raw) => setField(key as string, raw)}
              keyboardType="decimal-pad"
              placeholder="—"
              placeholderTextColor="#64748b"
              style={[styles.input, { width: 110 }]}
            />
          </View>
        ))}
        <PrimaryButton icon="scan" label="Save Scan" onPress={saveScan} />
      </Panel>

      {latest ? (
        <Panel title="History" icon="bar-chart">
          {state.bodyHistory.map((scan, index) => (
            <View key={scan.recordedAt} style={styles.offerCard}>
              <Text style={styles.sectionLabel}>
                {new Date(scan.recordedAt).toISOString().slice(0, 10)}{index === 0 ? " · latest" : ""}
              </Text>
              {FIELDS.filter(({ key }) => typeof (scan as unknown as Record<string, unknown>)[key as string] === "number").map(({ key, label }) => (
                <View key={key as string} style={styles.progressHeader}>
                  <Text style={styles.muted}>{label}</Text>
                  <Text style={styles.bodyText}>{String((scan as unknown as Record<string, number>)[key as string])}</Text>
                </View>
              ))}
            </View>
          ))}
        </Panel>
      ) : null}
    </View>
  );
}
