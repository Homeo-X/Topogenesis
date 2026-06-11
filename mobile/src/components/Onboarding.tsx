import React, { useState } from "react";
import { ScrollView, Text, TextInput, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { styles } from "../theme";
import { PrimaryButton, GhostButton, Segmented } from "./common";
import { QuestDomain } from "../types";
import { domains } from "../engine";

/**
 * Minimal first-run flow (shown once, gated by AppState.hasOnboarded).
 * Three steps: orient -> safety/consent framing -> set first goal.
 * Deliberately bounded: it orients and captures the first goal, nothing more.
 * The safety framing is not legal text; it sets honest expectations and is the
 * right place for the wellbeing posture to be stated up front.
 */
export function Onboarding({
  onComplete
}: {
  onComplete: (goal: { text: string; domain: QuestDomain } | null) => void;
}) {
  const [step, setStep] = useState(0);
  const [goal, setGoal] = useState("");
  const [domain, setDomain] = useState<QuestDomain>("craft");

  return (
    <SafeAreaView style={styles.shell}>
      <View style={styles.header}>
        <View>
          <Text style={styles.eyebrow}>Forge System OS</Text>
          <Text style={styles.title}>{step === 0 ? "Welcome" : step === 1 ? "How this works" : "Your first goal"}</Text>
        </View>
        <View style={styles.levelBadge}>
          <Text style={styles.levelText}>{step + 1}/3</Text>
        </View>
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentPad}>
        {step === 0 ? (
          <View style={styles.panel}>
            <Text style={styles.panelTitle}>A System for real goals</Text>
            <Text style={styles.bodyText}>
              Forge turns your real goals into small quests. You check in on how you're doing, the System suggests
              quests that fit, you act, and you tell it what happened. Over time it learns what works for you.
            </Text>
            <Text style={styles.bodyText}>
              The progression — levels, stats, skills, a world that reacts — is real: it reflects what you actually do.
            </Text>
          </View>
        ) : null}

        {step === 1 ? (
          <View style={styles.panel}>
            <Text style={styles.panelTitle}>What the System will and won't do</Text>
            <Text style={styles.bodyText}>
              It offers; you decide. Every quest can be accepted, altered, delayed, or rejected. Failure is treated as
              data, never as a verdict about you. The System never shames, never coerces, and never claims certainty it
              doesn't have.
            </Text>
            <Text style={styles.bodyText}>
              It is not medical, legal, or financial advice, and not a substitute for professional support. You can turn
              off surprise quests, physical, social, or outdoor quests, cap quest risk, and adjust notifications at any time in Settings.
            </Text>
          </View>
        ) : null}

        {step === 2 ? (
          <View style={styles.panel}>
            <Text style={styles.panelTitle}>Set one goal to begin</Text>
            <Text style={styles.bodyText}>One goal is enough. You can add or change goals later.</Text>
            <TextInput
              value={goal}
              onChangeText={setGoal}
              placeholder="e.g. Build the Forge System mobile app"
              placeholderTextColor="#64748b"
              style={styles.input}
            />
            <Text style={styles.sectionLabel}>Main area</Text>
            <Segmented
              value={domain}
              onChange={(value) => setDomain(value as QuestDomain)}
              options={domains.map((item) => ({ value: item.id, label: item.label }))}
            />
          </View>
        ) : null}
      </ScrollView>

      <View style={styles.contentPad}>
        {step < 2 ? (
          <PrimaryButton icon="flag" label="Continue" onPress={() => setStep((s) => s + 1)} />
        ) : (
          <PrimaryButton
            icon="flag"
            label="Begin"
            onPress={() => onComplete(goal.trim() ? { text: goal.trim(), domain } : null)}
          />
        )}
        {step === 2 ? (
          <GhostButton icon="close-circle" label="Skip for now" onPress={() => onComplete(null)} />
        ) : null}
      </View>
    </SafeAreaView>
  );
}
