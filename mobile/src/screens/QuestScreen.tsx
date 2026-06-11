import React, { useState } from "react";
import { Alert, Pressable, Text, TextInput, View } from "react-native";
import { EvidenceType, OutcomeType, Quest, TimerEvidence } from "../types";
import { domainColors, rankColors, styles } from "../theme";
import { AnimatedMeter, Chip, GhostButton, Panel, PrimaryButton, Segmented } from "../components/common";
import { evidenceTypes, outcomes } from "../uiConstants";
import { useStore } from "../state/store";
import { playCue } from "../feedback";
import { ACTIVITY_CATEGORIES, ActivityCategory, primaryDomainOf } from "../activities";
import { locationEnabled, sampleCellOnce } from "../location";
import { domains, MAX_ACTIVE_QUESTS } from "../engine";

export function QuestScreen() {
  const { state, getState, dispatch } = useStore();
  const [generating, setGenerating] = useState(false);
  const [selectedActivity, setSelectedActivity] = useState<ActivityCategory | null>(null);
  const active = state.activeQuests;
  const offers = state.questOffers ?? [];
  const logFull = active.length >= MAX_ACTIVE_QUESTS;
  const hasActiveGoal = state.profile.goals.some((g) => g.status === "active");

  async function requestQuest() {
    setGenerating(true);
    try {
      await dispatch({ type: "GENERATE_QUEST" });
    } finally {
      setGenerating(false);
    }
  }

  function acceptWithConsent(questId: string) {
    const selected = offers.find((offer) => offer.id === questId);
    if (selected && selected.riskTier >= 3) {
      Alert.alert(
        "High-risk quest",
        "This one is higher risk, or uses a setting you turned off. Only accept if you really want to.",
        [
          { text: "Cancel", style: "cancel" },
          { text: "Accept", style: "destructive", onPress: () => void dispatch({ type: "ACCEPT_QUEST", questId, confirmedRisk: true }) }
        ]
      );
      return;
    }
    void dispatch({ type: "ACCEPT_QUEST", questId });
  }

  // Activity door: pick "what you want to do," see who it builds, get a quest.
  // If an active goal already lives in one of the activity's domains, focus it;
  // otherwise create a light goal in the activity's primary domain. Either way
  // a generation follows, so the tap always ends in offers.
  async function questFromActivity(activity: ActivityCategory) {
    setGenerating(true);
    try {
      const activityDomains = activity.builds.map((b) => b.domain);
      const existing = state.profile.goals.find((g) => g.status === "active" && activityDomains.includes(g.domain));
      let targetGoalId = existing?.id;
      if (!targetGoalId) {
        await dispatch({ type: "ADD_GOAL", text: `Make real progress in ${activity.label.toLowerCase()}`, domain: primaryDomainOf(activity) });
        // getState reads the freshly committed state (safe post-await).
        targetGoalId = getState().profile.goals[0]?.id;
      }
      await dispatch({ type: "GENERATE_QUEST", goalId: targetGoalId });
      setSelectedActivity(null);
    } finally {
      setGenerating(false);
    }
  }

  return (
    <View>
      {active.length > 0 ? (
        <Panel title={`Active Quests (${active.length}/${MAX_ACTIVE_QUESTS})`} icon="flag">
          <Text style={styles.muted}>
            {logFull
              ? "Your quest log is full. Resolve one to take another."
              : "Running quests live here. You can take more from the offers below."}
          </Text>
        </Panel>
      ) : null}

      {active.map((quest) => (
        <ActiveQuestCard key={quest.id} quest={quest} />
      ))}

      {!logFull ? (
        <Panel title="Browse by Activity" icon="compass">
          <Text style={styles.muted}>Or start from what you want to do — each one builds part of who you're becoming.</Text>
          <View style={styles.questMeta}>
            {ACTIVITY_CATEGORIES.map((activity) => (
              <Pressable key={activity.id} onPress={() => setSelectedActivity(selectedActivity?.id === activity.id ? null : activity)}>
                <Chip label={activity.label} active={selectedActivity?.id === activity.id} />
              </Pressable>
            ))}
          </View>
          {selectedActivity ? (
            <View>
              <Text style={styles.bodyText}>{selectedActivity.blurb}</Text>
              <Text style={styles.muted}>
                Builds: {selectedActivity.builds.map((b) => domains.find((d) => d.id === b.domain)?.label ?? b.domain).join(" · ")}
              </Text>
              <PrimaryButton icon="flag" label={generating ? "Working..." : `Quest in ${selectedActivity.label}`} onPress={() => void questFromActivity(selectedActivity)} />
            </View>
          ) : null}
        </Panel>
      ) : null}

      {offers.length === 0 && active.length === 0 && !hasActiveGoal ? (
        <Panel title="No Goal Yet" icon="flag">
          <Text style={styles.bodyText}>
            Add a goal on the System tab first. I build quests from your goal, your latest scan, and your history.
          </Text>
        </Panel>
      ) : offers.length === 0 && !logFull && hasActiveGoal ? (
        <Panel title="Get a Quest" icon="sparkles">
          <Text style={styles.bodyText}>
            {generating
              ? "Looking at your goal, latest scan, and history to put some quests together..."
              : "I'll suggest a few quests based on your goal, latest scan, and history."}
          </Text>
          <PrimaryButton icon="flag" label={generating ? "Working..." : "Get Quests"} onPress={requestQuest} />
        </Panel>
      ) : null}

      {offers.length > 0 && !logFull ? (
        <Panel title="Pick a Quest" icon="sparkles">
          {state.systemSignals.lastEvent && offers[0]?.candidateId?.startsWith("special_") ? (
            <View style={styles.offerCard}>
              <Text style={styles.offerRank}>
                {state.systemSignals.lastEvent.kind === "hidden"
                  ? "Hidden Quest"
                  : state.systemSignals.lastEvent.kind === "boss"
                    ? "Boss Quest"
                    : "Emergency Quest"}
              </Text>
              <Text style={styles.bodyText}>{state.systemSignals.lastEvent.reason}</Text>
            </View>
          ) : null}
          <Text style={styles.bodyText}>Pick what fits — you can run up to {MAX_ACTIVE_QUESTS} at once.</Text>
          {offers.map((offer, index) => (
            <View key={offer.id} style={styles.offerCard}>
              <View style={styles.progressHeader}>
                <Text style={styles.offerRank}>{index === 0 ? "System Pick" : `Option ${index + 1}`}</Text>
                <Chip label={`${Math.round(offer.acceptanceScore * 100)}% fit`} />
              </View>
              <Text style={styles.questText}>{offer.title}</Text>
              <View style={styles.questMeta}>
                <Chip label={`Rank ${offer.rank}`} tone={rankColors[offer.rank]} />
                <Chip label={`${offer.timeLimitMinutes} min`} />
                <Chip label={`Risk ${offer.riskTier}`} />
                <Chip label={offer.domain} tone={domainColors[offer.domain]} />
              </View>
              <Text style={styles.bodyText}>{offer.objective}</Text>
              {offer.councilNote ? <Text style={styles.muted}>{offer.councilNote}</Text> : null}
              <View style={styles.inlineActions}>
                <PrimaryButton icon="checkmark-circle" label="Accept" onPress={() => acceptWithConsent(offer.id)} />
                <GhostButton icon="close-circle" label="Reject" onPress={() => void dispatch({ type: "REJECT_QUEST", questId: offer.id })} />
              </View>
            </View>
          ))}
          <GhostButton icon="refresh" label={generating ? "Generating..." : "Regenerate Offers"} onPress={requestQuest} />
        </Panel>
      ) : null}
    </View>
  );
}

/**
 * One running quest: detail, step checklist, and its own evidence/outcome
 * form. A React instance per quest means every quest's steps, note, evidence
 * selection, and effort timer are isolated by construction — no cross-quest
 * state bleed possible.
 */
function ActiveQuestCard({ quest }: { quest: Quest }) {
  const { state, dispatch, interpretOutcomeNote } = useStore();
  const [outcome, setOutcome] = useState<OutcomeType>("COMPLETED_FULL");
  const [evidence, setEvidence] = useState<EvidenceType>("self_report");
  const [artifactUri, setArtifactUri] = useState("");
  const [timerStartedAt, setTimerStartedAt] = useState<number | null>(null);
  const [timerCapture, setTimerCapture] = useState<TimerEvidence | null>(null);
  const [note, setNote] = useState("");
  const [interpreting, setInterpreting] = useState(false);
  const [checkedSteps, setCheckedSteps] = useState<Record<string, boolean>>({});
  const completedSteps = quest.activityPlan.steps.filter((step) => checkedSteps[step.id]).length;
  const stepProgress = Math.round((completedSteps / Math.max(1, quest.activityPlan.steps.length)) * 100);

  function toggleTimer() {
    if (timerStartedAt === null) {
      setTimerStartedAt(Date.now());
      return;
    }
    const endedAt = Date.now();
    setTimerCapture({
      startedAt: timerStartedAt,
      endedAt,
      plannedMinutes: quest.timeLimitMinutes,
      actualMinutes: Math.max(1, Math.round((endedAt - timerStartedAt) / 60000))
    });
    setTimerStartedAt(null);
  }

  async function commitOutcome(finalOutcome: OutcomeType) {
    const extraEvidence: Array<{ kind: EvidenceType; note?: string; artifactUri?: string; timer?: TimerEvidence; cellId?: string }> = [];
    if (timerCapture) extraEvidence.push({ kind: "timer", timer: timerCapture });
    if (artifactUri.trim() && evidence !== "artifact") extraEvidence.push({ kind: "artifact", artifactUri: artifactUri.trim() });
    // Location evidence: consent-gated, bounded by a timeout, never blocking
    // the submission on failure. Cell-level only — never raw coordinates.
    if (locationEnabled(state)) {
      const cellId = await sampleCellOnce();
      if (cellId) {
        const novel = !state.world.discoveredCells.includes(cellId);
        extraEvidence.push({
          kind: "location",
          cellId,
          note: novel ? "Completed in newly charted territory." : "Completed at a known location."
        });
        void dispatch({ type: "DISCOVER_CELLS", cells: [cellId] });
      }
    }
    void dispatch({
      type: "SUBMIT_OUTCOME",
      outcome: finalOutcome,
      evidence,
      note,
      questId: quest.id,
      extraEvidence: extraEvidence.length
        ? extraEvidence
        : artifactUri.trim() && evidence === "artifact"
          ? [{ kind: "artifact", artifactUri: artifactUri.trim() }]
          : undefined
    });
    void playCue(state, "complete");
  }

  async function submitWithInterpretation() {
    // If the LLM is off or the note is empty, interpretOutcomeNote returns null
    // and we submit exactly what the user selected (inform-only degrades to none).
    setInterpreting(true);
    let suggestion = null as Awaited<ReturnType<typeof interpretOutcomeNote>>;
    try {
      suggestion = await interpretOutcomeNote(outcome, note);
    } finally {
      setInterpreting(false);
    }

    if (!suggestion) {
      void commitOutcome(outcome);
      return;
    }

    // Suggest-with-confirmation: the System proposes, the user decides.
    if (suggestion.differsFromSelected) {
      const suggestedLabel = outcomes.find((o) => o.id === suggestion!.suggestedOutcome)?.label ?? suggestion.suggestedOutcome;
      Alert.alert(
        "System read your note differently",
        `${suggestion.reasoning}\n\nBlocker: ${suggestion.blocker}\n\nLog this as "${suggestedLabel}" instead of what you picked?`,
        [
          { text: "Keep mine", onPress: () => void commitOutcome(outcome) },
          { text: `Use "${suggestedLabel}"`, onPress: () => void commitOutcome(suggestion!.suggestedOutcome) }
        ]
      );
      return;
    }

    // Same outcome, but the model named a blocker worth surfacing (inform-only).
    Alert.alert("System note", `Blocker: ${suggestion.blocker}\n${suggestion.reasoning}`, [
      { text: "Submit", onPress: () => void commitOutcome(outcome) }
    ]);
  }

  return (
    <View>
      <Panel title={quest.title} icon="flag">
        <View style={styles.questMeta}>
          <Chip label={`Rank ${quest.rank}`} tone={rankColors[quest.rank]} />
          <Chip label={`${quest.timeLimitMinutes} min`} />
          <Chip label={`Risk ${quest.riskTier}`} />
          <Chip label={quest.domain} tone={domainColors[quest.domain]} />
        </View>
        <Text style={styles.sectionLabel}>Objective</Text>
        <Text style={styles.questText}>{quest.objective}</Text>
        <Text style={styles.sectionLabel}>Why this matters</Text>
        <Text style={styles.bodyText}>{quest.activityPlan.stakes}</Text>
        <Text style={styles.sectionLabel}>What To Do</Text>
        <Text style={styles.bodyText}>{quest.activityPlan.intent}</Text>
        <View style={styles.progressHeader}>
          <Text style={styles.muted}>Step progress</Text>
          <Text style={styles.muted}>{completedSteps}/{quest.activityPlan.steps.length}</Text>
        </View>
        <AnimatedMeter percent={stepProgress} />
        <View style={styles.planList}>
          {quest.activityPlan.steps.map((step) => (
            <Pressable
              key={step.id}
              onPress={() => setCheckedSteps((current) => ({ ...current, [step.id]: !current[step.id] }))}
              style={({ pressed }) => [
                styles.planStep,
                checkedSteps[step.id] && styles.planStepDone,
                pressed && styles.pressed
              ]}
            >
              <View style={styles.stepTime}>
                <Text style={styles.stepTimeText}>{checkedSteps[step.id] ? "v" : step.minutes}</Text>
                <Text style={styles.stepTimeUnit}>{checkedSteps[step.id] ? "done" : "min"}</Text>
              </View>
              <View style={styles.stepBody}>
                <Text style={styles.stepTitle}>{step.label}</Text>
                <Text style={styles.bodyText}>{step.instruction}</Text>
                <Text style={styles.stepOutput}>Output: {step.output}</Text>
              </View>
            </Pressable>
          ))}
        </View>
        <Text style={styles.sectionLabel}>Success Criteria</Text>
        {quest.activityPlan.successCriteria.map((criterion) => (
          <Text key={criterion} style={styles.criteriaLine}>- {criterion}</Text>
        ))}
        <Text style={styles.sectionLabel}>Fallback</Text>
        <Text style={styles.bodyText}>{quest.activityPlan.fallback}</Text>
        <Text style={styles.antiAvoidance}>{quest.activityPlan.antiAvoidanceRule}</Text>
        <Text style={styles.sectionLabel}>Proof</Text>
        <Text style={styles.bodyText}>{quest.proofRequired}</Text>
        <Text style={styles.sectionLabel}>Reward</Text>
        <Text style={styles.bodyText}>{quest.rewards.xp} XP and a small stat boost</Text>
      </Panel>

      <Panel title={`Evidence · ${quest.title}`} icon="document-text">
        <Segmented
          value={outcome}
          onChange={(value) => setOutcome(value as OutcomeType)}
          options={outcomes.map((item) => ({ value: item.id, label: item.label }))}
        />
        <Segmented
          value={evidence}
          onChange={(value) => setEvidence(value as EvidenceType)}
          options={evidenceTypes.map((item) => ({ value: item.id, label: item.label }))}
        />
        <TextInput
          value={note}
          onChangeText={setNote}
          placeholder="What happened?"
          placeholderTextColor="#64748b"
          multiline
          style={styles.input}
        />
        {evidence === "artifact" ? (
          <TextInput
            value={artifactUri}
            onChangeText={setArtifactUri}
            placeholder="Link or file reference (optional)"
            placeholderTextColor="#64748b"
            autoCapitalize="none"
            style={styles.input}
          />
        ) : null}
        <GhostButton
          icon="timer"
          label={timerStartedAt !== null ? "Stop timer (recording...)" : timerCapture ? `Timer captured: ${timerCapture.actualMinutes} min — tap to redo` : "Start effort timer"}
          onPress={toggleTimer}
        />
        {timerCapture ? (
          <Text style={styles.muted}>Timer evidence attached: {timerCapture.actualMinutes} min actual vs {timerCapture.plannedMinutes} min planned.</Text>
        ) : null}
        <PrimaryButton
          icon="cloud-upload"
          label={interpreting ? "Reading note..." : "Submit Outcome"}
          onPress={() => void submitWithInterpretation()}
        />
      </Panel>
    </View>
  );
}
