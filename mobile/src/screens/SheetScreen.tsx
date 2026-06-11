import React from "react";
import { Text, View } from "react-native";

import { styles } from "../theme";
import { AnimatedMeter, Panel } from "../components/common";
import { useStore } from "../state/store";
import { visibleSkillsByStat } from "../skills";
import { domains, domainToStat } from "../engine";
import { QUEST_ARCHETYPES, domainLevelFromPoints, unlockedArchetypes } from "../questArchetypes";

export function SheetScreen() {
  const { state } = useStore();
  const xpPercent = Math.min(100, Math.round((state.progression.xp / state.progression.xpToNext) * 100));
  const statRows = Object.entries(state.progression.stats);
  const visibleSkills = visibleSkillsByStat(state.progression.unlockedSkills);

  return (
    <View>
      <Panel title="Character Sheet" icon="person">
        <Text style={styles.bigNumber}>Level {state.progression.level}</Text>
        <Text style={styles.bodyText}>
          Class: {state.progression.classPath.inferred ?? "Initiate"}
        </Text>
        <AnimatedMeter percent={xpPercent} />
        <Text style={styles.muted}>{state.progression.xp} / {state.progression.xpToNext} XP</Text>
      </Panel>

      <Panel title="Stats" icon="bar-chart">
        {statRows.map(([name, value]) => (
          <View key={name} style={styles.statRow}>
            <Text style={styles.statName}>{name}</Text>
            <Text style={styles.statValue}>{value}</Text>
          </View>
        ))}
      </Panel>

      <Panel title="Quest Mastery" icon="book">
        <Text style={styles.muted}>Each area levels with play. Leveling unlocks deeper quest types; the early ones stay as repeatable habits.</Text>
        {domains.map(({ id, label }) => {
          const points = state.progression.stats[domainToStat[id]] ?? 0;
          const level = domainLevelFromPoints(points);
          const open = unlockedArchetypes(id, level);
          const total = (QUEST_ARCHETYPES[id] ?? []).length;
          const tier = open.length ? open[open.length - 1].subcategory ?? "Foundations" : "Foundations";
          return (
            <View key={id} style={styles.statRow}>
              <Text style={styles.statName}>{label}</Text>
              <Text style={styles.statValue}>Lv {level} · {tier} · {open.length}/{total}</Text>
            </View>
          );
        })}
      </Panel>

      <Panel title="Unlocked Skills" icon="sparkles">
        {Object.keys(visibleSkills).length === 0 ? (
          <Text style={styles.bodyText}>No skills unlocked yet.</Text>
        ) : null}
        {Object.entries(visibleSkills).map(([stat, skills]) => {
          const statValue = state.progression.stats[stat as keyof typeof state.progression.stats] ?? 0;
          return (
            <View key={stat} style={{ marginBottom: 10 }}>
              <Text style={styles.sectionLabel}>{stat} - {statValue}</Text>
              {skills.map((skill) => (
                <Text key={skill.id} style={styles.criteriaLine}>
                  * {skill.label}{skill.functional ? " - active" : ""}
                </Text>
              ))}
            </View>
          );
        })}
      </Panel>

      <Panel title="How I'm Tuning Difficulty" icon="pulse">
        {Object.entries(state.profile.difficultyCalibration).map(([name, value]) => (
          <View key={name} style={styles.statRow}>
            <Text style={styles.statName}>{name}</Text>
            <Text style={styles.statValue}>{Math.round(value * 100)}</Text>
          </View>
        ))}
        <Text style={styles.sectionLabel}>What Works for You</Text>
        {(state.profile.knownPatterns.length ? state.profile.knownPatterns : ["Nothing confirmed yet."]).map((item) => (
          <Text key={item} style={styles.criteriaLine}>- {item}</Text>
        ))}
        <Text style={styles.sectionLabel}>What You Tend to Avoid</Text>
        {(state.profile.resistancePatterns.length ? state.profile.resistancePatterns : ["Nothing noticed yet."]).map((item) => (
          <Text key={item} style={styles.criteriaLine}>- {item}</Text>
        ))}
        <Text style={styles.sectionLabel}>Recent Adjustments</Text>
        {(state.adaptationLog.length ? state.adaptationLog : ["No adjustments yet."]).map((item, index) => (
          <Text key={`${item}_${index}`} style={styles.criteriaLine}>- {item}</Text>
        ))}
      </Panel>
    </View>
  );
}
