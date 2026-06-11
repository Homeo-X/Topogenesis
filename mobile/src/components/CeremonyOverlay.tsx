import React, { useEffect, useRef } from "react";
import { Animated, Modal, Pressable, Text, View } from "react-native";
import { CeremonyEvent } from "../types";
import { styles } from "../theme";
import { motionReduced } from "../motion";
import { playCue, tap } from "../feedback";
import { useStore } from "../state/store";

/**
 * The ceremony: a full-screen moment for level-ups, domain mastery, and skill
 * unlocks. This is where "anime feeling from real mechanics" gets its payoff —
 * the event is always real (engine-emitted, never decorative), and the screen
 * gives it weight. Respects reduce-motion (content appears without scaling),
 * fires a success haptic and the level-up cue (both preference-gated), and
 * dismisses on tap so it never holds the player hostage.
 */
export function CeremonyOverlay({ event }: { event: CeremonyEvent }) {
  const { state, dispatch } = useStore();
  const scale = useRef(new Animated.Value(motionReduced() ? 1 : 0.7)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    void tap(state, "success");
    void playCue(state, event.kind === "level" ? "levelup" : "complete");
    if (motionReduced()) {
      opacity.setValue(1);
      return;
    }
    Animated.parallel([
      Animated.spring(scale, { toValue: 1, useNativeDriver: true, friction: 5 }),
      Animated.timing(opacity, { toValue: 1, duration: 220, useNativeDriver: true })
    ]).start();
    // Intentionally run once per ceremony event.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [event.at]);

  function dismiss() {
    void dispatch({ type: "PATCH", patch: { pendingCeremony: null } });
  }

  return (
    <Modal transparent animationType="none" visible onRequestClose={dismiss}>
      <Pressable style={styles.ceremonyBackdrop} onPress={dismiss}>
        <Animated.View style={[styles.ceremonyCard, { opacity, transform: [{ scale }] }]}>
          <Text style={styles.ceremonyTitle}>{event.title}</Text>
          {event.lines.map((line) => (
            <Text key={line} style={styles.ceremonyLine}>{line}</Text>
          ))}
          <Text style={styles.muted}>Tap anywhere to continue</Text>
        </Animated.View>
      </Pressable>
    </Modal>
  );
}
