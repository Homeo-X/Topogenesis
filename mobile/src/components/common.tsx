import React, { useEffect, useRef } from "react";
import { Animated, Easing, Pressable, Switch, Text, TextInput, View } from "react-native";
import { IconName, isNightMode, styles, Tab } from "../theme";
import { motionReduced } from "../motion";

/**
 * Press-feedback scale animation. Returns an animated style + press handlers.
 * Honors the global reduce-motion preference (no animation when reduced).
 * Self-contained (no store dependency) to avoid circular imports.
 */
export function usePressScale(scaleTo = 0.96) {
  const scale = useRef(new Animated.Value(1)).current;
  const animate = (to: number) => {
    if (motionReduced()) return;
    Animated.timing(scale, { toValue: to, duration: 90, easing: Easing.out(Easing.quad), useNativeDriver: true }).start();
  };
  return {
    style: { transform: [{ scale }] },
    onPressIn: () => animate(scaleTo),
    onPressOut: () => animate(1)
  };
}

export function useEntrance() {
  const value = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(value, {
      toValue: 1,
      duration: 360,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false
    }).start();
  }, [value]);

  return {
    opacity: value,
    transform: [
      {
        translateY: value.interpolate({
          inputRange: [0, 1],
          outputRange: [10, 0]
        })
      }
    ]
  };
}

export function AnimatedMeter({ percent, variant = "normal" }: { percent: number; variant?: "normal" | "threat" }) {
  const value = useRef(new Animated.Value(percent)).current;

  useEffect(() => {
    Animated.timing(value, {
      toValue: Math.max(0, Math.min(100, percent)),
      duration: 420,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false
    }).start();
  }, [percent, value]);

  const width = value.interpolate({
    inputRange: [0, 100],
    outputRange: ["0%", "100%"]
  });

  return (
    <View style={styles.progressTrack}>
      <Animated.View style={[variant === "threat" ? styles.threatFill : styles.progressFill, { width }]} />
    </View>
  );
}

export function SystemMessage({ text }: { text: string }) {
  const entrance = useEntrance();
  const pulse = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 1400, easing: Easing.inOut(Easing.quad), useNativeDriver: false }),
        Animated.timing(pulse, { toValue: 0, duration: 1400, easing: Easing.inOut(Easing.quad), useNativeDriver: false })
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [pulse]);

  const borderColor = pulse.interpolate({
    inputRange: [0, 1],
    outputRange: isNightMode() ? ["rgba(34, 211, 238, 0.42)", "#67e8f9"] : ["rgba(14, 165, 233, 0.36)", "#38bdf8"]
  });

  return (
    <Animated.View style={[styles.systemMessage, entrance, { borderColor }]}>
      <Text style={styles.systemLabel}>SYSTEM MESSAGE</Text>
      <Text style={styles.systemText}>{text}</Text>
    </Animated.View>
  );
}

export function Panel({ title, icon, children }: { title: string; icon: IconName; children: React.ReactNode }) {
  const entrance = useEntrance();
  return (
    <Animated.View style={[styles.panel, entrance]}>
      <View style={styles.panelHeader}>
        <Icon name={icon} size={18} color={isNightMode() ? "#67e8f9" : "#0284c7"} />
        <Text style={styles.panelTitle}>{title}</Text>
      </View>
      {children}
    </Animated.View>
  );
}

export function Metric({ label, value, setValue }: { label: string; value: number; setValue: (value: number) => void }) {
  return (
    <View style={styles.metric}>
      <View style={styles.metricHeader}>
        <Text style={styles.sectionLabel}>{label}</Text>
        <Text style={styles.muted}>{Math.round(value * 100)}%</Text>
      </View>
      <View style={styles.stepRow}>
        <IconButton icon="remove" onPress={() => setValue(Math.max(0, value - 0.1))} />
        <AnimatedMeter percent={Math.round(value * 100)} />
        <IconButton icon="add" onPress={() => setValue(Math.min(1, value + 0.1))} />
      </View>
    </View>
  );
}

export function Segmented({
  value,
  options,
  onChange
}: {
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <View style={styles.segmentWrap}>
      {options.map((option) => (
        <Pressable
          key={option.value}
          onPress={() => onChange(option.value)}
          style={({ pressed }) => [
            styles.segment,
            value === option.value && styles.segmentActive,
            pressed && styles.pressed
          ]}
        >
          <Text style={[styles.segmentText, value === option.value && styles.segmentTextActive]}>
            {option.label}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}

export function ToggleRow({ label, value, onValueChange }: { label: string; value: boolean; onValueChange: (value: boolean) => void }) {
  return (
    <View style={styles.toggleRow}>
      <Text style={styles.bodyText}>{label}</Text>
      <Switch
        value={value}
        onValueChange={onValueChange}
        thumbColor={value ? "#38bdf8" : isNightMode() ? "#7fb4c4" : "#94a3b8"}
        trackColor={isNightMode() ? { false: "#0f2a43", true: "#0e7490" } : { false: "#dbeafe", true: "#bae6fd" }}
      />
    </View>
  );
}

export function PrimaryButton({ icon, label, onPress }: { icon: IconName; label: string; onPress: () => void }) {
  const press = usePressScale();
  return (
    <Pressable onPress={onPress} onPressIn={press.onPressIn} onPressOut={press.onPressOut}>
      <Animated.View style={[styles.primaryButton, press.style]}>
        <Icon name={icon} size={18} color="#032332" />
        <Text style={styles.primaryButtonText}>{label}</Text>
      </Animated.View>
    </Pressable>
  );
}

export function GhostButton({ icon, label, onPress }: { icon: IconName; label: string; onPress: () => void }) {
  const press = usePressScale();
  return (
    <Pressable onPress={onPress} onPressIn={press.onPressIn} onPressOut={press.onPressOut}>
      <Animated.View style={[styles.ghostButton, press.style]}>
        <Icon name={icon} size={18} color={isNightMode() ? "#67e8f9" : "#0284c7"} />
        <Text style={styles.ghostButtonText}>{label}</Text>
      </Animated.View>
    </Pressable>
  );
}

export function IconButton({ icon, onPress }: { icon: IconName; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}>
      <Icon name={icon} size={18} color={isNightMode() ? "#e0fbff" : "#075985"} />
    </Pressable>
  );
}

/** Compact inline action: several fit on one row, unlike full-width buttons. */
export function MiniAction({ label, onPress, tone }: { label: string; onPress: () => void; tone?: "danger" | "primary" }) {
  const color = tone === "danger" ? "#f87171" : tone === "primary" ? "#67e8f9" : (isNightMode() ? "#9fd8e8" : "#0e7490");
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        {
          borderWidth: 1,
          borderColor: color,
          borderRadius: 8,
          paddingHorizontal: 10,
          paddingVertical: 6
        },
        pressed && styles.pressed
      ]}
    >
      <Text style={{ color, fontSize: 12, fontWeight: "700" }}>{label}</Text>
    </Pressable>
  );
}

export function Chip({ label, active, tone }: { label: string; active?: boolean; tone?: string }) {
  return (
    <View style={[styles.chip, active ? styles.navButtonActive : null, tone ? { borderColor: tone, borderWidth: 1 } : null]}>
      <Text style={[styles.chipText, tone ? { color: tone } : null]}>{label}</Text>
    </View>
  );
}

export function NavButton({
  tab,
  current,
  icon,
  label,
  setTab
}: {
  tab: Tab;
  current: Tab;
  icon: IconName;
  label: string;
  setTab: (tab: Tab) => void;
}) {
  const active = tab === current;
  return (
    <Pressable onPress={() => setTab(tab)} style={({ pressed }) => [styles.navButton, active && styles.navButtonActive, pressed && styles.pressed]}>
      <Icon name={icon} size={20} color={active ? (isNightMode() ? "#67e8f9" : "#0284c7") : (isNightMode() ? "#7fb4c4" : "#64748b")} />
      <Text style={[styles.navText, active && styles.navTextActive]} numberOfLines={1}>{label}</Text>
    </Pressable>
  );
}

export function Icon({ name, size, color }: { name: IconName; size: number; color: string }) {
  const glyphs: Record<IconName, string> = {
    timer: "(t)",
    add: "+",
    "add-circle": "+",
    "bar-chart": "#",
    book: "B",
    "checkmark-circle": "✓",
    "close-circle": "x",
    "cloud-upload": "^",
    compass: "C",
    "document-text": "D",
    flag: "F",
    map: "M",
    moon: "☾",
    person: "P",
    pulse: "~",
    refresh: "R",
    remove: "-",
    scan: "S",
    "shield-checkmark": "✓",
    sparkles: "*",
    terminal: ">"
  };
  const display = name === "checkmark-circle" || name === "shield-checkmark" ? "v" : glyphs[name];
  return <Text style={{ color, fontSize: size, fontWeight: "900", width: size + 4, textAlign: "center" }}>{display}</Text>;
}
