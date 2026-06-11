import React, { useState } from "react";
import { Modal, Pressable, ScrollView, StatusBar, Text, TextInput, View } from "react-native";
import { disableSystemNotifications, enableDailySystemNotifications } from "../notifications";
import { styles } from "../theme";
import { GhostButton, Panel, PrimaryButton, Segmented, ToggleRow } from "./common";
import { useStore } from "../state/store";

/**
 * Slide-in settings drawer launched from the header hamburger. Holds everything
 * that isn't day-to-day content: account/cloud sync, appearance, permissions,
 * and the local-data reset. Closes on backdrop tap or Android hardware back.
 */
export function SettingsMenu({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const { state, dispatch, auth } = useStore();
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authMsg, setAuthMsg] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const prefs = state.profile.preferences;
  const topPad = (StatusBar.currentHeight ?? 24) + 12;

  async function setNotificationConsent(value: boolean) {
    const result = value
      ? await enableDailySystemNotifications(state.host)
      : { enabled: false, message: await disableSystemNotifications() };
    void dispatch({
      type: "PATCH",
      patch: {
        profile: {
          ...state.profile,
          preferences: { ...state.profile.preferences, notificationConsent: value ? result.enabled : false }
        },
        lastSystemMessage: result.message
      }
    });
  }

  function patchPreference(
    key: "physicalQuests" | "socialQuests" | "outdoorQuests" | "hapticsEnabled" | "reduceMotion" | "nightMode" | "soundEnabled" | "surpriseQuests" | "hiddenQuests" | "bossQuests" | "alarmStyleNotifications" | "locationQuests",
    value: boolean
  ) {
    void dispatch({
      type: "PATCH",
      patch: { profile: { ...state.profile, preferences: { ...state.profile.preferences, [key]: value } } }
    });
  }

  async function handleAuth(kind: "in" | "up") {
    if (!authEmail.trim() || !authPassword) {
      setAuthMsg("Enter an email and password first.");
      return;
    }
    setAuthBusy(true);
    const res =
      kind === "in"
        ? await auth.signIn(authEmail.trim(), authPassword)
        : await auth.signUp(authEmail.trim(), authPassword);
    setAuthMsg(res.message);
    setAuthBusy(false);
    if (res.ok) setAuthPassword("");
  }

  async function handleSignOut() {
    setAuthBusy(true);
    await auth.signOut();
    setAuthBusy(false);
    setAuthMsg("Signed out. The app still works fully offline.");
  }

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.menuRoot}>
        <View style={[styles.menuDrawer, { paddingTop: topPad }]}>
          <View style={styles.menuHeader}>
            <View>
              <Text style={styles.eyebrow}>Forge System OS</Text>
              <Text style={styles.menuTitle}>Settings</Text>
            </View>
            <Pressable
              onPress={onClose}
              style={({ pressed }) => [styles.themeToggle, pressed && styles.pressed]}
              accessibilityRole="button"
              accessibilityLabel="Close menu"
            >
              <Text style={styles.themeToggleText}>x</Text>
            </Pressable>
          </View>

          <ScrollView style={styles.menuScroll} contentContainerStyle={styles.menuContent}>
            {auth.configured && (
              <Panel title="Account & Cloud Sync" icon="cloud-upload">
                {auth.email ? (
                  <>
                    <Text style={styles.bodyText}>
                      Signed in as {auth.email}. Your progress backs up to the cloud automatically.
                    </Text>
                    <View style={styles.rowGap}>
                      <GhostButton icon="close-circle" label={authBusy ? "Working..." : "Sign Out"} onPress={handleSignOut} />
                    </View>
                  </>
                ) : (
                  <>
                    <Text style={styles.muted}>
                      Create an account or sign in to back up your progress and restore it after a reinstall or on another device.
                    </Text>
                    <Text style={styles.sectionLabel}>Email</Text>
                    <TextInput
                      value={authEmail}
                      onChangeText={setAuthEmail}
                      placeholder="you@example.com"
                      placeholderTextColor="#64748b"
                      autoCapitalize="none"
                      autoCorrect={false}
                      keyboardType="email-address"
                      style={styles.input}
                    />
                    <Text style={styles.sectionLabel}>Password</Text>
                    <TextInput
                      value={authPassword}
                      onChangeText={setAuthPassword}
                      placeholder="At least 6 characters"
                      placeholderTextColor="#64748b"
                      autoCapitalize="none"
                      autoCorrect={false}
                      secureTextEntry
                      style={styles.input}
                    />
                    <PrimaryButton icon="person" label={authBusy ? "Working..." : "Sign In"} onPress={() => handleAuth("in")} />
                    <View style={styles.rowGap}>
                      <GhostButton icon="add-circle" label="Create Account" onPress={() => handleAuth("up")} />
                    </View>
                  </>
                )}
                {Boolean(authMsg) && <Text style={styles.routeStatus}>{authMsg}</Text>}
              </Panel>
            )}

            <Panel title="Appearance" icon="moon">
              <ToggleRow
                label="Night mode"
                value={Boolean(prefs.nightMode)}
                onValueChange={(value) => patchPreference("nightMode", value)}
              />
            </Panel>

            <Panel title="Permissions" icon="shield-checkmark">
              <ToggleRow label="Notifications" value={prefs.notificationConsent} onValueChange={setNotificationConsent} />
              <ToggleRow label="Physical quests" value={prefs.physicalQuests} onValueChange={(value) => patchPreference("physicalQuests", value)} />
              <ToggleRow label="Social quests" value={prefs.socialQuests} onValueChange={(value) => patchPreference("socialQuests", value)} />
              <ToggleRow
                label="Location quests (on-demand only)"
                value={prefs.locationQuests}
                onValueChange={(value) => patchPreference("locationQuests", value)}
              />
              <ToggleRow label="Surprise quests (all)" value={prefs.surpriseQuests} onValueChange={(value) => patchPreference("surpriseQuests", value)} />
              <ToggleRow label="Hidden quests" value={prefs.hiddenQuests} onValueChange={(value) => patchPreference("hiddenQuests", value)} />
              <ToggleRow label="Boss quests" value={prefs.bossQuests} onValueChange={(value) => patchPreference("bossQuests", value)} />
              <ToggleRow label="Alarm-style notifications" value={prefs.alarmStyleNotifications} onValueChange={(value) => patchPreference("alarmStyleNotifications", value)} />
              <Text style={styles.muted}>Emergency wording</Text>
              <Segmented
                value={prefs.emergencyWording}
                onChange={(value) => void dispatch({ type: "PATCH", patch: { profile: { ...state.profile, preferences: { ...prefs, emergencyWording: value as "direct" | "calm" } } } })}
                options={[{ value: "direct", label: "Direct" }, { value: "calm", label: "Calm" }]}
              />
              <Text style={styles.muted}>Highest quest risk offered</Text>
              <Segmented
                value={String(prefs.maxRiskTier)}
                onChange={(value) => void dispatch({ type: "PATCH", patch: { profile: { ...state.profile, preferences: { ...prefs, maxRiskTier: Number(value) } } } })}
                options={[0, 1, 2, 3, 4].map((tier) => ({ value: String(tier), label: String(tier) }))}
              />
              <ToggleRow label="Sound effects" value={prefs.soundEnabled} onValueChange={(value) => patchPreference("soundEnabled", value)} />
              <ToggleRow label="Outdoor / place quests" value={prefs.outdoorQuests} onValueChange={(value) => patchPreference("outdoorQuests", value)} />
              <ToggleRow label="Haptic feedback" value={prefs.hapticsEnabled} onValueChange={(value) => patchPreference("hapticsEnabled", value)} />
              <ToggleRow label="Reduce motion" value={prefs.reduceMotion} onValueChange={(value) => patchPreference("reduceMotion", value)} />
            </Panel>

            <GhostButton
              icon="refresh"
              label="Reset Local Data"
              onPress={() => {
                onClose();
                void dispatch({ type: "RESET" });
              }}
            />
          </ScrollView>
        </View>

        <Pressable style={styles.menuBackdrop} onPress={onClose} accessibilityRole="button" accessibilityLabel="Close menu" />
      </View>
    </Modal>
  );
}
