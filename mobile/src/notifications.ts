import * as Notifications from "expo-notifications";
import Constants from "expo-constants";
import { Platform } from "react-native";
import { AppState, HostState, Quest } from "./types";

// expo-notifications push token auto-registration throws in Expo Go (SDK 53+).
// All notification calls are no-ops in that environment; the APK is unaffected.
const isExpoGo = Constants.executionEnvironment === "storeClient";

const scanCategoryId = "forge_system_scan";
const questCategoryId = "forge_system_quest";
// Android 8+ refuses to post any notification that isn't bound to a channel.
const androidChannelId = "forge_system_default";

type NotificationKind = "daily_scan" | "post_scan" | "quest_window";

if (!isExpoGo) {
  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      shouldPlaySound: false,
      shouldSetBadge: false,
      shouldShowBanner: true,
      shouldShowList: true
    })
  });
}

export async function configureNotificationCategories() {
  if (Platform.OS === "web" || isExpoGo) return;
  try {
    await ensureAndroidChannel();
    await Notifications.setNotificationCategoryAsync(scanCategoryId, [
      {
        identifier: "OPEN_SCAN",
        buttonTitle: "Open Check-in",
        options: { opensAppToForeground: true }
      }
    ]);
    await Notifications.setNotificationCategoryAsync(questCategoryId, [
      {
        identifier: "OPEN_QUEST",
        buttonTitle: "Open Quest",
        options: { opensAppToForeground: true }
      }
    ]);
  } catch {
    // Channel/category setup is best-effort; never let it crash the app.
  }
}

/**
 * Android 8+ requires an explicit notification channel before anything can be
 * posted or scheduled; without one, scheduling silently fails or throws on a
 * real device. iOS and web have no channels, so this is a no-op there.
 */
async function ensureAndroidChannel(alarmStyle = false) {
  if (Platform.OS !== "android") return;
  await Notifications.setNotificationChannelAsync(androidChannelId, {
    name: "Forge System",
    importance: alarmStyle ? Notifications.AndroidImportance.MAX : Notifications.AndroidImportance.DEFAULT,
    showBadge: false
  });
}

export async function enableDailySystemNotifications(host?: HostState, alarmStyle = false) {
  if (Platform.OS === "web" || isExpoGo) {
    return { enabled: false, message: "Notifications available in the installed app, not Expo Go." };
  }

  try {
    const permission = await requestNotificationAccess();
    if (!permission) {
      return { enabled: false, message: "Notification permission was not granted." };
    }
    await ensureAndroidChannel(alarmStyle);

    await configureNotificationCategories();
    await scheduleDailyScan(host);
    return { enabled: true, message: `Daily check-in reminder set for ${preferredScanHour(host)}:00.` };
  } catch {
    return { enabled: false, message: "Couldn't set up notifications on this device." };
  }
}

export async function schedulePostScanQuestPrompt(state: AppState) {
  if (!canSchedule(state)) return "Notifications disabled.";
  try {
    await configureNotificationCategories();
    await cancelScheduledKind("post_scan");
    const seconds = state.host.focus < 0.35 ? 20 * 60 : state.host.challengeReadiness > 0.7 ? 5 * 60 : 12 * 60;
    await Notifications.scheduleNotificationAsync({
      content: {
        title: "Forge System OS",
        body: "Check-in saved. A quest is ready whenever you want to pick one.",
        categoryIdentifier: questCategoryId,
        data: { kind: "post_scan" as NotificationKind }
      },
      trigger: {
        type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL,
        seconds,
        channelId: androidChannelId
      }
    });
    return `Quest reminder set for ${Math.round(seconds / 60)} minutes from now.`;
  } catch {
    return "";
  }
}

export async function scheduleQuestWindowNotification(state: AppState, quest: Quest) {
  if (!canSchedule(state)) return "Notifications disabled.";
  try {
    await configureNotificationCategories();
    await cancelScheduledKind("quest_window");
    const seconds = quest.mode === "recover" ? 20 * 60 : quest.riskTier >= 3 ? 0 : Math.max(90, Math.min(15 * 60, quest.timeLimitMinutes * 30));
    if (seconds <= 0) return "High-risk quest selected; no reminder set — continue in the app when you're ready.";
    await Notifications.scheduleNotificationAsync({
      content: {
        title: "Quest window open",
        body: `${quest.title}: ${quest.timeLimitMinutes} min ${quest.difficultyBand} path.`,
        categoryIdentifier: questCategoryId,
        data: { kind: "quest_window" as NotificationKind, questId: quest.id }
      },
      trigger: {
        type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL,
        seconds,
        channelId: androidChannelId
      }
    });
    return `Quest reminder set for ${Math.round(seconds / 60)} minutes from now.`;
  } catch {
    return "";
  }
}

export async function disableSystemNotifications() {
  if (Platform.OS === "web") return "Notifications disabled.";
  try {
    await Notifications.cancelAllScheduledNotificationsAsync();
  } catch {
    // Already clear or unavailable — nothing to do.
  }
  return "System notifications disabled.";
}

async function scheduleDailyScan(host?: HostState) {
  await cancelScheduledKind("daily_scan");
  await Notifications.scheduleNotificationAsync({
    content: {
      title: "Forge System OS",
      body: dailyScanBody(host),
      categoryIdentifier: scanCategoryId,
      data: { kind: "daily_scan" as NotificationKind }
    },
    // DAILY repeats every day at the given hour/minute and is supported on
    // Android. (CALENDAR is iOS-only and throws on Android.)
    trigger: {
      type: Notifications.SchedulableTriggerInputTypes.DAILY,
      hour: preferredScanHour(host),
      minute: 0,
      channelId: androidChannelId
    }
  });
}

async function requestNotificationAccess() {
  const current = await Notifications.getPermissionsAsync();
  if (current.granted) return true;
  const requested = await Notifications.requestPermissionsAsync();
  return requested.granted;
}

function canSchedule(state: AppState) {
  return Platform.OS !== "web" && !isExpoGo && Boolean(state.profile.preferences.notificationConsent);
}

async function cancelScheduledKind(kind: NotificationKind) {
  const scheduled = await Notifications.getAllScheduledNotificationsAsync();
  await Promise.all(
    scheduled
      .filter((item) => item.content.data?.kind === kind)
      .map((item) => Notifications.cancelScheduledNotificationAsync(item.identifier))
  );
}

function preferredScanHour(host?: HostState) {
  if (!host) return 9;
  if (host.recoveryNeed > 0.65 || host.energy < 0.35) return 10;
  if (host.challengeReadiness > 0.7 && host.focus > 0.55) return 8;
  return 9;
}

function dailyScanBody(host?: HostState) {
  if (!host) return "Do a quick check-in and I'll suggest a quest that fits.";
  if (host.recoveryNeed > 0.65 || host.energy < 0.35) return "Start gently: check in, then pick a restful quest.";
  if (host.challengeReadiness > 0.7 && host.focus > 0.55) return "You've got room today. Check in, then open a quest.";
  return "Do a quick check-in and I'll suggest a quest that fits.";
}
