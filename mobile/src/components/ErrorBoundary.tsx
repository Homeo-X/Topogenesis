import React from "react";
import { Pressable, ScrollView, Text, View } from "react-native";
import { styles } from "../theme";

interface Props {
  children: React.ReactNode;
  onReset?: () => void;
}

interface State {
  error: Error | null;
}

/**
 * Root crash container. A render error in any screen is caught here and shown
 * as a recoverable panel instead of white-screening the app. "Reset" lets the
 * user clear the error (and optionally local state via onReset) and continue.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error) {
    // Surface for debugging; in production this is where telemetry would hook in.
    console.error("Forge System crashed:", error);
  }

  handleReset = () => {
    this.props.onReset?.();
    this.setState({ error: null });
  };

  render() {
    if (this.state.error) {
      return (
        <View style={styles.shell}>
          <ScrollView contentContainerStyle={styles.contentPad}>
            <View style={styles.panel}>
              <Text style={styles.panelTitle}>System Fault</Text>
              <Text style={styles.bodyText}>
                The console hit an unexpected error and paused to protect your data. Your saved progress is intact.
              </Text>
              <Text style={styles.muted}>{this.state.error.message}</Text>
              <Pressable onPress={this.handleReset} style={({ pressed }) => [styles.primaryButton, pressed && styles.pressed]}>
                <Text style={styles.primaryButtonText}>Reset View</Text>
              </Pressable>
            </View>
          </ScrollView>
        </View>
      );
    }
    return this.props.children;
  }
}
