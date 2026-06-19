import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.homeorealm.app',
  appName: 'HomeoRealm',
  webDir: 'dist',
  server: {
    androidScheme: 'https',
  },
  android: {
    minWebViewVersion: 88,
  },
};

export default config;
