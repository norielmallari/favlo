import { Stack } from 'expo-router';
import { Platform, View } from 'react-native';

import { ThemeProvider } from '@/theme/Theme';

function replaceLocalhost(): boolean {
  if (Platform.OS !== 'web' || typeof window === 'undefined') {
    return false;
  }
  if (window.location.hostname !== 'localhost') {
    return false;
  }
  const url = new URL(window.location.href);
  url.hostname = '127.0.0.1';
  window.location.replace(url.toString());
  return true;
}

export default function RootLayout() {
  if (replaceLocalhost()) {
    return null;
  }
  return (
    <ThemeProvider>
      <View style={{ flex: 1 }}>
        <Stack screenOptions={{ headerShown: false }} />
      </View>
    </ThemeProvider>
  );
}
