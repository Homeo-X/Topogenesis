import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

/**
 * Vite plugin that redirects api.js → api.local.ts and stubs out
 * Node built-ins (fs, path) so sim-core's contentLoader doesn't crash.
 */
function standalonePlugin(): Plugin {
  const fsStub = path.resolve(__dirname, 'src/stubs/fs.ts');
  const pathStub = path.resolve(__dirname, 'src/stubs/path.ts');
  const apiLocal = path.resolve(__dirname, 'src/api.local.ts');

  return {
    name: 'homeorealm-standalone',
    enforce: 'pre',
    resolveId(id, importer) {
      // Remap api.js → api.local.ts
      if (id.endsWith('api.js') && importer && !importer.includes('api.local')) {
        return apiLocal;
      }
      // Stub Node built-ins
      if (id === 'fs' || id === 'node:fs') return fsStub;
      if (id === 'path' || id === 'node:path') return pathStub;
      return null;
    },
  };
}

export default defineConfig({
  plugins: [standalonePlugin(), react()],
  resolve: {
    alias: {
      '@homeorealm/sim-core': path.resolve(__dirname, '../sim-core/src/index.ts'),
    },
  },
  build: {
    outDir: 'dist',
    target: 'es2020',
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'three-vendor': ['three', '@react-three/fiber', '@react-three/drei'],
          'sim-core': ['../sim-core/src/index.ts'],
        },
      },
    },
  },
});
