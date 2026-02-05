// Vite config migrated to ESM for top-level await and import.meta support
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig(async () => {
  const plugins = [react()];
  // Dynamic imports for plugins
  try {
    const cartographer = await import('@replit/vite-plugin-cartographer');
    plugins.push(cartographer.default());
  } catch {}
  try {
    const devBanner = await import('@replit/vite-plugin-dev-banner');
    plugins.push(devBanner.default());
  } catch {}
  return {
    plugins,
    sourcemap: false,
    resolve: {
      alias: {
        '@': path.join(process.cwd(), 'client', 'src'),
        '@shared': path.join(process.cwd(), 'shared'),
        '@assets': path.join(process.cwd(), 'attached_assets'),
      },
    },
    root: path.join(process.cwd(), 'client'),
    build: {
      outDir: path.join(process.cwd(), 'dist/public'),
      chunkSizeWarningLimit: 1000,
    },
    optimizeDeps: {
      exclude: ['lightningcss'],
    },
  };
});
