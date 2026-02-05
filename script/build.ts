import esbuild from 'esbuild';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function build() {
  try {
    console.log('Building application...');
    
    await esbuild.build({
      entryPoints: [join(__dirname, '../server/index.ts')],
      bundle: true,
      platform: 'node',
      target: 'node18',
      format: 'cjs',
      outfile: join(__dirname, '../dist/index.cjs'),
      external: [
        'better-sqlite3',
        'drizzle-orm',
        'express',
        'cors',
        'dotenv',
        'multer',
        'waveform-data',
        'lightningcss',
        'vite',
      ],
      sourcemap: true,
      minify: false,
    });

    console.log('✓ Build completed successfully!');
  } catch (error) {
    console.error('✗ Build failed:', error);
    process.exit(1);
  }
}

build();
