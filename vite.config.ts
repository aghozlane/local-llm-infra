import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// base './' : rend le build utilisable sur GitHub Pages sous /local-llm-infra/
export default defineConfig({
  plugins: [react()],
  base: './',
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    globals: false,
  },
});
