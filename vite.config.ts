import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  base: '/uni/',
  plugins: [react(), tailwindcss()],
  server: { host: true, port: 3000 },
  preview: { host: true, port: 3000 },
});
