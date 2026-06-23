/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: { DEFAULT: '#1d4ed8', dark: '#1e3a8a', light: '#3b82f6' },
        danger:  { DEFAULT: '#dc2626', light: '#fca5a5' },
        warning: { DEFAULT: '#d97706', light: '#fcd34d' },
        success: { DEFAULT: '#16a34a', light: '#86efac' },
      },
    },
  },
  plugins: [],
}
