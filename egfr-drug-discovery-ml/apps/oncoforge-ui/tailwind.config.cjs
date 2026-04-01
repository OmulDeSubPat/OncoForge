/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      opacity: {
        6: "0.06",
        8: "0.08",
        12: "0.12",
        15: "0.15",
        45: "0.45",
        65: "0.65",
        78: "0.78",
      },
      fontFamily: {
        display: ['"Space Grotesk"', "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "SFMono-Regular", "ui-monospace", "monospace"],
      },
      colors: {
        forge: {
          bg: "#050816",
          surface: "#0c1424",
          elevated: "#101b31",
          line: "rgba(120, 139, 182, 0.22)",
          text: "#edf8ff",
          muted: "#94a3b8",
          cyan: "#24d6ea",
          teal: "#28d7b8",
          green: "#40d98f",
          amber: "#f9c74f",
          red: "#ff6b7a",
          blue: "#73a6ff",
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(36, 214, 234, 0.16), 0 24px 80px rgba(2, 8, 23, 0.54)",
        soft: "0 12px 40px rgba(2, 8, 23, 0.34)",
      },
      backgroundImage: {
        "grid-fine":
          "linear-gradient(rgba(148, 163, 184, 0.09) 1px, transparent 1px), linear-gradient(90deg, rgba(148, 163, 184, 0.09) 1px, transparent 1px)",
      },
      keyframes: {
        drift: {
          "0%, 100%": { transform: "translate3d(0, 0, 0) scale(1)" },
          "50%": { transform: "translate3d(0, -10px, 0) scale(1.02)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "0.72" },
          "50%": { opacity: "1" },
        },
        scan: {
          "0%": { transform: "translateX(-120%)" },
          "100%": { transform: "translateX(120%)" },
        },
      },
      animation: {
        drift: "drift 7s ease-in-out infinite",
        pulseSoft: "pulseSoft 3.8s ease-in-out infinite",
        scan: "scan 2.8s linear infinite",
      },
    },
  },
  plugins: [],
};
