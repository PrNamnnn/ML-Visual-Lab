import type { Config } from "tailwindcss";

const v = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        bg: v("bg"),
        panel: v("panel"),
        raised: v("raised"),
        hover: v("hover"),
        accent: v("accent"),
        line: v("line"),
        "line-strong": v("line-strong"),
        fg: v("fg"),
        muted: v("muted"),
        faint: v("faint"),
        blue: v("blue"),
        pink: v("pink"),
        green: v("green"),
        amber: v("amber"),
        red: v("red"),
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      borderRadius: { DEFAULT: "6px" },
    },
  },
  plugins: [],
} satisfies Config;
