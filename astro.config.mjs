import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

export default defineConfig({
  site: "https://llm-curriculum.vercel.app",
  trailingSlash: "ignore",
  integrations: [mdx()],
  markdown: { shikiConfig: { theme: "github-dark-dimmed" } },
});
