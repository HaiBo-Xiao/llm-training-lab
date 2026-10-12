import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';

export default defineConfig({
  output: 'static',
  trailingSlash: 'ignore',
  integrations: [mdx(), react()],
  markdown: {
    processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
    shikiConfig: { theme: 'github-light' },
  },
  devToolbar: { enabled: false },
  vite: { server: { watch: { usePolling: true, interval: 500 } } },
});
