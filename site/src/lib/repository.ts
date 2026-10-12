import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chapters } from '../data/chapters';

const root = resolve(process.cwd(), '..');
const report = 'reports/m0-20260910-025345/';

// 仅发布明确选定的源码和实验材料；读取发生在构建时。
export const resources = [
  ...chapters.map(chapter => ({ slug: chapter.slug, title: chapter.title, path: `scripts/${chapter.file}`, language: 'python' as const })),
  { slug: 'environment-check', title: '环境检查脚本', path: 'scripts/00_check_training.py', language: 'python' as const },
  { slug: 'plan', title: '完整执行计划', path: 'PLAN.md', language: 'markdown' as const },
  { slug: 'tasks', title: '进度与验收证据', path: 'TASKS.md', language: 'markdown' as const },
  { slug: 'acceptance', title: '阶段验收标准', path: 'ACCEPTANCE.md', language: 'markdown' as const },
  { slug: 'environment', title: '环境与存储约定', path: 'ENVIRONMENT.md', language: 'markdown' as const },
  { slug: 'report-template', title: '实验报告模板', path: 'reports/TEMPLATE.md', language: 'markdown' as const },
  { slug: 'm0-report', title: 'M0 完整报告', path: `${report}REPORT.md`, language: 'markdown' as const },
  { slug: 'm0-output', title: 'M0 原始输出', path: `${report}output.txt`, language: 'text' as const },
  { slug: 'm0-config', title: 'M0 实验配置', path: `${report}run.json`, language: 'json' as const },
  { slug: 'm0-rebuild', title: 'M0 重建输出', path: `${report}rebuild_output.txt`, language: 'text' as const },
];

export function readResource(path: string): string {
  if (!resources.some(resource => resource.path === path)) throw new Error('未登记的展示材料');
  return readFileSync(resolve(root, path), 'utf8');
}
