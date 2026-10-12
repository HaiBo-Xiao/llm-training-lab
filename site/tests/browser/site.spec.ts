import { test, expect, type Page } from '@playwright/test';

async function openPage(page: Page, path: string) {
  await page.goto(path);
  await expect(page.locator('astro-island[ssr]')).toHaveCount(0);
}

async function openLab(page: Page) {
  await page.locator('.lab-disclosure > summary').click();
}

test('主页、章节、原始材料与搜索可正常访问', async ({ page, request }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await openPage(page, '/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('让每一次计算');
  await page.getByRole('button', { name: '位置 2' }).click();
  await expect(page.locator('.figure-caption')).toContainText('0、1、2');
  await page.screenshot({ path: '.playwright/home-desktop.png', fullPage: true });
  await page.getByRole('button', { name: '搜索章节' }).click();
  await page.getByRole('searchbox').fill('attention');
  await expect(page.locator('.search-results a:visible')).toHaveCount(1);
  await page.locator('.search-results a:visible').click();
  await expect(page).toHaveURL(/\/learn\/attention\//);
  for (const path of ['/learn/', '/learn/tokens/', '/learn/batches/', '/learn/forward/', '/learn/loss/', '/experiments/', '/roadmap/', '/source/attention/', '/source/m0-report/', '/raw/attention.txt']) {
    const response = await request.get(path);
    expect(response.status(), path).toBe(200);
  }
  const source = await request.get('/raw/attention.txt');
  expect(await source.text()).toContain('def apply_causal_mask');
  expect(errors).toEqual([]);
});

test('注意力交互能验证遮罩的作用，播放与恢复工作正常', async ({ page }) => {
  await openPage(page, '/learn/attention/');
  await openLab(page);
  await page.getByRole('button', { name: '汇总信息' }).click();
  const first = page.locator('[data-output-row="0"] strong');
  const second = page.locator('[data-output-row="1"] strong');
  const third = page.locator('[data-output-row="2"] strong');
  await expect(first).toHaveText('[1.000, 0.000]');
  await expect(second).toHaveText('[0.330, 0.670]');
  const slider = page.getByRole('slider', { name: '位置 2 的 V 值' });
  await slider.fill('100');
  await expect(first).toHaveText('[1.000, 0.000]');
  await expect(second).toHaveText('[0.330, 0.670]');
  await expect(third).toHaveText('[50.597, 50.597]');
  await page.getByRole('checkbox', { name: '启用因果遮罩' }).uncheck();
  await expect(first).not.toHaveText('[1.000, 0.000]');
  await expect(second).not.toHaveText('[0.330, 0.670]');
  await page.getByRole('button', { name: '重置', exact: true }).click();
  await expect(slider).toHaveValue('2');
  await expect(page.getByRole('checkbox', { name: '启用因果遮罩' })).toBeChecked();
  await page.getByRole('button', { name: '播放完整过程' }).click();
  await expect(page.getByRole('button', { name: '暂停播放' })).toBeVisible();
  await expect(page.getByRole('button', { name: '播放完整过程' })).toBeVisible({ timeout: 10000 });
  await expect(page.getByRole('button', { name: '汇总信息' })).toHaveAttribute('aria-pressed', 'true');
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: '.playwright/attention-desktop.png', fullPage: true });
  await page.locator('.chapter-reading > summary').click();
  await page.getByText('2. 把未来分数设为 0，和设为负无穷，效果相同吗？', { exact: true }).click();
  await expect(page.getByText('不同。exp(0) = 1', { exact: false })).toBeVisible();
});

test('数据练习切换后仍保持输入、目标与分数一致', async ({ page }) => {
  await openPage(page, '/learn/tokens/');
  await openLab(page);
  await page.getByLabel('输入一段练习词表中的文字').fill('鱼小猫');
  await expect(page.locator('.mini-result')).toContainText('「鱼」');
  await page.getByLabel('输入一段练习词表中的文字').fill('未知');
  await expect(page.getByRole('alert')).toContainText('只使用');
  await openPage(page, '/learn/batches/');
  await openLab(page);
  await page.getByRole('combobox', { name: '起始样本编号 k₀' }).selectOption('1');
  await expect(page.locator('.array-row').first()).toContainText('[ 13, 14, 15 ]');
  await openPage(page, '/learn/forward/');
  await openLab(page);
  await page.getByRole('button', { name: '鱼 ID 3' }).click();
  await expect(page.locator('.mini-result')).toContainText('预测：猫');
  await expect(page.locator('.logit-row strong')).toHaveText(['0.0', '0.0', '3.0', '1.0']);
});

test('手机布局无横向溢出，导航与交互可操作', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openPage(page, '/');
  await page.screenshot({ path: '.playwright/home-mobile.png', fullPage: true });
  await page.locator('.mobile-nav summary').click();
  await page.getByRole('navigation', { name: '移动端导航' }).getByRole('link', { name: '学习手册' }).click();
  await expect(page).toHaveURL(/\/learn\//);
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    for (const path of ['/', '/learn/attention/', '/learn/tokens/', '/learn/batches/', '/learn/forward/', '/learn/loss/', '/experiments/', '/roadmap/']) {
      await openPage(page, path);
      const fits = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
      if (!fits) console.log(await page.locator('body *').evaluateAll(elements => elements.filter(el => el.getBoundingClientRect().right > innerWidth + 1).slice(0, 12).map(el => ({ element: el.tagName, class: el.className, right: el.getBoundingClientRect().right, text: el.textContent?.slice(0,60) }))));
      expect(fits, `${width}px: ${path}`).toBe(true);
    }
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await openPage(page, '/learn/attention/');
  await openLab(page);
  await page.getByRole('slider', { name: '位置 2 的 V 值' }).fill('100');
  await expect(page.locator('[data-output-row="2"] strong')).toHaveText('[50.597, 50.597]');
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: '.playwright/attention-mobile.png', fullPage: true });
});

test('19 个 Manim 片段均可加载，能选节、播放、倍速与按文字定位', async ({ page, request }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const [chapter, count] of [['tokens', 4], ['batches', 6], ['forward', 4], ['attention', 5]] as const) {
    await openPage(page, `/learn/${chapter}/`);
    const cinema = page.locator('lesson-cinema');
    await expect(cinema).toHaveAttribute('data-ready', 'true');
    await expect(cinema.locator('[data-index]')).toHaveCount(count);
    await expect(page.locator('.chapter-reading')).not.toHaveAttribute('open');
    for (let index = 0; index < count; index++) {
      await cinema.locator('[data-index]').nth(index).click();
      await expect(cinema.locator('[data-index]').nth(index)).toHaveAttribute('aria-pressed', 'true');
      await expect.poll(() => cinema.locator('video').evaluate((video: HTMLVideoElement) => video.readyState)).toBeGreaterThanOrEqual(2);
      const expected = JSON.parse((await cinema.getAttribute('data-lessons'))!)[index];
      expect(await cinema.locator('video').evaluate((video: HTMLVideoElement) => ({ width: video.videoWidth, height: video.videoHeight, error: video.error?.code ?? null }))).toEqual({ width: expected.width, height: expected.height, error: null });
      const poster = await cinema.locator('video').getAttribute('poster');
      const captions = await cinema.locator('track').getAttribute('src');
      expect((await request.get(poster!)).ok()).toBe(true);
      expect(await (await request.get(captions!)).text()).toContain('WEBVTT');
    }
  }
  await openPage(page, '/learn/attention/#attention-mask');
  const cinema = page.locator('lesson-cinema');
  const video = cinema.locator('video');
  await expect(cinema.locator('[data-title]')).toHaveText('屏蔽未来位置');
  await expect(video).toHaveAttribute('src', '/animations/attention-mask.mp4');
  await video.evaluate((element: HTMLVideoElement) => element.play());
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.currentTime)).toBeGreaterThan(0.3);
  await page.getByLabel('动画播放速度').selectOption('1.5');
  expect(await video.evaluate((element: HTMLVideoElement) => element.playbackRate)).toBe(1.5);
  await cinema.locator('.cinema-transcript > summary').click();
  const last = cinema.locator('[data-cues] button').last();
  const time = Number(await last.getAttribute('data-time'));
  await last.click();
  expect(await video.evaluate((element: HTMLVideoElement) => element.paused)).toBe(true);
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.currentTime)).toBeCloseTo(time + 1, 1);
  await expect(cinema.locator('[data-description]')).toContainText('exp(0) = 1');
  await page.getByLabel('连播本章').check();
  await video.evaluate(async (element: HTMLVideoElement) => { element.currentTime = element.duration - 0.1; await element.play(); });
  await expect(video).toHaveAttribute('src', '/animations/attention-softmax.mp4');
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.currentTime)).toBeGreaterThan(0.2);
  await page.getByRole('button', { name: '上一小节' }).click();
  await expect(video).toHaveAttribute('src', '/animations/attention-mask.mp4');
  expect(await video.evaluate((element: HTMLVideoElement) => element.paused)).toBe(true);
  await cinema.locator('.cinema-transcript > summary').click();
  await page.evaluate(() => { (document.activeElement as HTMLElement)?.blur(); window.scrollTo({ top: 0, behavior: 'instant' }); });
  await page.screenshot({ path: '.playwright/manim-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: '.playwright/manim-mobile.png', fullPage: true });
  await openPage(page, '/learn/attention/#手算位置-1-的输出');
  await expect(page.locator('.chapter-reading')).toHaveAttribute('open');
  expect(errors).toEqual([]);
});
