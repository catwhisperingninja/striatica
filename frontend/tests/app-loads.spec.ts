import { test, expect } from '@playwright/test'

test.describe('App loads and renders', () => {
  test('loads dataset and shows point count in status bar', async ({ page }) => {
    await page.goto('/')
    // Wait for dataset to load — status bar shows "Points: 24,576"
    await expect(page.locator('text=/Points:\\s*[\\d,]+/')).toBeVisible({ timeout: 10000 })
  })

  test('canvas element exists and is sized', async ({ page }) => {
    await page.goto('/')
    // Two <canvas> exist: the main R3F scene and the minimap. .first() is the
    // main scene canvas (rendered before the minimap in App.tsx).
    const canvas = page.locator('canvas').first()
    await expect(canvas).toBeVisible({ timeout: 10000 })
    // R3F mounts the canvas at its default 300x150 intrinsic size, then its
    // resize observer grows it to fill the container a frame later. Poll the
    // box until layout settles so we measure the real size, not the default.
    await expect
      .poll(async () => (await canvas.boundingBox())?.width ?? 0, { timeout: 5000 })
      .toBeGreaterThan(200)
    await expect
      .poll(async () => (await canvas.boundingBox())?.height ?? 0, { timeout: 5000 })
      .toBeGreaterThan(200)
  })

  test('TopBar renders with color mode toggle', async ({ page }) => {
    await page.goto('/')
    // "Cluster"/"Local Dim" text also appears in cluster rows and the debug
    // console, so target the TopBar color-mode buttons specifically.
    await expect(page.getByRole('button', { name: 'Cluster', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Local Dim' })).toBeVisible()
  })

  test('NavPanel renders with clusters', async ({ page }) => {
    await page.goto('/')
    // Wait for data load, then clusters should appear
    await expect(page.locator('text=/Points:\\s*[\\d,]+/')).toBeVisible({ timeout: 10000 })
    // At least one cluster row should be visible (there are 51 "Cluster N" rows)
    await expect(page.locator('text=/Cluster \\d+/').first()).toBeVisible({ timeout: 5000 })
  })
})
