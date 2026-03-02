import { test, expect } from '@playwright/test'

test.describe('App loads and renders', () => {
  test('loads dataset and shows point count in status bar', async ({ page }) => {
    await page.goto('/')
    // Wait for dataset to load — status bar shows point count
    await expect(page.locator('text=/\\d+.*points/')).toBeVisible({ timeout: 10000 })
  })

  test('canvas element exists and is sized', async ({ page }) => {
    await page.goto('/')
    const canvas = page.locator('canvas')
    await expect(canvas).toBeVisible({ timeout: 10000 })
    const box = await canvas.boundingBox()
    expect(box!.width).toBeGreaterThan(200)
    expect(box!.height).toBeGreaterThan(200)
  })

  test('TopBar renders with color mode toggle', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=Cluster')).toBeVisible()
    await expect(page.locator('text=Local Dim')).toBeVisible()
  })

  test('NavPanel renders with clusters', async ({ page }) => {
    await page.goto('/')
    // Wait for data load, then clusters should appear
    await expect(page.locator('text=/\\d+.*points/')).toBeVisible({ timeout: 10000 })
    // At least one cluster row should be visible
    await expect(page.locator('text=/Cluster \\d+/')).toBeVisible({ timeout: 5000 })
  })
})
