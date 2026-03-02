import { test, expect } from '@playwright/test'

test.describe('Cluster selection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    // Wait for data load
    await expect(page.locator('text=/\\d+.*points/')).toBeVisible({ timeout: 10000 })
  })

  test('clicking a cluster selects it', async ({ page }) => {
    const firstCluster = page.locator('text=/Cluster \\d+/').first()
    await firstCluster.click()
    // The cluster row should get highlighted styling (bg-white/5 or similar)
    // Check debug console shows selectedClusters changed
    await page.keyboard.press('`') // open debug console
    await expect(page.locator('text=/clusters.*\\{/')).toBeVisible({ timeout: 2000 })
  })

  test('shift-click selects multiple clusters', async ({ page }) => {
    const clusters = page.locator('text=/Cluster \\d+/')
    const count = await clusters.count()
    if (count < 2) {
      test.skip()
      return
    }

    await clusters.nth(0).click()
    await clusters.nth(1).click({ modifiers: ['Shift'] })

    // Debug console should show 2 clusters selected
    await page.keyboard.press('`')
    const clustersRow = page.locator('text=/clusters.*\\{/')
    await expect(clustersRow).toBeVisible({ timeout: 2000 })
    // Should contain a comma (meaning 2+ IDs)
    const text = await clustersRow.textContent()
    expect(text).toMatch(/\d+,\s*\d+/)
  })

  test('clicking without shift replaces selection', async ({ page }) => {
    const clusters = page.locator('text=/Cluster \\d+/')
    const count = await clusters.count()
    if (count < 2) {
      test.skip()
      return
    }

    await clusters.nth(0).click()
    await clusters.nth(1).click() // no shift — should replace, not add

    await page.keyboard.press('`')
    const clustersRow = page.locator('text=/clusters.*\\{/')
    await expect(clustersRow).toBeVisible({ timeout: 2000 })
    // Should contain only one ID (no comma)
    const text = await clustersRow.textContent()
    // Either single ID or empty
    expect(text).not.toMatch(/\d+,\s*\d+/)
  })
})
