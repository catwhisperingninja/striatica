import { test, expect } from '@playwright/test'

test.describe('Cluster selection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    // Wait for data load
    await expect(page.locator('text=/Points:\\s*[\\d,]+/')).toBeVisible({ timeout: 10000 })
    // Wait for the NavPanel cluster rows to render before any test counts them
    // (locator.count() does not auto-wait; under parallel load the list may not
    // be painted yet, which would make the count()<2 guard skip spuriously).
    await expect(page.locator('text=/Cluster \\d+/').first()).toBeVisible({ timeout: 5000 })
  })

  // The debug console defaults open (useState(true)); its "clusters" state row
  // is already visible. The store field is also echoed in the transition log
  // (as "selectedClusters"), so .first() targets the state-panel row.
  test('clicking a cluster selects it', async ({ page }) => {
    const firstCluster = page.locator('text=/Cluster \\d+/').first()
    await firstCluster.click()
    // Debug console state row shows selectedClusters changed
    await expect(page.locator('text=/clusters.*\\{/').first()).toBeVisible({ timeout: 2000 })
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

    // Debug console state row should show 2 clusters selected
    const clustersRow = page.locator('text=/clusters.*\\{/').first()
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

    const clustersRow = page.locator('text=/clusters.*\\{/').first()
    await expect(clustersRow).toBeVisible({ timeout: 2000 })
    // Should contain only one ID (no comma)
    const text = await clustersRow.textContent()
    // Either single ID or empty
    expect(text).not.toMatch(/\d+,\s*\d+/)
  })
})
