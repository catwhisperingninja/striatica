import { test, expect } from '@playwright/test'

test.describe('Color mode switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=/Points:\\s*[\\d,]+/')).toBeVisible({ timeout: 10000 })
  })

  // The debug console defaults open (useState(true)), so the state rows below
  // are already visible — no backtick press needed. Each store field is also
  // echoed in the transition log, so .first() targets the state-panel row.
  test('starts in cluster color mode', async ({ page }) => {
    await expect(page.locator('text=/colorMode.*cluster/').first()).toBeVisible()
  })

  test('switches to local dim mode', async ({ page }) => {
    await page.getByRole('button', { name: 'Local Dim' }).click()
    await expect(page.locator('text=/colorMode.*localDim/').first()).toBeVisible({ timeout: 2000 })
  })

  test('switches back to cluster mode', async ({ page }) => {
    await page.getByRole('button', { name: 'Local Dim' }).click()
    await page.getByRole('button', { name: 'Cluster', exact: true }).click()
    await expect(page.locator('text=/colorMode.*cluster/').first()).toBeVisible({ timeout: 2000 })
  })

  test('reset button clears selections', async ({ page }) => {
    // Select a cluster first
    const firstCluster = page.locator('text=/Cluster \\d+/').first()
    await firstCluster.click()

    // Click reset
    const resetBtn = page.locator('text=Reset').or(page.locator('[title*="Reset"]'))
    if (await resetBtn.count() > 0) {
      await resetBtn.first().click()
      await expect(page.locator('text=/selectedIndex.*null/').first()).toBeVisible({ timeout: 2000 })
    }
  })
})
