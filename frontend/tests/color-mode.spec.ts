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
    // Assert the STATE-PANEL colorMode row, not the transition log. The log entry
    // "colorMode: cluster → localDim" also matches /colorMode.*cluster/, so the old
    // .first() regex passed even if the switch-back did nothing. The state row's
    // label span has exact text "colorMode" (the log message never does), so this
    // filter selects that single row. "cluster" and "localDim" share no substring
    // and neither is a substring of the label, so the row can satisfy exactly one
    // value — asserting both present-new and absent-old couples the test to the click.
    const colorModeRow = page
      .locator('div.justify-between')
      .filter({ has: page.getByText('colorMode', { exact: true }) })
    await expect(colorModeRow).toContainText('cluster', { timeout: 2000 })
    await expect(colorModeRow).not.toContainText('localDim')
  })

  // FIXME: doubly vacuous — the `if (count>0)` guard yields zero assertions when the
  // reset button is absent, and /selectedIndex.*null/ is pre-satisfied on a fresh page.
  // Phase 1 fix: click reset unconditionally, then assert the state-panel clusters row
  // returns to '(none)' (as the switch-back tests assert their state rows).
  test.fixme('reset button clears selections', async ({ page }) => {
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
