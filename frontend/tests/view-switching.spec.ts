import { test, expect } from '@playwright/test'

test.describe('View mode switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=/Points:\\s*[\\d,]+/')).toBeVisible({ timeout: 10000 })
  })

  // The debug console renders each store field in both the state panel and the
  // transition log, so a "viewMode …" regex can match two elements. .first()
  // targets the state-panel row (rendered before the log). The console defaults
  // open (useState(true)), so no backtick press is needed to reveal it.
  test('starts in Point Cloud view', async ({ page }) => {
    await expect(page.locator('text=/viewMode.*pointCloud/').first()).toBeVisible({ timeout: 2000 })
  })

  test('can switch to Circuits view', async ({ page }) => {
    await page.getByRole('button', { name: 'Circuits', exact: true }).click()
    await expect(page.locator('text=/viewMode.*circuits/').first()).toBeVisible({ timeout: 2000 })
    // Circuit panel should show "Select Circuit"
    await expect(page.locator('text=Select Circuit')).toBeVisible()
  })

  test('can switch back to Point Cloud', async ({ page }) => {
    await page.getByRole('button', { name: 'Circuits', exact: true }).click()
    await page.getByRole('button', { name: 'Point Cloud' }).click()
    await expect(page.locator('text=/viewMode.*pointCloud/').first()).toBeVisible({ timeout: 2000 })
  })
})
