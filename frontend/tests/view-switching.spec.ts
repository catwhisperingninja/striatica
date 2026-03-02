import { test, expect } from '@playwright/test'

test.describe('View mode switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=/\\d+.*points/')).toBeVisible({ timeout: 10000 })
  })

  test('starts in Point Cloud view', async ({ page }) => {
    await page.keyboard.press('`')
    await expect(page.locator('text=/viewMode.*pointCloud/')).toBeVisible({ timeout: 2000 })
  })

  test('can switch to Circuits view', async ({ page }) => {
    await page.locator('text=Circuits').click()
    await page.keyboard.press('`')
    await expect(page.locator('text=/viewMode.*circuits/')).toBeVisible({ timeout: 2000 })
    // Circuit panel should show "Select Circuit"
    await expect(page.locator('text=Select Circuit')).toBeVisible()
  })

  test('can switch back to Point Cloud', async ({ page }) => {
    await page.locator('text=Circuits').click()
    await page.locator('text=Point Cloud').click()
    await page.keyboard.press('`')
    await expect(page.locator('text=/viewMode.*pointCloud/')).toBeVisible({ timeout: 2000 })
  })
})
