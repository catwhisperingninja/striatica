import { test, expect } from '@playwright/test'

test.describe('Debug console', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=/\\d+.*points/')).toBeVisible({ timeout: 10000 })
  })

  test('toggles open/closed with backtick key', async ({ page }) => {
    // Debug console should be visible by default (open: true)
    await expect(page.locator('text=Debug Console')).toBeVisible()

    // Press backtick to close
    await page.keyboard.press('`')
    await expect(page.locator('text=Debug Console')).not.toBeVisible()

    // Press backtick to reopen
    await page.keyboard.press('`')
    await expect(page.locator('text=Debug Console')).toBeVisible()
  })

  test('shows current state values', async ({ page }) => {
    await expect(page.locator('text=viewMode')).toBeVisible()
    await expect(page.locator('text=colorMode')).toBeVisible()
    await expect(page.locator('text=selectedIndex')).toBeVisible()
    await expect(page.locator('text=edgeThreshold')).toBeVisible()
  })

  test('logs state transitions', async ({ page }) => {
    // Switch color mode to trigger a log entry
    await page.locator('text=Local Dim').click()

    // Log should show colorMode transition
    await expect(page.locator('text=/colorMode.*cluster.*localDim/')).toBeVisible({ timeout: 2000 })
  })

  test('copy button works', async ({ page }) => {
    const copyBtn = page.locator('text=Copy')
    await expect(copyBtn).toBeVisible()
    await copyBtn.click()
    await expect(page.locator('text=Copied!')).toBeVisible()
  })

  test('clear button empties log', async ({ page }) => {
    // Generate some log entries
    await page.locator('text=Circuits').click()
    await page.locator('text=Point Cloud').click()

    // Clear the log
    const clearBtn = page.locator('text=Clear')
    await clearBtn.click()

    // Should show empty state
    await expect(page.locator('text=Interact with the scene...')).toBeVisible()
  })
})
