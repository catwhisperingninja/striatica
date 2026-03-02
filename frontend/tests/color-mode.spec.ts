import { test, expect } from '@playwright/test'

test.describe('Color mode switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=/\\d+.*points/')).toBeVisible({ timeout: 10000 })
  })

  test('starts in cluster color mode', async ({ page }) => {
    await page.keyboard.press('`')
    await expect(page.locator('text=/colorMode.*cluster/')).toBeVisible()
  })

  test('switches to local dim mode', async ({ page }) => {
    await page.locator('text=Local Dim').click()
    await page.keyboard.press('`')
    await expect(page.locator('text=/colorMode.*localDim/')).toBeVisible({ timeout: 2000 })
  })

  test('switches back to cluster mode', async ({ page }) => {
    await page.locator('text=Local Dim').click()
    await page.locator('text=Cluster').click()
    await page.keyboard.press('`')
    await expect(page.locator('text=/colorMode.*cluster/')).toBeVisible({ timeout: 2000 })
  })

  test('reset button clears selections', async ({ page }) => {
    // Select a cluster first
    const firstCluster = page.locator('text=/Cluster \\d+/').first()
    await firstCluster.click()

    // Click reset
    const resetBtn = page.locator('text=Reset').or(page.locator('[title*="Reset"]'))
    if (await resetBtn.count() > 0) {
      await resetBtn.first().click()
      await page.keyboard.press('`')
      await expect(page.locator('text=/selectedIndex.*null/')).toBeVisible({ timeout: 2000 })
    }
  })
})
