import { test, expect } from '@playwright/test'

test.describe('Circuit view interactions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=/\\d+.*points/')).toBeVisible({ timeout: 10000 })
    // Switch to circuits view
    await page.locator('text=Circuits').click()
    await expect(page.locator('text=Select Circuit')).toBeVisible({ timeout: 3000 })
  })

  test('circuit manifest loads and shows entries', async ({ page }) => {
    // Should show at least one circuit entry with node/edge counts
    await expect(page.locator('text=/\\d+ nodes/')).toBeVisible({ timeout: 5000 })
  })

  test('clicking a circuit loads it and shows controls', async ({ page }) => {
    // Wait for manifest
    const entry = page.locator('text=/\\d+ nodes/').first()
    await expect(entry).toBeVisible({ timeout: 5000 })
    // Click the circuit entry (click parent div)
    await entry.locator('..').click()

    // Should now show edge threshold slider and back button
    await expect(page.locator('text=Edge threshold')).toBeVisible({ timeout: 5000 })
    await expect(page.locator('text=←')).toBeVisible()
  })

  test('edge threshold slider updates debug state', async ({ page }) => {
    // Load a circuit
    const entry = page.locator('text=/\\d+ nodes/').first()
    await expect(entry).toBeVisible({ timeout: 5000 })
    await entry.locator('..').click()
    await expect(page.locator('text=Edge threshold')).toBeVisible({ timeout: 5000 })

    // Open debug console
    await page.keyboard.press('`')
    const thresholdRow = page.locator('text=/edgeThreshold/')
    await expect(thresholdRow).toBeVisible()

    // Move the slider
    const slider = page.locator('input[type="range"]')
    await slider.fill('0.5')

    // Debug console should reflect the change
    await expect(page.locator('text=/edgeThreshold.*0\\.50/')).toBeVisible({ timeout: 2000 })
  })

  test('back button returns to circuit selector', async ({ page }) => {
    const entry = page.locator('text=/\\d+ nodes/').first()
    await expect(entry).toBeVisible({ timeout: 5000 })
    await entry.locator('..').click()
    await expect(page.locator('text=Edge threshold')).toBeVisible({ timeout: 5000 })

    // Click back
    await page.locator('text=←').click()
    await expect(page.locator('text=Select Circuit')).toBeVisible({ timeout: 3000 })
  })

  test('clicking a circuit node selects it and triggers fly-to', async ({ page }) => {
    // Load a circuit
    const entry = page.locator('text=/\\d+ nodes/').first()
    await expect(entry).toBeVisible({ timeout: 5000 })
    await entry.locator('..').click()
    await expect(page.locator('text=Edge threshold')).toBeVisible({ timeout: 5000 })

    // Open debug console
    await page.keyboard.press('`')

    // Click a node in the node list (look for activation values like "0.85")
    const nodeItem = page.locator('text=/^0\\.\\d{2}\\s/')
    const nodeCount = await nodeItem.count()
    if (nodeCount === 0) {
      test.skip()
      return
    }
    await nodeItem.first().click()

    // selectedIndex should be non-null and flyTarget should be set
    await expect(page.locator('text=/selectedIndex.*\\d/')).toBeVisible({ timeout: 2000 })
    await expect(page.locator('text=/flyTarget.*\\[/')).toBeVisible({ timeout: 2000 })
  })
})
