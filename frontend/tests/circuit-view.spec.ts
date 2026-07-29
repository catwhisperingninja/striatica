import { test, expect } from '@playwright/test'

test.describe('Circuit view interactions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('text=/Points:\\s*[\\d,]+/')).toBeVisible({ timeout: 10000 })
    // Switch to circuits view ("Circuits" text also appears in the debug
    // console, so target the TopBar view button specifically).
    await page.getByRole('button', { name: 'Circuits', exact: true }).click()
    await expect(page.locator('text=Select Circuit')).toBeVisible({ timeout: 3000 })
  })

  test('circuit manifest loads and shows entries', async ({ page }) => {
    // Should show at least one circuit entry with node/edge counts
    // (the manifest lists 10, so .first() picks the first entry).
    await expect(page.locator('text=/\\d+ nodes/').first()).toBeVisible({ timeout: 5000 })
  })

  test('clicking a circuit loads it and shows controls', async ({ page }) => {
    // Wait for manifest
    const entry = page.locator('text=/\\d+ nodes/').first()
    await expect(entry).toBeVisible({ timeout: 5000 })
    // Click the circuit entry (click parent div)
    await entry.locator('..').click()

    // Should now show edge threshold slider and back button. The "←" glyph also
    // appears inside some feature explanations, so target the back button.
    await expect(page.locator('text=Edge threshold')).toBeVisible({ timeout: 5000 })
    await expect(page.getByRole('button', { name: '←' })).toBeVisible()
  })

  test('edge threshold slider updates debug state', async ({ page }) => {
    // Load a circuit
    const entry = page.locator('text=/\\d+ nodes/').first()
    await expect(entry).toBeVisible({ timeout: 5000 })
    await entry.locator('..').click()
    await expect(page.locator('text=Edge threshold')).toBeVisible({ timeout: 5000 })

    // Debug console defaults open (useState(true)) — its edgeThreshold row is
    // already visible, no backtick press needed.
    const thresholdRow = page.locator('text=/edgeThreshold/').first()
    await expect(thresholdRow).toBeVisible()

    // Move the slider
    const slider = page.locator('input[type="range"]')
    await slider.fill('0.5')

    // Debug console state row should reflect the change (.first() targets the
    // state-panel row; the value is also echoed in the transition log).
    await expect(page.locator('text=/edgeThreshold.*0\\.50/').first()).toBeVisible({ timeout: 2000 })
  })

  test('back button returns to circuit selector', async ({ page }) => {
    const entry = page.locator('text=/\\d+ nodes/').first()
    await expect(entry).toBeVisible({ timeout: 5000 })
    await entry.locator('..').click()
    await expect(page.locator('text=Edge threshold')).toBeVisible({ timeout: 5000 })

    // Click back (target the back button — "←" also appears in feature text)
    await page.getByRole('button', { name: '←' }).click()
    await expect(page.locator('text=Select Circuit')).toBeVisible({ timeout: 3000 })
  })

  test('clicking a circuit node selects it and triggers fly-to', async ({ page }) => {
    // Load a circuit
    const entry = page.locator('text=/\\d+ nodes/').first()
    await expect(entry).toBeVisible({ timeout: 5000 })
    await entry.locator('..').click()
    await expect(page.locator('text=Edge threshold')).toBeVisible({ timeout: 5000 })

    // Debug console defaults open (useState(true)) — no backtick press needed.

    // Click a node in the node list (look for activation values like "0.85").
    // A loaded circuit MUST render nodes — an empty node list is a rendering
    // regression, so assert rather than skip. Wait for the first node (count()
    // does not auto-wait) before asserting the count is non-zero.
    const nodeItem = page.locator('text=/^0\\.\\d{2}\\s/')
    await expect(nodeItem.first()).toBeVisible({ timeout: 5000 })
    expect(await nodeItem.count()).toBeGreaterThan(0)
    await nodeItem.first().click()

    // selectedIndex should be non-null and flyTarget should be set (.first()
    // targets the state-panel row; both are also echoed in the transition log).
    await expect(page.locator('text=/selectedIndex.*\\d/').first()).toBeVisible({ timeout: 2000 })
    await expect(page.locator('text=/flyTarget.*\\[/').first()).toBeVisible({ timeout: 2000 })
  })
})
