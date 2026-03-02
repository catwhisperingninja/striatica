# striatica/pipeline/banner.py
"""ASCII banner and terminal UI helpers for the striat CLI."""

from __future__ import annotations

import os
import sys
import time

# ── Banner ──────────────────────────────────────────────────────────────

BANNER = """
  ░▒▓  s t r i a t i c a  ≡≡≡≡≡  ▓▒░
  🔬 geometric atlas for machine intelligence
"""

TAGLINE = "  🔬 geometric atlas for machine intelligence"
VERSION_LINE = "  ⚡ v0.1.0"


def print_banner() -> None:
    """Print the striatica ASCII banner to stderr (so it doesn't pollute piped output)."""
    out = sys.stderr

    # Use the detailed banner — the block-character letters should
    # render in any modern terminal with Unicode support.
    out.write(BANNER)
    out.write(f"{VERSION_LINE}\n\n")
    out.flush()


# ── Progress Bar ────────────────────────────────────────────────────────

class ProgressBar:
    """Minimal, dependency-free terminal progress bar.

    Usage:
        bar = ProgressBar(total=24, label="Downloading features")
        for i in range(24):
            do_work()
            bar.update(i + 1)
        bar.finish()
    """

    BAR_CHARS = "░▒▓█"  # striation-themed fill

    def __init__(
        self,
        total: int,
        label: str = "",
        width: int = 30,
        emoji: str = "🧬",
    ) -> None:
        self.total = total
        self.label = label
        self.width = width
        self.emoji = emoji
        self.start_time = time.monotonic()
        self._last_line_len = 0

    def update(self, current: int, suffix: str = "") -> None:
        """Redraw the progress bar at `current` out of `self.total`."""
        frac = current / self.total if self.total > 0 else 1.0
        filled = int(self.width * frac)
        remaining = self.width - filled

        # Build the bar with striation gradient at the leading edge
        if filled >= self.width:
            bar = "█" * self.width
        elif filled > 0:
            bar = "█" * (filled - 1) + "▓" + "░" * remaining
        else:
            bar = "░" * self.width

        elapsed = time.monotonic() - self.start_time
        if current > 0 and current < self.total:
            eta = (elapsed / current) * (self.total - current)
            time_str = f"ETA {eta:.0f}s"
        elif current >= self.total:
            time_str = f"{elapsed:.0f}s"
        else:
            time_str = "..."

        pct = frac * 100
        line = f"\r  {self.emoji} {self.label} ▐{bar}▌ {pct:5.1f}% ({current}/{self.total}) {time_str} {suffix}"

        # Clear any leftover characters from a longer previous line
        padding = max(0, self._last_line_len - len(line))
        sys.stderr.write(line + " " * padding)
        sys.stderr.flush()
        self._last_line_len = len(line)

    def finish(self, message: str = "") -> None:
        """Complete the progress bar and move to next line."""
        self.update(self.total)
        if message:
            sys.stderr.write(f"  {message}")
        sys.stderr.write("\n")
        sys.stderr.flush()


# ── Step Printing ───────────────────────────────────────────────────────

# Mapping step names to futuristic emojis
STEP_EMOJIS = {
    "download":  "🛰️",   # satellite — pulling data from orbit
    "vectors":   "🧬",   # DNA helix — decoder weight structure
    "reduce":    "🔮",   # crystal ball — seeing hidden structure
    "cluster":   "🌐",   # network — grouping features
    "dimension": "📐",   # ruler — measuring local geometry
    "vgt":       "📈",   # chart — growth curves
    "assemble":  "⚡",   # lightning — final assembly
    "circuits":  "🔗",   # link — connecting features
    "frontend":  "🖥️",   # monitor — launching the viz
    "done":      "✅",   # check — completion
    "cached":    "💾",   # floppy — already on disk
}


def step_header(step: str, label: str) -> None:
    """Print a styled step header."""
    emoji = STEP_EMOJIS.get(step, "▸")
    sys.stderr.write(f"\n  {emoji}  {label}\n")
    sys.stderr.write(f"  {'─' * (len(label) + 4)}\n")
    sys.stderr.flush()


def step_done(elapsed: float) -> None:
    """Print step completion with timing."""
    sys.stderr.write(f"     done in {elapsed:.0f}s\n")
    sys.stderr.flush()


def step_cached(filename: str) -> None:
    """Print a cache-hit message."""
    emoji = STEP_EMOJIS["cached"]
    sys.stderr.write(f"  {emoji}  Using cached {filename}\n")
    sys.stderr.flush()
