# striatica/pipeline/banner.py
"""ASCII banner and terminal UI helpers for the striat CLI.

Rainbow gradient output inspired by Turborepo's build logs.
Uses ANSI 256-color escapes; degrades to plain text when color
is unsupported (NO_COLOR, TERM=dumb, or piped output).
"""

from __future__ import annotations

import os
import sys
import time


# ── Color System ─────────────────────────────────────────────────────

# Rainbow palette — 12 steps through the spectrum (ANSI 256-color codes)
_RAINBOW = [
    196,  # red
    202,  # orange
    208,  # dark orange
    214,  # gold
    220,  # yellow
    118,  # lime
    48,   # green
    43,   # teal
    39,   # cyan
    33,   # blue
    129,  # purple
    171,  # magenta
]

# Accent colors for different message types
_COLORS = {
    "dim":     "\033[38;5;242m",   # dark gray
    "white":   "\033[38;5;255m",   # bright white
    "green":   "\033[38;5;48m",    # success green
    "red":     "\033[38;5;196m",   # error red
    "yellow":  "\033[38;5;220m",   # warning yellow
    "cyan":    "\033[38;5;39m",    # info cyan
    "magenta": "\033[38;5;171m",   # highlight magenta
    "bold":    "\033[1m",
    "reset":   "\033[0m",
}


def _color_enabled() -> bool:
    """Check if the terminal supports color output."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not hasattr(sys.stderr, "isatty"):
        return False
    return sys.stderr.isatty()


_USE_COLOR = _color_enabled()


def _c(color_key: str) -> str:
    """Return an ANSI escape for the given color key, or empty string if no color."""
    if not _USE_COLOR:
        return ""
    return _COLORS.get(color_key, "")


def _reset() -> str:
    return _COLORS["reset"] if _USE_COLOR else ""


def _rainbow_text(text: str) -> str:
    """Apply a rainbow gradient across each character of text."""
    if not _USE_COLOR:
        return text
    result = []
    chars = [ch for ch in text]
    n = len(chars)
    for i, ch in enumerate(chars):
        if ch == " " or ch == "\n":
            result.append(ch)
            continue
        color_idx = int(i / max(n - 1, 1) * (len(_RAINBOW) - 1))
        result.append(f"\033[38;5;{_RAINBOW[color_idx]}m{ch}")
    result.append(_COLORS["reset"])
    return "".join(result)


def _gradient_line(text: str, start_idx: int = 0) -> str:
    """Apply rainbow gradient starting from a given palette index."""
    if not _USE_COLOR:
        return text
    result = []
    visible = 0
    for ch in text:
        if ch in (" ", "\n"):
            result.append(ch)
        else:
            color_idx = (start_idx + visible) % len(_RAINBOW)
            result.append(f"\033[38;5;{_RAINBOW[color_idx]}m{ch}")
            visible += 1
    result.append(_COLORS["reset"])
    return "".join(result)


def _step_color(step_num: int) -> str:
    """Get a rainbow color for a pipeline step number."""
    if not _USE_COLOR:
        return ""
    idx = step_num % len(_RAINBOW)
    return f"\033[38;5;{_RAINBOW[idx]}m"


# ── Banner ──────────────────────────────────────────────────────────

TAGLINE = "  geometric atlas for machine intelligence"
VERSION_LINE = "  v0.1.0"

_BANNER_ART = "  ░▒▓  s t r i a t i c a  ≡≡≡≡≡  ▓▒░"


def print_banner() -> None:
    """Print the striatica ASCII banner with rainbow gradient."""
    out = sys.stderr

    if _USE_COLOR:
        out.write("\n")
        out.write(_rainbow_text(_BANNER_ART))
        out.write("\n")
        out.write(f"  {_c('cyan')}🔬{_reset()} {_c('dim')}{TAGLINE}{_reset()}\n")
        out.write(f"  {_c('yellow')}⚡{_reset()} {_c('dim')}{VERSION_LINE}{_reset()}\n\n")
    else:
        out.write(f"\n{_BANNER_ART}\n")
        out.write(f"  🔬{TAGLINE}\n")
        out.write(f"  ⚡{VERSION_LINE}\n\n")
    out.flush()


# ── Progress Bar ────────────────────────────────────────────────────

class ProgressBar:
    """Minimal, dependency-free terminal progress bar with gradient fill.

    Usage:
        bar = ProgressBar(total=24, label="Downloading features")
        for i in range(24):
            do_work()
            bar.update(i + 1)
        bar.finish()
    """

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

    def _gradient_bar(self, filled: int) -> str:
        """Build a progress bar with rainbow gradient fill."""
        if not _USE_COLOR:
            if filled >= self.width:
                return "█" * self.width
            elif filled > 0:
                return "█" * (filled - 1) + "▓" + "░" * (self.width - filled)
            else:
                return "░" * self.width

        parts = []
        for i in range(self.width):
            if i < filled:
                color_idx = int(i / max(self.width - 1, 1) * (len(_RAINBOW) - 1))
                parts.append(f"\033[38;5;{_RAINBOW[color_idx]}m█")
            else:
                parts.append(f"{_c('dim')}░")
        parts.append(_reset())
        return "".join(parts)

    def update(self, current: int, suffix: str = "") -> None:
        """Redraw the progress bar at `current` out of `self.total`."""
        frac = current / self.total if self.total > 0 else 1.0
        filled = int(self.width * frac)

        bar = self._gradient_bar(filled)

        elapsed = time.monotonic() - self.start_time
        if current > 0 and current < self.total:
            eta = (elapsed / current) * (self.total - current)
            time_str = f"ETA {eta:.0f}s"
        elif current >= self.total:
            time_str = f"{elapsed:.0f}s"
        else:
            time_str = "..."

        pct = frac * 100
        pct_color = _c("green") if pct >= 100 else _c("white")

        line = (
            f"\r  {self.emoji} {_c('white')}{self.label}{_reset()} "
            f"▐{bar}▌ "
            f"{pct_color}{pct:5.1f}%{_reset()} "
            f"{_c('dim')}({current}/{self.total}) {time_str}{_reset()} "
            f"{suffix}"
        )

        # Clear any leftover characters from a longer previous line
        # Count only visible chars for padding (strip ANSI)
        import re
        visible_len = len(re.sub(r'\033\[[^m]*m', '', line))
        padding = max(0, self._last_line_len - visible_len)
        sys.stderr.write(line + " " * padding)
        sys.stderr.flush()
        self._last_line_len = visible_len

    def finish(self, message: str = "") -> None:
        """Complete the progress bar and move to next line."""
        self.update(self.total)
        if message:
            sys.stderr.write(f"  {message}")
        sys.stderr.write("\n")
        sys.stderr.flush()


# ── Step Printing ───────────────────────────────────────────────────

STEP_EMOJIS = {
    "download":  "🛰️",
    "vectors":   "🧬",
    "reduce":    "🔮",
    "cluster":   "🌐",
    "dimension": "📐",
    "vgt":       "📈",
    "assemble":  "⚡",
    "circuits":  "🔗",
    "frontend":  "🖥️",
    "done":      "✅",
    "cached":    "💾",
}

# Track step count for rainbow cycling
_step_counter = 0


def step_header(step: str, label: str) -> None:
    """Print a styled step header with rainbow-cycled accent color."""
    global _step_counter
    emoji = STEP_EMOJIS.get(step, "▸")
    color = _step_color(_step_counter)
    _step_counter += 1

    line_char = "─"
    sys.stderr.write(f"\n  {emoji}  {color}{_c('bold')}{label}{_reset()}\n")
    sys.stderr.write(f"  {_c('dim')}{line_char * (len(label) + 4)}{_reset()}\n")
    sys.stderr.flush()


def step_done(elapsed: float) -> None:
    """Print step completion with timing."""
    sys.stderr.write(f"     {_c('green')}done{_reset()} {_c('dim')}in {elapsed:.0f}s{_reset()}\n")
    sys.stderr.flush()


def step_cached(filename: str) -> None:
    """Print a cache-hit message."""
    emoji = STEP_EMOJIS["cached"]
    sys.stderr.write(f"  {emoji}  {_c('dim')}Using cached{_reset()} {_c('cyan')}{filename}{_reset()}\n")
    sys.stderr.flush()


# ── Info / Status Helpers ───────────────────────────────────────────

def info(label: str, value: str, emoji: str = "▸") -> None:
    """Print a key-value info line with colored value."""
    sys.stderr.write(f"  {emoji} {_c('dim')}{label:<14}{_reset()} {_c('white')}{value}{_reset()}\n")
    sys.stderr.flush()


def success(message: str) -> None:
    """Print a success message."""
    sys.stderr.write(f"\n  {_c('green')}✅  {message}{_reset()}\n")
    sys.stderr.flush()


def error(message: str) -> None:
    """Print an error message."""
    sys.stderr.write(f"  {_c('red')}❌  {message}{_reset()}\n")
    sys.stderr.flush()


def warn(message: str) -> None:
    """Print a warning message."""
    sys.stderr.write(f"  {_c('yellow')}⚠️   {message}{_reset()}\n")
    sys.stderr.flush()


def detail(message: str) -> None:
    """Print a dim detail line (indented)."""
    sys.stderr.write(f"     {_c('dim')}{message}{_reset()}\n")
    sys.stderr.flush()


def separator() -> None:
    """Print a rainbow separator line."""
    line = "━" * 48
    sys.stderr.write(f"  {_gradient_line(line)}\n")
    sys.stderr.flush()


def reset_step_counter() -> None:
    """Reset the step counter (useful between runs in batch mode)."""
    global _step_counter
    _step_counter = 0
