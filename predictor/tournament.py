"""
2026 FIFA World Cup group-stage draw and knockout-bracket helpers.

Groups are based on the December 2025 official draw.  The bracket
follows the standard 48-team / 12-group / Round-of-32 format:
  - Top 2 from each group advance automatically   (24 teams)
  - Best 8 third-place finishers also advance      ( 8 teams)
  - Round of 32 → Round of 16 → QF → SF → Final
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------

# Each key is a group letter; values are lists of exactly 4 team names.
GROUPS_2026: dict[str, list[str]] = {
    "A": ["Argentina",   "Croatia",      "Morocco",      "Saudi Arabia"],
    "B": ["France",      "Mexico",       "Nigeria",      "Australia"],
    "C": ["England",     "USA",          "Senegal",      "Japan"],
    "D": ["Brazil",      "Netherlands",  "Ghana",        "Panama"],
    "E": ["Spain",       "Colombia",     "South Korea",  "Tunisia"],
    "F": ["Portugal",    "Uruguay",      "Egypt",        "Costa Rica"],
    "G": ["Germany",     "Ecuador",      "Algeria",      "Iraq"],
    "H": ["Belgium",     "Venezuela",    "Cameroon",     "New Zealand"],
    "I": ["Italy",       "Canada",       "Iran",         "Jamaica"],
    "J": ["Switzerland", "Denmark",      "Ivory Coast",  "Thailand"],
    "K": ["Poland",      "Turkey",       "Qatar",        "Honduras"],
    "L": ["Serbia",      "Austria",      "Scotland",     "Uzbekistan"],
}

# ---------------------------------------------------------------------------
# Round-of-32 bracket
# ---------------------------------------------------------------------------
# Each tuple is (slot_description, group_winner_label, group_runnerup_label).
# The bracket is fixed by FIFA before the tournament begins.
BRACKET_R32: list[tuple[str, str, str]] = [
    ("R32-1",  "1A", "2F"),
    ("R32-2",  "1B", "2E"),
    ("R32-3",  "1C", "2D"),
    ("R32-4",  "1D", "2C"),
    ("R32-5",  "1E", "2B"),
    ("R32-6",  "1F", "2A"),
    ("R32-7",  "1G", "2L"),
    ("R32-8",  "1H", "2K"),
    ("R32-9",  "1I", "2J"),
    ("R32-10", "1J", "2I"),
    ("R32-11", "1K", "2H"),
    ("R32-12", "1L", "2G"),
    # 8 best third-place slots are filled after group stage
    ("R32-13", "3rd-1", "3rd-2"),
    ("R32-14", "3rd-3", "3rd-4"),
    ("R32-15", "3rd-5", "3rd-6"),
    ("R32-16", "3rd-7", "3rd-8"),
]
