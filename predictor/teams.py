"""
2026 FIFA World Cup team ELO ratings.

ELO ratings are calibrated to reflect team strength heading into the
2026 tournament.  A difference of 400 ELO points corresponds to
roughly a 10-to-1 win-probability ratio in a decisive match.
"""

# fmt: off
TEAM_RATINGS: dict[str, float] = {
    # CONMEBOL – South America (6 spots)
    "Argentina":    2090,
    "Brazil":       1955,
    "Uruguay":      1780,
    "Colombia":     1770,
    "Ecuador":      1680,
    "Venezuela":    1630,

    # UEFA – Europe (16 spots)
    "France":       2005,
    "England":      1960,
    "Spain":        1945,
    "Portugal":     1920,
    "Germany":      1900,
    "Netherlands":  1870,
    "Belgium":      1850,
    "Italy":        1840,
    "Croatia":      1810,
    "Switzerland":  1790,
    "Denmark":      1750,
    "Poland":       1740,
    "Turkey":       1730,
    "Serbia":       1660,
    "Austria":      1650,
    "Scotland":     1640,

    # CAF – Africa (9 spots)
    "Senegal":      1720,
    "Morocco":      1710,
    "Algeria":      1580,
    "Tunisia":      1570,
    "Egypt":        1560,
    "Nigeria":      1550,
    "Ghana":        1540,
    "Ivory Coast":  1525,
    "Cameroon":     1530,

    # AFC – Asia (8 spots)
    "Japan":        1700,
    "South Korea":  1690,
    "Iran":         1600,
    "Australia":    1670,
    "Saudi Arabia": 1490,
    "Iraq":         1440,
    "Qatar":        1430,
    "Uzbekistan":   1410,

    # CONCACAF – North/Central America & Caribbean (6 spots)
    "USA":          1760,
    "Mexico":       1755,
    "Canada":       1615,
    "Costa Rica":   1520,
    "Honduras":     1510,
    "Panama":       1500,

    # OFC – Oceania (1 spot)
    "New Zealand":  1480,

    # Inter-continental play-off (2 spots)
    "Jamaica":      1460,
    "Thailand":     1380,
}
# fmt: on
