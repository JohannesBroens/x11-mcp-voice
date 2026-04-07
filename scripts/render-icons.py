#!/usr/bin/env python3
"""Render Nox state icons from inline SVG to PNG."""
import os
import sys

try:
    import cairosvg
except ImportError:
    print("Install cairosvg: pip install cairosvg")
    sys.exit(1)

ICONS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "x11_mcp_voice", "icons")
SIZE = 48

SVGS = {
    "nox-somnus": '''<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
        <polygon points="60,10 105,35 105,85 60,110 15,85 15,35" fill="#E95420" opacity="0.4"/>
        <polygon points="60,22 95,42 95,78 60,98 25,78 25,42" fill="#F47B4A" opacity="0.1"/>
        <text x="52" y="36" font-family="monospace" font-size="9" font-weight="bold" fill="#1a1a2e" opacity="0.5">Z</text>
        <text x="60" y="30" font-family="monospace" font-size="7" font-weight="bold" fill="#1a1a2e" opacity="0.35">z</text>
        <text x="66" y="25" font-family="monospace" font-size="5" font-weight="bold" fill="#1a1a2e" opacity="0.2">z</text>
        <path d="M 36 56 Q 46 62 56 56" stroke="#1a1a2e" stroke-width="3" fill="none" stroke-linecap="round"/>
        <path d="M 64 56 Q 74 62 84 56" stroke="#1a1a2e" stroke-width="3" fill="none" stroke-linecap="round"/>
        <rect x="50" y="74" width="20" height="3" rx="1.5" fill="#1a1a2e" opacity="0.3"/>
    </svg>''',

    "nox-excito": '''<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
        <polygon points="60,10 105,35 105,85 60,110 15,85 15,35" fill="#F9A03F"/>
        <polygon points="60,22 95,42 95,78 60,98 25,78 25,42" fill="#FFD166" opacity="0.45"/>
        <text x="55" y="36" font-family="monospace" font-size="14" font-weight="bold" fill="#1a1a2e" opacity="0.6">!</text>
        <rect x="34" y="46" width="18" height="18" rx="2" fill="#1a1a2e"/>
        <rect x="68" y="46" width="18" height="18" rx="2" fill="#1a1a2e"/>
        <circle cx="43" cy="55" r="4" fill="#FFD166"/>
        <circle cx="77" cy="55" r="4" fill="#FFD166"/>
        <circle cx="60" cy="78" r="5" fill="#1a1a2e" opacity="0.7"/>
    </svg>''',

    "nox-ausculto": '''<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
        <polygon points="60,10 105,35 105,85 60,110 15,85 15,35" fill="#4FC1E8"/>
        <polygon points="60,22 95,42 95,78 60,98 25,78 25,42" fill="#6DD5F5" opacity="0.25"/>
        <g transform="translate(55, 22)">
            <rect x="0" y="0" width="6" height="9" rx="3" fill="#1a1a2e" opacity="0.5"/>
            <path d="M -2 7 Q 3 13 8 7" stroke="#1a1a2e" stroke-width="1.3" fill="none" opacity="0.5"/>
            <line x1="3" y1="12" x2="3" y2="15" stroke="#1a1a2e" stroke-width="1.3" opacity="0.5"/>
        </g>
        <g transform="rotate(10, 60, 60)">
            <rect x="34" y="46" width="16" height="14" rx="4" fill="#1a1a2e"/>
            <rect x="66" y="46" width="16" height="14" rx="4" fill="#1a1a2e"/>
            <circle cx="46" cy="52" r="3" fill="#4FC1E8"/>
            <circle cx="78" cy="52" r="3" fill="#4FC1E8"/>
            <path d="M 46 74 Q 56 82 68 74" stroke="#1a1a2e" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        </g>
        <path d="M 102 50 Q 112 60 102 70" stroke="#1a1a2e" stroke-width="2.5" fill="none" stroke-linecap="round" opacity="0.4"/>
        <path d="M 107 46 Q 118 60 107 74" stroke="#1a1a2e" stroke-width="1.5" fill="none" stroke-linecap="round" opacity="0.25"/>
    </svg>''',

    "nox-cogito": '''<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
        <polygon points="60,10 105,35 105,85 60,110 15,85 15,35" fill="#AC92EB"/>
        <polygon points="60,22 95,42 95,78 60,98 25,78 25,42" fill="#C4ADEB" opacity="0.25"/>
        <g transform="translate(52, 20)">
            <polygon points="0,0 14,0 7,7" fill="#1a1a2e" opacity="0.5"/>
            <polygon points="0,14 14,14 7,7" fill="#1a1a2e" opacity="0.5"/>
            <circle cx="7" cy="5" r="1" fill="#AC92EB" opacity="0.6"/>
            <circle cx="7" cy="10" r="1" fill="#AC92EB" opacity="0.6"/>
        </g>
        <rect x="34" y="46" width="18" height="14" rx="2" fill="#1a1a2e"/>
        <rect x="68" y="46" width="18" height="14" rx="2" fill="#1a1a2e"/>
        <circle cx="43" cy="49" r="3.5" fill="#AC92EB"/>
        <circle cx="77" cy="49" r="3.5" fill="#AC92EB"/>
        <line x1="67" y1="40" x2="87" y2="37" stroke="#1a1a2e" stroke-width="2.5" stroke-linecap="round" opacity="0.4"/>
        <path d="M 50 74 Q 55 72 60 74 Q 65 76 70 74" stroke="#1a1a2e" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    </svg>''',

    "nox-dico": '''<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
        <polygon points="60,10 105,35 105,85 60,110 15,85 15,35" fill="#8CC152"/>
        <polygon points="60,22 95,42 95,78 60,98 25,78 25,42" fill="#A8D86E" opacity="0.25"/>
        <g transform="translate(51, 22)" opacity="0.5">
            <rect x="0" y="3" width="5" height="7" fill="#1a1a2e"/>
            <polygon points="5,3 12,0 12,13 5,10" fill="#1a1a2e"/>
            <path d="M 14 2 Q 18 6.5 14 11" stroke="#1a1a2e" stroke-width="1.5" fill="none"/>
        </g>
        <rect x="34" y="46" width="16" height="16" rx="2" fill="#1a1a2e"/>
        <rect x="68" y="46" width="16" height="16" rx="2" fill="#1a1a2e"/>
        <rect x="36" y="48" width="4" height="4" fill="#8CC152" opacity="0.5"/>
        <rect x="70" y="48" width="4" height="4" fill="#8CC152" opacity="0.5"/>
        <ellipse cx="60" cy="76" rx="12" ry="8" fill="#1a1a2e" opacity="0.8"/>
        <ellipse cx="60" cy="78" rx="7" ry="4" fill="#6B8E3A" opacity="0.4"/>
        <path d="M 74 72 Q 80 76 74 80" stroke="#1a1a2e" stroke-width="1.5" fill="none" opacity="0.3"/>
        <path d="M 46 72 Q 40 76 46 80" stroke="#1a1a2e" stroke-width="1.5" fill="none" opacity="0.3"/>
    </svg>''',

    "nox-impero": '''<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
        <polygon points="60,10 105,35 105,85 60,110 15,85 15,35" fill="#E95420"/>
        <polygon points="60,22 95,42 95,78 60,98 25,78 25,42" fill="#FF7043" opacity="0.35"/>
        <g transform="translate(52, 18)" opacity="0.5">
            <polygon points="0,0 0,14 4,10.5 7,16 9,15 6,9 11,8" fill="#1a1a2e"/>
        </g>
        <rect x="32" y="50" width="20" height="10" rx="1" fill="#1a1a2e"/>
        <rect x="68" y="50" width="20" height="10" rx="1" fill="#1a1a2e"/>
        <circle cx="42" cy="55" r="3" fill="#FF5722"/>
        <circle cx="78" cy="55" r="3" fill="#FF5722"/>
        <circle cx="42" cy="55" r="1.2" fill="#FFD166"/>
        <circle cx="78" cy="55" r="1.2" fill="#FFD166"/>
        <rect x="30" y="44" width="24" height="3" rx="1" fill="#1a1a2e" opacity="0.7"/>
        <rect x="66" y="44" width="24" height="3" rx="1" fill="#1a1a2e" opacity="0.7"/>
        <rect x="44" y="74" width="32" height="3" rx="1" fill="#1a1a2e" opacity="0.6"/>
    </svg>''',

    "nox-erratum": '''<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
        <polygon points="60,10 105,35 105,85 60,110 15,85 15,35" fill="#ED5565"/>
        <polygon points="60,22 95,42 95,78 60,98 25,78 25,42" fill="#FF8A80" opacity="0.25"/>
        <text x="52" y="36" font-family="monospace" font-size="14" font-weight="bold" fill="#1a1a2e" opacity="0.5">&#x26A0;</text>
        <line x1="36" y1="46" x2="54" y2="64" stroke="#1a1a2e" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="54" y1="46" x2="36" y2="64" stroke="#1a1a2e" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="66" y1="46" x2="84" y2="64" stroke="#1a1a2e" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="84" y1="46" x2="66" y2="64" stroke="#1a1a2e" stroke-width="3.5" stroke-linecap="round"/>
        <rect x="44" y="76" width="32" height="3" rx="1" fill="#1a1a2e" opacity="0.5"/>
    </svg>''',
}


def main():
    os.makedirs(ICONS_DIR, exist_ok=True)
    for name, svg in SVGS.items():
        out_path = os.path.join(ICONS_DIR, f"{name}.png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=out_path,
                         output_width=SIZE, output_height=SIZE)
        print(f"  {out_path}")
    print(f"Rendered {len(SVGS)} icons to {ICONS_DIR}/")


if __name__ == "__main__":
    main()
