"""Crops of the CHE 341 Exam 2 equation sheet for the practice page.

Each crop is one section of the sheet with the key equation(s) boxed in red.
Coordinates are PDF points on the Exam 2 page (letter, 612 x 792).

Regenerate the PNGs after the sheet changes:
    python -m pchemapps.exam2_sheet "path/to/CHE341 Exam 1 and 2 Equations 26FA.pdf"
"""
import sys
from pathlib import Path

ASSET_DIR = Path(__file__).parent / "assets" / "exam2_sheet"
# 0-based pages of the combined PDF: (page, x0, x1, caption)
EXAM2 = (2, 30, 598, "Exam 2 equation sheet")
EXAM1 = (0, 62, 578, "Exam 1 equation sheet")

# Section bands (y0, y1)
CONSTANTS = (100, 156)
GASES = (180, 210)
FIRST_LAW = (225, 312)
REACTIONS = (346, 414)
SECOND_LAW = (438, 524)
INTEGRALS = (587, 627)

# key: (band, [boxes (x0, y0, x1, y1)]) on the Exam 2 sheet, or (band, boxes, EXAM1)
CROPS = {
    "R_lbar": (CONSTANTS, [(279, 120, 413, 138)]),
    "L_bar": (CONSTANTS, [(56, 123, 128, 138)]),
    "ideal_gas": (GASES, [(53, 188, 133, 203)]),
    "Z": (GASES, [(518, 182, 566, 209)]),
    "U_ig": (GASES, [(147, 182, 505, 209)]),
    "first_law": (FIRST_LAW, [(56, 239, 117, 255)]),
    "w_mech": (FIRST_LAW, [(292, 237, 371, 256)]),
    "first_law_w_mech": (FIRST_LAW, [(56, 239, 117, 255), (292, 237, 371, 256)]),
    "dS_surr": (SECOND_LAW, [(223, 494, 408, 521)]),
    "w_mech_elec": (FIRST_LAW, [(292, 237, 371, 256), (295, 267, 398, 285)]),
    "C_V": (FIRST_LAW, [(414, 224, 566, 262)]),
    "dH_const_P": (FIRST_LAW, [(55, 268, 191, 285)]),
    "C_molar": (FIRST_LAW, [(57, 294, 112, 310), (206, 294, 259, 310)]),
    "hess": (REACTIONS, [(293, 345, 547, 372)]),
    "S_pressure": (REACTIONS, [(305, 373, 428, 392)]),
    "S_rxn_Q": (REACTIONS, [(300, 391, 420, 411)]),
    "dS_def": (SECOND_LAW, [(35, 440, 119, 469)]),
    "dS_volume": (SECOND_LAW, [(33, 467, 220, 494)]),
    "dS_pressure": (SECOND_LAW, [(33, 493, 204, 521)]),
    "dS_surr_univ": (SECOND_LAW, [(223, 494, 408, 521), (415, 444, 481, 461)]),
    "dS_univ": (SECOND_LAW, [(415, 444, 481, 461)]),
    "dS_def_surr": (SECOND_LAW, [(35, 440, 119, 469), (223, 494, 408, 521)]),
    "boltzmann_S": (SECOND_LAW, [(245, 444, 304, 461)]),
    "int_dx_x": (INTEGRALS, [(72, 587, 156, 626)]),
    # Exam 1 sheet
    "q_trans_rot": ((617, 713), [(68, 677, 186, 712), (242, 685, 300, 713), (457, 685, 512, 713)], EXAM1),
    "rigid_rotor_B": ((458, 517), [(433, 489, 516, 516), (498, 460, 563, 487)], EXAM1),
    "particle_box": ((517, 548), [(195, 518, 290, 546)], EXAM1),
}


def path(key: str) -> Path:
    return ASSET_DIR / f"{key}.png"


def caption(key: str) -> str:
    return (CROPS[key][2] if len(CROPS[key]) > 2 else EXAM2)[3]


def build(pdf: str, dpi: int = 200) -> None:
    import pymupdf
    from PIL import Image, ImageDraw

    doc = pymupdf.open(pdf)
    pages = {}
    for n in (EXAM1[0], EXAM2[0]):
        pix = doc[n].get_pixmap(dpi=dpi)
        pages[n] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    s = dpi / 72
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    pages[EXAM2[0]].save(ASSET_DIR / "page.png")
    for key, ((y0, y1), boxes, *sheet) in CROPS.items():
        n, X0, X1, _ = sheet[0] if sheet else EXAM2
        im = pages[n].copy()
        draw = ImageDraw.Draw(im)
        for bx0, by0, bx1, by1 in boxes:
            draw.rectangle([bx0 * s, by0 * s, bx1 * s, by1 * s], outline=(220, 30, 30), width=4)
        im.crop((int(X0 * s), int(y0 * s), int(X1 * s), int(y1 * s))).save(path(key))


if __name__ == "__main__":
    build(sys.argv[1])
