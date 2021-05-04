"""Copy the figures embedded in the paper PDF into figures/original/.

    pip install pymupdf
    python scripts/extract_paper_figures.py

PyMuPDF is only needed for this script, so it is not in requirements.txt.
"""

from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "An intelligent traffic congestion detection approach based on fuzzy inference system.pdf"
OUT = ROOT / "figures" / "original"

# Embedded images in reading order (page, index on page) -> file name.
NAMES = {
    (3, 0): "fig1_fis_architecture.png",
    (3, 1): "fig2_max_min_composition.png",
    (3, 2): "fig3_network_map.png",
    (4, 0): "fig4_model_schematic.png",
    (4, 1): "fig5_rule_editor.png",
    (5, 0): "fig6_rule_viewer.png",
    (5, 1): "fig7_surface_I.png",
    (6, 0): "fig7_surface_II.png",
    (6, 1): "fig7_surface_III.png",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    for page_no in range(len(doc)):
        for i, img in enumerate(doc[page_no].get_images(full=True)):
            name = NAMES.get((page_no + 1, i))
            if name is None:
                continue
            pix = fitz.Pixmap(doc, img[0])
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(OUT / name)
            print("wrote", (OUT / name).relative_to(ROOT), f"{pix.width}x{pix.height}")


if __name__ == "__main__":
    main()
