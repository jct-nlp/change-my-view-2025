"""Rebuild the combined report PDF from the six report_draft chapters.

Pipeline: concatenate chapters (with a title-page YAML header and page
breaks between chapters) -> pandoc (markdown -> typst) -> typst.compile
(typst -> pdf). Run from this directory with the project .venv active:

    source ../../../.venv/bin/activate
    python make_pdf.py
"""
import pathlib
import re
import pypandoc
import typst

HERE = pathlib.Path(__file__).parent
REPORT_DRAFT = HERE.parent
DEGREE_COMPLETION = REPORT_DRAFT.parent

CHAPTERS = [
    "00_ABSTRACT.md",
    "01_INTRODUCTION.md",
    "02_METHODS.md",
    "03_RESULTS.md",
    "04_DISCUSSION.md",
    "05_REFERENCES.md",
]

# Pandoc always emits equal percentage widths for pipe-table columns (markdown
# tables carry no width info). Override specific tables, identified by a
# unique string from their header row, with hand-picked widths instead.
TABLE_COLUMN_OVERRIDES = {
    # Table 1 (Introduction, §1.3): RQ label / Results§ / Discussion§ are
    # short labels; the research-question text needs the remaining space.
    r"table\.header\(\[\], \[Research question\], \[Results\], \[Discussion\],\)":
        "(8%, 72%, 10%, 10%)",
}

def apply_table_column_overrides(typ_text):
    for anchor, new_columns in TABLE_COLUMN_OVERRIDES.items():
        pattern = re.compile(
            r"columns: \([^)]*\)(,\s*\n\s*align:[^\n]*\n\s*" + anchor + r")"
        )
        typ_text, n = pattern.subn(lambda m: f"columns: {new_columns}" + m.group(1), typ_text)
        if n == 0:
            print(f"WARNING: table-column override anchor not found: {anchor!r}")
    return typ_text

def build():
    front = (HERE / "00_frontmatter.yaml").read_text(encoding="utf-8")
    parts = [front]
    for i, fname in enumerate(CHAPTERS):
        text = (REPORT_DRAFT / fname).read_text(encoding="utf-8")
        if i > 0:
            parts.append("\n```{=typst}\n#pagebreak()\n```\n\n")
        parts.append(text)
    combined_md = HERE / "combined.md"
    combined_md.write_text("\n".join(parts), encoding="utf-8")

    combined_typ = HERE / "combined.typ"
    pypandoc.convert_file(
        str(combined_md), "typst", outputfile=str(combined_typ),
        extra_args=["--standalone", "--toc"],
    )

    # Pandoc wraps each markdown image in a captioned #figure(), which Typst
    # auto-numbers ("Figure N:") independently of our own hand-written
    # "**Figure N.**" caption text -- producing double numbering that often
    # disagrees (our numbers follow the figures' established filename IDs,
    # not document order). Suppress Typst's own counter/label for image
    # figures so only our caption text shows.
    typ_text = combined_typ.read_text(encoding="utf-8")
    suppress_rule = "#show figure.where(kind: image): set figure(numbering: none)\n"
    typ_text = suppress_rule + typ_text
    typ_text = apply_table_column_overrides(typ_text)
    combined_typ.write_text(typ_text, encoding="utf-8")

    # figures/ is referenced as "figures/report/..." relative to degree_completion/;
    # a symlink here makes that path resolve from this build directory too.
    figures_link = HERE / "figures"
    if not figures_link.exists():
        figures_link.symlink_to(DEGREE_COMPLETION / "figures")

    out_pdf = HERE / "combined.pdf"
    result, warnings = typst.compile_with_warnings(str(combined_typ), output=str(out_pdf))
    for w in warnings:
        print("WARN:", w)

    final_pdf = REPORT_DRAFT / "Report_Dikla_Vilge.pdf"
    final_pdf.write_bytes(out_pdf.read_bytes())
    print("Wrote", final_pdf)

if __name__ == "__main__":
    build()
