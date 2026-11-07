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
    "06_APPENDIX.md",
]

# Chapters whose page should flip to landscape (applied just before that
# chapter's own page-break, so the flip takes effect starting on its first
# page). Only chapters that are the last in the document need no "flip back".
LANDSCAPE_CHAPTERS = {"06_APPENDIX.md"}

# Pandoc always emits equal percentage widths for pipe-table columns (markdown
# tables carry no width info), and a shared borderless style (set globally in
# the preamble) with one hline under the header only. Override specific
# tables, identified by a unique string from their header row, with
# hand-picked column widths and/or a per-row stroke instead.
TABLE_OVERRIDES = {
    # Table 1 (Introduction, §1.3): RQ label / Results§ / Discussion§ are
    # short labels; the research-question text needs the remaining space.
    r"table\.header\(\[\], \[Research question\], \[Results\], \[Discussion\],\)": dict(
        columns="(8%, 72%, 10%, 10%)",
    ),
    # Table 2a-2c (Appendix A, landscape): short categorical columns (source
    # notebook / column count / trusted-untrusted) stay narrow; the two
    # free-text columns get the room the wider landscape page provides.
    # These three are also the only tables with a line between every row
    # (not just under the header), per explicit request for the appendix only.
    r"table\.header\(\[#strong\[Feature / feature group\]\]": dict(
        columns="(26%, 32%, 14%, 18%, 10%)",
        stroke='(y: 0.4pt + rgb("#c9c8c0"), x: none)',
    ),
    # Table 4 (Methods §2.5): "Matching procedure" is a long free-text
    # description; Sample/n/r are all short labels or numbers.
    r"table\.header\(\[Sample\], \[Matching procedure\]": dict(
        columns="(16%, 52%, 10%, 22%)",
    ),
    # Table 5 (Methods §2.6): "Strategy" is a one-/two-word label; the
    # definitions are full sentences and need nearly all the room.
    r"table\.header\(\[Strategy\], \[Definition\],\)": dict(
        columns="(20%, 80%)",
    ),
    # Tables 6 (split across two pages as Table 6 / Table 6 continued) and 7
    # (Methods §2.9) all share this exact header text: row-label column
    # narrower, the two dataset columns split the rest evenly.
    r"table\.header\(\[\], \[Home dataset\], \[Winning Arguments \(WA\)\],\)": dict(
        columns="(18%, 41%, 41%)",
    ),
    # Table 8 (Results §3.1.2): model names are long descriptive phrases;
    # the four numeric columns are all short values.
    r"table\.header\(\[Model\]": dict(
        columns="(40%, 15%, 15%, 15%, 15%)",
    ),
    # Table 11 (Results §3.2.2): "Feature set" labels are the longest
    # content in the table; widen it a bit at the other two columns' expense.
    r"table\.header\(\[Feature set\], \[PR-AUC \(mean ± SD\)\], \[ROC-AUC \(mean\)\],\)": dict(
        columns="(45%, 28%, 27%)",
    ),
    # Table 12 (Results §3.2.3): "Sample" labels are the longest content;
    # the other four columns are all short values, spread evenly.
    r"table\.header\(\[Sample\], \[n\], \[Lift without": dict(
        columns="(28%, 18%, 18%, 18%, 18%)",
    ),
    # Table 15 (Results §3.3.3): "Feature set" labels (including the long
    # "Robust core (...)" row) are the longest content; "Pairwise accuracy"
    # is a single short value/range per row.
    r"table\.header\(\[Feature set\], \[Pairwise accuracy\],\)": dict(
        columns="(70%, 30%)",
    ),
}

def apply_table_overrides(typ_text):
    for anchor, overrides in TABLE_OVERRIDES.items():
        replacement = f"columns: {overrides['columns']}"
        if "stroke" in overrides:
            replacement += f",\n    stroke: {overrides['stroke']}"
        pattern = re.compile(
            r"columns: \([^)]*\)(,\s*\n\s*align:[^\n]*\n\s*" + anchor + r")"
        )
        typ_text, n = pattern.subn(
            lambda m: replacement + m.group(1), typ_text, count=overrides.get("count", 0)
        )
        if n == 0:
            print(f"WARNING: table override anchor not found: {anchor!r}")
    return typ_text

def build():
    front = (HERE / "00_frontmatter.yaml").read_text(encoding="utf-8")
    parts = [front]
    for i, fname in enumerate(CHAPTERS):
        text = (REPORT_DRAFT / fname).read_text(encoding="utf-8")
        if i > 0:
            typst_directives = "#pagebreak()\n"
            if fname in LANDSCAPE_CHAPTERS:
                typst_directives += "#set page(flipped: true)\n"
            parts.append(f"\n```{{=typst}}\n{typst_directives}```\n\n")
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
    # "**Figure N.**" caption text -- producing double numbering. Our own
    # numbers follow document reading order (figure filenames carry a
    # different, unrelated numbering from when they were generated for the
    # presentation deck, not meant to be shown in the report). Suppress
    # Typst's own counter/label for image figures so only our caption text
    # shows.
    typ_text = combined_typ.read_text(encoding="utf-8")
    rules = (
        "#show figure.where(kind: image): set figure(numbering: none)\n"
        # Narrow table columns otherwise hyphenate mid-word (e.g. "Logistic
        # Regres-sion", "note-book(s)"); disabling hyphenation in table
        # cells makes any needed wrap fall between words instead.
        "#show table.cell: set text(hyphenate: false)\n"
    )
    typ_text = rules + typ_text
    typ_text = apply_table_overrides(typ_text)
    combined_typ.write_text(typ_text, encoding="utf-8")

    # figures/ is referenced as "figures/report/..." relative to degree_completion/;
    # a symlink here makes that path resolve from this build directory too.
    figures_link = HERE / "figures"
    if not figures_link.exists():
        figures_link.symlink_to(DEGREE_COMPLETION / "figures")

    # pdf_tags=False: Typst's default tagged-PDF output (a full accessibility
    # structure tree) has been observed to render as blank pages in Adobe
    # Reader, even though the content is intact (other viewers render it
    # fine). A plain, untagged PDF avoids that failure mode.
    out_pdf = HERE / "combined.pdf"
    result, warnings = typst.compile_with_warnings(
        str(combined_typ), output=str(out_pdf), pdf_tags=False
    )
    for w in warnings:
        print("WARN:", w)

    final_pdf = REPORT_DRAFT / "Dikla-Vilge-Predicting-Persuasive-Comments-on-rChangeMyView-Report.pdf"
    final_pdf.write_bytes(out_pdf.read_bytes())
    print("Wrote", final_pdf)

if __name__ == "__main__":
    build()
