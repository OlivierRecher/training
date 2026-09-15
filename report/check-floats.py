#!/usr/bin/env python3
"""Report floats that TeX pushed past a heading, out of the section that declares them.

Float drift is invisible in the .tex sources: \\input only says "place it here if
you can", and where it really lands is decided at typesetting time. But the build
records the answer --- .aux gives every float its number and page, .toc gives every
heading its number and page --- so the drift is detectable without reading the PDF
by eye. The page itself is only used to order a float and a heading that share it.

    python3 check-floats.py /tmp/build
"""
import os, re, subprocess, sys

BUILD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/build"
HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(BUILD, "main.pdf")

def norm(s):
    s = re.sub(r"\\[A-Za-z]+\s*|[{}$\\]|~", " ", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()

# --- where each float is declared, and under which heading ------------------
main = re.sub(r"(?m)%.*$", "", open(os.path.join(HERE, "main.tex")).read())
decl, seen = [], set()
for sec in re.findall(r"\\input\{(sections/[^}]+)\}", main):
    path = os.path.join(HERE, sec + ".tex")
    if not os.path.exists(path):
        continue
    head = None
    for line in open(path):
        if line.lstrip().startswith("%"):
            continue
        m = re.search(r"\\(?:sub)?section\*?\{(.+?)\}\s*$", line)
        if m:
            head = m.group(1)
        m = re.search(r"\\input\{((?:figures|tables)/[^}]+)\}", line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); decl.append((m.group(1), head))

# --- where each float and each heading landed -------------------------------
aux = open(os.path.join(BUILD, "main.aux")).read()
land = {l: (n, int(p)) for l, n, p in
        re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux)}

heads = []
for kind, num, title, pg in re.findall(
        r"\\contentsline \{(chapter|section|subsection)\}"
        r"\{\\numberline \{([\d.]+)\}(.*?)\}\{(\d+)\}",
        open(os.path.join(BUILD, "main.toc")).read()):
    heads.append((num, norm(title), int(pg)))
by_title = {h[1]: i for i, h in enumerate(heads)}

pages = {}
def page_lines(p):                       # printed page -> its lines, cached
    if p not in pages:
        off = int(os.environ.get("PAGE_OFFSET", "1"))
        out = subprocess.run(["pdftotext", "-f", str(p + off), "-l", str(p + off),
                              PDF, "-"], capture_output=True, text=True).stdout
        pages[p] = [l.strip() for l in out.splitlines()]
    return pages[p]

drift = []
for ffile, head in decl:
    fpath = os.path.join(HERE, ffile + ".tex")
    if not os.path.exists(fpath):
        continue
    m = re.search(r"\\label\{((?:fig|tab):[^}]+)\}", open(fpath).read())
    if not m or m.group(1) not in land:
        continue                          # not a float, or not in the build
    num, fpg = land[m.group(1)]
    i = by_title.get(norm(head))
    if i is None:
        continue
    kind = "Figure" if m.group(1).startswith("fig") else "Table"
    lines = page_lines(fpg)
    try:
        at = next(k for k, l in enumerate(lines) if l.startswith(f"{kind} {num}:"))
    except StopIteration:
        continue
    # any heading that comes after the declaring one, yet is printed above the float
    jumped = [h for h in heads[i + 1:] if h[2] == fpg
              and any(k < at and l == h[0] for k, l in enumerate(lines))]
    jumped += [h for h in heads[i + 1:] if h[2] < fpg]
    if jumped:
        drift.append((kind, num, m.group(1), fpg, heads[i], jumped))

for kind, num, lbl, fpg, h, jumped in drift:
    print(f"  {kind} {num} ({lbl}) landed p{fpg}, declared under {h[0]} {h[1]!r}")
    for j in jumped:
        print(f"      pushed past {j[0]} {j[1]!r}, which starts p{j[2]}")
print(f"\n{len(drift)} float(s) pushed past a heading." if drift
      else "\nNo float was pushed past a heading.")
