#!/usr/bin/env python3
"""Check that every float sits where Olivier placed it, in the sources.

Rule (2026-09-29): every in-build figure and table is `[H]`, and its \\input is
its own paragraph, placed right after the first paragraph that cites it
(\\ref, \\panelref or \\panelnum). Several floats first cited by the same
paragraph follow it one after the other. The floats in BEFORE are the explicit
exceptions: they sit right before their first citing paragraph instead.

Prints nothing and exits 0 when everything is in place.
Usage: python3 check-placement.py   (from report/)
"""
import re
import sys

BEFORE = {"fig-overview", "fig-worked-example", "fig-scaling-ladder"}

problems = []
files = re.findall(r'\\input\{(sections/[^}]+)\}', open('main.tex').read())
for f in files:
    paras = [p.strip() for p in re.split(r'\n\s*\n', open(f + '.tex').read())]
    floats = {}
    for i, p in enumerate(paras):
        for m in re.finditer(r'\\input\{((?:figures|tables)/[^}]+)\}', p):
            src = open(m.group(1) + '.tex').read()
            if 'begin{figure' not in src and 'begin{table' not in src:
                continue
            name = m.group(1).split('/')[1]
            if p != m.group(0):
                problems.append(f"{f}: {name} is not a paragraph of its own")
            if re.search(r'\\begin\{(figure|table)\}(?!\[H\])', src):
                problems.append(f"{m.group(1)}.tex: not [H]")
            floats[i] = (name, re.search(r'\\label\{([^}]+)\}', src).group(1))
    for i, (name, lab) in floats.items():
        pat = r'(ref|panelref|panelnum)\{' + re.escape(lab) + r'\}'
        cites = [j for j, q in enumerate(paras) if j not in floats and re.search(pat, q)]
        if not cites:
            problems.append(f"{f}: {name} is never cited in its own section file")
            continue
        c = cites[0]
        if name in BEFORE:
            ok = all(k in floats for k in range(i + 1, c))
            where = "right before"
        else:
            ok = c < i and all(k in floats for k in range(c + 1, i))
            where = "right after"
        if not ok:
            problems.append(f"{f}: {name} should sit {where} the paragraph "
                            f"starting '{paras[c][:60]}...'")

print('\n'.join(problems))
sys.exit(1 if problems else 0)
