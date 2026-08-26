# -*- coding: utf-8 -*-
"""Twelve edits to 'Presentation Stage - Finale.pptx', from Olivier's review pass.

Reads the deck in place and writes it back. Run from the repository root.
"""
import copy
import sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

DECK = "Présentation Stage - Finale.pptx"

# ---------------------------------------------------------------- palette
TITLE   = "232959"
BODY    = "33363D"
SECOND  = "47515C"
RULE    = "387190"
BASE    = "4E97B0"
GAIN    = "3F8F52"
GAINBAR = "5FA83C"
CAVEAT  = "D4760A"
LOSS    = "C0392B"
MUTED   = "8A939E"
TRACK   = "E8EAEE"
NEUTRAL = "F4F6F8"
LINE    = "C7CFD6"
CONTEXT = "5B4B8A"          # NEW: the context, and only the context
CONTEXTG= "EDEAF4"

CAL   = "Calibri (MS)"
CALB  = "Calibri (MS) Bold"
CALI  = "Calibri (MS) Italics"
GEOB  = "Georgia Bold"

prs = Presentation(DECK)
S = lambda n: prs.slides[n - 1]


# ---------------------------------------------------------------- helpers
def walk(shapes):
    for sh in shapes:
        if sh.__class__.__name__ == "GroupShape":
            yield sh
            yield from walk(sh.shapes)
        else:
            yield sh


def sh(slide, sid):
    for s in walk(slide.shapes):
        if s.shape_id == sid:
            return s
    raise KeyError(f"shape {sid} not on slide")


def retext(shape, *texts, para=0):
    """Assign texts to the runs of one paragraph, cloning/dropping runs."""
    p = shape.text_frame.paragraphs[para]
    runs = p.runs
    if not runs:
        raise ValueError("paragraph has no run to copy formatting from")
    while len(p.runs) < len(texts):
        p._p.append(copy.deepcopy(runs[-1]._r))
    runs = p.runs
    for r, t in zip(runs, texts):
        r.text = t
    for r in runs[len(texts):]:
        r._r.getparent().remove(r._r)


def kill(slide, *sids):
    for sid in sids:
        s = sh(slide, sid)
        s._element.getparent().remove(s._element)


def move(slide, sid, *, dx=0.0, dy=0.0, top=None, left=None, height=None, width=None):
    s = sh(slide, sid)
    if left is not None:
        s.left = Inches(left)
    elif dx:
        s.left = Emu(s.left + Inches(dx))
    if top is not None:
        s.top = Inches(top)
    elif dy:
        s.top = Emu(s.top + Inches(dy))
    if height is not None:
        s.height = Inches(height)
    if width is not None:
        s.width = Inches(width)


def refill(slide, sid, rgb):
    s = sh(slide, sid)
    s.fill.solid()
    s.fill.fore_color.rgb = RGBColor.from_string(rgb)


def box(slide, x, y, w, h, runs, size=13.0, font=CAL, color=BODY, bold=None,
        italic=None, align=PP_ALIGN.LEFT, wrap=True, anchor=MSO_ANCHOR.TOP):
    """runs: str, or list of (text, {font/size/color/bold/italic}) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, over in runs:
        r = p.add_run()
        r.text = text
        f = r.font
        f.name = over.get("font", font)
        f.size = Pt(over.get("size", size))
        f.bold = over.get("bold", bold)
        f.italic = over.get("italic", italic)
        f.color.rgb = RGBColor.from_string(over.get("color", color))
    return tb


def rect(slide, x, y, w, h, fill=None, line=None, lw=0.75):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(h))
    if fill:
        r.fill.solid()
        r.fill.fore_color.rgb = RGBColor.from_string(fill)
    else:
        r.fill.background()
    if line:
        r.line.color.rgb = RGBColor.from_string(line)
        r.line.width = Pt(lw)
    else:
        r.line.fill.background()
    r.shadow.inherit = False
    r.text_frame.text = ""
    return r


def chip(slide, x, y, w, h, text, fill, color, size=11.0, font=CALB):
    """A filled rectangle carrying centred text."""
    r = rect(slide, x, y, w, h, fill=fill)
    tf = r.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string(color)
    return r


def stamp(n, corpus, rest):
    """The corpus / protocol / baseline stamp, right-aligned on the eyebrow line."""
    box(S(n), 8.26, 0.52, 10.5, 0.34,
        [(corpus, {"font": CALB, "bold": True, "color": RULE}),
         ("   " + rest, {"font": CAL, "color": SECOND})],
        size=12.0, align=PP_ALIGN.RIGHT, wrap=False)


# ================================================================ item 11
# Straight quotes everywhere.
QMAP = {"“": '"', "”": '"', "‘": "'", "’": "'"}
for slide in prs.slides:
    for s in walk(slide.shapes):
        if s.__class__.__name__ == "GroupShape":
            continue
        frames = []
        if s.has_text_frame:
            frames.append(s.text_frame)
        elif s.has_table:
            frames += [c.text_frame for r in s.table.rows for c in r.cells]
        for tf in frames:
            for p in tf.paragraphs:
                for r in p.runs:
                    t = r.text
                    for a, b in QMAP.items():
                        t = t.replace(a, b)
                    if t != r.text:
                        r.text = t


# ================================================================ item 4
# The corpus, the protocol and the baseline, on every slide that reports one.
stamp(8,  "RAW DAY", "last 20% of each day  ·  baseline 69.2%")
stamp(9,  "MOBILE-USER DAY", "last 20% of each day  ·  baseline 45.5% at the 80% cut")
stamp(10, "RAW DAY, 5,000 USERS", "1,173 test users  ·  baseline recomputed per user")
stamp(11, "MOBILE-USER DAY", "400 training users")
stamp(12, "MOBILE-USER DAY", "last 20% of each day  ·  n = 1,747")
stamp(13, "MOBILE-USER DAY", "every position scored  ·  baseline 43.1%")
stamp(14, "MOBILE-USER DAY", "400 training users")
stamp(15, "MOBILE-USER DAY", "100 held-out users")
stamp(16, "MOBILE-USER DAY", "400 train / 100 test")
stamp(17, "MOBILE-USER DAY", "every position scored  ·  baseline 43.1%")
stamp(19, "MOBILE-USER DAY", "every position scored  ·  baseline 43.1%")
stamp(20, "MOBILE-USER DAY", "every position scored  ·  baseline 43.1%")
stamp(21, "2,000-USER DAY, THEN ELEVEN-DAY SET", "each tier against its own baseline")
stamp(22, "2,000-USER DAY", "training users per tier")
stamp(23, "LEVELLED CORPUS", "last 20% of each day  ·  baseline 76.5%")

# ... and said once, in words, on the context slide.
retext(sh(S(2), 32), "The same rule scores 69.2% on a raw day, 43.1% on mobile users.")
move(S(2), 32, height=0.72)
box(S(2), 13.57, 9.42, 4.97, 0.42,
    "so every result slide carries its corpus and its baseline, top right",
    size=13.0, font=CALI, italic=True, color=SECOND)


# ================================================================ item 5
# No "in-repertoire": top-1 is top-1.
retext(sh(S(10), 8), "Model (top-1)")
retext(sh(S(23), 16), "−1.6 pt under the persistence baseline, on the same sequences")


# ================================================================ item 6
retext(sh(S(15), 4),
       "Group detection rate, by how much of the day is given as context (100 test users)")


# ================================================================ item 10
# Slide 6 presents the two models instead of reminding us of a past test.
retext(sh(S(6), 3), "Qwen2.5-0.5B over Mistral-7B: 14× smaller, 11× faster")
retext(sh(S(6), 41), "14× smaller → ≈11× faster (5.5 h vs 29 min)")
retext(sh(S(6), 49),
       "Both were fine-tuned on the same corpus, with LoRA adapters over a frozen "
       "4-bit base, on the same free cloud GPU. Only the size of the base model changed.")
retext(sh(S(6), 15), "ON THIS TASK", para=0)
retext(sh(S(6), 15), "≈5.5 h for one training run", para=1)
retext(sh(S(6), 15), "best validation 41.1%, no gain over the rule", para=2)
retext(sh(S(6), 29), "ON THIS TASK", para=0)
retext(sh(S(6), 29), "≈29 min for the same run", para=1)


# ================================================================ item 12
# One colour for the context, and only for the context.
refill(S(7), 4, CONTEXT)                      # the 80% context bar
for sid in (2, 60):                           # its label, and the example
    shape = sh(S(7), sid)
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            try:
                if str(r.font.color.rgb) == CAVEAT:
                    r.font.color.rgb = RGBColor.from_string(CONTEXT)
            except Exception:
                pass
for sid in (7, 13, 19, 25):                   # slide 15, the context fractions
    for r in sh(S(15), sid).text_frame.paragraphs[0].runs:
        r.font.color.rgb = RGBColor.from_string(CONTEXT)
        r.font.bold = True
        r.font.name = CALB
        r.font.italic = False
# Plain English, while we are here.
retext(sh(S(7), 62), "one prediction per cell_id of the target, until the day is finished")
retext(sh(S(12), 16), "FIXED", para=0)
retext(sh(S(12), 16), "every hour replaced by 12h", para=1)


# ================================================================ item 2
# Slide 17: the hour ablation does not belong in the presentation.
kill(S(17), 41, 43, 44, 46, 47, 48, 49, 51, 53, 54, 55, 57, 59, 60)
for sid in (15, 17, 18, 20, 21, 22, 23, 25, 27, 28, 29, 31, 33, 34, 35, 37, 39, 40):
    move(S(17), sid, dx=4.61)
retext(sh(S(17), 9), "1.8 points")
retext(sh(S(17), 78), "T",
       "he groups are solid on their own: the hours a profile transitions on are "
       "27.4 points harder than the rest of its day. The failure is in the channel, "
       "not in the signal.")


# ================================================================ items 3 and 9
# Slide 18: the group stops being a token and becomes which weights run.
retext(sh(S(18), 4),
       "Three changes, and one thing deliberately left alone. Two tighten the "
       "transition windows; the third changes how the group reaches the model at all, "
       "and that is the one that moved the score.")
retext(sh(S(18), 41),
       "The base model is frozen and four adapters are trained. Which adapter answers "
       "is decided by the user's profile.")
box(S(18), 1.55, 7.82, 7.68, 0.22,
    [("BEFORE", {"font": CALB, "bold": True, "color": MUTED}),
     ("   the group was one token at the head of the sequence", {"font": CAL, "color": SECOND})],
    size=9.99, wrap=False)
chip(S(18), 1.55, 8.06, 7.68, 0.28,
     "<G3>   one set of weights for all four profiles, and attention free to ignore it",
     TRACK, MUTED, size=10.0, font=CAL)
move(S(18), 47, top=8.40)
retext(sh(S(18), 47), "AFTER   THE PROFILE CHOOSES THE WEIGHTS")
for sid in (42, 44):
    move(S(18), sid, top=8.62, height=0.24)
for sid in (48, 50, 53, 55, 58, 60, 63, 65):
    move(S(18), sid, top=8.86, height=0.28)
refill(S(18), 64, TRACK)                       # G3's track was orange by accident
retext(sh(S(18), 70), "NEW CODE   ", "four adapters over one frozen base, the group stops being a token")
retext(sh(S(18), 77),
       "Windows stay keyed by behavioural group, the one piece already measured as "
       "correct. Inside them persistence scores 19.0% against 46.4% outside, so they "
       "mark hours far harder than the rest of the day.")
retext(sh(S(18), 78), "PER-GROUP WINDOWS · HOW MUCH HARDER THE HOURS THEY MARK ARE")
move(S(18), 78, width=7.68)
retext(sh(S(18), 79), "27.4 pt harder")
move(S(18), 79, width=7.68)
retext(sh(S(18), 89), "THE POINT   ",
       "Two of the three changes are bookkeeping: a window too wide to mean anything, "
       "and peak hours set by a handful of heavy users. The third is structural, the "
       "group stops being text the attention can dilute and becomes which weights run.")


# ================================================================ item 1
# Slide 19 says what it measured; slide 21 shows where the gap is bought.
retext(sh(S(19), 4),
       "Top-1 on the mobile-user day, 400 training and 100 held-out users, every "
       "position of each test sequence scored.")
retext(sh(S(19), 48),
       "Weighted by how many users each profile holds the gain is +3.3 pt, because 28% "
       "of the population sits in the profile that needed no model. What the same "
       "configuration is worth on larger populations is on slide 21.")
retext(sh(S(21), 4),
       "Points gained over each tier's own persistence baseline. A corpus built for it: "
       "eleven day-files of 2,000 users, the architecture of slide 19 unchanged. "
       "Constant compute, every tier reads 8,000 training examples.")
retext(sh(S(21), 73), "where the curve starts: the model is the rule")
retext(sh(S(21), 78), "the entire edge, bought in four steps")
retext(sh(S(21), 83), "ten times more users buy nothing more")
rect(S(21), 9.60, 4.98, 8.40, 1.52, fill="FFFFFF", line=LINE)
rect(S(21), 9.60, 4.98, 0.11, 1.52, fill=GAINBAR)
box(S(21), 10.05, 5.10, 6.0, 0.28,
    "THE JUMP", size=13.0, font=CALB, bold=True, color=GAIN, wrap=False)
box(S(21), 10.05, 5.38, 7.6, 0.60,
    "+0.6  →  +4.2 pt", size=28.0, font=GEOB, bold=True, color=GAIN, wrap=False)
box(S(21), 10.05, 6.02, 7.60, 0.40,
    "bought between 400 and 1,900 training users, four times the population. "
    "Everything after it is flat.", size=13.0, font=CAL, color=BODY)
retext(sh(S(21), 89), "THE POINT    ",
       "The gap goes from +0.6 to +4.2 pt between 400 and 1,900 training users. The ten "
       "times more that follows buys +0.1 pt, and a paired bootstrap over 1,000 held-out "
       "users puts zero inside every interval. The curve does not slow down, it stops.")

# Slide 22: after slide 18 the group is an adapter, not a token.
retext(sh(S(22), 4),
       "G2 is the least populated of the four behavioural groups: 14% of the users. Its "
       "head count at each tier decides whether that profile has enough people for its "
       "own adapter to learn anything.")
retext(sh(S(22), 36), "Below the threshold there is nothing for a profile to learn from")
retext(sh(S(22), 37),
       "A profile can only be worth its own adapter if it has enough inhabitants for its "
       "rhythm to be estimated. One mechanism fits the timing of the jump: the smallest "
       "profile passes a hundred users exactly between the two tiers where the score moves.")
retext(sh(S(22), 52), "THE POINT    ",
       "Cross-training only pays once every profile is populated. The four groups were "
       "not a better idea at 1,000 users than at 500, they were simply large enough to "
       "be worth reading.")


# ================================================================ items 7 and 8
# Slide 23: no Gini, and show what levelling actually does to a sequence.
retext(sh(S(23), 4),
       "Global top-1, 100 held-out test users, reference 80% context protocol. Same "
       "500-user split, same architecture, every sequence stretched to exactly 200 records.")
kill(S(23), 23, 24, 26, 27, 29, 31, 32, 33, 35, 37, 38, 39, 41, 43, 44, 45, 47, 49)

box(S(23), 1.25, 5.68, 17.51, 0.48,
    "How a user is stretched to 200 records, and what it hands the trivial rule",
    size=20.0, font=CALB, bold=True, color=TITLE)
rect(S(23), 1.25, 6.26, 0.98, 0.04, fill=RULE)

# left: the mechanism
box(S(23), 1.25, 6.42, 8.0, 0.24,
    "NATURAL LENGTHS", size=11.0, font=CALB, bold=True, color=SECOND, wrap=False)
NAT = ["BKVCHE1", "BKVCHE1", "UKVSUM101", "BKVPAN1", "BKVPAN1", "DSOTKB2"]
for i, c in enumerate(NAT):
    chip(S(23), 1.25 + i * 1.30, 6.70, 1.22, 0.32, c, TRACK, BODY, size=9.5, font=CAL)
box(S(23), 1.25, 7.08, 8.0, 0.24,
    "19 to 200 records per user, mean 86", size=11.0, font=CALI, italic=True, color=SECOND)
box(S(23), 1.25, 7.38, 8.2, 0.24,
    [("↓   ", {"font": CALB, "bold": True, "color": CAVEAT}),
     ("each record repeated a few times, spread evenly along the day",
      {"font": CAL, "color": BODY})], size=12.0, wrap=False)
box(S(23), 1.25, 7.70, 8.0, 0.24,
    "LEVELLED TO 200", size=11.0, font=CALB, bold=True, color=CAVEAT, wrap=False)
LEV = [("BKVCHE1", 1), ("BKVCHE1", 0), ("BKVCHE1", 0), ("UKVSUM101", 1),
       ("UKVSUM101", 0), ("BKVPAN1", 1)]
x = 1.25
for c, first in LEV:
    chip(S(23), x, 7.98, 1.22, 0.32, c,
         "FCEEDC" if first else TRACK, BODY if first else MUTED, size=9.5, font=CAL)
    x += 1.30
box(S(23), 1.25, 8.36, 8.6, 0.24,
    "exactly 200 records for every user, so every user weighs 0.25% of the training",
    size=11.0, font=CALI, italic=True, color=SECOND)

# right: the consequence
box(S(23), 10.40, 6.42, 8.36, 0.24,
    "WHAT THAT DOES TO THE SCORE", size=11.0, font=CALB, bold=True, color=SECOND, wrap=False)
box(S(23), 10.40, 6.66, 8.36, 0.62,
    "2,262 of 4,000", size=30.0, font=GEOB, bold=True, color=CAVEAT, wrap=False)
box(S(23), 10.40, 7.34, 8.36, 0.80,
    "scored positions now ask for a copy of the record before. Any rule that repeats is "
    "right on all of them by construction, which is why the model and the baseline both "
    "rose by about thirty points.", size=13.5, font=CAL, color=BODY)
box(S(23), 10.40, 8.20, 8.36, 0.44,
    "On the 1,738 positions that are a real decision the baseline scores 45.9% and the "
    "model 42.2 to 44.8%.", size=13.0, font=CALI, italic=True, color=SECOND)

retext(sh(S(23), 54),
       "The absolute score is not the comparison; the gap on the same sequences is, and "
       "it changed sign. Copying also destroys the signal the profiles are cut on: a "
       "copied step never changes site, so every user drifts towards the homebody profile.")

prs.save(DECK)
print("saved", DECK)
