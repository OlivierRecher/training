# Build the internship report.
#   make        -> report.pdf (main.tex) via latexmk
#   make clean  -> remove LaTeX auxiliary files
#   make purge  -> also remove the PDF

MAIN := main

.PHONY: all clean purge
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

purge:
	latexmk -C $(MAIN).tex
