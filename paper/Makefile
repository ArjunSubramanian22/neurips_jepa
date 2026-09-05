.PHONY: tables claims pdf

PYTHON ?= python3

tables:
	$(PYTHON) sync_tables.py

claims: tables
	$(PYTHON) check_claims.py

pdf: claims
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
