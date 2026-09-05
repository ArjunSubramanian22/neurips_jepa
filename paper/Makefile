.PHONY: tables claims pdf

PYTHON ?= python3

tables:
	$(PYTHON) sync_tables.py

claims: tables
	$(PYTHON) check_claims.py

# CFP requires NeurIPS 2026 style. Prefer official pdflatex+bibtex; this
# machine currently has tectonic only, which still consumes neurips_2026.sty.
pdf: claims
	@if command -v pdflatex >/dev/null 2>&1; then \
		pdflatex -interaction=nonstopmode main.tex && \
		bibtex main && \
		pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic -X compile --keep-logs --keep-intermediates main.tex; \
	else \
		echo "Need pdflatex or tectonic on PATH" >&2; exit 1; \
	fi
