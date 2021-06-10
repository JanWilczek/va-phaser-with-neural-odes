MAKEFLAGS += -j1
LATEXFLAGS += --synctex=-1 -enable-write18
BATCHFLAG += -interaction=batchmode
TEXPATH += .:./lib//:

.PHONY: document beispiel desktop preview clean cleanall

document: thesis.pdf

desktop: thesis.pdf
	cp $< ~/Desktop/thesis-`date +'%y-%m-%d'`.pdf

preview: thesis.pdf
	xdg-open $< &

clean:
	- rm -f *.dvi *.ps *.log *.toc *.snm *.nav *.aux *.out *.bbl *.blg *.cb *.diff.tex *.lof *.lot *.lol *.cb2 *.fls *.fdb* *.synctex *.auxlock
	
cleanall: clean
	- rm -f *.pdf

debug: BATCHFLAG =
debug: cleanall thesis.pdf

test: debug
	bash test.sh thesis.log

thesis.pdf:		config_packages_macros.tex config.tex \
				titlepage.tex abstract.tex acknowledgements.tex appendix.tex preface.tex statement.tex \
				chapter_01.tex chapter_02.tex chapter_03.tex chapter_appendix.tex chapter_conclusions.tex \

%.pdf: %.tex
	TEXINPUTS=$(TEXPATH) pdflatex $(BATCHFLAG) $(LATEXFLAGS) $<
	bibtex $(<:.tex=) || true
	TEXINPUTS=$(TEXPATH) pdflatex $(BATCHFLAG) $(LATEXFLAGS) $<
	TEXINPUTS=$(TEXPATH) pdflatex $(BATCHFLAG) $(LATEXFLAGS) $<
