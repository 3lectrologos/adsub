TARGET = adsub

.PHONY: ${TARGET}.pdf all clean

all: ${TARGET}.pdf

${TARGET}.pdf: ${TARGET}.tex
	latexmk -pdf -xelatex -use-make $<

clean:
	latexmk -CA
