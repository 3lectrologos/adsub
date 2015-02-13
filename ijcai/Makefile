all: ijcai15.tex ijcai15.bib
	latexmk -silent -pdf ijcai15.tex

fetch:
	cd figures; python fetch.py

clean:
	latexmk -C
