all: ijcai15.tex ijcai15.bib
	cd figures; python fetch.py
	latexmk -silent -pdf ijcai15.tex

clean:
	latexmk -C
