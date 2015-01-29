#!/usr/bin/env bash

latexmk -pdf influence_grid.tex
latexmk -c influence_grid.tex

latexmk -pdf maxcut_grid.tex
latexmk -c maxcut_grid.tex