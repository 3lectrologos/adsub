#!/usr/bin/env bash

python influence.py -m gr -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.1 -k 10 -g 2 -w 4
python influence.py -m gr -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.1 -k 10 -g 3 -w 4
python influence.py -m gr -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.1 -k 10 -g 5 -w 4