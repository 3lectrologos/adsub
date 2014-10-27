#!/usr/bin/env bash

MODEL=SNAP_GR
COST=1
K=10
WORKERS=4

python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.01 -k $K -g $COST -w $WORKERS
python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.05 -k $K -g $COST -w $WORKERS
python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.1 -k $K -g $COST -w $WORKERS
python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.15 -k $K -g $COST -w $WORKERS
python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.2 -k $K -g $COST -w $WORKERS
python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.3 -k $K -g $COST -w $WORKERS
python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.4 -k $K -g $COST -w $WORKERS
python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.5 -k $K -g $COST -w $WORKERS
python influence.py -m $MODEL -n 1000 -nn 10000 -na 1000 -ni 100 -p 0.6 -k $K -g $COST -w $WORKERS