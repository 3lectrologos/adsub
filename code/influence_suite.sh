#!/usr/bin/env bash

#!/usr/bin/env bash

MODELS="ATC B_A CAIDA DBLP_CITE DIGG EGO_FB EMAIL E_R EUROROAD GNUTELLA GPLUS HAMSTERSTER PGP PHD POWER_GRID PROTEIN SNAP_GR STELZL W_S"

for M in $MODELS
do
python influence_plots.py -m $M -n 1000
done