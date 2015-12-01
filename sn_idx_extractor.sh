#!/bin/bash
for x in {a..z}
do
    wget www.sparknotes.com/lit/index_${x}.html
done
mkdir sn_indices
mv index_* sn_indices
