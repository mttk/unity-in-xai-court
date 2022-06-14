#!/bin/bash

python run.py --model-name DBERT --gpu $1 --lr $4 --conicity 0 --tying 5 --l2 0 --save-dir $2 --data $3
python run.py --model-name DBERT --gpu $1 --lr $4 --conicity 0 --tying 10 --l2 0 --save-dir $2 --data $3