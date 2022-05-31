#!/bin/bash

python run.py --model-name JWA --gpu $1 --lr $4 --conicity 1 --tying 0 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr $4 --conicity 5 --tying 0 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr $4 --conicity 10 --tying 0 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr $4 --conicity 0 --tying 5 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr $4 --conicity 0 --tying 10 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr $4 --conicity 0 --tying 20 --l2 0 --save-dir $2 --data $3