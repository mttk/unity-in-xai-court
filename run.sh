#!/bin/bash

python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0 --tying 0 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0 --tying 0.1 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0 --tying 0.3 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0 --tying 0.5 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0 --tying 1 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0 --tying 0 --l2 1e-4 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0 --tying 0 --l2 1e-3 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0 --tying 0 --l2 1e-2 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0.1 --tying 0 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0.3 --tying 0 --l2 0 --save-dir $2 --data $3
python run.py --model-name JWA --gpu $1 --lr 2e-3 --conicity 0.5 --tying 0 --l2 0 --save-dir $2 --data $3