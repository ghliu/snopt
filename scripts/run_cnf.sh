GAS="      --result-dir result/gas       --problem gas       --nhidden 160 --batch-size 1000 --eval-itr 500 --seminorm --dataset-ratio 0.1"
MINIBOONE="--result-dir result/miniboone --problem miniboone --nhidden 860 --batch-size 128  --eval-itr 200"


if [ "$1" == miniboone ]; then
  python3 main_cnf.py $MINIBOONE --epoch 20 --optimizer Adam  --lr 0.003 --seed 0
  python3 main_cnf.py $MINIBOONE --epoch 25 --optimizer SGD   --lr 0.01  --seed 0 --momentum 0.9
  python3 main_cnf.py $MINIBOONE --epoch 10 --optimizer SNOpt --lr 0.005 --seed 0 --momentum 0.9 --snopt-freq 250 --snopt-eps 0.1
fi

if [ "$1" == gas ]; then
  python3 main_cnf.py $GAS --epoch 30 --optimizer Adam  --lr 0.002 --seed 0
  python3 main_cnf.py $GAS --epoch 30 --optimizer SGD   --lr 0.01  --seed 0 --momentum 0.9
  python3 main_cnf.py $GAS --epoch 20 --optimizer SNOpt --lr 0.01  --seed 0 --momentum 0.9 --snopt-freq 250 --snopt-eps 0.05
fi
