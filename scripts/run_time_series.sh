
CHART="--result-dir result/char-traj --epoch 400 --batch-size 512 --problem CharT --nhidden 32 --seminorm"
ARTWR="--result-dir result/art-wr    --epoch 600 --batch-size 256 --problem ArtWR --nhidden 64 --seminorm"
SPOAD="--result-dir result/spo-ad    --epoch 10  --batch-size 128 --problem SpoAD --nhidden 32 --seminorm"


if [ "$1" == char-traj ]; then
    python3 main_time_series.py $CHART --seed 0 --optimizer Adam  --lr 0.001 --lr2 0.001 --lr-gamma 0.5 --milestones 300 600 900
    python3 main_time_series.py $CHART --seed 0 --optimizer SGD   --lr 0.001 --lr2 0.001 --lr-gamma 0.7 --milestones 300 600 900
    python3 main_time_series.py $CHART --seed 0 --optimizer SNOpt --lr 0.003 --lr2 0.003 --lr-gamma 0.7 --milestones 450 750 900 \
      --snopt-freq 100 --snopt-eps 0.05 --snopt-step-size 1.0 --l2-norm 0.0001
fi


if [ "$1" == art-wr ]; then
    python3 main_time_series.py $ARTWR --seed 0 --optimizer Adam  --lr 0.001 --lr2 0.001 --lr-gamma 0.5 --milestones 200 375 550
    python3 main_time_series.py $ARTWR --seed 0 --optimizer SGD   --lr 0.003 --lr2 0.001
    python3 main_time_series.py $ARTWR --seed 0 --optimizer SNOpt --lr 0.003 --lr2 0.003 \
      --snopt-freq 100 --snopt-eps 0.1 --snopt-step-size 1.0
fi


if [ "$1" == spo-ad ]; then
    python3 main_time_series.py $SPOAD --seed 0 --optimizer Adam  --lr 0.003 --lr2 0.003
    python3 main_time_series.py $SPOAD --seed 0 --optimizer SGD   --lr 0.01  --lr2 0.003
    python3 main_time_series.py $SPOAD --seed 0 --optimizer SNOpt --lr 0.005 --lr2 0.003 \
     --snopt-freq 100 --snopt-eps 0.05 --snopt-step-size 1.0
fi
