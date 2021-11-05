
MNIST="--result-dir result/mnist --problem mnist   --batch-size 128 --epoch 5 "
SVHN=" --result-dir result/svhn  --problem SVHN    --batch-size 128 --epoch 10"
CIFAR="--result-dir result/cifar --problem cifar10 --batch-size 128 --epoch 20"


if [ "$1" == mnist ]; then
    python3 main_img_clf.py $MNIST --optimizer Adam  --lr 0.001 --seed 0
    python3 main_img_clf.py $MNIST --optimizer SNOpt --lr 0.02 --snopt-eps 0.05 --snopt-freq 100 --seed 0
    python3 main_img_clf.py $MNIST --optimizer SGD   --lr 0.02 --momentum 0.9 --seed 0
fi

if [ "$1" == svhn ]; then
    python3 main_img_clf.py $SVHN --optimizer Adam  --lr 0.0005 --l2-norm 0.001 --seed 0
    python3 main_img_clf.py $SVHN --optimizer SNOpt --lr 0.01   --l2-norm 0.0001 --snopt-eps 0.05 --snopt-freq 100 --seed 0
    python3 main_img_clf.py $SVHN --optimizer SGD   --lr 0.02   --momentum 0.9 --seed 0
fi

if [ "$1" == cifar10 ]; then
    python3 main_img_clf.py $CIFAR --optimizer Adam  --lr 0.001 --l2-norm 0.001 --seed 0
    python3 main_img_clf.py $CIFAR --optimizer SNOpt --lr 0.01  --snopt-eps 0.03 --snopt-freq 100 --seed 0
    python3 main_img_clf.py $CIFAR --optimizer SGD   --lr 0.01  --momentum 0.9 --seed 0
fi


ADAPTIVE="--result-dir result/cifar-t1 --problem cifar10 --batch-size 128 --epoch 20 --adaptive-t1 feedback --t1-update-freq 50 --t1-lr 0.1"
if [ "$1" == cifar10-t1-optimize ]; then
    python3 main_img_clf.py $ADAPTIVE --optimizer Adam --lr 0.001 --t1 1.0  --t1-reg 0.0006 --seed 0
    python3 main_img_clf.py $ADAPTIVE --optimizer Adam --lr 0.001 --t1 0.05 --t1-reg 0.0004 --seed 0
fi

