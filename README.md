
## train
version 2: we use train set datas to fit vine copula model.
Also, we use supervised label to try to improve classification accuracy.

    CUDA_VISIBLE_DEVICES=0 python train_base.py --device cuda:0 >cifar10_nt_v2.txt
    CUDA_VISIBLE_DEVICES=1 python train_AAD.py --device cuda:0 >tiny_AADnt.txt


    CUDA_VISIBLE_DEVICES=1 python train_trades.py --device cuda:0 > nt_cifar10_trades.txt
    CUDA_VISIBLE_DEVICES=1 python train_mart.py --device cuda:0 > nt_cifar10_mart.txt
    CUDA_VISIBLE_DEVICES=1 python train_AAD.py --device cuda:0 > nt_cifar10_beta1.txt

if rpy3.8 environment:"OSError: cannot load library '/data/diska/liulin/anaconda3/envs/rpy3.8/lib/R/lib/libR.so': /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /data/diska/liulin/anaconda3/envs/rpy3.8/lib/R/lib/../../libicuuc.so.73)"

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    watch --color -n1 gpustat -cpu

## test

    CUDA_VISIBLE_DEVICES=0 python test_cafi.py --device cuda:0 >test_cifar10_nt_beta0.txt

## print vinemodel


## the pre-trained copula model in the training set /testing set

## attack methods

### FWA
[Github](https://github.com/watml/fast-wasserstein-adversarial)

Dependency: PyTorch 1.5.1 with CUDA 10.2, scipy 1.5.0, and advertorc 0.2.3
before runing the procedure, it is required to install the sparse tensor package:
    cd sparse_tensor
    python setup.py install
