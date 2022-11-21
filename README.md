# PyTorch 배워보기

> 3년을 넘게 사용하던 `Tensorflow`를 뒤로하고, `PyTorch`를 배워보기로 하였습니다. TF도 턱걸이로 배워서, 실무에서 활용할 때마다 눈물이 앞을 가렸는데 새롭게 (다시 말해, 또!) PyTorch를 배워야 하다니... (털썩...) 그리하여 제가 공부하는 PyTorch 자료를 한 곳에 집중해보고자 합니다. 도움이 되었으면 하네요. 모두 즐거운(?) 공부(흑...흑...) 되세요.

## TODO

- [ ] Pytorch Lightning
- [ ] fastai
- [ ] PyTorch Geometric

## Ref.

- [ ] [Kunal Sawarkar, Deep Learning with PyTorch Lightning, Packt, 2022](https://www.packtpub.com/product/deep-learning-with-pytorch-lightning/9781800561618)
- [ ] [Daniel Voigt Godoy, Deep Learning with PyTorch Step-by-Step A Beginner’s Guide, 2022](https://leanpub.com/pytorch)
- [ ] [Sebastian Raschka et al., Machine Learning with PyTorch and Scikit-Learn, Packt, 2022](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312)


## Setup

* Python >= 3.10
* CUDA >= 11.6

```
// Virtualenv recommendation
$ python3 -m venv .venv
$ source .venv/bin/activate
$ python -m pip install --upgrade pip
$ python -m pip install --upgrade setuptools
$ pip install torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
```
