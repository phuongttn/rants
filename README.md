# RanTS: Random Token Sparsification
Code repository for the paper: Random Token Sparsification for ViT-based Hand Representation
<p align="center">
  <a href="https://github.com/phuongttn">Truong Thi Ngoc Phuong</a>,
  <a href="https://github.com/nttbdrk25">Thanh Tuan Nguyen</a>,
   <a href="https://github.com/nttbdrk25">Duy-Dinh Le</a>,
   <a href="https://github.com/nttbdrk25">Thanh Phuong Nguyen</a>,
</p>
INSTALLATION

First you need to clone the repo:

```bash
git clone --recursive https://github.com/phuongttn/rants.git
cd rants

We recommend creating a virtual environment for RanTS. You can use venv:
python3.10 -m venv .rants
source .rants/bin/activate

or alternatively conda:
conda create --name rants python=3.10
conda activate rants

Then, you can install the rest of the dependencies:
pip install -e .
