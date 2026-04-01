# RanTS: Random Token Sparsification
Code repository for the paper: Random Token Sparsification for ViT-based Hand Representation
<p align="center">
  <a href="https://github.com/phuongttn">Truong Thi Ngoc Phuong</a>,
  <a href="https://github.com/nttbdrk25">Thanh Tuan Nguyen</a>,
   <a href="https://github.com/nttbdrk25">Duy-Dinh Le</a>,
   <a href="https://github.com/nttbdrk25">Thanh Phuong Nguyen</a>,
</p>
<strong>Abstract:</strong> Transformer-based models have become the dominant paradigm for hand pose estimation (HPE) and hand mesh recovery (HMR) due to their strong capability in modeling global spatial relationships. However, they have been challenging in real deployments since the quadratic computational complexity of the transformer-based encoders leads to high training cost and scalability limitations. In consideration of the spatial distribution of hand pixels in real images, it can be realized that the hand region typically occupies a small fraction in the images. So, taking into account the whole spatial patterns of these images would be substantial redundancy for the token representation of the transformer encoders. To this end, Random Token Sparsification (RanTS) is proposed to eliminate a large proportion of the redundant tokens during training. Thereby, RanTS can sharply
reduce the computational cost of the token-based description while preserving the discriminative features for hand estimation.Experimental results have verified the efficacy of our simple strategy of token sparsification. For instance, with 25% rate of token elimination (i.e., one-fourth token reduction), RanTS for hand pose estimation obtained 84.2% AUC on HO3D-v2 [1], nearly the same 84.5% with full token representation, while the computational cost of RanTS is significantly decreased by about 11.83% training cost and 15% GPU RAM consumption. 

<strong> INSTALLATION </strong>

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
