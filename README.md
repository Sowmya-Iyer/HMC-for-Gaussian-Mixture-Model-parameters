[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![Pytorch 1.3](https://img.shields.io/badge/pytorch-1.3.1-blue.svg)](https://pytorch.org/)
# HMC-for-Gaussian-Mixture-Model-parameters
GMM descibes a set of points in the 2D plane. Implemented a posterior sampler using the Hamiltion Monte Carlo (HMC) algorithm.


### Relevant codes:
    `models\hmc.py`
    `main_hmc.py` 
    `models\gmm.py` 
    
  In general, given a model (say, our GMM) with parameters  $W$ and a training dataset $D$.
A Bayesian sampler of this model obtains $m$ samples $W_{t} \sim P(W|D)$, where $t \in \{0,\ldots,m-1\}$ is the sample index. <br />
To achieve this via HMC, we need two measurements, the potential energy $U(W)$
and the kinetic energy $K(\Phi)$, where $\Phi \sim \mathcal{N}(0, R)$ is the
auxiliary momentum in HMC algorithm randomly sampled from zero-mean Gaussian
distribution with covariance matrix $R$. <br />

Given an arbitrary dataset $\mathcal{D}$, we have $$U(W) = -\log P(\mathcal{D}|W) + Z_{U},$$ and $$K(\Phi) = 0.5 \cdot \Phi^\mathsf{T} R^{-1} \Phi + Z_{K},$$ where $-\log P(\mathcal{D}|W)$ is negative
log-likelihood (mean) of model parameter on dataset $\mathcal{D}$ and $Z_{U},
Z_{K}$ are arbitrary constants. <br />
Thus, we can regard the total energy as $$H(W, \Phi) = -\log P(\mathcal{D}|W) + 0.5 \cdot \Phi^{T} R^{-1} \Phi.$$

![plot](https://github.com/Sowmya-Iyer/HMC-for-Gaussian-Mixture-Model-parameters/blob/main/figures/HMC_alg.png)

 You can run the code using default arguments:
 
 `python main hmc.py`
 
When the samples are generated from the posterior, a large number of samples are rejected. The
efficiency of HMC is that it has a higher acceptance rate and hence more efficient.
HMC propose samplers fro Metropolitan-Hastings based acceptance scheme with high acceptance probability.

It can be observed that the samples are accepted with probability very close 100% or 100% to one under warm-up.
 
 **Table for HMC Sampling**: Pairs of odd IDs come from same GMM clusters, while pairs of
even IDs come from different GMM clusters. There is a warm-up phase where we perform some steps of
maximum likelihood.

![results](https://github.com/Sowmya-Iyer/HMC-for-Gaussian-Mixture-Model-parameters/blob/main/figures/HMC%20result.png)



    
