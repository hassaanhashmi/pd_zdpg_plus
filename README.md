# Critic-Free Deterministic Policy Gradients for One-Step MDPs
Actor-only policy learning with zeroth-order gradient representations of the Critic outperforms (heuristics of) selecting feasible Critic models. We establish this in this work on model-free learning of optimal deterministic resource allocations in wireless systems via action space exploration. Check out the paper for our PD-ZDPG+ algorithm: https://arxiv.org/abs/2108.10352

Deterministic Policy Gradient via Action-Space Exploration:
<br />
<p align="left">
<img src="zdpg+.png" alt="Deterministic Policy Gradient via Action-Space Exploration" width="70% align="middle">
</p> 
<br />                                                                                                                 
<p align="center">
<img src="awgn_multi_plot.png" alt="Performance of all methods on AWGN channel" width="70% align="middle">
</p>  
<br /><br />

Before running the experiments, please clone ```gym-cstr-optim``` from [here](https://github.com/hassaanhashmi/gym-cstr-optim). Afterwards, run the following:

```
pip install -e gym-cstr-optim
sudo apt-get install texlive-latex-recommended 
sudo apt install texlive-latex-extra
sudo apt install dvipng
sudo apt install cm-super
```
<br />
