# SGD in Gaussian Processes

## Introduction

* In this project, we apply stochastic gradient descent (SGD) algorithm and its variants to accelerate and improve Gaussian process (GP) inference.
* We provide code for implementing **sgGP** described in *Stochastic Gradient Descent in Correlated Settings: A Study on Gaussian Processes* by Hao Chen, Lili Zheng, Raed Al Kontar, Garvesh, Raskutti.

## Contributions

* We prove minibatch SGD converges to a critical point of the empirical loss function and recovers model hyperparameters with rate $'O(\frac{1}{K})'$ up to a statistical error term depending on the minibatch size.
* We prove that the conditional expectation of the loss function given covariates satisfies a relaxed property of strong convexity, which guarantees the $'O(\frac{1}{K})'$ optimization error bound.
* Computationally, we are able to scale to dataset sizes previously unexplored in GPs in a fraction of time needed for competing methods. Meanwhile statistically, we find that the induced regularization imposed by SGD improves generalization in GPs, specifically in large data settings.  

## Problem Setup

### Model

We consider the Gaussian process model

<!-- $$\begin{aligned}
    f\sim \mathcal{GP}(0, \sigma_f^2k(\cdot,\cdot)),&\quad \mathbf{x}_1,\dots,\mathbf{x}_n\overset{\text{i.i.d.}}{\sim}\mathbb{P},\\
    y_i=f(\mathbf{x}_i) + \epsilon_i,&\quad \epsilon_i\overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_{\epsilon}^2),\quad 1\leq i\leq n,
\end{aligned}$$ -->

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;f\sim&space;\mathcal{GP}(0,&space;\sigma_f^2k(\cdot,\cdot)),&\quad&space;\mathbf{x}_1,\dots,\mathbf{x}_n\overset{\text{i.i.d.}}{\sim}\mathbb{P},\\&space;y_i=f(\mathbf{x}_i)&space;&plus;&space;\epsilon_i,&\quad&space;\epsilon_i\overset{\text{i.i.d.}}{\sim}&space;\mathcal{N}(0,&space;\sigma_{\epsilon}^2),\quad&space;1\leq&space;i\leq&space;n,&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;f\sim&space;\mathcal{GP}(0,&space;\sigma_f^2k(\cdot,\cdot)),&\quad&space;\mathbf{x}_1,\dots,\mathbf{x}_n\overset{\text{i.i.d.}}{\sim}\mathbb{P},\\&space;y_i=f(\mathbf{x}_i)&space;&plus;&space;\epsilon_i,&\quad&space;\epsilon_i\overset{\text{i.i.d.}}{\sim}&space;\mathcal{N}(0,&space;\sigma_{\epsilon}^2),\quad&space;1\leq&space;i\leq&space;n,&space;\end{aligned}" title="\begin{aligned} f\sim \mathcal{GP}(0, \sigma_f^2k(\cdot,\cdot)),&\quad \mathbf{x}_1,\dots,\mathbf{x}_n\overset{\text{i.i.d.}}{\sim}\mathbb{P},\\ y_i=f(\mathbf{x}_i) + \epsilon_i,&\quad \epsilon_i\overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_{\epsilon}^2),\quad 1\leq i\leq n, \end{aligned}" /></a>

where $\mathbf{x}_i\in \mathcal{X}\subset \mathbb{R}^D$ is the input, $y_i$ is the output, $k(\cdot,\cdot): \mathcal{X}\rightarrow \mathbb{R}$ is some kernel function, $\sigma_f^2$ is the signal variance and $\sigma_{\epsilon}^2$ is the noise variance.

### Loss function

<!-- $$\begin{aligned}
    \ell(\boldsymbol{\theta};\mathbf{X}_n, \mathbf{y}_n) & = -\frac{1}{n}\log p(\mathbf{y}_n| \mathbf{X}_n,\boldsymbol{\theta}) \\
    & = \frac{1}{2n}[\mathbf{y}_n^\top\mathbf{K}_n^{-1}(\boldsymbol{\theta})\mathbf{y}_n+\log|\mathbf{K}_n(\boldsymbol{\theta})|+n\log (2\pi)],
\end{aligned}$$ -->

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\ell(\boldsymbol{\theta};\mathbf{X}_n,&space;\mathbf{y}_n)&space;&&space;=&space;-\frac{1}{n}\log&space;p(\mathbf{y}_n|&space;\mathbf{X}_n,\boldsymbol{\theta})&space;\\&space;&&space;=&space;\frac{1}{2n}[\mathbf{y}_n^\top\mathbf{K}_n^{-1}(\boldsymbol{\theta})\mathbf{y}_n&plus;\log|\mathbf{K}_n(\boldsymbol{\theta})|&plus;n\log&space;(2\pi)],&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\ell(\boldsymbol{\theta};\mathbf{X}_n,&space;\mathbf{y}_n)&space;&&space;=&space;-\frac{1}{n}\log&space;p(\mathbf{y}_n|&space;\mathbf{X}_n,\boldsymbol{\theta})&space;\\&space;&&space;=&space;\frac{1}{2n}[\mathbf{y}_n^\top\mathbf{K}_n^{-1}(\boldsymbol{\theta})\mathbf{y}_n&plus;\log|\mathbf{K}_n(\boldsymbol{\theta})|&plus;n\log&space;(2\pi)],&space;\end{aligned}" title="\begin{aligned} \ell(\boldsymbol{\theta};\mathbf{X}_n, \mathbf{y}_n) & = -\frac{1}{n}\log p(\mathbf{y}_n| \mathbf{X}_n,\boldsymbol{\theta}) \\ & = \frac{1}{2n}[\mathbf{y}_n^\top\mathbf{K}_n^{-1}(\boldsymbol{\theta})\mathbf{y}_n+\log|\mathbf{K}_n(\boldsymbol{\theta})|+n\log (2\pi)], \end{aligned}" /></a>

and we aim to eistimate the hyperparameters $`&sigma;<sub>f</sub>^2`$ and $`&sigma;<sub>&epsilon;</sub>^2`$ using minibatch SGD. Here $\boldsymbol{\theta}=(\sigma_{f}^2,\sigma_{\epsilon}^2)^\top\in\mathbb{R}^{2}$,  $(\mathbf{X}_n, \mathbf{y}_n)=((\mathbf{x}_1^\top,\dotsc,\mathbf{x}_n^\top)^\top,(y_1,\dotsc,y_n)^\top)$, $\mathbf{K}_n(\boldsymbol{\theta})=\theta_1\mathbf{K}_{f,n}+\theta_{2}\mathbf{I}_n\in \mathbb{R}^{n\times n}$ and $\mathbf{K}_{f,n}\in\mathbb{R}^{n\times n}$ is the kernel matrix of $k(\cdot,\cdot)$ evaluated at $\mathbf{X}_n$, i.e. $(\mathbf{K}_{f,n})_{i,j}=k(\mathbf{x}_i,\mathbf{x}_j)$.


## Theoreticl Guarantee of Convergence

### Assumptions
We assume the kernel function $`k`$ has exponential eigendecay, and each parameter iterate and stochastic gradient is bounded.
* **Exponential eigendecay**. The eigenvalues of kernel function $k(\cdot,\cdot)$ w.r.t. probability measure $\mathbb{P}$ are $\{Ce^{-bj}\}_{j=0}^{\infty}$.
* **Bounded iterates**. The true parameter ￼$\boldsymbol{\theta}^*$ and iterates $\boldsymbol{\theta}^{(k)}$￼ lie in ￼$[\boldsymbol{\theta}_{\min}, \boldsymbol{\theta}_{\max}]$.
* **Bounded stochastic gradient**. $\|g(\boldsymbol{\theta}^{(k)};\boldsymbol{X}_{\xi_{k+1}},\boldsymbol{y}_{\xi_{k+1}})\|_2\leq G$.

### Convergence of parameter iterates

<!-- $$\begin{aligned}
    (\theta^{(K)}_{1}-\theta^*_1)^2&\leq  C\left[\frac{G^2}{(K+1)}+{m^{-\frac{1}{2}+\varepsilon}}\right],\\
    (\theta^{(K)}_{1}-\theta^*_1)^2&\leq  C\left[\frac{G^2}{(K+1)}+{m^{-\frac{1}{2}+\varepsilon}}\right].
\end{aligned}$$ -->

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;(\theta^{(K)}_{1}-\theta^*_1)^2&\leq&space;C\left[\frac{G^2}{(K&plus;1)}&plus;{m^{-\frac{1}{2}&plus;\varepsilon}}\right],\\&space;(\theta^{(K)}_{1}-\theta^*_1)^2&\leq&space;C\left[\frac{G^2}{(K&plus;1)}&plus;{m^{-\frac{1}{2}&plus;\varepsilon}}\right].&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;(\theta^{(K)}_{1}-\theta^*_1)^2&\leq&space;C\left[\frac{G^2}{(K&plus;1)}&plus;{m^{-\frac{1}{2}&plus;\varepsilon}}\right],\\&space;(\theta^{(K)}_{1}-\theta^*_1)^2&\leq&space;C\left[\frac{G^2}{(K&plus;1)}&plus;{m^{-\frac{1}{2}&plus;\varepsilon}}\right].&space;\end{aligned}" title="\begin{aligned} (\theta^{(K)}_{1}-\theta^*_1)^2&\leq C\left[\frac{G^2}{(K+1)}+{m^{-\frac{1}{2}+\varepsilon}}\right],\\ (\theta^{(K)}_{1}-\theta^*_1)^2&\leq C\left[\frac{G^2}{(K+1)}+{m^{-\frac{1}{2}+\varepsilon}}\right]. \end{aligned}" /></a>

### Convergence of full gradient

<!-- $$\begin{aligned}
\|\nabla \ell(\boldsymbol{\theta}^{(K)})\|_2^2\leq C\left[\frac{G^2}{{K+1}}+{m^{-\frac{1}{2}+\varepsilon}}\right].
\end{aligned}$$ -->

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\|\nabla&space;\ell(\boldsymbol{\theta}^{(K)})\|_2^2\leq&space;C\left[\frac{G^2}{{K&plus;1}}&plus;{m^{-\frac{1}{2}&plus;\varepsilon}}\right].&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\|\nabla&space;\ell(\boldsymbol{\theta}^{(K)})\|_2^2\leq&space;C\left[\frac{G^2}{{K&plus;1}}&plus;{m^{-\frac{1}{2}&plus;\varepsilon}}\right].&space;\end{aligned}" title="\begin{aligned} \|\nabla \ell(\boldsymbol{\theta}^{(K)})\|_2^2\leq C\left[\frac{G^2}{{K+1}}+{m^{-\frac{1}{2}+\varepsilon}}\right]. \end{aligned}" /></a>

## Numerical results

## Prerequisite

* [R](https://www.r-project.org/)
* [dplyr](https://github.com/tidyverse/dplyr)
* [RANN](https://github.com/jefferislab/RANN)

## Datasets

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
* [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)
