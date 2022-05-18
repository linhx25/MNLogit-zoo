# Multinomial Logit models

## 1. MNLogit model
$$U_{ijt}=\alpha+\beta X_{ijt}+\epsilon_{ijt}$$

where $\epsilon_{ijt}$ is distributed double exponential with $f(\epsilon_{ijt})=e^{-\epsilon_{ijt}}e^{-e^{-\epsilon_{ijt}}}$, $X$ are the features.

The conditional probability that each individual $i$ choose brand $y_{it}=c$ is:

$$Prob(y_{it}=c)=\frac{\exp (\alpha+\beta X_{ict})}{\sum_{j}\exp (\alpha+\beta X_{ijt})}$$

The likelihood is: 
<p align="center">
  <img 
    width="300"
    height="100"
    src="https://latex.codecogs.com/svg.image?\begin{aligned}&space;&space;L&=\prod_{i=1}^{I}\prod_{t=1}^{T}\prod_{j=1}^{J}&space;Prob(y_{it}=c)^{\mathbf{1}_{y_{it}=c}}\\&space;&space;&=\prod_{i=1}^{I}\prod_{t=1}^{T}\prod_{j=1}^{J}&space;\biggl(\frac{\exp&space;(\alpha&plus;\beta&space;X_{ict})}{\sum_{j}\exp&space;(\alpha&plus;\beta&space;X_{ijt})}\biggr)^{\mathbf{1}_{y_{it}=c}}&space;&space;&space;&space;\end{aligned}"
  >
</p>
<!-- $$
\begin{aligned}
  L&=\prod_{i=1}^{I}\prod_{t=1}^{T}\prod_{j=1}^{J} Prob(y_{it}=c)^{\mathbf{1}_{y_{it}=c}}\\
  &=\prod_{i=1}^{I}\prod_{t=1}^{T}\prod_{j=1}^{J} \biggl(\frac{\exp (\alpha+\beta X_{ict})}{\sum_{j}\exp (\alpha+\beta X_{ijt})}\biggr)^{\mathbf{1}_{y_{it}=c}}    
\end{aligned}
$$ -->

where $I,T,J$ is the number of customers, number of periods, number of choices, respectively. In this work, I set last choice as a baseline model ($\alpha_{-1}=0$).

## 2. Latent Class MNLogit
To introduce heterogeneity of customer, we extend the MNLogit model with $s$ customer segments:
$$U_{ijt}=\alpha_s+\beta_s X_{ijt}+\epsilon_{ijt}$$

The conditional probability that each individual $i$ choose brand
$y_{it}=c$ is:

$$Prob(y_{it|s}=c)=\frac{\exp (\alpha_s+\beta_s X_{ict})}{\sum_{j}\exp (\alpha_s+\beta_s X_{ijt})}$$

The individual conditional likelihood is: 
$$
L_{i|s}=\prod_{i=1}^{T}\prod_{j=1}^{J}Prob(y_{it|s}=c)^{\mathbf{1}_{y_{it}=c}}
$$

Hence the unconditional log-likelihood is: 

<p align="center">
  <img 
    width="400"
    height="200"
    src="https://latex.codecogs.com/svg.image?\begin{aligned}&space;&space;LL&=\log\biggl(\prod_{i=1}^{I}\sum_{s=1}^{S}L_{i|s}\pi_s\biggr)&space;&space;=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}L_{i|s}\pi_s\biggr)\\&space;&space;&=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}&space;&space;\prod_{i=1}^{T}\prod_{j=1}^{J}Prob(y_{it|s}=c)^{\mathbf{1}_{y_{it}=c}}\pi_s\biggr)\\&space;&space;&=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}&space;&space;\prod_{i=1}^{T}\prod_{j=1}^{J}&space;&space;\bigl(\frac{\exp&space;(\alpha_s&plus;\beta_s&space;X_{ict})}{\sum_{j}\exp&space;(\alpha_s&plus;\beta_s&space;X_{ijt})}\bigr)^{\mathbf{1}_{y_{it}=c}}\pi_s\biggr)\\&space;&space;&=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}&space;&space;\prod_{i=1}^{T}\prod_{j=1}^{J}&space;&space;\bigl(\frac{e^{\alpha_s&plus;\beta_s&space;X_{ict}}}{\sum_{j}e^{\alpha_s&plus;\beta_s&space;X_{ijt}}}\bigr)^{\mathbf{1}_{y_{it}=c}}&space;&space;\frac{e^{Z_{i}'\gamma_s}}{\sum_{s'}e^{Z_{i}'\gamma_s}}\biggr)\\\end{aligned}"
  >
</p>

<!-- $$\begin{aligned}
  LL&=\log\biggl(\prod_{i=1}^{I}\sum_{s=1}^{S}L_{i|s}\pi_s\biggr)
  =\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}L_{i|s}\pi_s\biggr)\\
  &=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}
  \prod_{i=1}^{T}\prod_{j=1}^{J}Prob(y_{it|s}=c)^{\mathbf{1}_{y_{it}=c}}\pi_s\biggr)\\
  &=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}
  \prod_{i=1}^{T}\prod_{j=1}^{J}
  \bigl(\frac{\exp (\alpha_s+\beta_s X_{ict})}{\sum_{j}\exp (\alpha_s+\beta_s X_{ijt})}\bigr)^{\mathbf{1}_{y_{it}=c}}\pi_s\biggr)\\
  &=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}
  \prod_{i=1}^{T}\prod_{j=1}^{J}
  \bigl(\frac{e^{\alpha_s+\beta_s X_{ict}}}{\sum_{j}e^{\alpha_s+\beta_s X_{ijt}}}\bigr)^{\mathbf{1}_{y_{it}=c}}
  \frac{e^{Z_{i}'\gamma_s}}{\sum_{s'}e^{Z_{i}'\gamma_s}}\biggr)\\
\end{aligned}$$  -->

where $Z_i$ is the latent feature for inferring the latent segments, $\pi_s$ is the segment size (the probability of the segment), $\gamma_s$ are the corresponding parameters. In this work, I set last choice as a baseline model ($\alpha_{-1}=0$) and the last segment as a baseline segment ($\gamma_{-1}=0$).

## 3. MNLogit with state dependence 
To introduce dynamics, we extend the MNLogit model with state dependence $Y_{ijt-1}$:

$$U_{ijt}=\alpha_s+\beta_{1s} X_{ijt}+\beta_{2s} Y_{ijt-1}+\epsilon_{ijt}$$

The conditional probability that each individual $i$ choose brand
$y_{it}=c$ is:

$$Prob(y_{it|s}=c)=\frac{\exp (\alpha_s+\beta_{1s} X_{ict}+\beta_{2s} Y_{ict-1})}{\sum_{j}\exp (\alpha_s+\beta_{1s} X_{ijt}+\beta_{2s} Y_{ijt-1})}$$

The individual conditional likelihood is: 

$$
  L_{i|s}=\prod_{i=1}^{T}\prod_{j=1}^{J}Prob(y_{it|s}=c)^{\mathbf{1}_{y_{it}=c}}
$$

Hence the unconditional log-likelihood is:

<p align="center">
  <img 
    width="500"
    height="260"
    src="https://latex.codecogs.com/svg.image?\begin{aligned}&space;&space;LL&=\log\biggl(\prod_{i=1}^{I}\sum_{s=1}^{S}L_{i|s}\pi_s\biggr)&space;&space;=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}L_{i|s}\pi_s\biggr)\\&space;&space;&=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}&space;&space;\prod_{i=1}^{T}\prod_{j=1}^{J}Prob(y_{it|s}=c)^{\mathbf{1}_{y_{it}=c}}\pi_s\biggr)\\&space;&space;&=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}&space;&space;\prod_{i=1}^{T}\prod_{j=1}^{J}&space;&space;\bigl(\frac{\exp&space;(\alpha_s&plus;\beta_{1s}&space;X_{ict}&plus;\beta_{2s}&space;Y_{ict-1})}{\sum_{j}\exp&space;(\alpha_s&plus;\beta_{1s}&space;X_{ijt}&plus;\beta_{2s}&space;Y_{ijt-1})}\bigr)^{\mathbf{1}_{y_{it}=c}}\pi_s\biggr)\\&space;&space;&=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}&space;&space;\prod_{i=1}^{T}\prod_{j=1}^{J}&space;&space;\bigl(\frac{e^{\alpha_s&plus;\beta_{1s}&space;X_{ict}&plus;\beta_{2s}&space;Y_{ict-1}}}{\sum_{j}e^{\alpha_s&plus;\beta_{1s}&space;X_{ijt}&plus;\beta_{2s}&space;Y_{ijt-1}}}\bigr)^{\mathbf{1}_{y_{it}=c}}&space;&space;\frac{e^{Z_{i}'\gamma_s}}{\sum_{s'}e^{Z_{i}'\gamma_s}}\biggr)\\\end{aligned}"
  >
</p>

<!-- $$\begin{aligned}
  LL&=\log\biggl(\prod_{i=1}^{I}\sum_{s=1}^{S}L_{i|s}\pi_s\biggr)
  =\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}L_{i|s}\pi_s\biggr)\\
  &=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}
  \prod_{i=1}^{T}\prod_{j=1}^{J}Prob(y_{it|s}=c)^{\mathbf{1}_{y_{it}=c}}\pi_s\biggr)\\
  &=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}
  \prod_{i=1}^{T}\prod_{j=1}^{J}
  \bigl(\frac{\exp (\alpha_s+\beta_{1s} X_{ict}+\beta_{2s} Y_{ict-1})}{\sum_{j}\exp (\alpha_s+\beta_{1s} X_{ijt}+\beta_{2s} Y_{ijt-1})}\bigr)^{\mathbf{1}_{y_{it}=c}}\pi_s\biggr)\\
  &=\sum_{i=1}^{I}\biggl(\log\sum_{s=1}^{S}
  \prod_{i=1}^{T}\prod_{j=1}^{J}
  \bigl(\frac{e^{\alpha_s+\beta_{1s} X_{ict}+\beta_{2s} Y_{ict-1}}}{\sum_{j}e^{\alpha_s+\beta_{1s} X_{ijt}+\beta_{2s} Y_{ijt-1}}}\bigr)^{\mathbf{1}_{y_{it}=c}}
  \frac{e^{Z_{i}'\gamma_s}}{\sum_{s'}e^{Z_{i}'\gamma_s}}\biggr)\\
\end{aligned}$$ -->

