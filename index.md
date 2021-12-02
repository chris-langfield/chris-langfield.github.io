---
layout: page
title: 
---

I am a Research Software Engineer at Princeton University, based in the Program for Computational and Applied Mathematics. I work full-time on [ASPIRE](https://github.com/ComputationalCryoEM/ASPIRE-Python), an open-source Python package which processes and reconstructs cryo-electron microscopy data. ASPIRE uses numerical libraries such as NumPy, SciPy, and [FINUFFT](https://github.com/flatironinstitute/finufft). The problem of single-particle cryo-EM is to reconstruct the 3D energy density (and hence, structure) of a macromolecule given many noisy 2-dimensional projections of this function at unknown rotations:

![Image](https://quicklatex.com/cache3/74/ql_671de6d2466dfee3a005ddf9e6285b74_l3.png)

_Image formation model in cryo-EM: The function_ $$\phi : \mathbb{R}^3 \into \mathbb{R}$$ _is estimated from thousands of projection images_ $$I_i$$ _(1)_

The experimental image can be modeled via a convolution of this projection with the microscope's [point spread function](https://en.wikipedia.org/wiki/Point_spread_function) and finally the addition of Gaussian noise. 

This challenging problem is still an area of active research, and many biological molecules have been reconstructed to resolutions of just a few angstroms (`10^-10` meters), including the [COVID-19 spike protein](https://www.ebi.ac.uk/emdb/EMD-11526). Many fields both within and adjacent to mathematics have applications to cryo-EM, including signal processing, computer vision, machine learning, and even [group theory](https://arxiv.org/abs/1712.10163).

### Previous work 

I worked in the [Cognitive Neuroscience Division](http://www.columbianeuroresearch.org/taub/res-cognitive.html) at Columbia University Irving Medical Center from 2018-2021. 


---------
(1) Singer, A. (2018). Mathematics for cryo-electron microscopy. In B. Sirakov, P. N. de Souza, & M. Viana (Eds.), Invited Lectures (pp. 4013-4032). (Proceedings of the International Congress of Mathematicians, ICM 2018; Vol. 4). World Scientific Publishing Co. Pte Ltd.
