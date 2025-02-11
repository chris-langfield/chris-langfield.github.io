---
layout: post
title: "A 'wedding cake' probability distribution -- Part 2"
author: "Chris Langfield"
categories: math
tags: [math]
---

In [part 1](https://chris-langfield.github.io/wedding-cake) we explored the discrete probability distribution generated by a process of "double selection" from a set of integers. First we pick $$X_0 \sim \text{unif}(1\dots k)$$, and then we pick $$X_1 \sim \text{unif}(1\dots X_0)$$. The probability distribution of $$X_1$$ is

$$ Pr(X_1 = s) = \frac{1}{k} (H_k - H_{s-1}) $$

We proposed a sequence of probability distributions -- let's now call them $$p^k_m$$ -- corresponding to repeating this process $$m$$ times. That is, we pick an integer $$X_0$$ uniformly from $$1 \dots k$$, then we pick an integer $$X_1$$ uniformly from $$1 \dots X_0$$, pick $$X_2$$ from $$1 \dots X_1$$ and so on. Define 

$$p^k_0(s) = \frac{1}{k}$$

i.e. the base case of this process is simply the discrete uniform distribution. Then,

$$p^k_1(s) = \frac{1}{k} (H_k - H_{s-1})$$

This is what was derived in the last post. I mentioned that it is quite difficult to continue finding general expressions for $$p^k_m(s)$$ directly. However, there is a pattern with these distributions. First, recall:

$$Pr(X_m = s) = \sum_{i=1}^k Pr(X_m = s | X_{m-1} = i) Pr(X_{m-1} = i) $$

Using the same reasoning as in Part 1, we argue that 

$$
Pr(X_m=s|X_{m-1}=i) = \begin{cases} 
    \frac{1}{i} & s\leq i \\
    0 & s > i \\
  \end{cases}
$$

for every $$m$$. This is due to the fact that once $$X_{m-1}$$ has been chosen, $$X_m$$ is sampled *uniformly* from $$1\dots X_{m-1}$$

Therefore,

$$p^k_m(s) = \sum_{i=s}^k \frac{1}{i} p^k_{m-1}(i)$$

This recursive relationship can be expanded into $$k$$ equations:
$$
p^k_m(1) = 1 \cdot p_{m-1}(1) + \bigg(\frac{1}{2}\bigg)p^k_{m-1}(2) + \bigg(\frac{1}{3}\bigg)p^k_{m-1}(3) + \dots + \bigg(\frac{1}{k-1}\bigg)p^k_{m-1}(k-1) + \bigg(\frac{1}{k}\bigg)p^k_{m-1}(k) 
$$

$$
p^k_m(2) = 0  + \bigg(\frac{1}{2}\bigg) p^k_{m-1}(2) + \bigg(\frac{1}{3}\bigg)p^k_{m-1}(3) + \dots + \bigg(\frac{1}{k-1}\bigg)p^k_{m-1}(k-1) + \bigg(\frac{1}{k}\bigg)p^k_{m-1}(k) 
$$

$$
p^k_m(3) = 0  + 0 + \bigg(\frac{1}{3}\bigg)p^k_{m-1}(3) + \dots + \bigg(\frac{1}{k-1}\bigg)p^k_{m-1}(k-1) + \bigg(\frac{1}{k}\bigg)p^k_{m-1}(k) 
$$

$$
p^k_m(4) = 0 + 0 + 0  + \dots + \bigg(\frac{1}{k-1}\bigg)p^k_{m-1}(k-1) + \bigg(\frac{1}{k}\bigg)p^k_{m-1}(k) 
$$

$$
...
$$

$$
p^k_m(k) = 0 + 0 + 0  + \dots + 0 + \bigg(\frac{1}{k}\bigg)p^k_{m-1}(k) 
$$

I'm insisting on keeping the superscript $$k$$ because it's important to remember that these distributions are parametrized by $$k$$, i.e. the size of the initial sample space, as well as by $$m$$. We can think of the equations above as a matrix multiplication, making the following expression equivalent:

$$
    \begin{pmatrix}
      p^k_m(1) \\
      p^k_m(2) \\
      \dots \\
      p^k_m(k)
    \end{pmatrix}
    =
    \begin{pmatrix}
      1 & \frac{1}{2} & \frac{1}{3} & \dots & \frac{1}{k-1} & \frac{1}{k} \\
      0 & \frac{1}{2} & \frac{1}{3} & \dots  & \frac{1}{k-1} & \frac{1}{k} \\
      0 & 0 & \frac{1}{3} & \dots & \frac{1}{k-1} & \frac{1}{k} \\
      \vdots & \vdots & \vdots & \ddots & \frac{1}{k-1} & \frac{1}{k} \\
      0 & 0 & 0 & 0 & 0 & \frac{1}{k} 
    \end{pmatrix}
    \cdot
    \begin{pmatrix}
      p^k_{m-1}(1) \\
      p^k_{m-1}(2) \\
      \dots \\
      p^k_{m-1}(k)\\
    \end{pmatrix}
$$

Rather than thinking of the probability distributions $$p^k_0, p^k_1, \dots p^k_m$$, then, we can think of a sequence of *probability vectors* $$\mathbf{P}^k_0, \mathbf{P}^k_1, \dots \mathbf{P}^k_m$$ satisfying

$$
\mathbf{P}^k_m = \mathbf{W}\mathbf{P}^k_{m-1}
$$

Where $$\mathbf{W}$$ is the transition matrix shown above. Then

$$\mathbf{P}^k_m = \mathbf{W}^m \mathbf{P}^k_0$$

Since we have that $$p^k_0(s) = \frac{1}{k}$$ for all $$s$$, $$\mathbf{P}^k_0 = \big(\frac{1}{k}, \frac{1}{k}, \dots \frac{1}{k} \big)$$. 

This expression, along with the base vector $$\mathbf{P}^k_0$$, defines a *matrix difference equation*, a well-studied topic (see for reference Ch. 7 in [Cull, Flahive and Robson: Difference Equations: From Rabbits to Chaos](https://link.springer.com/book/10.1007/0-387-27645-9)). 

## Pascal and Inverse Pascal

We take the standard approach to solving a matrix difference equation. Notice that $$\mathbf{W}$$ is an upper-triangular matrix, which means its eigenvalues can be read off the diagonal. These are $$1,\frac{1}{2}, \frac{1}{3}, \dots \frac{1}{k}$$. Because there are $$k$$ distinct eigenvalues, $$\mathbf{W}$$ [is diagonalizable](https://en.wikipedia.org/wiki/Diagonalizable_matrix). That means it can be written in the form

$$
\mathbf{W} = \mathbf{Q} \mathbf{D} \mathbf{Q}^{-1}
$$

Where $$\mathbf{D} = \text{diag}(1,\frac{1}{2}, \dots \frac{1}{k})$$ and the columns of $$Q$$ are the corresponding eigenvectors of $$\mathbf{W}$$. This is particularly useful in this case because we can write, for example:

$$
\mathbf{W}^2 = (\mathbf{Q}\mathbf{D}\mathbf{Q}^{-1} ) (\mathbf{Q}\mathbf{D}\mathbf{Q}^{-1}) = \mathbf{Q} \mathbf{D} (\mathbf{Q}^{-1}\mathbf{Q}) \mathbf{D} \mathbf{Q}^{-1} = \mathbf{Q}\mathbf{D}\mathbf{I}\mathbf{D}\mathbf{Q}^{-1} = \mathbf{Q}\mathbf{D}^2\mathbf{Q}^{-1}
$$

In general,

$$
\mathbf{W}^m = \mathbf{Q}\mathbf{D}^m\mathbf{Q}^{-1}
$$

With $$\mathbf{D}$$ being a diagonal matrix, $$\mathbf{D}^m$$ is easy to compute: $$\mathbf{D}^m = \text{diag}(1, \frac{1}{2^m}, \frac{1}{3^m}, \dots \frac{1}{k^m})$$

First we must find $$\mathbf{Q}$$ and $$\mathbf{Q}^{-1}$$:

> **Proposition**
>
>*The matrix $$\mathbf{Q}$$ whose columns are the eigenvectors of $$\mathbf{W}$$ is the **inverse Pascal matrix**, whose entries are given by:*
>
>$$
>Q_{ij} = \begin{cases} 
>    (-1)^{j-i} \binom{j-1}{i-1}, & i \leq j \\
>    0 & i > j \\
>  \end{cases}
>$$
>
>*That is,*
>
>$$
>    \mathbf{Q}
>    =
>    \begin{pmatrix}
>      1 & -1 & 1 & -1 & 1 & -1 & \dots \\
>      0 &  1 &-2 &  3 &-4 &  5 & \dots \\
>      0 & 0 &  1 & -3 & 6 &-10 & \dots \\
>      0 & 0 & 0 &   1 &-4 & 10 & \dots \\
>      0 & 0 & 0 &   0 & 1 & -5 & \dots \\
>      \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots
>    \end{pmatrix}
>$$
><details><summary>Click to expand proof</summary>
    The entries of the matrix are given by
    $$
    W_{ij} = \begin{cases} 
        \frac{1}{j}, & i \leq j \\
        0 & i > j \\
        \end{cases}
    $$
    We propose that W has eigenvalue-eigenvector pairs
    $$
    \lambda_a = \frac{1}{a}, \quad (e_a)_r = \begin{cases}
        (-1)^{a-1+r}\binom{a-1}{r-1} & 1 \leq r \leq a \\
        0 & a < r \leq k
        \end{cases}
    $$
    We must show 
    $$
    \mathbf{W}\cdot \mathbf{e}_a = \frac{1}{a} \mathbf{e}_a, \quad a = 1, 2, \dots k
    $$
    Or equivalently,
    $$
    (We_a)_r = \frac{1}{a} (e_a)_r, \quad 1 \leq r \leq a
    $$
    Note that the following identities are true for the binomial coefficients:
    $$
    \binom{n-1}{k-1} = \frac{k}{n} \binom{n}{k} \quad (\ast)\\
    \sum_{i=0}^{n} (-1)^i \binom{n}{i} = 0 \quad (\dagger) \\
    \sum_{i=0}^{D} (-1)^i \binom{n}{i} = (-1)^{D} \binom{n-1}{D}, \quad D<n \quad (\ddagger)
    $$
    The first identity is easily verified. See https://math.stackexchange.com/questions/887960 for the latter two. Expanding the previous expression:
    $$
    (We_a)_r = \sum_{l=r}^a \frac{1}{l} (-1)^{a-1+l} \binom{a-1}{l-1} \\
    = (-1)^{a-1} \sum_{l=r}^a \frac{1}{l} (-1)^l \binom{a-1}{l-1} \\
    =  \frac{(-1)^{a-1}}{a} \sum_{l=r}^a (-1)^l \binom{a}{l} \quad (\ast) \\
    = \frac{(-1)^{a-1}}{a} \bigg( \sum_{l=0}^a (-1)^l \binom{a}{l} - \sum_{l=0}^{r-1} (-1)^l \binom{a}{l} \bigg) \\
    = \frac{(-1)^{a}}{a} \sum_{l=0}^{r-1} (-1)^l \binom{a}{l} \quad (\dagger) \\
    = \frac{(-1)^{a}}{a} (-1)^{r-1} \binom{a-1}{r-1} \quad (\ddagger) \\
    = \frac{1}{a} (-1)^{a-1+r} \binom{a-1}{r-1} = \frac{1}{a}(e_a)_r 
    $$
    Therefore, the columns of $$\mathbf{Q}$$ are the vectors $$\mathbf{e}_a, \quad a = 1,2,\dots k$$.◼️
></details>

$$\mathbf{Q}^{-1}$$ is of course just the inverse of the matrix above:

>**Proposition**
>
>*$$\mathbf{Q}^{-1}$$ is equal to the **Pascal matrix**, whose entries are given by:*
>
>$$
>Q^{-1}_{ij} = \begin{cases} 
>    \binom{j-1}{i-1}, & i \leq j \\
>    0 & i > j \\
>  \end{cases}
>$$
>
>*That is,*
>
>$$
>    \mathbf{Q}^{-1}
>    =
>    \begin{pmatrix}
>      1 & 1 & 1 & 1 & 1 & 1 & \dots \\
>      0 & 1 & 2 & 3 & 4 & 5 & \dots \\
>      0 & 0 & 1 & 3 & 6 & 10 & \dots \\
>      0 & 0 & 0 & 1 & 4 & 10 & \dots \\
>      0 & 0 & 0 & 0 & 1 & 5 & \dots \\
>      \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots
>    \end{pmatrix}
>$$
>
> [This proof](https://proofwiki.org/wiki/Inverse_of_Pascal%27s_Triangle_expressed_as_Matrix) can be slightly modified to arrive at the result.

With this information, we can begin to work on computing $$\mathbf{P}^k_m$$. First, we would like to find an expression for the entries of $$\mathbf{W}^m = \mathbf{Q}\mathbf{D}^m\mathbf{Q}^{-1}$$. The first matrix product, $$\mathbf{D}^m\mathbf{Q}^{-1}$$, is:

$$
(D^mQ^{-1})_{ij} = \sum_{l = 1}^k (D^m)_{il} (Q^{-1})_{lj} = \sum_{l=1}^k \delta_{il} \bigg(\frac{1}{l}\bigg)^m \binom{j-1}{l-1} = \binom{j-1}{i-1}\bigg(\frac{1}{i}\bigg)^m, \quad i \leq j 
$$

Then,

$$
W^m_{ij} = (QD^m Q^{-1})_{ij} = \sum_{l=1}^k (-1)^{l-i} \binom{l-1}{i-1} \binom{j-1}{l-1} \bigg(\frac{1}{l}\bigg)^m, \quad i \leq j 
$$

Note that because $W^m_{ij}$ is an upper triangular matrix, $W^m_{ij} = 0$ for $i>j$. This expression can be slightly simplified:

>**Proposition**
>
>*The following identity holds*
>
>$$  
>W^m_{ij} = \sum_{l=1}^k (-1)^{l-i} \binom{l-1}{i-1} \binom{j-1}{l-1} \bigg(\frac{1}{l}\bigg)^m = \binom{j-1}{i-1} \sum_{r=0}^{j-i} (-1)^{r} \binom{j-i}{r} \bigg(\frac{1}{r + i}\bigg)^m 
>$$
>
>*for $$i \leq j$$.*
>
><details><summary>Click to expand proof</summary>
    The following identity is true for the binomial coefficients:
    $$
    \binom{n}{m}\binom{m}{k} = \binom{n}{k}\binom{n-k}{m-k}
    $$
    (J. Gross, lecture notes for combinatorial mathematics http://www.cs.columbia.edu/~cs4205/files/CM4.pdf). Therefore, 
    $$
    \binom{j-1}{l-1}\binom{l-1}{i-1} = \binom{j-1}{i-1}\binom{j-1-(i-1)}{l-1-(i-1)} = \binom{j-1}{i-1}\binom{j-i}{l-i}
    $$
    Then the lefthand side of the equation is equal to
    $$
    \binom{j-1}{i-1} \sum_{l=1}^k (-1)^{l-i} \binom{j-i}{l-i} \bigg(\frac{1}{l}\bigg)^m
    $$
    We only allow terms where $$0 \leq l-i \leq j-i$$ so the limits of summation can be rewritten:
    $$
    \binom{j-1}{i-1} \sum_{l=i}^j (-1)^{l-i} \binom{j-i}{l-i} \bigg(\frac{1}{l}\bigg)^m
    $$
    Reindexing via $$r=l-i$$ gives us the righthand side of the equation. ◼️
>    </details>

The $$m$$'th probability vector $$\mathbf{P}^k_m$$ has components $$(p^k_m(1), p^k_m(2) \dots p^k_m(k))$$. Therefore, we can find an expression for the probability distribution $$p^k_m(s)$$  for $$s \in S = \{1, 2, \dots k\}$$ by multiplying the $$s$$'th row vector of $$\mathbf{W}^m$$ by the base vector $\mathbf{P}^k_0 = (\frac{1}{k}, \frac{1}{k}, \dots \frac{1}{k})$:

$$
p^k_m(s) = \frac{1}{k} \sum_{l=1}^{k} W^m_{sl} = \frac{1}{k} \sum_{l=s}^k W^m_{sl}
$$

Note that we can start the summation at column $$s$$ because $W^m_{ij}$ is upper triangular. Therefore the $s$'th row has zeros in columns 1 up through $s-1$. This is the "workable but unsatisfying" closed form I mentioned in the last post. Although the mathematical expression is not very clean, it is straightforward to implement the computation of $W^m_{ij}$ in Python. Let's take a look at the distributions for different values of $$m$$:

![Figure_1](https://user-images.githubusercontent.com/34426450/147511054-a5c78919-622b-4cab-be77-184e21535184.png)

The code is available on [GitHub](https://github.com/chris-langfield/Wedding-Cake-Distribution). This matches our intuition. As $$m$$ grows larger, the tail of the distribution shrinks rapidly. At $$m=5$$ the probability of picking $$s=1$$ is more than 75%. 

Strictly, at this point, the problem has been solved. For any choice of $$k$$ or $$m$$ we can compute the probability distribution $$p^k_m(s): S \mapsto \mathbb{R}$$. We can make a quick definition to simplify things for Part 3. If the distribution of a random variable $$X$$ over $$S$$ is $$p^k_m(s)$$, then we'll say that $$X \sim W(k,m)$$. That is $$X$$ is distributed according to a Wedding Cake distribution with parameters $$k$$ and $$m$$. There is one more interesting result I would like to add:

>**Proposition**
>
>*The sum expressing the entries of $$\mathbf{W}^m$$ has an integral representation, specifically:*
>
>$$  
>W^m_{ij} = \binom{j-1}{i-1} \int_0^1 \int_0^1 \dots \int_0^1 (x_1 x_2 \dots x_m)^{i-1} (1-x_1 x_2 \dots x_m)^{j-i} dx_1 dx_2 \dots dx_m 
>$$
>
>*for $$i\leq j$$.*
><details><summary>Click to expand proof</summary> 
    Using the fact that
    $$
    \frac{1}{r+i} = \int_0^1 x^{r+i-1} dx
    $$
    We write
    $$ 
    W^m_{ij} = \binom{j-1}{i-1} \sum_{r=0}^{j-i} \binom{j-i}{r} (-1)^r \bigg(\int_0^1 x^{r+i-1} dx \bigg)^m \\ 
    = \binom{j-1}{i-1} \sum_{r=0}^{j-i} \binom{j-i}{r} (-1)^r \int_0^1 x_1^{r+i-1} dx_1 \int_0^1 x_2^{r+i-1} dx_2 \dots \int_0^1 x_m^{r+i-1} dx_m \\
    = \binom{j-1}{i-1} \sum_{r=0}^{j-i} \binom{j-i}{r} (-1)^r \int_0^1 \int_0^1 \dots \int_0^1 x_1^{r+i-1} x_2^{r+i-1} \dots x_m^{r+i-1} dx_1 dx_2 \dots dx_m \\
    = \binom{j-1}{i-1} \int_0^1 \int_0^1 \dots \int_0^1 (x_1 x_2 \dots x_m)^{i-1} \sum_{r=0}^{j-i} \binom{j-i}{r} (-1)^r (x_1 x_2 \dots x_m)^r dx_1 dx_2 \dots dx_m
    $$
    Via the binomial theorem:
    $$
    \sum_{r=0}^{j-i} \binom{j-i}{r} (-1)^r (x_1 x_2 \dots x_m)^r = (1-x_1 x_2 \dots x_m)^{j-i}
    $$
    And the result follows directly. ◼️
></details>

If we take $m=1$, we have

$$
W^1_{ij} = \binom{j-1}{i-1} \int_0^1 x^{i-1} (1-x)^{j-i} dx = \binom{j-1}{i-1} \text{B}(i, j-i+1)
$$

where $$B(x,y)$$ is the Beta function. Note that the definition of the beta function requires that for real arguments $$x$$ and $$y$$, $$x>0$$ and $$y>0$$. $$W^m_{ij} = 0$$ for $i>j$, so this expression is always valid. Then we have an identity

$$
p^k_1(s) = \frac{1}{k} \sum_{l=s}^k W^1_{sl} = \frac{1}{k} \sum_{l=s}^k \binom{l-1}{s-1} \text{B}(s, l-s+1) = \frac{1}{k} (H_k - H_{s-1})
$$

where the last expression is the original formula for $p^k_1(s)$ derived in part 1. 
