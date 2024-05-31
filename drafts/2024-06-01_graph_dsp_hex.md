---
layout: post
title: "Graph Signal Processing and Regular Grids"
author: "Chris Langfield"
categories: math
tags: [math]
---

In this post I compare the [graph Fourier transform](https://en.wikipedia.org/wiki/Graph_Fourier_transform) with the standard discrete Fourier transform in 1D and on square and hexagonal 2D grids. Primarily I wanted to explore some of the theoretical and computational building blocks for processing 2D images (square or hexagonal) using Graph Signal Processing, a relatively recent field in digital signal processing and machine learning. 

I'm using [PyGSP](https://pygsp.readthedocs.io/en/stable/) and my own [hexfft](https://github.com/chris-langfield/hexfft). See the bottom of the post for a list of resources I consulted.

# Graph Signal Processing

Graph signal processing is a framework for processing signals whose domains are the vertices of graphs. There are already many introductions to the topic ([1](https://arxiv.org/abs/1211.0053), [2](https://infoscience.epfl.ch/record/256648?ln=en), [3](https://sybernix.medium.com/introduction-to-graph-signal-processing-ab9c0fde4d51), [4](https://balcilar.medium.com/struggling-signals-from-graph-34674e699df8)) so I'll just very quickly summarize: A graph $\mathcal{G}$ can be defined as

$$
\mathcal{G} = \\{ \mathcal{V}, \mathcal{E}, \mathbf{W} \\}
$$

where $\mathcal{V}$ is a set of vertices and $\mathcal{E}$ is a set of edges between the vertices. If the graph $\mathcal{G}$ has $V = |\mathcal{V}|$ vertices, the *adjacency matrix* $\mathbf{W}$ is a $V$ by $V$ matrix where $W_{ij}$ is nonzero when vertices $i$ and $j$ have an edge connecting them, and zero otherwise. To start with, I'm only considering *unweighted* graphs, where all the nonzero entries of $\mathbf{W}$ are 1. Additionally, graphs can be *directed* meaning that there can be a directed edge from vertex $i$ to vertex $j$ but not the other way around, and that the edges could have different weights. For now I consider only undirected graphs (the adjacency matrix $\mathbf{W}$ is symmetric in this case). 

A graph signal $f$, then is a map:

$$
f : \mathcal{V} \rightarrow \mathbb{R}
$$

## Example in `PyGSP`

We can define a graph based on the adjacency matrix $\mathbf{W}$. Plotting a constant signal allows us to just see the graph's geometry.

```python
import pygsp

W = np.array([[0., 0., 1., 1., 0.],
       [0., 0., 1., 1., 1.],
       [1., 1., 0., 0., 1.],
       [1., 1., 0., 0., 0.],
       [0., 1., 1., 0., 0.]])

g = pygsp.graphs.Graph(W)
g.set_coordinates()
g.plot_signal(np.ones(5))
```
![graph](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/a652e7b0-d084-490c-98ce-b8948179ede9)

We can define a random signal on this graph by specifying a number for each vertex:

```python
f = np.random.randn(5)
print(f)
```
```
[ 1.01241896  1.66358438 -0.20836631  0.6641997   2.31801111]
```

Plot this signal on its domain:

```python
fig, ax = plt.subplots()
g.plot_signal(f, ax=ax)
ax.set_title("Random signal")
```

![randgraphsignal](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/86cd687b-2283-462e-96fb-b584c761afc5)

## The Graph Fourier Transform

In analogy with the fact that the continuous space [Fourier modes are eigenfunctions of the the Laplacian operator](https://www.math.ucla.edu/~tao/preprints/fourier.pdf), the Graph Fourier Transform is defined as the decomposition of a graph signal into the eigenvectors of the [graph Laplacian matrix](https://en.wikipedia.org/wiki/Laplacian_matrix). This is defined as:

$$
\mathbf{L} = \mathbf{D} - \mathbf{W}
$$

where $\mathbf{W}$ is the adjacency matrix defined above. $D$ is the *degree matrix*, a diagonal matrix whose $i$'th diagonal entry is the number of edges connecting to vertex $i$. In other words,

$$
D_{ii} = \sum_{j} W_{ij}
$$

Because $L$ is a real, symmetric matrix, it has $V$ (the number of vertices) real and orthogonal eigenvectors $u_{i}$, each of which is also a graph signal. This collection of eigenvectors can be seen as a kind of basis for signals defined on the graph $\mathcal{G}$. The eigenvectors $\lambda_i$ are thought of as the "graph frequency" values. 

The graph Fourier transform is then defined as:

$$
\hat{f} (\lambda_i) = \sum_j f(j) u_i^{*} (j)
$$

$\hat{f}$ is considered to be a function defined at the graph frequency values $\lambda_i$ and is thus a vector in $\mathbb{R}^V$.

# The 1D DFT as the graph Fourier transform on a ring

Does the graph Fourier transform have any connection to the discrete Fourier transform? We can take the simple example of a signal defined as a 1D timeseries. We can think of the domain (equidistant points) as a graph where the vertices lie in a line, each one only connected to its two neighbors. We can also connect the first and last nodes to enforce a periodic boundary condition. Let's take as an example a sine wave sampled at 32 points. 

We can build this in `PyGSP`:

```python
N = 32
rg = pygsp.graphs.Ring(N)
sine = np.sin(2*np.pi*np.arange(N)/N)
rg.plot_signal(sine)
```
![sine_ring](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/647d1a9e-ec7c-4667-a104-46a49b11b3a5)

This is what the adjacency matrix of this ring graph looks like:

```python
fig, ax = plt.subplots()
ax.matshow(rg.W.toarray())
ax.set_title("N=32 Ring graph adjacency matrix")
```
![ringadjacency](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/e0f607f3-b324-449c-b43d-74b99d665910)

We can mathematically show that the graph Laplacian eigenvectors are equivalent to the Fourier modes, however we'll find that the eigenvalues (the "graph frequency") do not correspond to our usual notion of spatial frequency.

We can see that the adjacency matrix $\mathbf{W}$ has the following form:

$$
    \mathbf{W} = 
    \begin{pmatrix}
      0 & 1 & 0 & 0 & \cdots & 1 \\
      1 & 0 & 1 & 0 & \cdots & 0 \\
      0 & 1 & 0 & 1 & \cdots & 0 \\
      \cdots \\
      0 & \cdots & & 1 & 0 & 1 \\
      1 & & \cdots & & 1 & 0 \\
    \end{pmatrix}
$$

Because each row is a circular shift of the top row $(0, 1, 0, \cdots, 1)$, $\mathbf{W}$ is a [circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix). It therefore immediately follows that the eigenvectors $u_k$ are

$$
u_k = \big(1, \exp(\frac{2 \pi i k}{V}), \exp(\frac{2 \pi i (2k)}{V}), \exp(\frac{2 \pi i (3k)}{V}), \cdots , \exp(\frac{2 \pi i (nk)}{V}), \cdots, \exp(\frac{2 \pi i (V-1)k}{V}) \big)
$$


Let's compute the graph Fourier basis and the regular 1D Fourier basis and compare. The `compute_fourier_basis()` method in `PyGSP` automatically computes the eigendecomposition of the graph Laplacian. The eigenvectors are stored in the array `Graph.U`:

```python
rg.compute_fourier_basis()
rg.plot_signal(rg.U[:, 3])
```








