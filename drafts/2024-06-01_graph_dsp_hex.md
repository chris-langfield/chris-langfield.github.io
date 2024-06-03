---
layout: post
title: "Graph Signal Processing and Lattice Fourier Transforms"
author: "Chris Langfield"
categories: math
tags: [math]
---

In this post I compare the [graph Fourier transform](https://en.wikipedia.org/wiki/Graph_Fourier_transform) with the standard discrete Fourier transform in 1D and on square and hexagonal 2D grids. Primarily I wanted to explore some of the theoretical and computational building blocks for processing 2D images (square or hexagonal) using Graph Signal Processing, a relatively recent field in digital signal processing and machine learning. 

I'm using [PyGSP](https://pygsp.readthedocs.io/en/stable/) and my own [hexfft](https://github.com/chris-langfield/hexfft). See the bottom of the post for a list of resources I consulted.

# Graph Signal Processing

Graph signal processing is a framework for processing signals whose domains are the vertices of graphs. There are already many introductions to the topic ([1](https://arxiv.org/abs/1211.0053), [2](https://infoscience.epfl.ch/record/256648?ln=en), [3](https://sybernix.medium.com/introduction-to-graph-signal-processing-ab9c0fde4d51), [4](https://balcilar.medium.com/struggling-signals-from-graph-34674e699df8)) so I'll just very quickly summarize: A graph $\mathcal{G}$ can be defined as

$$
\mathcal{G} = ( \mathcal{V}, \mathcal{E}, \mathbf{W} )
$$

where $\mathcal{V}$ is a set of vertices and $\mathcal{E}$ is a set of edges between the vertices. If the graph $\mathcal{G}$ has $V = \vert\mathcal{V}\vert$ vertices, the *adjacency matrix* $\mathbf{W}$ is a $V$ by $V$ matrix where $W_{ij}$ is nonzero when vertices $i$ and $j$ have an edge connecting them, and zero otherwise. To start with, I'm only considering *unweighted* graphs, where all the nonzero entries of $\mathbf{W}$ are 1. Additionally, graphs can be *directed* meaning that there can be a directed edge from vertex $i$ to vertex $j$ but not the other way around, and that the edges could have different weights. For now I consider only undirected graphs (the adjacency matrix $\mathbf{W}$ is symmetric in this case). 

A graph signal $f$, then is a map

$$
f : \mathcal{V} \rightarrow \mathbb{R}
$$

## Example in PyGSP

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

where $\mathbf{W}$ is the adjacency matrix defined above. $\mathbf{D}$ is the *degree matrix*, a diagonal matrix whose $i$'th diagonal entry is the number of edges connecting to vertex $i$. In other words,

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

Since $\mathbf{D} = \text{diag}(2, 2, 2, \cdots)$ (each vertex has two neighbors), we have

$$
    \mathbf{L} = 
    \begin{pmatrix}
      2 & -1 & 0 & 0 & \cdots & -1 \\
      -1 & 2 & -1 & 0 & \cdots & 0 \\
      0 & -1 & 2 & -1 & \cdots & 0 \\
      \cdots \\
      0 & \cdots & & -1 & 2 & -1 \\
      -1 & & \cdots & & -1 & 2 \\
    \end{pmatrix}
$$

Because each row is a circular shift of the top row $(2, -1, 0, \cdots, -1)$, $\mathbf{W}$ is a symmetric and [circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix). It therefore immediately follows that the eigenvectors $u_k$ are

$$
u_k = \bigg(1, \exp\big(\frac{2 \pi i k}{N}\big), \exp\big(\frac{2 \pi i (2k)}{N}\big), \exp\big(\frac{2 \pi i (3k)}{N}\big), \cdots , \exp\big(\frac{2 \pi i (nk)}{N}), \cdots, \exp\big(\frac{2 \pi i (N-1)k}{N}\big) \bigg)
$$

Also using the properties of symmetric circulant matrices, the eigenvalues of $\mathbf{L}$ are

$$
\lambda_k = 2 - \exp\big(\frac{2 \pi i k}{N}\big) - \exp\big( \frac{2 \pi i (N-1)}{N} \big) = 2\big(1-\cos(\frac{2 \pi k}{N})\big)
$$

Note that these eigenvalues, the "graph spectrum" do not correspond to the eigenvalues of the regular Laplacian ($\pi^2 k^2$ where $k$ is the frequency).

![eigsring](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/ca62029a-2271-47d6-81cc-ecdc22ad16e0)

Because $\mathbf{L}$ has real eigenvalues, the set of imaginary and real components of the eigenvectors $u_k$ will span $R^{N}$ and we can select a real basis from among multiples of this set of vectors, which have the form

$$
s_k = \bigg(0, \sin\big(\frac{2 \pi k}{N}\big), \sin\big(\frac{2 \pi (2k)}{N}\big), \sin\big(\frac{2 \pi (3k)}{N}\big), \cdots , \sin\big(\frac{2 \pi (nk)}{N}), \cdots, \sin\big(\frac{2 \pi (N-1)k}{N}\big) \bigg)
$$

$$
c_k = \bigg(1, \cos\big(\frac{2 \pi k}{N}\big), \cos\big(\frac{2 \pi (2k)}{N}\big), \cos\big(\frac{2 \pi (3k)}{N}\big), \cdots , \cos\big(\frac{2 \pi (nk)}{N}), \cdots, \cos\big(\frac{2 \pi (N-1)k}{N}\big) \bigg)
$$

Let's numerically compute the graph Fourier basis and compare with the regular 1D Fourier basis. The `compute_fourier_basis()` method in `PyGSP` automatically computes the eigendecomposition of the graph Laplacian. The eigenvectors are stored in the array `Graph.U`:

```python
fig, ax = plt.subplots()
rg.plot_signal(rg.U[:, 3], ax=ax)
ax.set_title("3rd graph Laplacian eigenvector")
```
![ring_eigenmode](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/52de25b9-09eb-4fbf-85b0-c700f0a1c87a)

This eigenvector appears to be a sinusoid with a frequency of 2, a promising start for a graph Fourier basis. Because the graph Fourier eigenvectors are just 1D vectors of length 32, we can plot them on a line and compare with the standard Fourier modes. First we compute the Fourier basis functions for $N=32$:

```python
Uf = np.zeros((N, N), np.complex128)
for i in range(N):
    Uf[:, i] = np.exp(-2.j * np.pi * i * np.arange(N) / N)
```

Now plot them next to each other:
> A note on normalization: `PyGSP` returns the 0-frequency (DC component) as a constant $\frac{1}{\sqrt{N}}$, and the numerically computed eigenvectors are scaled to have amplitude $\frac{1}{4}$. They were normalized for the purposes of plotting.

![compare_1d_gft_fourier](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/743417a5-baad-4c06-bbf2-8bf8dd8b0090)

I arranged the graph eigenvectors (blue) and the Fourier modes (red) so that we can see our intuition is confirmed. For each spatial frequency, there are two graph Laplacian eigenvectors: one sine and one cosine (sometimes with a sign flip). It seems that the graph Fourier decomposition for a signal on a ring graph is largely an equivalent decomposition to the 1D Fourier decomposition.

As a last step, we can check our analytical expression for the eigenvalues above against the numerically computed ones:

```python
eigs_numerical = np.sort(np.linalg.eigvals(rg.L.toarray()))
eigs_analytic = np.sort(np.array([2*(1-np.cos(2*np.pi * k / N)) for k in range(N)]))
np.allclose(eigs_numerical-eigs_analytic)
```

# The graph Fourier transform on a 2D grid and the 2D DFT

We would like to try this on a 2D grid. If we can think of a 1D timeseries as a graph signal on a ring graph, can we process an image as a signal defined on a lattice? Is the graph Fourier transform comparable to the 2D discrete Fourier transform? 

The simplest initial approach will be to construct a [square lattice graph](https://en.wikipedia.org/wiki/Lattice_graph), where each vertex connects to four others (with careful consideration of edge cases). Graph based techniques for image processing have of course progressed far beyond this (see [here](http://www.arxiv.org/abs/1211.0053), Example 2, where 8 neighbors instead of 4 are used). 

This graph is already implemented in `PyGSP`, and we can take a look.

```python
N1, N2 = 6, 6
sg = pygsp.graphs.Grid2d(N1, N2)
sg.plot_signal(np.ones(N1*N2))
```

![2ggrid](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/07775d21-11ff-491d-8801-8a4823855114)

We choose the convention of labelling the 144 vertices starting at the bottom right, and proceeding right along each row, wrapping back to the left for each row. So the bottom row consists of vertices 0, 2, ... 11, the next-from bottom row 12, 13, ... 23, etc.

The adjacency matrix $\mathbf{W}$ looks like this:

![adj6grid2d](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/4e9bbf55-45c1-4d17-bd5d-34e7ad7faed8)

The first thing to notice is that this is not (yet) a circulant matrix as in the 1D case. However, looking at the structure of the matrix above, we can see that something may be missing. Recall that the sum across row $i$ of the adjacency matrix is the *degree* of vertex $i$, i.e. the number of vertices connecting to it. Note that quite a few of the rows in the adjacency matrix above have less than 4 nonzero entries. This is because this implementation of the graph does not include periodic boundary connections. The top 6 rows and bottom 6 rows correspond to the bottom and top row of vertices in the graph respectively, which all lack a connection down or up respectively. Additionally every 6 rows, a pair of rows have only 3 connections. These correspond to the first and last vertex of each row in the graph, which lack a left and a right connection, respectively. 

For an $N_1$ by $N_2$ square lattice graph, we can populate the periodicity conditions as follows:

```python
import scipy
circ = np.zeros(N1*N2)
circ[[1, N2, N1*N2-N2, N1*N2-1]] = 1
adj = scipy.linalg.circulant(circ).T
```

(we keep $N_1 = N_2$ in what follows for simplicity.)

The result:

![6x6square_periodic](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/df4606fd-436e-4f41-b198-de44fe1e97ed)

Note that this graph can no longer be embedded in 2D Euclidean space as its geometry is that of a torus. Inspection of the new adjacency matrix will show that it fills in the missing connections between the top and bottom, and the sides of the graph, such that each vertex has 4 edges. Lucky for us, this is now a circulant matrix. This makes the eigenvectors and eigenvalues of the graph Laplacian easy to derive. Since $\mathbf{D} = \text{diag}(4, 4, 4, ...)$, the Laplacian is the circulant matrix repetition of the vector

$$
(4, -1, 0, 0, \cdots, 0, -1, 0, \cdots, 0, -1, 0, \cdots, 0, -1)^T
$$

with 4 at entry 0, and -1's at entries $1$, $N$, $N^2-N$, and $N^2-1$. The eigenvectors of an $N^2$ by $N^2$ circulant matrix will be:

$$
u_k = \bigg(1, \exp\big(\frac{2 \pi i k}{N^2}\big), \exp\big(\frac{2 \pi i (2k)}{N^2}\big), \exp\big(\frac{2 \pi i (3k)}{N^2}\big), \cdots , \exp\big(\frac{2 \pi i (nk)}{N^2}), \cdots, \exp\big(\frac{2 \pi i (N^2-1)k}{N^2}\big) \bigg)
$$

The eigenvalues are (again using the properties listed in the [Wikipedia article](https://en.wikipedia.org/wiki/Circulant_matrix) for circulant matrices):

$$
\lambda_k = 4  - \exp\big(\frac{2 \pi i k}{N^2}\big)  - \exp\big(\frac{2 \pi i k N}{N^2}\big) -  \exp\big(\frac{2 \pi i k (N^2 - N)}{N^2}\big) - \exp\big( \frac{2 \pi i k (N^2-1)}{N} \big)
$$

This can be simplified via some algebra with the complex exponentials, and we found that the eigenvalues are real as expected:

$$
\lambda_k = 4 - \exp\big(\frac{2 \pi i k}{N^2}\big)  - \exp\big(\frac{2 \pi i k}{N}\big) -  \exp\big(\frac{2 \pi i k N}{N} \big) \exp\big(\frac{-2 \pi i k}{N}\big) - \exp\big(\frac{2 \pi i k N^2}{N^2}\big) \exp\big( -\frac{2 \pi i k}{N^2} \big)
$$

$$
= 4 - \bigg(\exp\big(\frac{2 \pi i k}{N^2}\big) + \exp\big( -\frac{2 \pi i k}{N^2} \big)\bigg) + \bigg(\exp\big(\frac{2 \pi i k}{N}\big) + \exp\big( -\frac{2 \pi i k}{N} \big)\bigg)
$$

$$
= 4 - \bigg( 2 \cos\big(\frac{2 \pi k}{N^2}\big) + 2 \cos \big(\frac{2 \pi k}{N}\big) \bigg)
$$

These eigenvectors $u_k$ don't appear to have any obvious 2 dimensional structure. They are regular sinusoids evaluated over a domain of $N^2$ sequential points. This is a consequence of the fact that in order to create the adjacency matrix, and hence compute the graph Laplacian, the vertices must be listed in one dimension. They are, however, graph signals according to the theory, defined on the periodic square lattice graph defined above. 

When we plot these eigenvectors and compare with the 2D Fourier modes, they are remarkably similar. We again see the phenomenon where the real Laplacian eigenfunctions appear to have a sine and a cosine wave for each spatial frequency.

![12x12_eigenvector_modes](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/e8acbe49-1a77-48ec-aabb-d1ee8536b071)

![12x12_fourier_modes](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/bce7efc0-368b-406a-b662-73241ad54de3)

Despite the fact that in computing the graph Laplacian, we have serialized the vertices, their geometric relation to each other is stored in the adjacency matrix $\mathbf{W}$. The code to reproduce the above plots is here. Note we cannot use the pre-written `pygsp.graphs.Grid2d` class, but have to create a graph from the circulant adjacency matrix defined above. 

```python
N=12
circ = np.zeros(N**2)
circ[[1, N, N**2-N, N**2-1]] = 1
adj = scipy.linalg.circulant(circ).T
psg = pygsp.graphs.Graph(W=adj)

fig, axs = plt.subplots(4, 4)
for i, ax in enumerate(axs.flat):
    ax.matshow(psg.U[:, i].reshape(N1, N2))
fig.suptitle("12x12 Square Graph: Eigenvector modes")

Uf = np.zeros((N1, N2, N1, N2), np.complex128)
x, y = np.meshgrid(np.arange(N1), np.arange(N2), indexing="xy")
for i in range(N1):
    for j in range(N2):
        Uf[i, j, :, :] = np.exp(2.j * np.pi * (i*x/N1+j*y/N2))
fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axs[i, j].matshow(np.real(Uf[i, j, :, :]))
fig.suptitle("12x12 Square Graph: Fourier modes")
```

This was so surprising to me that I was convinced my derivation of the eigenfunctions was wrong. To sanity check, we can compute the analytic forms derived above and take a look:

```python
analytic_U = np.zeros((N1*N2, N1*N2))
for k in range((N1*N2)):
    analytic_U[:, k] = np.exp(2.j * np.pi * k * np.arange(N1*N2)/(N1*N2))

fig, axs = plt.subplots(4, 4)
for i, ax in enumerate(axs.flat):
    ax.matshow(analytic_U[:, i].reshape(N1, N2))
fig.suptitle("12x12 Square Graph: Eigenvector modes (analytic)")
```
![square_eigenvec_12x12_analytic](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/433c2f16-4684-4536-a54c-4fc094f2c6b5)

The eigenfunctions are in a different order (I'm not totally sure about the order of the numerical eigen-decomposition performed in `pygsp`) but I'm convinced the math checks out.

The periodic boundary conditions are crucial for this to work. To demonstrate this, let's plot the graph Laplacian eigenfunctions *without* the periodic boundary conditions. In this case we'll use the `pygsp.graphs.Grid2d` class:

```python
N1, N2 = 12, 12
sg = pygsp.graphs.Grid2d(N1, N2)
fig, axs = plt.subplots(4, 4)
for i, ax in enumerate(axs.flat):
    ax.matshow(sg.U[:, i].reshape(N1, N2))
fig.suptitle("12x12 Square Graph: Eigenvector modes (no periodic boundary conditions)")
```
![12x12_no_periodic](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/f6442641-9a57-44ec-b5f2-5388c136fc9d)

These basis functions are totally different. Visually, they seem to encode spatial information about the graph's structure, but they aren't the Fourier modes. In fact, they more closely resemble the [Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) basis:

![dct](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/65c0c6d6-a407-4cdb-8b2d-f044011410d5)
>image courtesy of [Wikimedia](https://commons.wikimedia.org/wiki/File:DCT-8x8.png)

We can also see differences in the graph spectrum (the Laplacian eigenvalues). Here we compare the eigenvalues derived analytically above, the eigenvalues of the graph with periodic boundary conditions added and those of the graph without boundary conditions. (the first two are identical):

```python
eigs_noboundary = np.linalg.eigvals(sg.L.toarray())
eigs_boundary = np.linalg.eigvals(psg.L.toarray())
eigs_analytic = np.sort(
    np.array(
        [4-2*np.cos(2*np.pi*k/(N1**2)) - 2*np.cos(2*np.pi*k/N1) 
         for k in range(N1**2)]
    )
)

fig, ax = plt.subplots()
ax.plot(np.sort(eigs_noboundary)[::-1], label="numerical - no boundary conditions")
ax.plot(np.sort(eigs_boundary)[::-1], label="numerical - with periodic boundary conditions")
ax.plot(np.sort(eigs_analytic)[::-1], label="analytic - with periodic boundary conditions")
fig.suptitle("Graph Laplacian eigenvalues: 12x12 Square Grid Graph")
ax.legend()
```
![eigvals_comparison12x12](https://github.com/chris-langfield/chris-langfield.github.io/assets/34426450/ddbb7b04-59a5-4ae4-986f-3194b9d2ad8d)

# The graph Fourier transform on a triangular lattice and the hexagonal DFT

Lastly, because we know [I love hexagons](https://chris-langfield.github.io/hex), I want to work out the graph signal processing for a hexagonal grid and compare with the hexagonal discrete Fourier transform. 

We can represent a signal with 2D hexagonal sampling as being a graph signal over a [triangular grid graph](https://mathworld.wolfram.com/TriangularGridGraph.html). `PyGSP` does not currently implement this graph so I messily rolled my own as follows. I set the XY coordinates of the vertices for visualization using grid generation from `hexfft`. 

```python
from hexfft.array import generate_grid
import scipy
import pygsp
import numpy as np

def TriangleGrid2d(N1, N2, periodic=True):
    W = np.zeros((N1*N2, N1*N2))

    if periodic:
        c0 = np.zeros(N1*N2)
        c0[[1, N2-1, N2, N1*N2-N2, N1*N2-N2+1, N1*N2 - 1]] = 1.
        c1 = np.zeros(N1*N2)
        c1[[1, N2, N2+1, N1*N2-N2-1, N1*N2-N2, N1*N2-1]] = 1.

        even_block = scipy.linalg.circulant(c0).T
        odd_block = scipy.linalg.circulant(c1).T

        for i in range(N1):
            idx = slice(i*N1,(i+1)*N1)
            if i % 2 == 0:
                W[idx, :] = even_block[idx, :]
            else:
                W[idx, :] = odd_block[idx]

    else:
        for i in range(N1*N2-1):
            j = i + 1
            if j % N2 == 0:
                continue
            else:
                W[i, j] = 1.
                W[j, i] = 1.

        for i in range(N1):
            if i == 0:
                W[i, N2] = 1.
                for j in range(1, N2):
                    W[i + j, N2 + j - 1: N2 + j + 1] = 1.
            elif i == N1 - 1 and i % 2 == 0:
                W[i*N2, i*N2 - N2] = 1.
                for j in range(1, N2):
                    W[i*N2 + j, -N2 + i*N2 + j - 1: -N2 + i*N2 + j + 1] = 1.
            elif i % 2 == 0:
                W[i*N2, i*N2 + N2] = 1.
                W[i*N2, i*N2 - N2] = 1.
                for j in range(1, N2):
                    W[i*N2 + j, N2 + i*N2 + j - 1: N2 + i*N2 + j + 1] = 1.
                    W[i*N2 + j, -N2 + i*N2 + j - 1: -N2 + i*N2 + j + 1] = 1.
            
        W = W + W.T
        W[W > 0] = 1.

    x, y = generate_grid((N1, N2), "offset")
    coords = np.stack([x.flatten(), y.flatten()]).T
    return pygsp.graphs.Graph(W=W, coords=coords)
```

We learned from implementing the graph Fourier transform on the square lattice that boundary conditions are very important.  




