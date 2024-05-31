---
layout: post
title: "Graph Signal Processing and the hexagonal grid"
author: "Chris Langfield"
categories: math
tags: [math]
---

In this post I compare the [graph Fourier transform](https://en.wikipedia.org/wiki/Graph_Fourier_transform) with the standard discrete Fourier transform in 1D and on square and hexagonal 2D grids. Primarily I wanted to explore some of the theoretical and computational building blocks for processing 2D images (square or hexagonal) using Graph Signal Processing, a relatively recent field in digital signal processing and machine learning. 

I'm using [PyGSP](https://pygsp.readthedocs.io/en/stable/) and my own [hexfft](https://github.com/chris-langfield/hexfft). See the bottom of the post for a list of resources I consulted.

# Graph Signal Processing

Graph signal processing is a framework for processing signals whose domains are the vertices of graphs. There are already many excellent introductions to the topic ([1](https://arxiv.org/abs/1211.0053), [2](https://infoscience.epfl.ch/record/256648?ln=en), [3](https://sybernix.medium.com/introduction-to-graph-signal-processing-ab9c0fde4d51), [4](https://balcilar.medium.com/struggling-signals-from-graph-34674e699df8)) so I'll just very quickly summarize: A graph $\mathcal{G}$ can be defined as

$$
\mathcal{G} = \\{ \mathcal{V}, \mathcal{E}, \mathbf{W} \\}
$$

where $\mathcal{V}$ is a set of vertices and $\mathcal{E}$ is a set of edges between the vertices. If the graph $\mathcal{G}$ has $V = |\mathcal{V}|$ vertices, the *adjacency matrix* $\mathbf{W}$ is a $V$ by $V$ matrix where $W_{ij}$ is nonzero when vertices $i$ and $j$ have an edge connecting them, and zero otherwise. To start with, I'm only considering *unweighted* graphs, where all the nonzero entries of $\mathbf{W}$ are 1. Additionally, graphs can be *directed* meaning that there can be a directed edge from vertex $i$ to vertex $j$ but not the other way around, and that the edges could have different weights. For now I consider only undirected graphs (the adjacency matrix $\mathbf{W}$ is symmetric in this case). 

For example, 

