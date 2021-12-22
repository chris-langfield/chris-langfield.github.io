---
layout: post
title: "A 'wedding cake' probability distribution -- Part 1"
author: "Chris Langfield"
categories: math
tags: [math]
---

This is a writeup of some progress on a problem I first formulated a while back. The context was a bit silly. Practitioners of [procedural generation](https://www.reddit.com/r/proceduralgeneration/) think a lot about noise, specifically sampling from stochastic processes to create realistic looking heightmaps, textures, and other content. This [blog post](https://www.redblobgames.com/articles/noise/introduction.html) gives a sort of ground-up approach to the intuition of different types of noise, smoothing, etc. In one of the examples, the author does something interesting. Their first example is generating a 1-D heightmap by sampling uniformly from discrete values up to some maximum. In pseudocode,

```
for i=0, i < map_length, i++
  noise[i] = random_selection( {1, 2, 3, ... k} )
```

This predictably creates a jagged looking timeseries. The author proposes the following modification to smooth out the landscape:

```
for i=0, i < map_length, i++
  MAX = random_selection( {1, 2, 3, ... k} )
  noise[i] = random_selection( {1, 2, 3, ... MAX} )
```

On each iteration, we pick a random number uniformly, and then for our noise value, we sample uniformly from 1 up to that maximum. Naturally, this biases the noise towards lower values, resulting in a smoother heightmap.

The blog post goes on to describe noise colors, time-averaging, and other more advanced topics. But I was curious about this process of double-selection. What is the probability distribution for the integers 1 up through k when doing this?

### The simple case: double selection one time

Let's make the problem formal. We are first considering uniform selection from $$ S = \{1, 2, 3, ... k\} $$. Let $$X_0$$ be the corresponding random variable. Clearly $$X_0 \sim \unif(1,...k)$$. Now we use $$X_0$$ to define the sample space for a new random variable, $$X_1$$, sampling uniformly from the set $$\{s \in S: s \leq X_0\}$$. We want to find the probability mass function of $$X_1$$. 

Applying the law of total probability, we can write

$$ Pr(X_1=s) = \sum_{i=1}^{k} Pr(X_1=s, X_0=i) = \sum_{i=1}^{k} Pr(X_1=s | X_0=i)Pr(X_0=i) $$

We know $$Pr(X_0 = i) = \frac{1}{k}$$. We can also observe that

$$
Pr(X_2=s|X_1=i) = \begin{cases} 
    \frac{1}{i} & s\leq i \\
    0 & s > i \\
  \end{cases}
$$






