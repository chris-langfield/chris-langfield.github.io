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

Let's make the problem formal. We are first considering uniform selection from $$ S = \{1, 2, 3, ... k\} $$. Let $$X_0$$ be the corresponding random variable. Clearly $$X_0 \sim unif(1,...k)$$. Now we use $$X_0$$ to define the sample space for a new random variable, $$X_1$$, sampling uniformly from the set $$\{s \in S: s \leq X_0\}$$. We want to find the probability mass function of $$X_1$$. 

Applying the law of total probability, we can write

$$ Pr(X_1=s) = \sum_{i=1}^{k} Pr(X_1=s, X_0=i) = \sum_{i=1}^{k} Pr(X_1=s | X_0=i)Pr(X_0=i) $$

We know $$Pr(X_0 = i) = \frac{1}{k}$$. We can also observe that

$$
Pr(X_1=s|X_0=i) = \begin{cases} 
    \frac{1}{i} & s\leq i \\
    0 & s > i \\
  \end{cases}
$$

Filling in this information, we can say:

$$ Pr(X_1=s) = \frac{1}{k} \sum_{i=s}^k \frac{1}{i} $$

With some algebra and using some identities about harmonic numbers, we can rewrite this as

$$ Pr(X_1=s) = \frac{1}{k}(H_k-H_{s-1}) $$

and further show that it is indeed a probability distribution on $$S$$. 

Here's what this distribution looks like for $$k=30$$, contrasted with uniform selection. 

![uniformandp1](https://user-images.githubusercontent.com/34426450/147142098-766a71e9-15be-434c-a8a4-213b139865d2.png)

It does smooth out the distribution along a nice curve.

Just as in the blog post, we can sample from this distribution and compare to sampling uniformly (uniform on top, our double-selection method on the bottom). Here $$k=10$$.

![Screen Shot 2021-12-22 at 2 47 29 PM](https://user-images.githubusercontent.com/34426450/147147232-ec35b4ec-24e2-4a17-8a6d-1e61e6618296.png)

Nothing too crazy, but it does looks smoother. You could perhaps imagine doing this if you wanted a scrolling city skyline in the background of a game. I would be very curious to look into the properties of the spectra of these distributions. The double-selection process is still uncorrelated, so "white" noise, but it seems like some low pass filtering is being done here somehow (don't quote me). 

I took this in a different, more impractical, direction though. I wanted to know how things would look if we repeated this selection process $$m$$ times. That is, pick $$X_0$$ uniformly from $$S$$, then pick $$X_1$$ uniformly from $$\{1 ... X_0 \}$$, then pick $$X_2$$ uniformly from $$\{1 ... X_1\}$$, and so on. What is the probability distribution of $$X_m$$? Clearly, this process would converge to picking 1 with probability 1 at some point. So as $$m$$ approaches $$k$$ and beyond, the distribution becomes degenerate. But for $$m < k$$, it seems like we could define a sequence of distributions that bias more and more towards lower values in curves like the one we saw in the case $$m=1$$. What are these curves? I would like to know, but I haven't figured it out yet. My name for these are "wedding cake" distributions because they represent sampling repeatedly from a shrinking set of integers. 

Difficulties soon crop up when trying to find a general expression. We know that similarly to $$X_1$$, $$X_2$$'s distribution would satisfy:

$$Pr(X_2=s) = \sum_{i=1}^{k} Pr(X_2=s, X_1=i) = \sum_{i=1}^k Pr(X_2=s|X_1=i)Pr(X_1=i)$$

And substituting, again applying the fact that for certain values of $$i$$ and $$s$$

$$Pr(X_2=s|X_1=i) = 0$$

$$Pr(X_2=s) = \frac{1}{k} \sum_{i=s}^{k} \frac{1}{i} (H_k - H_{i-1})$$

This is not very nice, and I made no headway on simplifying it. This is where I left off a few years ago until I recently took another look at it. In Part 2, I'll go through finding a general expression for the "wedding cake" distributions that is workable, but still not totally satisfying.

Note: I would be very curious to know if there is literature already out there on this type of problem. I have not been able to find any.


