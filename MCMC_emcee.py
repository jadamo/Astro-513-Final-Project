# The emcee algorithm is an "Afine invariant ensemble sampler". The major
# difference to Metropolis-Hastings is that you use a suite of samplers of
# the parameter space you're using and evolve each sampler based on the
# state of the others
# Source: https://arxiv.org/pdf/1202.3665.pdf

# ---non-paralel Stretch-move algorithm---
# for k in (1, K):
#   get walker X_k
#   randomly get walker X_j (j != k)
#   get random number Z from distribution g(z)
#   Find potential new positon Y = X_j + Z[X_k - X_j]
#   get acceptance probability q (how?)
#   get random number r in [0, 1]
#   if r <= q update position to Y
#   else update position to X_k

# The way you're suppossed to parallelize the above is split the ensemble of
# samplers into 2 sets and evolve each of those in parallel

# ---paralel Stretch-move algorithm---
# for each half
#   for k in (1, K/2) <- can paralelize this loop
#       get walker X_k
#       randomly get walker X_j (j in [K/2, K] )
#       do the same as the non-parallel algorithm