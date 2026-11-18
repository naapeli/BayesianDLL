Hierarchical Models
===================

Models whose parameters share structure: groups that borrow strength from
each other through a common population distribution, correlated group
effects, and models with many interacting weights. The examples show how
the choice of coordinates decides whether NUTS samples well. They compare
the centered and the non-centered (``Affine``) forms of a group effect and
explain when each one is preferable, build correlated effects from
``LKJCholesky`` and ``CholeskyCovariance``, and write the likelihoods with
the blocks in ``BayesianDLL.Deterministic`` (``Index``, ``MatMul``,
``Concat`` and others). Each example plots its fit against the data.
