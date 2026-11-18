Gaussian Mixture Models
=======================

Mixtures of normal distributions fitted with NUTS. The likelihood of a mixture
does not change when its components are relabelled, so the posterior has one
copy of every mode per labelling. The examples break this symmetry with an
``Ordered`` prior on the component means, which keeps all chains on the same
labelling and lets R-hat and the other diagnostics be read as usual.

* *Two Component Gaussian Mixture* uses the ``Mixture`` likelihood, which sums
  over the components of every observation.
* *Mixture with Enumerated Assignments* writes the component label of every
  observation as a discrete parameter, sums it out with enumeration and
  recovers the probability that each observation belongs to each component.
