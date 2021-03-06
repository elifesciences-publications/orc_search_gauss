
###This code is associated with the paper from Igaev et al., "Microtubule assembly governed by tubulin allosteric gain in flexibility and lattice induced fit". eLife, 2018. http://dx.doi.org/10.7554/eLife.34353

# orc_search_gauss.py
A Python-based script for optimal reaction coordinate search. The method requires that projections of the trajectory (trajectories) onto a N-dimensional vector basis (*e.g.*, obtained by Principal Component Analysis) be provided.

# Installation
The script is Python-based -- hence, no istallation is needed. However, some dependencies are needed:
* `numpy`
* `scipy`
* `matplotlib`
* `overlap_NEW.so`

The library `overlap_NEW.so` has to be compiled from the Cython module `overlap_NEW.pyx` manually prior to the script usage.
```
python setup_overlap_NEW.py build_ext --inplace
```
