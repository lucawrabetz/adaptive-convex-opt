# Adaptive Convex Optimization

* Start with first approximation algorithm in section 2
    * 3 experiments 
    * 1 with simple quadratic data
    * 1 with more interesting quadratic data
    * 1 with a real dataset 
    * Start with a class for the k-center problem, make sure it works on that
    * Only parameter for this example is a_i, and L_i=1 for all
    * Try very far 'trivial' clusters
    * After the algorithm outputs centers, also assign each point to center

* Generating a_i for the class:
    * There are k clusters
    * Make there k clear clusters, maybe by perturbing k dummy centers, and controlling a distance parameter between them
