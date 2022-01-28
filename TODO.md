# Todo adaptive convex optimization

## 12/23

- initialize each U_i with a_i
- add box constraints to x variables (need to right a small function to define the box)
- change the separation to choose one point in the current incumbent where the would be cut is tight - use that point as xhat and add a cut

## Early January

- fixed numerical issues by adding more initial linear approximation constraint

## Experiments

### Scaled metric problem

- cleanup first: fix all prints, add debug option - refactor long functions
- generalize 2 dimensional stuff to n dimensions
