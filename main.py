import numpy as np
from numpy import linalg as lg
from gurobipy import *


def log(*args):
    """
    Logging for debugging purposes, will print each item with label on a new line
    Please pass an iterable of pairs (tuples, sets whatever as long as its of size 2)
    """
    for arg in args:
        print(arg[0] + ": ", arg[1])


def print_solution(U, points):
    """
    Print out a solution
    """
    print("centers:")
    for u in U:
        print(points[u])


def euclidian_distance(x, y):
    """
    Evaluate euclidian distance for two points x and y
    """
    return np.linalg.norm(x - y)


def pairwise_distances(points):
    """
    Return a matrix of pairwise distances between each point
    Also return the max distance
    """
    m = len(points)
    distance_matrix = np.empty((m, m))
    max_distance = -1

    for i in range(m):
        for j in range(m):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance = euclidian_distance(points[i], points[j])
                distance_matrix[i][j] = distance
                if distance > max_distance:
                    max_distance = distance

    return distance_matrix, max_distance


def next_index(U, U_bar, distance_matrix, max_distance, m):
    """
    Return next index in loop (max_{[m] \ U} min_{u in U} (L_l /2) * euclidian_distance(x_u, x_l))
    """
    minimums = [-1 for i in range(m)]

    # construct a list of the closest existing center for each point in U_bar
    for l in U_bar:
        temp_min = max_distance + 1
        temp_min_index = -1
        for u in U:
            if distance_matrix[l][u] < temp_min:
                temp_min = distance_matrix[l][u]
                temp_min_index = u
            minimums[l] = temp_min

    # find the point in U_bar that is farthest from its closest index
    temp_max = -1
    temp_max_index = -1
    for i in range(m):
        if minimums[i] > temp_max:
            temp_max_index = i
            temp_max = minimums[i]

    return temp_max_index, temp_max


def greedy_algorithm(k, l_constants, points, debug=False):
    """
    Main procedure:
        in :
            - number of clusters (k) - int value
            - lipshitz constants (l_i) - list of floats
            - points (a_i) - list of np.arrays (shape = (n,))
            - minimizers (x_i) - to be added
        out :
            - set of centers u - set of ints (indexes)
    """
    # initializations
    m = len(points)
    U = set()
    U_bar = set(range(m))
    # select a point a_i to be the center with the max L_i (j is an index)
    j = 0
    # add the point index to the set of centers
    U.add(j)
    U_bar.remove(j)

    # precompute distance matrix, get max distance
    distance_matrix, max_distance = pairwise_distances(points)

    while len(U) < k:
        if debug:
            log(["U", U], ["not in U", U_bar])
        j, max_min_distance = next_index(U, U_bar, distance_matrix, max_distance, m)
        if debug:
            log(["next j", j], ["max-min-distance", max_min_distance])
        U.add(j)
        U_bar.remove(j)

    return U

def initialize_oamodel(eta_lower, k, m, name):
    '''
    Initialize the master model
        - in:
            - eta_lower - lower bound to set an initial constraint
            - ints k, m, number of policies and functions
    '''
    # global list of xhat points for every i in [m]
    U = [[] for i in range(m)]

    # initialize model
    oa_model = Model('OA')
    oa_model.Params.lazyConstraints = 1

    # initialize eta variable
    eta = oa_model.addVar(vtype = GRB.CONTINUOUS, obj = 1, name = 'eta')
    x = {}
    z = {}

    # initialize x_i variables - centers
    for j in range(k):
        x[j] = oa_model.addVar(vtype = GRB.BINARY, name = 'x_'+str(j))

    # add variables z_ij - assigning point i to cluster j
    for i in range(m):
        for j in range(k):
            z[i, j] = oa_model.addVar(vtype = GRB.BINARY, name = 'z_'+str(i)+'_'+str(j))

    # add initial sanity constraint eta \geq initial lower bound
    oa_model.addConstr(eta >= eta_lower, name = 'initial_constr')

    # add constraints for z_ij to sum to 1 over js, for every i
    for i in range(m):
        oa_model.addConstr(quicksum(z[i, j] for j in range(k)) == 1)

    # update model and write to initial file for debug
    oa_model.update()
    oa_model.write(name)

    # load data into the model for callback - variables, and U
    oa_model._eta = eta
    oa_model._z = z
    oa_model._x = x
    oa_model._U = U

    print("initialized model")

    return oa_model

def prep_cut(xhat_i, x_j, a_i):
    '''
    Prep an 'optimality cut' to master model
    Specific variables for the metric example (fi(x): euclidian_distance(ai, x))
        -fi_xhat_i: euclidian_distance(ai, xhat_i)
        -fi_xhat_i_gradient: this is just 2xhat_i
        -fxgi_transpose_xj_xhi: (fi_xhat_i_gradient)^T dot (x_j - xhat_i)
        -rhs: fi_xhat_i + fxgi_transpose_xj_xhi
    Return the rhs - we'll add the cut in the main callback algorithm
    '''
    print(xhat_i, x_j, a_i)
    # compute rhs
    fi_xhat_i = euclidian_distance(a_i, xhat_i)
    fi_xhat_i_gradient = 2 * xhat_i
    fxgi_transpose_xj_xhi = np.dot(fi_xhat_i_gradient, (x_j - xhat_i))
    rhs = fi_xhat_i + fxgi_transpose_xj_xhi

    print(fi_xhat_i, fi_xhat_i_gradient, fxgi_transpose_xj_xhi, rhs)

    # return the RHS for the cut (which will simply be eta \geq rhs)
    return rhs
    pass

def separation_algorithm(model, where):
    '''
    Add a cut for every tight variable (z_ij == 1) in current incumbent
    '''
    # when we have an incumbent (check that MIPSOL doesn't need to be MINLPSOL or something)
    if where == GRB.Callback.MIPSOL:
        # retrieve necessary variables
        x_sol = oa_model.cbGetSolution(oa_model._x)
        z_sol = oa_model.cbGetSolution(oa_model._z)
        eta_sol = oa_model.cbGetSolution(oa_model._eta)

        # separation algorithm
        # for i in range(m):
        #     for xhat_i in U[i]:
        #         for j in range(k):
        #             if (): continue

def outer_approximation(k, l_constants, points, name, debug=False):
    '''
    Exact algorithm
        in :
            - number of clusters (k) - int value
            - lipshitz constants (l_i) - list of floats
            - points (a_i) - list of np.arrays (shape = (n,))
            - minimizers (x_i) - to be added
        out :
            - set of centers u - set of ints (indexes)
    '''
    # initialize the model with variables, lower bound and set-partitioning constraints
    oa_model = initialize_oamodel(0, k, len(points), name)
    print(oa_model)


    # optimize, passing callback function to model
    oa_model.optimize(add_cut)

# directories
_DAT = "dat"

# some test instances
triangle = {
    "k": 2,
    "l_constants": [1 for i in range(3)],
    "points": [np.array([0, 0]), np.array([0, 1]), np.array([1, 1])],
}

obvious_clusters = {
    "k": 5,
    "l_constants": [1 for i in range(20)],
    "points": [
        np.array([0, 1]),
        np.array([0, 2]),
        np.array([0, 3]),
        np.array([0, 4]),
        np.array([1000, 1]),
        np.array([1000, 2]),
        np.array([1000, 3]),
        np.array([1000, 4]),
        np.array([2000, 1]),
        np.array([2000, 2]),
        np.array([2000, 3]),
        np.array([2000, 4]),
        np.array([3000, 1]),
        np.array([3000, 2]),
        np.array([3000, 3]),
        np.array([3000, 4]),
        np.array([4000, 1]),
        np.array([4000, 2]),
        np.array([4000, 3]),
        np.array([4000, 4]),
    ],
}


if __name__ == "__main__":
    # U = greedy_algorithm(
    #     obvious_clusters["k"],
    #     obvious_clusters["l_constants"],
    #     obvious_clusters["points"],
    #     True,
    # )
    # print_solution(U, obvious_clusters["points"])
    # outer_approximation(
    #     triangle["k"],
    #     triangle["l_constants"],
    #     triangle["points"],
    #     "triangle-initial.lp",
    #     True,
    # )

    test_points = [
        np.array([0, 1]),
        np.array([0, 2]),
        np.array([0, 3]),
    ]

    prep_cut(test_points[0], test_points[1], test_points[2])

