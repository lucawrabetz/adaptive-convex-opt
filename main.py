import numpy as np
from gurobipy import *
from numpy import linalg as lg


def log(*args):
    """
    Logging for debugging purposes, will print each item with label on a new line
    Please pass an iterable of pairs (tuples, sets whatever as long as its of size 2)
    """
    for arg in args:
        print(arg[0] + ": ", arg[1])


def print_solution(U, points):
    """
    Print out a solution - only for approximation algorithm (greedy)
    """
    print("centers:")
    for u in U:
        print(points[u])


def euclidian_distance(x, y):
    """
    Evaluate euclidian distance for two points x and y
    """
    return (np.linalg.norm(x - y)) ** 2


def gradient_euclidian_distance(a, x_hat):
    """
    Return gradient value for euclidian distance function on points a and x_hat
    """
    return 2 * (x_hat - a)


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

def compute_box_bounds(points, m):
    '''
    Compute upper and lower bounds for x variables
    Loop through all points and maintain max and min value for each dimension
        Inputs:
            - the m points
        Outputs:
            - lb_zero, ub_zero: lower and upper bound on 0th dimension
            - lb_one, ub_one: lower and upper bound on first dimension
    '''
    lb_zero = np.inf
    lb_one = np.inf
    ub_zero = np.NINF
    ub_one = np.NINF

    for i in range(m):
        if points[i][0] > ub_zero: ub_zero = points[i][0]
        if points[i][0] < lb_zero: lb_zero = points[i][0]
        if points[i][1] > ub_one: ub_one = points[i][1]
        if points[i][1] < lb_one: lb_one = points[i][1]

    return lb_zero, lb_one, ub_zero, ub_one


def initialize_oamodel(eta_lower, points, k, m, name):
    """
    Initialize the master model
        - in:
            - eta_lower - lower bound to set an initial constraint
            - ints k, m, number of policies and functions
        - out:
            - initialized model oa_model
        - notes:
            - eta is just a single continuous GRBVAR
            - z is a multidict of binary GRBVARs, indexed point i to cluster j
            - x is a multicict of continuous GRBVARs, indexed center of j, dimension in n
    """
    # initialize U
    U = [[] for i in range(m)]
    for i in range(m):
        point = np.array(points[i])
        point += np.array([1, 1])
        U[i].append(point)

    print(U)

    # choose big M
    distances, M = pairwise_distances(points)
    print('M: ' + str(M))

    # initialize model
    oa_model = Model("OA")
    oa_model.Params.lazyConstraints = 1

    # initialize eta variable
    eta = oa_model.addVar(vtype=GRB.CONTINUOUS, obj=1, name="eta")
    x = {}
    z = {}

    lb_zero, lb_one, ub_zero, ub_one = compute_box_bounds(points, m)

    # initialize x_i variables - centers, and z_ij variables - point i assigned to cluster j
    for j in range(k):
        x[j, 0] = oa_model.addVar(
            vtype=GRB.CONTINUOUS, obj=0, lb=lb_zero, ub=ub_zero, name="x_" + str(j) + "_" + str(0)
        )
        x[j, 1] = oa_model.addVar(
            vtype=GRB.CONTINUOUS, obj=0, lb=lb_one, ub=ub_one, name="x_" + str(j) + "_" + str(1)
        )
        for i in range(m):
            z[i, j] = oa_model.addVar(
                vtype=GRB.BINARY, name="z_" + str(i) + "_" + str(j)
            )

    # add initial sanity constraint eta \geq initial lower bound
    oa_model.addConstr(eta >= eta_lower, name="initial_constr")

    # add constraints for z_ij to sum to 1 over js, for every i
    for i in range(m):
        oa_model.addConstr(quicksum(z[i, j] for j in range(k)) == 1)

    # one run of separation algorithm
    # for i in range(m):
    #     xhat_i = U[i][0]
    #     for j in range(k):
    #         # import pdb; pdb.set_trace()
    #         intercept, gradient_slope = prep_cut(xhat_i, points[i])
    #         oa_model.addConstr(
    #             eta
    #             >= intercept
    #             + (gradient_slope[0] * (x[j, 0] - xhat_i[0]))
    #             + (gradient_slope[1] * (x[j, 1] - xhat_i[1]))
    #             - M * (1 - z[i, j])
    #         )

    # update model and write to initial file for debug
    oa_model.update()
    oa_model.write(name)

    # load data into the model for callback - variables, and U
    oa_model._eta = eta
    oa_model._z = z
    oa_model._x = x
    oa_model._U = U
    oa_model._m = m
    oa_model._k = k
    oa_model._points = points
    oa_model._M = M

    print("initialized model")

    return oa_model


def prep_cut(xhat_i, a_i):
    """
    Prep an 'optimality cut' to master model
    Inputs:
        - parameters xhat_i, point a_i (to define function_i)
    Output:
        - returns intercept slope for affine rhs
    Specific intermediate variables for the metric example (fi(x): euclidian_distance(ai, x))
        -fi_xhat_i: euclidian_distance(ai, xhat_i)
        -fi_xhat_i_gradient: this is just 2xhat_i
    """
    # compute affine function parameters
    intercept = euclidian_distance(a_i, xhat_i)
    gradient_slope = gradient_euclidian_distance(a_i, xhat_i)
    # return the RHS for the cut (which will simply be eta \geq rhs)
    return intercept, gradient_slope


def separation_algorithm(model, where):
    """
    Add a cut for every current incumbent, if the cut is 'tight'
    """
    # when we have an incumbent (check that MIPSOL doesn't need to be MINLPSOL or something)
    if where == GRB.Callback.MIPSOL:
        # retrieve necessary variables
        x = model._x
        z = model._z
        eta = model._eta
        x_sol = model.cbGetSolution(x)
        z_sol = model.cbGetSolution(z)
        eta_sol = model.cbGetSolution(eta)
        U = model._U
        k = model._k
        m = model._m
        points = model._points
        M = model._M

        # separation algorithm
        # first we find an i, xhat_l, and j where a would-be-cut is tight
        for i in range(m):
            for xhat_i in U[i]:
                for l in range(k):
                    # import pdb; pdb.set_trace()
                    intercept, gradient_slope = prep_cut(xhat_i, points[i])
                    xl_array = np.array([x_sol[l, 0], x_sol[l, 1]])

                    lhs = eta_sol
                    rhs = (
                        intercept
                        + np.dot(gradient_slope, (xl_array - xhat_i))
                        - M * (1 - z_sol[i, l])
                    )

                    if lhs == rhs: continue

                    xl_array

                    # when we find a tight cut:
                    # add xhat_l to U_i
                    U[i].append(xl_array)

                    intercept, gradient_slope = prep_cut(xl_array, points[i])
                    # add a cut based on xhat_l, gradient_slope, intercept
                    # gradient_slope and intercept have been recomputed for xl_array
                    # add these cuts for every variable x_j
                    for j in range(k):
                        model.cbLazy(
                         eta
                         >= intercept
                         + (gradient_slope[0] * (x[j, 0] - xl_array[0]))
                         + (gradient_slope[1] * (x[j, 1] - xl_array[1]))
                         - M * (1 - z[i, j])
                        )
                        # print(
                        #  "added cut: intercept - "
                        #  + str(intercept)
                        #  + ", gradient - "
                        #  + "["
                        #  + str(gradient_slope[0])
                        #  + ", "
                        #  + str(gradient_slope[1])
                        #  + "] , j - "
                        #  + str(j)
                        #  + ", x_hat_l - "
                        #  + "["
                        #  + str(xl_array[0])
                        #  + ", "
                        #  + str(xl_array[1])
                        #  + "], i - "
                        #  + str(i)
                        #  + ":"
                        # )
                        print("cut: "
                         + "eta >= " + str(intercept)
                         + " + " + str(gradient_slope[0]) + " * x[" + str(j) + ", 0] - " + str(xl_array[0])
                         + " + " + str(gradient_slope[1]) + " * x[" + str(j) + ", 1] - " + str(xl_array[1])
                         + " - " + str(M) + " * (1 - z[" + str(i) + ", " + str(j) + "])")

                    # break out of separation algorithm
                    return


def outer_approximation(k, l_constants, points, name, debug=False):
    """
    Exact algorithm
        in :
            - number of clusters (k) - int value
            - lipshitz constants (l_i) - list of floats
            - points (a_i) - list of np.arrays (shape = (n,))
            - minimizers (x_i) - to be added
        out :
            - set of centers u - set of ints (indexes)
    """
    # initialize the model with variables, lower bound and set-partitioning constraints
    oa_model = initialize_oamodel(0, points, k, len(points), name)
    print(oa_model)

    # optimize, passing callback function to model
    oa_model.optimize(separation_algorithm)

    # print solution
    if oa_model.status == GRB.Status.OPTIMAL:
        x = oa_model._x
        eta = oa_model._eta
        z = oa_model._z

        for j in range(k):
            print(
                "center "
                + str(j + 1)
                + ", ["
                + str(x[j, 0].x)
                + ", "
                + str(x[j, 1].x)
                + "], assigned: "
            )
            for i in range(oa_model._m):
                if z[i, j].x == 1: print(' point - ' + str(points[i]))
        print("objective: " + str(oa_model.objVal))
        print("eta: " + str(eta.x))
    elif oa_model.status == GRB.Status.INFEASIBLE:
        print("Infeasible")
    elif oa_model.status == GRB.Status.UNBOUNDED:
        print("Unbounded")
    else:
        print("unkown error")


# directories
_DAT = "dat"

# some test instances
triangle = {
    "k": 2,
    "l_constants": [1 for i in range(3)],
    "points": [np.array([0, 0]), np.array([0, 1]), np.array([2, 2])],
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
    print("main loop")
    # TESTING APPROXIMATION
    # U = greedy_algorithm(
    #     obvious_clusters["k"],
    #     obvious_clusters["l_constants"],
    #     obvious_clusters["points"],
    #     True,
    # )
    # print_solution(U, obvious_clusters["points"])

    # TESTING OA
    # outer_approximation(
    #     obvious_clusters["k"],
    #     obvious_clusters["l_constants"],
    #     obvious_clusters["points"],
    #     "obvious-clusters-initial.lp",
    #     True,
    # )
    outer_approximation(
        triangle["k"],
        triangle["l_constants"],
        triangle["points"],
        "triangle-initial.lp",
        True,
    )

    # SMALL TESTS
    # test_points = [
    #     np.array([0, 1]),
    #     np.array([0, 3]),
    # ]

    # prep_cut(test_points[0], test_points[1])
