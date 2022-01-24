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


def log_cut(intercept, gradient, i, j, x_hat, M):
    """
    Logging for debugging purposes for cuts
    """
    print("cut: "
     + "eta >= " + str(intercept)
     + " + " + str(gradient[0]) + " * x[" + str(j) + ", 0] - " + str(x_hat[0])
     + " + " + str(gradient[1]) + " * x[" + str(j) + ", 1] - " + str(x_hat[1])
     + " - " + str(M) + " * (1 - z[" + str(i) + ", " + str(j) + "])")

def print_solution(U, max_min_distance, j, points):
    """
    Print out a solution - only for approximation algorithm (greedy)
    """
    print("centers:")
    for u in U:
        print(points[u])

    print("")
    log(["farthest point from center", points[j]], ["distance", max_min_distance])
    print("")


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
            - objective value - max min distance
            - j - the point that is farthest from its center
    """
    # initializations
    log(["greedy algorithm", "\n"])
    m = len(points)
    U = set()
    U_bar = set(range(m))
    # select a point a_i to be the center with the max L_i (j is an index)
    j = 0
    # add the point index to the set of centers
    U.add(j)
    U_bar.remove(j)
    iteration = 1

    # precompute distance matrix, get max distance
    distance_matrix, max_distance = pairwise_distances(points)

    while len(U) < k:

        if debug:
            log(["iteration", iteration], ["U", U], ["not in U", U_bar])

        j, max_min_distance = next_index(U, U_bar, distance_matrix, max_distance, m)

        if debug:
            log(["next j", j], ["max-min-distance", max_min_distance])

        U.add(j)
        U_bar.remove(j)
        iteration += 1

        if debug:
            print("\n")

    # use next_index just to find objective value (don't actually need another index)
    j, max_min_distance = next_index(U, U_bar, distance_matrix, max_distance, m)

    return U, max_min_distance, j


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


def constraint_points(point, alpha):
    """
    compute gradients to add initial constraints defining linear lower bounds
        - in:
            - point (np array) - this defines the function and minimizer
            - alpha (int) - step size
            - n (int) - number of relative points (+n and -n) (just two dimensions now)
        - out:
            - relative_points (list) - all the relative points at which we'll add cuts
            - there will be 2n relative points
    """

    constraint_points = []

    # dimension 0
    increment = np.array([alpha, 0])
    # add the relative point (point + increment) 
    constraint_points.append(point + increment)
    # add the relative point (point - increment) 
    constraint_points.append(point - increment)

    # dimension 1
    increment = np.array([0, alpha])
    # add the relative point (point + increment) 
    constraint_points.append(point + increment)
    # add the relative point (point - increment) 
    constraint_points.append(point - increment)

    return constraint_points


def prep_cut(xhat, a_i):
    """
    prep an 'optimality cut' to master model
    inputs:
        - parameters xhat (relative point), point a_i (to define function_i)
    output:
        - returns intercept and gradient for affine rhs
    """
    # compute affine function parameters
    intercept = euclidian_distance(a_i, xhat)
    gradient_slope = gradient_euclidian_distance(a_i, xhat)

    # return the rhs (data) for the cut (eta \geq rhs)
    return intercept, gradient_slope

def add_linear_lower_bounds(oa_model, eta, x, z, alphas, points, m, k, M, debug):
    """
    add initial linear approximations at 'a few' points around global minimizer
    Add constraints for every f_i at the relative points:
    (x_hat) a_i - alpha e_j and a_i + alpha e_j for j = 1,…,n
    alpha is some (small) scalar
    inputs:
        - oa_model, eta, x, z, so we can add cuts directly here
        - alphas - list of alphas (step sizes)
        - points - the m points (instance)
        - m, k, M - ints
    output:
        - returns nothing, just adds cuts to oa_model
    """
    for alpha in alphas:
        for i in range(m):
            # compute the relative points for this i (x_hats)
            relative_points = constraint_points(points[i], alpha)
            if debug: log(["point", str(points[i])], ["relative_points", relative_points])

            for x_hat in relative_points:
                # compute gradient and intercept for cut
                intercept, gradient = prep_cut(x_hat, points[i])
                if debug: log(["x_hat", x_hat], ["intercept", intercept], ["gradient", gradient])

                for j in range(k):
                    oa_model.addConstr(
                     eta
                     >= intercept
                     + (gradient[0] * (x[j, 0] - x_hat[0]))
                     + (gradient[1] * (x[j, 1] - x_hat[1]))
                     - M * (1 - z[i, j])
                    )

                    if debug: log_cut(intercept, gradient, i, j, x_hat, M)

                if debug: print("\n")

            if debug: print("\n")


def initialize_oamodel(eta_lower, points, k, m, name, debug):
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
    # initialize U and M
    U = [[points[i]] for i in range(m)]
    distances, M = pairwise_distances(points)

    if debug:
        log(["M", M], ["U", U])

    # initialize model
    oa_model = Model("OA")
    oa_model.Params.lazyConstraints = 1

    if ~(debug):
        oa_model.setParam("OutputFlag", 0)

    # initialize eta variable
    eta = oa_model.addVar(vtype=GRB.CONTINUOUS, obj=1, name="eta")
    x = {}
    z = {}

    # add box constraints (simply lower and upper bounds in this case)
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

    # add constraints for each center to be assigned at least one point
    for j in range(k):
        oa_model.addConstr(quicksum(z[i, j] for i in range(m)) >= 1)

    alphas = [1]

    if debug: log(["adding initial linear approximation cuts", "\n"])

    add_linear_lower_bounds(oa_model, eta, x, z, alphas, points, m, k, M, debug)

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
    oa_model._debug = debug

    if debug: log(["initialized model", "\n"])

    return oa_model


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
        debug = model._debug

        if debug: print("separation algorithm"); log(["U", U])

        # first we find an i, xhat_l, and j where a would-be-cut is tight
        for i in range(m):
            for xhat_i in U[i]:
                new_points = []
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

                    if lhs == rhs:

                        if debug: log(["xl_hat", xl_array])

                        # when we find a tight cut add xhat_l to U_i
                        new_points.append(xl_array)
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

                            if debug: log_cut(intercept, gradient_slope, i, j, xl_array, M)

            model._U[i].extend(new_points)

        if debug: print("\n")

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
    print("outer approximation algorithm")
    # initialize the model with variables, lower bound and set-partitioning constraints
    oa_model = initialize_oamodel(0, points, k, len(points), name, debug)

    # optimize, passing callback function to model
    oa_model.optimize(separation_algorithm)

    # print solution
    if oa_model.status == GRB.Status.OPTIMAL:
        x = oa_model._x
        eta = oa_model._eta
        z = oa_model._z

        print("centers:")
        for j in range(k):
            print(
                "["
                + str(x[j, 0].x)
                + ", "
                + str(x[j, 1].x)
                + "]"
            )

            if debug:
                print("assigned: ")
                for i in range(oa_model._m):
                    if z[i, j].x > 0.5: print(' point - ' + str(points[i]))
                print("\n")
        print("\n")
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
    "c_scaling": [1 for i in range(3)],
    "points": [np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 3, 0])],
}

obvious_clusters = {
    "k": 5,
    "l_constants": [1 for i in range(20)],
    "c_scaling": [1 for i in range(20)],
    "points": [
        np.array([0, 0, 0]),
        np.array([0, 0, 1]),
        np.array([0, 0, 2]),
        np.array([0, 0, 3]),
        np.array([0, 10, 0]),
        np.array([0, 10, 1]),
        np.array([0, 10, 2]),
        np.array([0, 10, 3]),
        np.array([0, 20, 0]),
        np.array([0, 20, 1]),
        np.array([0, 20, 2]),
        np.array([0, 20, 3]),
        np.array([0, 30, 0]),
        np.array([0, 30, 1]),
        np.array([0, 30, 2]),
        np.array([0, 30, 3]),
        np.array([0, 40, 0]),
        np.array([0, 40, 1]),
        np.array([0, 40, 2]),
        np.array([0, 40, 3]),
    ],
}

if __name__ == "__main__":
    # TESTING APPROXIMATION
    U, max_min_distance, j = greedy_algorithm(
        obvious_clusters["k"],
        obvious_clusters["l_constants"],
        obvious_clusters["points"],
        False,
    )
    print_solution(U, max_min_distance, j, obvious_clusters["points"])

    # TESTING OA
    outer_approximation(
        obvious_clusters["k"],
        obvious_clusters["l_constants"],
        obvious_clusters["points"],
        "obvious-clusters-initial.lp",
        True,
    )
    # outer_approximation(
    #     triangle["k"],
    #     triangle["l_constants"],
    #     triangle["points"],
    #     "triangle-initial.lp",
    #     True,
    # )

    # SMALL TESTS
    # test_points = [
    #     np.array([0, 1]),
    #     np.array([0, 3]),
    # ]

    # prep_cut(test_points[0], test_points[1])
