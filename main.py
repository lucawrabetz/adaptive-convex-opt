import json
import sys
import os
import time
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gurobipy import *
from numpy import linalg as lg
from regression_generation import gen_ls_data

# DIRECTORIES
DAT = "dat"
isdir = os.path.isdir(DAT)
if not isdir:
    os.mkdir(DAT)
MODELS = os.path.join(DAT, "models")
EXPERIMENTS = os.path.join(DAT, "experiments")
isdir = os.path.isdir(MODELS)
if not isdir:
    os.mkdir(MODELS)
isdir = os.path.isdir(EXPERIMENTS)
if not isdir:
    os.mkdir(EXPERIMENTS)

# SIMPLE INSTANCE NAMES
TRIANGLE2 = "triangle_2d"
TRIANGLE3 = "triangle_3d"
OBVIOUS_CLUSTERS2 = "obvious_clusters_2d"
OBVIOUS_CLUSTERS3 = "obvious_clusters_3d"


# COLUMN LIST FOR EXPERIMENT RESULTS - ALWAYS USE ALL OF THEM, NAN IF CALC NOT INCLUDED
COLUMNS = ["instance_id", "problem_type", "k", "n", "m", "kappa", "obj_greedy", "time_greedy", "obj_mip", "time_mip", "obj_qp", "time_qp", "g_mip_ratio", "g_qp_ratio", "mip_qp_ratio"]
PROBLEM_TYPES = {0: "scaled_k_center",
                 1: "batch_regression",
                 2: "logistic_regression"}

def log(*args):
    """
    Logging for debugging purposes, will print each item with label on a new line
    Please pass an iterable of pairs (tuples, sets whatever as long as its of size 2)
    """
    for arg in args:
        print(arg[0] + ": ", arg[1])


def log_sep(num_lines=1):
    """
    Small utility logging to separate blocks in printin
    """
    print("\n")
    for i in range(num_lines):
        print("<   -------------------------------------------------   >")
    print("\n")


def log_cut(intercept, gradient, i, j, x_hat, M):
    """
    Logging for debugging purposes for cuts
    """

    middle_part = ""
    for n in range(x_hat.shape[0]):
        temp_string = (
            " + "
            + str(gradient[n])
            + " * (x["
            + str(j)
            + ", "
            + str(n)
            + "] - "
            + str(x_hat[n])
            + ")"
        )
        middle_part += temp_string

    print(
        "cut: "
        + "eta >= "
        + str(intercept)
        + middle_part
        + " - "
        + str(M)
        + " * (1 - z["
        + str(i)
        + ", "
        + str(j)
        + "])"
    )


def log_instance(instance):
    """
    Logging for debugging purposes for instances
    """
    log(["instance", instance["name"]])
    log(
        ["k", instance["k"]],
        ["n", len(instance["points"][0])],
        ["m", len(instance["points"])],
    )
    log(["C", instance["c_scaling"]])
    log(["points", instance["points"]])
    print("\n")


def print_solution(centers, max_min_distance, i, j, points):
    """
    Print out a solution
    """
    print("centers:")
    for u in centers:
        print(u)

    print("\n")
    log(
        ["farthest point from center", points[i]],
        ["center", centers[j]],
        ["distance", max_min_distance],
    )


def compute_constants(instance):
    """
    Calculates the lipschitz and convexity constants, returns a list with the Lipschitz constants first and convexity constants second
    parameters:
    self is used to get the A_list and B_list

    """
    # Initialize list of lipschitz constants
    m = len(instance["b_list"])
    A_list = instance["A_list"]
    L = []
    mu = []
    kappa = 0
    for i in range(m):
        # Calculate the matrix for eigenvalue calculation
        Eig_Calc = np.matmul(np.transpose(A_list[i]), A_list[i])
        Eigenvalues = np.linalg.eig(Eig_Calc)[0]
        L.append(max(Eigenvalues))
        mu.append(min(Eigenvalues))
        k_temp = max(Eigenvalues) / min(Eigenvalues)
        if k_temp > kappa:
            kappa = k_temp

    instance["constants"] = [L, mu, kappa]


def euclidian_distance(a, x, C):
    """
    Evaluate euclidian distance for two points x and y
        - convention: C is the scaling factor for point (i) which is x
    """
    return (C / 2) * ((lg.norm(x - a)) ** 2)


def gradient_euclidian_distance(a, x_hat, C):
    """
    Return gradient value for euclidian distance function on points a and x_hat
        - note: order (of a, x_hat), does matter here - x_hat is the input point (x), a is the point that defines the function f_i (a_i)
    """
    return C * (x_hat - a)


def regression_value(A, b, x):
    """
    Evaluate regression error for A, x, b
        - note that in the algorithms, A, b should be fixed for the function f_i
        - x is the input
        - 0.5 * because jourdain had it in his code - also putting it in MIP and qp
    """
    return (0.5 * (lg.norm(np.matmul(A, x) - b)) ** 2)


def pairwise_distances(points, c_scaling):
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
                distance = euclidian_distance(points[i], points[j], c_scaling[i])
                distance_matrix[i][j] = distance
                if distance > max_distance:
                    max_distance = distance

    return distance_matrix, max_distance


def pairwise_distances_regression(A_list, b_list, x_list):
    """
    Return a matrix of pairwise distances between each point
    Also return the max distance
    """
    m = len(A_list)
    distance_matrix = np.empty((m, m))
    max_distance = -1

    for i in range(m):
        for j in range(m):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance = regression_value(A_list[i], b_list[i], x_list[j])
                distance_matrix[i][j] = distance
                if distance > max_distance:
                    max_distance = distance

    return distance_matrix, max_distance


def objective_matrix(points, c_scaling, centers):
    """
    Return a matrix of distances between each point (rows) and each center (columns)
    Also return the max min distance (max distance of a point from its closest center)
    """
    m = len(points)
    k = len(centers)
    max_min_distance = -1

    distances = np.zeros((m, k))
    i_final = -1
    j_final = -1

    for i in range(m):
        for j in range(k):
            distances[i][j] = euclidian_distance(
                np.array(points[i]), np.array(centers[j]), c_scaling[i]
            )

    for i in range(m):

        minimum = np.inf
        j_temp = -1
        i_temp = -1

        for j in range(k):
            if minimum > distances[i][j]:
                minimum = distances[i][j]
                j_temp = j
                i_temp = i

        if max_min_distance < minimum:
            max_min_distance = minimum
            i_final = i_temp
            j_final = j_temp

    return distances, max_min_distance, i_final, j_final


def objective_matrix_regression(centers, A_list, b_list):
    """
    Return a matrix of distances between each point (rows) and each center (columns)
    Also return the max min distance (max distance of a point from its closest center)
    """
    m = len(A_list)
    k = len(centers)
    max_min_distance = -1

    distances = np.zeros((m, k))
    i_final = -1
    j_final = -1

    for i in range(m):
        for j in range(k):
            distances[i][j] = regression_value(
               A_list[i], b_list[i], np.array(centers[j])
            )

    for i in range(m):

        minimum = np.inf
        j_temp = -1
        i_temp = -1

        for j in range(k):
            if minimum > distances[i][j]:
                minimum = distances[i][j]
                j_temp = j
                i_temp = i

        if max_min_distance < minimum:
            max_min_distance = minimum
            i_final = i_temp
            j_final = j_temp

    return distances, max_min_distance, i_final, j_final


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


def greedy_algorithm(instance, debug=False):
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
    log(["greedy algorithm, instance", instance["name"] + "\n"])

    k = instance["k"]
    c_scaling = instance["c_scaling"]
    points = instance["points"]
    name = instance["name"]

    m = len(points)
    U = set()
    U_bar = set(range(m))
    # select a point a_i to be the center with the max L_i (j is an index)
    j = -1
    max_c = -1
    for i in range(m):
        if c_scaling[i] > max_c:
            j = i
            max_c = c_scaling[i]
    # add the point index to the set of centers
    U.add(j)
    U_bar.remove(j)
    iteration = 1

    # precompute distance matrix, get max distance
    distance_matrix, max_distance = pairwise_distances(points, c_scaling)

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

    centers = [points[u] for u in U]

    distance_matrix, max_min_distance, i_final, j_final = objective_matrix(
        points, c_scaling, centers
    )
    print("")
    print_solution(centers, max_min_distance, i_final, j_final, points)

    return max_min_distance


def greedy_algorithm_regression(instance, debug=False):
    """
    Main procedure:
        in :
            - number of clusters (k) - int value
            - matrices (A_list) - list of np.arrays
            - vectors (b_list) - list of np.arrays
            - minimizers (x_list) - to be added
        out :
            - set of centers u - set of ints (indexes)
            - objective value - max min distance
            - j - the point that is farthest from its center
    """
    # initializations
    log(["greedy algorithm, instance", instance["name"] + "\n"])

    k = instance["k"]
    A_list = instance["A_list"]
    b_list = instance["b_list"]
    points = instance["minimizers"]
    name = instance["name"]

    m = len(points)
    U = set()
    U_bar = set(range(m))
    # select a point a_i to be the center with the max L_i (j is an index)
    j = -1
    max_c = -1
    for i in range(m):
        if instance["constants"][0][i] > max_c:
            j = i
            max_c = instance["constants"][0][i]
    # add the point index to the set of centers
    U.add(j)
    U_bar.remove(j)
    iteration = 1

    # precompute distance matrix, get max distance
    distance_matrix, max_distance = pairwise_distances_regression(A_list, b_list, points)

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

    centers = [points[u] for u in U]

    distance_matrix, max_min_distance, i_final, j_final = objective_matrix_regression(
        centers, A_list, b_list
    )
    print("")
    print_solution(centers, max_min_distance, i_final, j_final, points)

    return max_min_distance


def compute_box_bounds(points, m, n):
    """
    Compute upper and lower bounds for x variables
    Loop through all points and maintain max and min value for each dimension
        Inputs:
            - the m points
        Outputs:
            - lb - list, lower bound for each dimension
            - ub - list, upper bound for each dimension
    """
    lb = [np.inf for l in range(n)]
    ub = [np.NINF for l in range(n)]

    for i in range(m):
        for l in range(n):
            if points[i][l] > ub[l]:
                ub[l] = points[i][l]
            if points[i][l] < lb[l]:
                lb[l] = points[i][l]

    return lb, ub


def constraint_points(point, alpha):
    """
    compute points at which initial constraints defining linear lower bounds will be added
        - in:
            - point (np array) - this defines the function and minimizer
            - alpha (int) - step size
            - n (int) - number of relative points (+n and -n) (just two dimensions now)
        - out:
            - relative_points (list) - all the relative points at which we'll add cuts
            - there will be 2n relative points
    """

    constraint_points = []

    for n in range(point.shape[0]):
        # create the increment vector (zeros with alpha at position n)
        increment = np.zeros_like(point)
        increment[n] = alpha

        # add the relative point (point + increment)
        constraint_points.append(point + increment)
        # add the relative point (point - increment)
        constraint_points.append(point - increment)

    return constraint_points


def prep_cut(a_i, x_hat, C_i):
    """
    prep an 'optimality cut' to master model
    inputs:
        - parameters - point a_i (to define function_i), x_hat (relative point/input to f())
    output:
        - returns intercept and gradient for affine rhs
    """
    # compute affine function parameters
    intercept = euclidian_distance(a_i, x_hat, C_i)
    gradient = gradient_euclidian_distance(a_i, x_hat, C_i)

    # return the rhs (data) for the cut (eta \geq rhs)
    return intercept, gradient


def add_linear_lower_bounds(
    oa_model, eta, x, z, alphas, points, c_scaling, m, k, M, debug
):
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
    N = points[0].shape[0]

    for alpha in alphas:
        for i in range(m):
            # compute the relative points for this i (x_hats)
            relative_points = constraint_points(points[i], alpha)
            if debug:
                log(["point", str(points[i])], ["relative_points", relative_points])

            counter = 0
            for x_hat in relative_points:
                # compute gradient and intercept for cut
                intercept, gradient = prep_cut(points[i], x_hat, c_scaling[i])
                if debug:
                    log(
                        ["point", str(points[i])],
                        ["x_hat", x_hat],
                        ["intercept", intercept],
                        ["gradient", gradient],
                    )
                for j in range(k):
                    constraint_name = (
                        "initial_linear_"
                        + str(i)
                        + "_"
                        + str(alpha)
                        + "_"
                        + str(j)
                        + "_"
                        + str(counter)
                    )
                    if debug:
                        log(["constraint name", constraint_name])

                    oa_model.addConstr(
                        eta
                        >= intercept
                        + quicksum(
                            (gradient[n] * (x[j, n] - x_hat[n])) for n in range(N)
                        )
                        - M * (1 - z[i, j]),
                        name=constraint_name,
                    )

                    if debug:
                        log_cut(intercept, gradient, i, j, x_hat, M)

                counter += 1

                if debug:
                    print("\n")

            if debug:
                print("\n")


def initialize_oamodel(eta_lower, points, c_scaling, k, m, name, debug):
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
    distances, M = pairwise_distances(points, c_scaling)

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
    lb, ub = compute_box_bounds(points, m, points[0].shape[0])

    # initialize x_i variables - centers, and z_ij variables - point i assigned to cluster j
    for j in range(k):
        for n in range(points[0].shape[0]):
            x[j, n] = oa_model.addVar(
                vtype=GRB.CONTINUOUS,
                obj=0,
                lb=lb[n],
                ub=ub[n],
                name="x_" + str(j) + "_" + str(n),
            )
        for i in range(m):
            z[i, j] = oa_model.addVar(
                vtype=GRB.BINARY, name="z_" + str(i) + "_" + str(j)
            )

    # add initial sanity constraint eta \geq initial lower bound
    # oa_model.addConstr(eta >= eta_lower, name="initial_constr")

    # add constraints for z_ij to sum to 1 over js, for every i
    for i in range(m):
        oa_model.addConstr(quicksum(z[i, j] for j in range(k)) == 1)

    # add constraints for each center to be assigned at least one point
    for j in range(k):
        oa_model.addConstr(quicksum(z[i, j] for i in range(m)) >= 1)

    alphas = [
        0,
        0.0000000000000001,
        0.0000000000000005,
        0.000000001,
        0.000000005,
        0.00001,
        0.00005,
        0.1,
        0.25,
        0.5,
        0.75,
        1,
        1.25,
        1.5,
        1.75,
        2,
        3,
        4,
    ]

    if debug:
        log(["adding initial linear approximation cuts", "\n"])

    add_linear_lower_bounds(
        oa_model, eta, x, z, alphas, points, c_scaling, m, k, M, debug
    )

    oa_model.update()
    lpfile_name = name + ".lp"
    oa_model.write(os.path.join(MODELS, lpfile_name))

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
    oa_model._c_scaling = c_scaling

    if debug:
        log(["initialized model", "\n"])

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
        c_scaling = model._points
        M = model._M
        debug = model._debug
        N = points[0].shape[0]

        if debug:
            print("separation algorithm")
            log(["U", U])

        # first we find an i, xhat_l, and j where a would-be-cut is tight
        for i in range(m):
            for xhat_i in U[i]:
                new_points = []
                for l in range(k):
                    # import pdb; pdb.set_trace()
                    intercept, gradient = prep_cut(points[i], xhat_i, c_scaling[i])
                    xl_array = np.zeros(N)
                    for n in range(N):
                        xl_array[n] = x_sol[l, n]
                    lhs = eta_sol
                    rhs = (
                        intercept
                        + np.dot(gradient, (xl_array - xhat_i))
                        - M * (1 - z_sol[i, l])
                    )

                    if lhs == rhs:

                        if debug:
                            log(["xl_hat", xl_array])

                        # when we find a tight cut add xhat_l to U_i
                        new_points.append(xl_array)
                        intercept, gradient = prep_cut(
                            points[i], xl_array, c_scaling[i]
                        )

                        # add a cut based on xhat_l, gradient_slope, intercept
                        # gradient_slope and intercept have been recomputed for xl_array
                        # add these cuts for every variable x_j
                        if debug:
                            log(["intercept", intercept], ["gradient", gradient])
                        for j in range(k):
                            model.cbLazy(
                                eta
                                >= intercept
                                + quicksum(
                                    (gradient[n] * (x[j, n] - xl_array[n]))
                                    for n in range(N)
                                )
                                - M * (1 - z[i, j])
                            )

                            if debug:
                                log_cut(intercept, gradient, i, j, xl_array, M)

            model._U[i].extend(new_points)

        if debug:
            print("\n")


def outer_approximation(instance, debug=False):
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
    k = instance["k"]
    c_scaling = instance["c_scaling"]
    points = instance["points"]
    name = instance["name"]
    print("outer approximation algorithm, instance: " + instance["name"])

    # initialize the model with variables, lower bound and set-partitioning constraints
    oa_model = initialize_oamodel(0, points, c_scaling, k, len(points), name, debug)

    # optimize, passing callback function to model
    oa_model.optimize(separation_algorithm)

    # print solution
    if oa_model.status == GRB.Status.OPTIMAL:
        x = oa_model._x
        eta = oa_model._eta
        z = oa_model._z

        if debug:
            print("centers: ")
        centers = [[] for j in range(k)]
        for j in range(k):
            point_str = "["

            for n in range(points[0].shape[0]):
                point_str += str(x[j, n].x)
                centers[j].append(x[j, n].x)
                point_str += " "

            if debug:
                print(point_str + "]")
                print("assigned: ")
                for i in range(oa_model._m):
                    if z[i, j].x > 0.5:
                        print(" point - " + str(points[i]))

        print("\n")
        distance_matrix, max_min_distance, i_final, j_final = objective_matrix(
            points, c_scaling, centers
        )
        print_solution(centers, max_min_distance, i_final, j_final, points)
        print("eta: " + str(eta.x))

        return eta.x

    elif oa_model.status == GRB.Status.INFEASIBLE:
        print("Infeasible")
        return None
    elif oa_model.status == GRB.Status.UNBOUNDED:
        print("Unbounded")
        return None
    else:
        print("unkown error")
        return None


def qp_model(instance, debug=False):
    """
    Initialize the master model
        - in:
            - eta_lower - lower bound to set an initial constraint
            - ints k, m, number of policies and functions
        - out:
            - initialized model qp_model
        - notes:
            - eta is just a single continuous GRBVAR
            - z is a multidict of binary GRBVARs, indexed point i to cluster j
            - x is a multicict of continuous GRBVARs, indexed center of j, dimension in n
    """
    k = instance["k"]
    c_scaling = instance["c_scaling"]
    points = instance["points"]
    name = instance["name"]
    m = len(points)
    print("quadratic model, instance: " + instance["name"])

    # initialize M
    distances, M = pairwise_distances(points, c_scaling)

    if debug:
        log(["M", M])

    # initialize model
    qp_model = Model("QP")

    if ~(debug):
        qp_model.setParam("OutputFlag", 0)

    # initialize eta variable
    eta = qp_model.addVar(vtype=GRB.CONTINUOUS, obj=1, name="eta")
    x = {}
    z = {}

    # add box constraints (simply lower and upper bounds in this case)
    lb, ub = compute_box_bounds(points, m, points[0].shape[0])

    # initialize x_i variables - centers, and z_ij variables - point i assigned to cluster j
    for j in range(k):
        for n in range(points[0].shape[0]):
            x[j, n] = qp_model.addVar(
                vtype=GRB.CONTINUOUS,
                obj=0,
                lb=lb[n],
                ub=ub[n],
                name="x_" + str(j) + "_" + str(n),
            )
        for i in range(m):
            z[i, j] = qp_model.addVar(
                vtype=GRB.BINARY, name="z_" + str(i) + "_" + str(j)
            )

    # add constraints for z_ij to sum to 1 over js, for every i
    for i in range(m):
        qp_model.addConstr(quicksum(z[i, j] for j in range(k)) == 1)

    # add constraints for each center to be assigned at least one point
    for j in range(k):
        qp_model.addConstr(quicksum(z[i, j] for i in range(m)) >= 1)

    for i in range(m):
        for j in range(k):
            qp_model.addConstr(
                eta
                >= quicksum(
                    ((c_scaling[i] / 2) * (x[j, n] - points[i][n]) ** 2) for n in range(points[0].shape[0])
                )
                - M * (1 - z[i, j])
            )

    qp_model.update()
    lpfile_name = name + "-qp.lp"
    qp_model.write(os.path.join(MODELS, lpfile_name))
    qp_model.optimize()

    # print solution
    if qp_model.status == GRB.Status.OPTIMAL:

        if debug:
            print("centers: ")
        centers = [[] for j in range(k)]
        for j in range(k):
            point_str = "["

            for n in range(points[0].shape[0]):
                point_str += str(x[j, n].x)
                centers[j].append(x[j, n].x)
                point_str += " "

            if debug:
                print(point_str + "]")
                print("assigned: ")
                for i in range(m):
                    if z[i, j].x > 0.5:
                        print(" point - " + str(points[i]))

        print("\n")
        distance_matrix, max_min_distance, i_final, j_final = objective_matrix(
            points, c_scaling, centers
        )
        print_solution(centers, max_min_distance, i_final, j_final, points)
        print("eta: " + str(eta.x))

        return eta.x

    elif qp_model.status == GRB.Status.INFEASIBLE:
        print("Infeasible")
        return None
    elif qp_model.status == GRB.Status.UNBOUNDED:
        print("Unbounded")
        return None
    else:
        print("unkown error")
        return None


def qp_model_regression(instance, debug=False):
    """
    Initialize the master model
        - in:
            - eta_lower - lower bound to set an initial constraint
            - ints k, m, number of policies and functions
        - out:
            - initialized model qp_model
        - notes:
            - eta is just a single continuous GRBVAR
            - z is a multidict of binary GRBVARs, indexed point i to cluster j
            - x is a multicict of continuous GRBVARs, indexed center of j, dimension in n
    """
    k = instance["k"]
    A_list = instance["A_list"]
    b_list = instance["b_list"]
    points = instance["minimizers"]
    N = points[0].shape[0]
    Ni = b_list[0].shape[0]
    name = instance["name"]
    m = len(points)
    print("quadratic model, instance: " + instance["name"])

    # initialize M
    distances, M = pairwise_distances_regression(A_list, b_list, points)

    if debug:
        log(["M", M])

    # initialize model
    qp_model = Model("QP")

    if ~(debug):
        qp_model.setParam("OutputFlag", 0)

    # initialize eta variable
    eta = qp_model.addVar(vtype=GRB.CONTINUOUS, obj=1, name="eta")
    x = {}
    z = {}

    # initialize x_i variables - centers, and z_ij variables - point i assigned to cluster j
    for j in range(k):
        for n in range(N):
            x[j, n] = qp_model.addVar(
                vtype=GRB.CONTINUOUS,
                obj=0,
                name="x_" + str(j) + "_" + str(n),
            )
        for i in range(m):
            z[i, j] = qp_model.addVar(
                vtype=GRB.BINARY, name="z_" + str(i) + "_" + str(j)
            )

    # add constraints for z_ij to sum to 1 over js, for every i
    for i in range(m):
        qp_model.addConstr(quicksum(z[i, j] for j in range(k)) == 1)

    # add constraints for each center to be assigned at least one point
    for j in range(k):
        qp_model.addConstr(quicksum(z[i, j] for i in range(m)) >= 1)

    for i in range(m):
        for j in range(k):
            qp_model.addConstr(
                eta
                >= quicksum(
                    0.5 * (quicksum(A_list[i][ni][n] * x[j, n] for n in range(N)) - b_list[i][ni]) ** 2 for ni in range(Ni)
                )
                - M * (1 - z[i, j])
            )

    qp_model.update()
    lpfile_name = name + "-qp.lp"
    qp_model.write(os.path.join(MODELS, lpfile_name))
    qp_model.optimize()

    # print solution
    if qp_model.status == GRB.Status.OPTIMAL:

        if debug:
            print("centers: ")
        centers = [[] for j in range(k)]
        for j in range(k):
            point_str = "["

            for n in range(N):
                point_str += str(x[j, n].x)
                centers[j].append(x[j, n].x)
                point_str += " "

            if debug:
                print(point_str + "]")
                print("assigned: ")
                for i in range(m):
                    if z[i, j].x > 0.5:
                        print(" point - " + str(points[i]))

        print("\n")
        distance_matrix, max_min_distance, i_final, j_final = objective_matrix_regression(
            centers, A_list, b_list
        )
        print_solution(centers, max_min_distance, i_final, j_final, points)
        print("eta: " + str(eta.x))

        import pdb; pdb.set_trace()
        return eta.x

    elif qp_model.status == GRB.Status.INFEASIBLE:
        print("Infeasible")
        return None
    elif qp_model.status == GRB.Status.UNBOUNDED:
        print("Unbounded")
        return None
    else:
        print("unkown error")
        return None


def mip_model(instance, debug=False):
    """
    Initialize the master model
        - in:
            - eta_lower - lower bound to set an initial constraint
            - ints k, m, number of policies and functions
        - out:
            - initialized model mip_model
        - notes:
            - eta is just a single continuous GRBVAR
            - z is a multidict of binary GRBVARs
            - y is a multicict of binary GRBVARs
    """
    k = instance["k"]
    c_scaling = instance["c_scaling"]
    points = instance["points"]
    name = instance["name"]
    m = len(points)
    print("MIP model, instance: " + instance["name"])

    # initialize M
    distances, max_min_distance = pairwise_distances(points, c_scaling)

    # initialize model
    mip_model = Model("MIP")

    if ~(debug):
        mip_model.setParam("OutputFlag", 0)

    # initialize eta variable and declare y and z
    eta = mip_model.addVar(vtype=GRB.CONTINUOUS, obj=1, name="eta")
    y = {}
    z = {}

    # initialize z_i variables and y_ij variables
    for i in range(m):
        z[i] = mip_model.addVar(vtype=GRB.BINARY, name="z_" + str(i))
        for j in range(m):
            y[i, j] = mip_model.addVar(
                vtype=GRB.BINARY, name="y_" + str(i) + "_" + str(j)
            )

    # add constraints for y_ij to sum to 1 over j in [m], for every i
    for i in range(m):
        mip_model.addConstr(quicksum(y[i, j] for j in range(m)) == 1)

    # y is bounded by z (can only assign points to existing centers)
    for i in range(m):
        for j in range(m):
            mip_model.addConstr(y[i, j] <= z[j])

    # exactly k centers
    mip_model.addConstr(quicksum(z[i] for i in range(m)) == k)

    # enforce max constraint
    for i in range(m):
        mip_model.addConstr(
            eta >= quicksum(distances[i][j] * y[i, j] for j in range(m))
        )

    mip_model.update()
    lpfile_name = name + "-mip.lp"
    mip_model.write(os.path.join(MODELS, lpfile_name))
    mip_model.optimize()

    # print solution
    if mip_model.status == GRB.Status.OPTIMAL:

        if debug:
            print("centers: ")

        centers = []

        for i in range(m):

            if z[i].x < 0.5:
                continue

            centers.append(points[i])

            if debug:
                print(str(points[i]))
                print("assigned: ")
                for j in range(m):
                    if y[j, i].x > 1:
                        print("  point - " + str(points[j]))

        print("\n")
        distance_matrix, max_min_distance, i_final, j_final = objective_matrix(
            points, c_scaling, centers
        )
        print_solution(centers, max_min_distance, i_final, j_final, points)
        print("eta: " + str(eta.x))

        return eta.x

    elif mip_model.status == GRB.Status.INFEASIBLE:
        print("Infeasible")
        return None
    elif mip_model.status == GRB.Status.UNBOUNDED:
        print("Unbounded")
        return None
    else:
        print("unkown error")
        return None


def mip_model_regression(instance, debug=False):
    """
    MIP model for batch regression
    """

    k = instance["k"]
    A_list = instance["A_list"]
    b_list = instance["b_list"]
    points = instance["minimizers"]
    name = instance["name"]
    m = len(points)
    print("MIP model, instance: " + instance["name"])

    # initialize M
    distances, max_min_distance = pairwise_distances_regression(A_list, b_list, points)

    # initialize model
    mip_model = Model("MIP")

    if ~(debug):
        mip_model.setParam("OutputFlag", 0)

    # initialize eta variable and declare y and z
    eta = mip_model.addVar(vtype=GRB.CONTINUOUS, obj=1, name="eta")
    y = {}
    z = {}

    # initialize z_i variables and y_ij variables
    for i in range(m):
        z[i] = mip_model.addVar(vtype=GRB.BINARY, name="z_" + str(i))
        for j in range(m):
            y[i, j] = mip_model.addVar(
                vtype=GRB.BINARY, name="y_" + str(i) + "_" + str(j)
            )

    # add constraints for y_ij to sum to 1 over j in [m], for every i
    for i in range(m):
        mip_model.addConstr(quicksum(y[i, j] for j in range(m)) == 1)

    # y is bounded by z (can only assign points to existing centers)
    for i in range(m):
        for j in range(m):
            mip_model.addConstr(y[i, j] <= z[j])

    # exactly k centers
    mip_model.addConstr(quicksum(z[i] for i in range(m)) == k)

    # enforce max constraint
    for i in range(m):
        mip_model.addConstr(
            eta >= quicksum(distances[i][j] * y[i, j] for j in range(m))
        )

    mip_model.update()
    lpfile_name = name + "-mip.lp"
    mip_model.write(os.path.join(MODELS, lpfile_name))
    mip_model.optimize()

    # print solution
    if mip_model.status == GRB.Status.OPTIMAL:

        if debug:
            print("centers: ")

        centers = []

        for i in range(m):

            if z[i].x < 0.5:
                continue

            centers.append(points[i])

            if debug:
                print(str(points[i]))
                print("assigned: ")
                for j in range(m):
                    if y[j, i].x > 1:
                        print("  point - " + str(points[j]))

        print("\n")
        distance_matrix, max_min_distance, i_final, j_final = objective_matrix_regression(
            centers, A_list, b_list
        )
        print_solution(centers, max_min_distance, i_final, j_final, points)
        print("eta: " + str(eta.x))

        return eta.x

    elif mip_model.status == GRB.Status.INFEASIBLE:
        print("Infeasible")
        return None
    elif mip_model.status == GRB.Status.UNBOUNDED:
        print("Unbounded")
        return None
    else:
        print("unkown error")
        return None


def append_date(exp_name):
    """
    Append today's date to experiment name
    """
    today = date.today()
    date_str = today.strftime("%m_%d_%y")

    name = exp_name + "-" + date_str
    return name


def check_make_dir(path, i):
    """
    Recursively check if an experiment directory exists, or create one with the highest number
        - example - if "path" string is "/dat/experiments/test-01_29_22", and there already exist:
            - "/dat/experiments/test-01_29_22-0"
            - "/dat/experiments/test-01_29_22-1"
            - "/dat/experiments/test-01_29_22-2"
        we have to create the dir "/dat/experiments/test-01_29_22-3"
    """

    isdir = os.path.isdir(path + "-" + str(i))

    # if the directory exists, call on the next i
    if isdir:
        return check_make_dir(path, i + 1)

    # base case - create directory for given i (and return final path)
    else:
        os.mkdir(path + "-" + str(i))
        return path + "-" + str(i)


def dump_instance(path, instance, i=None):
    """
    Dump instance to json file - k-center
    """
    if i != None:
        name = "instance-" + str(i) + ".json"
    else:
        name = "instance.json"

    file_path = os.path.join(path, name)
    f = open(file_path, "w")
    json.dump(instance, f, indent=4)
    f.close()


def dump_instance_regression(path, instance, i=None):
    """
    Dump instance to json file - regression
    """
    if i != None:
        name = "instance-" + str(i) + ".json"
    else:
        name = "instance.json"

    instance["A_list"] = [A.tolist() for A in instance["A_list"]]
    instance["b_list"] = [b.tolist() for b in instance["b_list"]]
    instance["minimizers"] = [x.tolist() for x in instance["minimizers"]]

    file_path = os.path.join(path, name)
    f = open(file_path, "w")
    json.dump(instance, f, indent=4)
    f.close()

    instance["A_list"] = [np.array(A) for A in instance["A_list"]]
    instance["b_list"] = [np.array(b) for b in instance["b_list"]]
    instance["minimizers"] = [np.array(x) for x in instance["minimizers"]]


def generate_instance(
    n, m, c_lower, c_upper, k, problem_type, name, kappa, exp_path=None, instance_num=None, debug=False
):
    """
    Exact algorithm
        in :
            - n, m, dimension and number of points - int values
            - c_upper and c_lower - bounds for uniform dist for scaling constants - int values
            - k - number of centers (k) - int
            - name - instance name - string
        out :
            - returns instance as a dict
            - additionally - write the instance to file "instance.json" - in directory created by check_make_dir
    """
    # generate instance as dictionary
    if problem_type == 0:
        instance = {
            "k": k,
            "kappa": 1,
            "c_scaling": list(np.random.uniform(c_lower, c_upper, m)),
            "points": [list(np.random.normal(0, 1, n)) for i in range(m)],
            "name": name
        }
    elif problem_type == 1:
        ni = 2 * n
        A_list, b_list, minimizers = gen_ls_data(m, n, ni, k)
        instance = {
            "k": k,
            "kappa": kappa,
            "A_list": A_list,
            "b_list": b_list,
            "minimizers": minimizers,
            "name": name
        }
        compute_constants(instance)

    if exp_path:
        # for an actual experiment - use the experiment dir if it exists
        # we are "in the loop" we already used check_make_dir outside the loop
        if problem_type == 0:
            dump_instance(exp_path, instance, instance_num)
        elif problem_type == 1:
            dump_instance_regression(exp_path, instance, instance_num)

        log(["instance written to directory", exp_path])
        if debug:
            log_instance(instance)
    else:
        # for a single run, no experiment name, single directory for instance
        # append date to name
        name = append_date(name)
        # create directory for experiment
        temp_path = os.path.join(EXPERIMENTS, name)
        experiment_path = check_make_dir(temp_path, 0)
        name = experiment_path.split("/")[-1]

        # dump instance to json file
        if problem_type == 0:
            dump_instance(experiment_path, instance, instance_num)
        elif problem_type == 1:
            dump_instance_regression(experiment_path, instance, instance_num)

        log(["instance written to directory", experiment_path])
        if debug:
            log_instance(instance)

    # return instance (first convert points back to np arrays if its problem type 0)
    if problem_type == 0:
        points_list = instance["points"]
        instance["points"] = [np.array(i) for i in points_list]

    return instance


def greedy_exact(instance, problem_type, qp=False, debug=False):
    """
    Run greedy vs exact
        in :
            - instance:
                - instance passed as a dict
                - instance name, to be found in a json file in the experiments folder
            - problem_type (int)
            - qp - whether to run qp as well or not
        out :
            - run greedy vs outer approximation algorithm
            - write results to results.csv file
    """

    log_sep(2)
    if debug:
        print("hello from experiment function\n")

    if type(instance) == str:
        # we are receiving a filename in this case
        # must load json to dict
        file_path = os.path.join(EXPERIMENTS, instance, "instance.json")
        f = open(file_path, "r")
        instance = json.load(f)
        points_list = instance["points"]
        instance["points"] = [np.array(i) for i in points_list]
        f.close()

    if problem_type == 0:
        start = time.perf_counter()
        eta_greedy = greedy_algorithm(instance, debug)
        greedy_time = time.perf_counter() - start
        log(["time", greedy_time])
    elif problem_type == 1:
        start = time.perf_counter()
        eta_greedy = greedy_algorithm_regression(instance, debug)
        greedy_time = time.perf_counter() - start
        log(["time", greedy_time])

    log_sep()

    if problem_type == 0:
        start = time.perf_counter()
        eta_mip = mip_model(instance, debug)
        mip_time = time.perf_counter() - start
        log(["time", mip_time])
    elif problem_type == 1:
        start = time.perf_counter()
        eta_mip = mip_model_regression(instance, debug)
        mip_time = time.perf_counter() - start
        log(["time", mip_time])

    log_sep()

    if eta_mip == None:
        print("optimal solution not found during mip optimization")

    if qp:
        if problem_type == 0:
            start = time.perf_counter()
            eta_qp = qp_model(instance, debug)
            qp_time = time.perf_counter() - start
            log(["time", qp_time])
        if problem_type == 1:
            start = time.perf_counter()
            eta_qp = qp_model_regression(instance, debug)
            qp_time = time.perf_counter() - start
            log(["time", qp_time])

        if eta_qp == None:
            print("optimal solution not found during qp optimization")

    log_sep(2)

    if qp and (problem_type == 0):
        results = [eta_greedy, greedy_time, eta_mip, mip_time, eta_qp, qp_time]
    else:
        results = [eta_greedy, greedy_time, eta_mip, mip_time]

    return results


def plot_experiment(experiment, experiment_name=None):
    """
    Create plots for an experiment
        - experiment - (string) name of experiment directory (full name not base!), read from results.csv
        - eperiment - (pd.DataFrame) use directly - in this case full name must be passed as a string
        - experiment_name - (string) experiment name for path
    """
    if type(experiment) == str:
        experiment_name = experiment  # assign the string to experiment name for later
        results_path = os.path.join(EXPERIMENTS, experiment, "results.csv")
        results_df = pd.read_csv(results_path)
    elif type(experiment) == pd.DataFrame:
        results_df = experiment
    else:
        return False

    print(results_df)

    # standard deviation
    sns.relplot(x="k", y="ratio", kind="line", ci="sd", style="m", data=results_df)
    check_dir = os.path.join(EXPERIMENTS, experiment_name, "figures")
    figure_dir = check_make_dir(check_dir, 0)
    figure_path_png = os.path.join(figure_dir, "ratio_plot.png")
    figure_path_jpg = os.path.join(figure_dir, "ratio_plot.jpg")
    plt.savefig(figure_path_png)
    plt.savefig(figure_path_jpg)

    # confidence interval
    sns.relplot(x="k", y="ratio", kind="line", style="m", data=results_df)
    figure_path_png = os.path.join(figure_dir, "ratio_plot-1.png")
    figure_path_jpg = os.path.join(figure_dir, "ratio_plot-1.jpg")
    plt.savefig(figure_path_png)
    plt.savefig(figure_path_jpg)
    # fmri = sns.load_dataset("fmri")

    return True


def single_rep(exp_name, exp_path, i, k, n, m, c_lower, c_upper, kappa, problem_type, qp):
    """
    Single rep of the experiment
    """
    run_results_full = [0 for i in COLUMNS]
    instance_id = exp_name + "-" + str(i)
    # generate instance
    instance = generate_instance(n, m, c_lower, c_upper, k, problem_type, instance_id, kappa, exp_path, i)

    run_results_full[0] = instance_id
    run_results_full[1] = PROBLEM_TYPES[problem_type]
    run_results_full[2] = k
    run_results_full[3] = n
    run_results_full[4] = m
    run_results_full[5] = kappa

    # run the algorithms
    run_results = greedy_exact(instance, problem_type, qp)

    run_results_full[6] = run_results[0]
    run_results_full[7] = run_results[1]
    run_results_full[8] = run_results[2]
    run_results_full[9] = run_results[3]

    if qp:
        run_results_full[10] = run_results[4]
        run_results_full[11] = run_results[5]
    else:
        run_results_full[10] = np.NaN
        run_results_full[11] = np.NaN

    if run_results[2] == None: run_results_full[12] = np.NaN
    else: run_results_full[12] = run_results[0] / run_results[2]

    if qp:
        if run_results[4] == None:
            run_results_full[13] = np.NaN
            run_results_full[14] = np.NaN
        else:
            run_results_full[13] = run_results[0] / run_results[4]
            run_results_full[14] = run_results[2] / run_results[4]

    else:
        run_results_full[13] = np.NaN
        run_results_full[14] = np.NaN

    return run_results_full


def greedy_exact_experiment(exp_name, problem_type, k_lower, k_upper, kappa_list, c_lower, c_upper, n_list, m_list, reps, qp=False):
    """
    Generate instances and run experiments from k_lower to k_upper
        - c_lower, c_upper, scaling factor upper and lower bounds
        - n, m dimension and number of points
        - reps number of instance for every (k, n, m) combo (want to average and stdev in results)
        - exp_name - will create a directory EXPERIMENTS/exp_name/ where the results and instances will go
        - file_name - the base file name for the instnce files
        - problem_type - metric k-center (0), batch regression (1), logistic regression (2)
    """
    exp_name = append_date(exp_name)
    temp_path = os.path.join(EXPERIMENTS, exp_name)
    exp_path = check_make_dir(temp_path, 0)
    exp_name = exp_path.split("/")[-1]
    problem_type_str = PROBLEM_TYPES[problem_type]

    if problem_type == 0:
        kappa_list = [1] # enforce condition number for k-center problem

    results = []
    instance_num = 1

    # COLUMNS = ["instance_id", "problem_type", "k", "n", "m", "kappa", "obj_greedy", "time_greedy", "obj_mip", "time_mip", "obj_qp", "time_qp", "g_mip_ratio", "g_qp_ratio", "mip_qp_ratio"]

    for kappa in kappa_list:
        for m in m_list:
            for n in n_list:
                for k in range(k_lower, k_upper+1):
                    for rep in range(reps):
                        if problem_type == 0:
                            run_results_full = single_rep(exp_name, exp_path, instance_num, k, n, m, c_lower, c_upper, kappa, problem_type, qp)
                            instance_num += 1
                            results.append(run_results_full)
                        elif problem_type == 1:
                            run_results_full = single_rep(exp_name, exp_path, instance_num, k, n, m, c_lower, c_upper, kappa, problem_type, qp)
                            instance_num += 1
                            results.append(run_results_full)

    # COLUMNS = ["instance_id", "problem_type", "k", "n", "m", "kappa", "obj_greedy", "time_greedy", "obj_mip", "time_mip", "obj_qp", "time_qp", "g_mip_ratio", "g_qp_ratio", "mip_qp_ratio"]
    results_df = pd.DataFrame(results, columns=COLUMNS)
    results_path = os.path.join(exp_path, "results.csv")
    print(results_df)
    results_df.to_csv(results_path)


def aggregate_experiments(new_name, experiments):
    """
    Aggregate the results from multiple experimental runs into one
        - new_name - target name for new directory with results
            - just name, not full path
            - full name tho, not just base name
        - list of experiments (by name) to include in aggragation
    """
    # create target directory for new results 
    temp_path = os.path.join(EXPERIMENTS, new_name)
    new_path = check_make_dir(temp_path, 0)
    new_name = new_path.split("/")[-1]
    frames = []

    # create a small log file to log experiments aggregated
    log_dict = {"experiment names": experiments}
    log_path = os.path.join(new_path, "log.json")
    f = open(log_path, "w")
    json.dump(log_dict, f, indent=4)
    f.close()

    # read in data frames
    for exp in experiments:
        path = os.path.join(EXPERIMENTS, exp, "results.csv")
        df = pd.read_csv(path)
        frames.append(df)

    final_frame = pd.concat(frames)
    print(final_frame)
    final_path = os.path.join(new_path, "results.csv")
    final_frame.to_csv(final_path)


def main():
    """
    Main Loop
        - when naming experiments, follow convention of just using the problem type as base
    """
    # args = sys.argv[1:]

    # instances = [TRIANGLE2, TRIANGLE3, OBVIOUS_CLUSTERS2, OBVIOUS_CLUSTERS3]
    # for i in instances:
    #     name = i
    #     greedy_exact(name, 0, True)
    #     name = i + "-scaled"
    #     greedy_exact(name, 0, True)
    #     import pdb; pdb.set_trace()

    # note - keeping track of runs that have fully finished, commented out with a name in caps variable
    EXP_1 = "scaled_k_center-02_12_22-0" # greedy_exact_experiment(name1, 0, k_lower, k_upper, [5], c_lower, c_upper, n_list, m_list, reps, False)

    name1 = PROBLEM_TYPES[0]
    name2 = PROBLEM_TYPES[1]
    n_list = [5, 10, 100]
    m_list = [20, 50, 100]
    n_list_2 = [10, 20, 150]
    m_list_2 = [30, 70, 110]
    n_list_qp = [5, 10, 20]
    m_list_qp = [20, 21, 22, 23, 24, 25]
    k_lower = 2
    k_upper = 19
    c_lower = 1
    c_upper = 10
    k_upper_qp = 2
    kappa_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    reps = 30

    greedy_exact_experiment(name2, 1, k_lower, k_upper, [1], c_lower, c_upper, n_list, m_list, reps, False)
    greedy_exact_experiment(name2, 1, k_lower, 10, kappa_list, c_lower, c_upper, [5], [20], reps, False)

    greedy_exact_experiment(name1, 0, k_lower, k_upper, [1], c_lower, c_upper, n_list_2, m_list_2, reps, False)
    greedy_exact_experiment(name1, 0, k_lower, k_upper, [1], c_lower, c_upper, n_list, m_list_2, reps, False)
    greedy_exact_experiment(name1, 0, k_lower, k_upper, [1], c_lower, c_upper, n_list_2, m_list, reps, False)

    greedy_exact_experiment(name2, 1, k_lower, k_upper, [5], c_lower, c_upper, n_list, m_list, reps, False)

    greedy_exact_experiment(name1, 0, k_lower, k_upper_qp, [5], c_lower, c_upper, n_list_qp, m_list_qp, reps, True)


if __name__ == "__main__":
    main()

