import numpy as np


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


def main(k, l_constants, points, debug=False):
    """
    Main procedure:
        in :
            - number of clusters (k) - int value
            - lipshitz constants (L_i) - list of floats
            - points (a_i) - list of np.arrays (shape = (n,))
            - minimizers (x_i) - to be added
        out :
            - set of centers U - set of ints (indexes)
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
    U = main(
        obvious_clusters["k"],
        obvious_clusters["l_constants"],
        obvious_clusters["points"],
        True,
    )
    print_solution(U, obvious_clusters["points"])
