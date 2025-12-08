import numpy as np
import heapq


def anonymize(walks: np.ndarray) -> np.ndarray:
    """
    Python version of C++ anonymize(...)

    Parameters
    ----------
    walks : np.ndarray, shape (num_walks, walk_len), dtype integer
        Original walks with node IDs.

    Returns
    -------
    anon_walks : np.ndarray, same shape as walks, dtype=np.uint32
        Anonymized walks where nodes are relabeled per-walk in order of first appearance,
        starting from 1.
    """
    walks = np.asarray(walks)
    shape, walk_len = walks.shape

    anon_walks = np.zeros((shape, walk_len), dtype=np.uint32)

    for i in range(shape):
        value_to_id = {}
        current_id = 1

        for k in range(walk_len):
            value = int(walks[i, k])

            if value not in value_to_id:
                value_to_id[value] = current_id
                current_id += 1

            anon_walks[i, k] = value_to_id[value]

    return anon_walks


def anonymize_neighbors(
    walks: np.ndarray,
    restarts: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
):
    """
    Python version of C++ anonymizeNeighbors(...)

    Parameters
    ----------
    walks : np.ndarray, shape (num_walks, walk_len), np.uint32
        Original random walks (node IDs).
    restarts : np.ndarray, shape (num_walks, walk_len), bool
        Restart flags corresponding to each step in `walks`.
    indptr : np.ndarray, shape (n_nodes + 1,), np.uint32
        CSR indptr array of the underlying graph.
    indices : np.ndarray, shape (n_edges,), np.uint32
        CSR indices array of the underlying graph (neighbors).

    Returns
    -------
    anon_walks : np.ndarray, shape (num_walks, walk_len), np.uint32
        Anonymized walk matrix (IDs assigned per-walk).
    new_walks : np.ndarray, shape (num_walks, walk_len), np.uint32
        New walk matrix, where after first seeing a node, its already-visited neighbors
        (with still-unvisited edges) are inserted immediately in anonymized ID order.
    new_restarts : np.ndarray, shape (num_walks, walk_len), bool
        Restart flags aligned with `new_walks`.
    neighbors : np.ndarray, shape (num_walks, walk_len), bool
        True at positions that come from the neighbor-expansion step (i.e., added by queue),
        False for original walk positions.
    """
    walks = np.asarray(walks, dtype=np.uint32)
    restarts = np.asarray(restarts, dtype=bool)
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)

    shape, walk_len = walks.shape
    n_edges = indices.shape[0]

    # allocate outputs
    anon_walks = np.zeros((shape, walk_len), dtype=np.uint32)
    new_walks = np.zeros((shape, walk_len), dtype=np.uint32)
    new_restarts = np.zeros((shape, walk_len), dtype=bool)
    neighbors = np.zeros((shape, walk_len), dtype=bool)

    for i in range(shape):
        value_to_id = {}
        id_to_value = {}
        visited = [False] * n_edges  # per-edge visited flags
        current_id = 1
        index = 0  # position in the *new* sequences

        for k in range(walk_len):
            value = int(walks[i, k])
            restart_flag = bool(restarts[i, k])

            if index == walk_len:
                # no more space to write into the row
                continue

            if value not in value_to_id:
                # assign a new anonymized ID
                value_to_id[value] = current_id
                id_to_value[current_id] = value
                current_id += 1

                # write the original step
                anon_walks[i, index] = value_to_id[value]
                new_walks[i, index] = value
                new_restarts[i, index] = restart_flag
                neighbors[i, index] = False
                index += 1

                # now expand neighbors with unvisited edges
                start = indptr[value]
                end = indptr[value + 1]

                # previous node in the ORIGINAL walk (C++ used 0 if k == 0)
                prev = int(walks[i, k - 1]) if k > 0 else 0

                # collect neighbor IDs (anonymized) in a min-heap
                min_queue = []
                for z in range(start, end):
                    neighbor = int(indices[z])
                    unvisited = not visited[z]
                    if k > 0:
                        unvisited = unvisited and (prev != neighbor)
                    if unvisited and (neighbor in value_to_id):
                        visited[z] = True
                        neighbor_id = value_to_id[neighbor]
                        heapq.heappush(min_queue, neighbor_id)

                # insert neighbors into sequence while we have space
                while min_queue and index < walk_len:
                    neighbor_id = heapq.heappop(min_queue)
                    orig_val = id_to_value[neighbor_id]
                    anon_walks[i, index] = neighbor_id
                    new_walks[i, index] = orig_val
                    new_restarts[i, index] = False
                    neighbors[i, index] = True
                    index += 1

            else:
                # value already known: just write it as-is
                anon_id = value_to_id[value]
                anon_walks[i, index] = anon_id
                new_walks[i, index] = value
                new_restarts[i, index] = restart_flag
                neighbors[i, index] = False
                index += 1

    return anon_walks, new_walks, new_restarts, neighbors


def _sample_neighbor(indices, weights, rng):
    """
    Sample an index into `indices` with unnormalized weights `weights`.
    Returns indices[pos].
    """
    weight_sum = weights.sum()
    if weight_sum <= 0:
        # all weights zero, fall back to uniform
        return rng.choice(indices)

    draw = rng.random() * weight_sum
    cumsum = 0.0
    for idx, w in zip(indices, weights):
        cumsum += w
        if draw <= cumsum:
            return idx
    # numerical fall-back
    return indices[-1]


def random_walks(indptr, indices, data, start_nodes, seed, n_walks, walk_len):
    """
    Python/Numpy version of C++ randomWalks.

    Args:
        indptr: 1D array, CSR indptr (len = n_nodes+1)
        indices: 1D array, CSR indices
        data: 1D array, edge weights (per CSR entry)
        start_nodes: 1D array of start nodes (length = n_nodes)
        seed: int
        n_walks: number of walks per start node
        walk_len: length of each walk

    Returns:
        walks: (n_walks * n_nodes, walk_len), dtype=np.uint32
    """
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    data = np.asarray(data, dtype=float)
    start_nodes = np.asarray(start_nodes, dtype=np.uint32)

    n_nodes = start_nodes.shape[0]
    shape = n_walks * n_nodes

    walks = np.empty((shape, walk_len), dtype=np.uint32)

    for i in range(shape):
        rng = np.random.default_rng(seed + i)
        draws = rng.random(walk_len - 1)  # in [0,1)

        step = int(start_nodes[i % n_nodes])
        walks[i, 0] = step

        for k in range(1, walk_len):
            start = indptr[step]
            end = indptr[step + 1]

            # no neighbors → stay in place
            if start == end:
                walks[i, k] = step
                continue

            neigh_idx = indices[start:end]
            weights = data[start:end].copy()
            # sample neighbor
            draw = draws[k - 1]
            next_step = _sample_neighbor(neigh_idx, weights, rng)
            step = int(next_step)
            walks[i, k] = step

    return walks


def random_walks_no_backtrack(indptr, indices, data, start_nodes, seed, n_walks, walk_len):
    """
    Python/Numpy version of C++ randomWalksNoBacktrack.
    Same as random_walks, but avoids immediately going back to the previous node.
    """
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    data = np.asarray(data, dtype=float)
    start_nodes = np.asarray(start_nodes, dtype=np.uint32)

    n_nodes = start_nodes.shape[0]
    shape = n_walks * n_nodes

    walks = np.empty((shape, walk_len), dtype=np.uint32)

    for i in range(shape):
        rng = np.random.default_rng(seed + i)
        draws = rng.random(walk_len - 1)

        step = int(start_nodes[i % n_nodes])
        walks[i, 0] = step

        for k in range(1, walk_len):
            start = indptr[step]
            end = indptr[step + 1]

            # no neighbors → stay
            if start == end:
                walks[i, k] = step
                continue

            neigh_idx = indices[start:end]
            weights = data[start:end].copy()

            if k >= 2:
                prev = int(walks[i, k - 2])

                # set weight to 0 for prev node
                for z in range(start, end):
                    if indices[z] == prev:
                        weights[z - start] = 0.0

            next_step = _sample_neighbor(neigh_idx, weights, rng)
            step = int(next_step)
            walks[i, k] = step

    return walks


def random_walks_restart(indptr, indices, data, start_nodes, seed, n_walks, walk_len, alpha):
    """
    Python/Numpy version of C++ randomWalksRestart.

    alpha = restart probability.
    Returns (walks, restarts)
    """
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    data = np.asarray(data, dtype=float)
    start_nodes = np.asarray(start_nodes, dtype=np.uint32)

    n_nodes = start_nodes.shape[0]
    shape = n_walks * n_nodes

    walks = np.empty((shape, walk_len), dtype=np.uint32)
    restarts = np.zeros((shape, walk_len), dtype=bool)

    for i in range(shape):
        rng = np.random.default_rng(seed + i)
        draws = rng.random(walk_len - 1)

        step = int(start_nodes[i % n_nodes])
        start_node = step
        walks[i, 0] = step

        restart_flag = False
        restarts[i, 0] = restart_flag

        for k in range(1, walk_len):
            start = indptr[step]
            end = indptr[step + 1]

            # no neighbors: restart with prob alpha
            if start == end:
                if rng.random() < alpha:
                    step = start_node
                    restart_flag = True
                walks[i, k] = step
                restarts[i, k] = restart_flag
                continue

            # possibly restart (but not if previous step was restart)
            if (not restarts[i, k - 1]) and (rng.random() < alpha):
                step = start_node
                restart_flag = True
            else:
                neigh_idx = indices[start:end]
                weights = data[start:end].copy()
                next_step = _sample_neighbor(neigh_idx, weights, rng)
                step = int(next_step)
                restart_flag = False

            walks[i, k] = step
            restarts[i, k] = restart_flag

    return walks, restarts


def random_walks_restart_no_backtrack(indptr, indices, data, start_nodes, seed, n_walks, walk_len, alpha):
    """
    Python/Numpy version of C++ randomWalksRestartNoBacktrack.
    Restart + no-backtracking combination.
    """
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    data = np.asarray(data, dtype=float)
    start_nodes = np.asarray(start_nodes, dtype=np.uint32)

    n_nodes = start_nodes.shape[0]
    shape = n_walks * n_nodes

    walks = np.empty((shape, walk_len), dtype=np.uint32)
    restarts = np.zeros((shape, walk_len), dtype=bool)

    for i in range(shape):
        rng = np.random.default_rng(seed + i)
        draws = rng.random(walk_len - 1)

        step = int(start_nodes[i % n_nodes])
        start_node = step
        walks[i, 0] = step

        restart_flag = False
        restarts[i, 0] = restart_flag

        for k in range(1, walk_len):
            start = indptr[step]
            end = indptr[step + 1]

            # no neighbors: restart with prob alpha
            if start == end:
                if rng.random() < alpha:
                    step = start_node
                    restart_flag = True
                walks[i, k] = step
                restarts[i, k] = restart_flag
                continue

            if k >= 2:
                # do not restart if previous step was restart
                if (not restarts[i, k - 1]) and (rng.random() < alpha):
                    step = start_node
                    restart_flag = True
                else:
                    prev = int(walks[i, k - 2])

                    neigh_idx = indices[start:end]
                    weights = data[start:end].copy()

                    # remove backtracking
                    for z in range(start, end):
                        if indices[z] == prev:
                            weights[z - start] = 0.0

                    next_step = _sample_neighbor(neigh_idx, weights, rng)
                    step = int(next_step)
                    restart_flag = False
            else:
                # k == 1
                if (not restarts[i, k - 1]) and (rng.random() < alpha):
                    step = start_node
                    restart_flag = True
                else:
                    neigh_idx = indices[start:end]
                    weights = data[start:end].copy()
                    next_step = _sample_neighbor(neigh_idx, weights, rng)
                    step = int(next_step)
                    restart_flag = False

            walks[i, k] = step
            restarts[i, k] = restart_flag

    return walks, restarts


def random_walks_periodic_restart(indptr, indices, data, start_nodes, seed, n_walks, walk_len, period):
    """
    Python/Numpy version of C++ randomWalksPeriodicRestart.
    Restart deterministically every `period` steps.
    """
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    data = np.asarray(data, dtype=float)
    start_nodes = np.asarray(start_nodes, dtype=np.uint32)

    n_nodes = start_nodes.shape[0]
    shape = n_walks * n_nodes

    walks = np.empty((shape, walk_len), dtype=np.uint32)

    for i in range(shape):
        rng = np.random.default_rng(seed + i)
        draws = rng.random(walk_len - 1)

        step = int(start_nodes[i % n_nodes])
        start_node = step
        walks[i, 0] = step

        for k in range(1, walk_len):
            start = indptr[step]
            end = indptr[step + 1]

            # no neighbors: periodic restart
            if start == end:
                if k % period == 0:
                    step = start_node
                walks[i, k] = step
                continue

            if k % period == 0:
                step = start_node
            else:
                neigh_idx = indices[start:end]
                weights = data[start:end].copy()
                next_step = _sample_neighbor(neigh_idx, weights, rng)
                step = int(next_step)

            walks[i, k] = step

    return walks


def random_walks_periodic_restart_no_backtrack(indptr, indices, data, start_nodes, seed, n_walks, walk_len, period):
    """
    Python/Numpy version of C++ randomWalksPeriodicRestartNoBacktrack.
    Periodic restart + no-backtracking.
    """
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    data = np.asarray(data, dtype=float)
    start_nodes = np.asarray(start_nodes, dtype=np.uint32)

    n_nodes = start_nodes.shape[0]
    shape = n_walks * n_nodes

    walks = np.empty((shape, walk_len), dtype=np.uint32)

    for i in range(shape):
        rng = np.random.default_rng(seed + i)
        draws = rng.random(walk_len - 1)

        step = int(start_nodes[i % n_nodes])
        start_node = step
        walks[i, 0] = step

        for k in range(1, walk_len):
            start = indptr[step]
            end = indptr[step + 1]

            # no neighbors: periodic restart
            if start == end:
                if k % period == 0:
                    step = start_node
                walks[i, k] = step
                continue

            if k >= 2:
                if k % period == 0:
                    step = start_node
                else:
                    prev = int(walks[i, k - 2])
                    neigh_idx = indices[start:end]
                    weights = data[start:end].copy()

                    # no backtracking
                    for z in range(start, end):
                        if indices[z] == prev:
                            weights[z - start] = 0.0

                    next_step = _sample_neighbor(neigh_idx, weights, rng)
                    step = int(next_step)
            else:
                if k % period == 0:
                    step = start_node
                else:
                    neigh_idx = indices[start:end]
                    weights = data[start:end].copy()
                    next_step = _sample_neighbor(neigh_idx, weights, rng)
                    step = int(next_step)

            walks[i, k] = step

    return walks


def n2v_random_walks(indptr, indices, data, start_nodes, seed, n_walks, walk_len, p, q):
    """
    Python/Numpy version of C++ n2vRandomWalks.

    node2vec bias:
      - weight /= p if candidate == prev (backtracking)
      - weight /= q if candidate not in neighbors(prev)
      - else keep weight as is (if candidate is a neighbor of prev)
    """
    indptr = np.asarray(indptr, dtype=np.uint32)
    indices = np.asarray(indices, dtype=np.uint32)
    data = np.asarray(data, dtype=float)
    start_nodes = np.asarray(start_nodes, dtype=np.uint32)

    n_nodes = start_nodes.shape[0]
    shape = n_walks * n_nodes

    walks = np.empty((shape, walk_len), dtype=np.uint32)

    for i in range(shape):
        rng = np.random.default_rng(seed + i)
        draws = rng.random(walk_len - 1)

        step = int(start_nodes[i % n_nodes])
        walks[i, 0] = step

        for k in range(1, walk_len):
            start = indptr[step]
            end = indptr[step + 1]

            # no neighbors → stay
            if start == end:
                walks[i, k] = step
                continue

            neigh_idx = indices[start:end]
            weights = data[start:end].copy()

            if k >= 2:
                prev = int(walks[i, k - 2])
                prev_start = indptr[prev]
                prev_end = indptr[prev + 1]
                prev_neighbors = indices[prev_start:prev_end]

                # adjust weights according to node2vec
                for local_idx, z in enumerate(range(start, end)):
                    neighbor = indices[z]
                    w = weights[local_idx]
                    if neighbor == prev:
                        w = w / p
                    else:
                        # check if neighbor is in neighbors(prev)
                        if neighbor in prev_neighbors:
                            # stay as w
                            pass
                        else:
                            w = w / q
                    weights[local_idx] = w
            # else: first step, no bias

            next_step = _sample_neighbor(neigh_idx, weights, rng)
            step = int(next_step)
            walks[i, k] = step

    return walks
