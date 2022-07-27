from codecs import latin_1_decode
import pyphi
import pickle
import numpy as np
from tqdm import tqdm_notebook
from pyphi.convert import nodes2indices as n2i
import scipy.io
from pathlib import Path
import string
import glob
from scipy.stats import norm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

pyphi.config.CACHE_REPERTOIRES = False


def pklthis(this, name):
    with open(name, "wb") as f:
        pickle.dump(this, f)


def loadpkl(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def strp(x):
    return strip_punct(str(x))


def pickle_name(mech, subsystem):
    state_name = strip_punct(str(subsystem.state))
    pickle_dir = Path(f"pickles/{state_name}/")
    mech_name = f"{strip_punct(str(subsystem.indices2nodes(mech)))}"
    return f"pickles/{state_name}/{mech_name}.pkl"


def save_concept(concept, subsystem):
    filename = pickle_name(concept.mechanism, subsystem)
    del concept.subsystem
    with open(filename, "wb") as f:
        pickle.dump(concept, f)
    return filename


def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i : i + 1] = l[i]
        i += 1
    return ltype(l)


def strip_punct(s):
    return str(
        s.translate(str.maketrans({key: None for key in string.punctuation})).replace(
            " ", ""
        )
    )


def strp(x):
    return strip_punct(str(x))


def i2n(subsystem, mech):
    return strp(subsystem.indices2nodes(mech))


def save_network(network, network_name=None):
    if network_name is None:
        network_filename = "network.pkl"
    else:
        network_filename = f"{network_name}_network.pkl"
    with open(network_filename, "wb") as f:
        pickle.dump(network, f)
    return print(f"Network saved to: {network_name}")


# Logistic function
def LogFunc(x, l, k, x0):
    y = l / (l + np.e ** (-k * (x - x0)))
    return y


# Gaussian function (picky XOR)
def Gauss(x, mu, si):
    y = np.exp(-0.5 * (((x - mu) / si) ** 2))
    return y


def NR(x, exponent, threshold):
    x_exp = x ** exponent
    y = x_exp / (threshold + x_exp)
    return y


# Network generator functions


def get_net(
    mech_func,
    weights,
    mu=None,
    si=None,
    exp=None,
    th=None,
    ths=None,
    l=None,
    k=None,
    x0=None,
    input_nodes=None,
    input_modifier=None,
    node_labels=None,
    network_name=None,
    pickle_network=True,
    state_domain=[-1, 1],
):

    """
    Returns a pyphi network (Logistic activation function)

    Args:
        mech_func: (list) list of mechanism function labels ('d' for selective-OR (Gauss), 'nr' for Naka-Rushton, 'l' for LogFunc)
        weights: (numpy array) matrix of node by node weights (x sends to y)
        mu = mean (Gauss)
        si = standard deviation (Gauss)
        exp = exponent (NR or MvsG)
        th = threshold (NR) or curve steepness (MvsG)
        x0 = midpoint value (LogFunc)
        l = max value (LogFunc)
        k = growth rate (LogFunc)
        gridsize = number of network nodes in the grid excluded inputs
    """

    weights = weights.T
    node_indices = [n for n in range(len(weights))]
    nodes_n = len(node_indices)

    if node_labels is None:
        node_labels = [string.ascii_uppercase[n] for n in range(len(weights))]

    # mechs_pset = list(pyphi.utils.powerset(range(nodes_n), nonempty=True))
    states = list(pyphi.utils.all_states(nodes_n))
    tpm = np.zeros([2 ** nodes_n, nodes_n])

    for s in range(len(states)):
        if state_domain == [-1, 1]:
            state = [2 * v - 1 for v in states[s]]
        else:
            state = states[s]

        tpm_line = []

        for z in node_indices:
            # g = Gaussian
            if mech_func[z] == "g":
                val = Gauss(
                    sum(state * np.array([weights[z][n] for n in node_indices])), mu, si
                )
            # nr = Naka Rushton, s = space
            elif mech_func[z] == "nr" or mech_func[z] == "s":
                if ths:
                    th = ths[z]
                input_sum = sum(state * weights[z])
                val = NR(input_sum, exp, th)
            # l = LogFunc
            elif mech_func[z] == "l":
                val = LogFunc(
                    sum(state * np.array([weights[z][n] for n in node_indices])),
                    l,
                    k,
                    x0,
                )

            # r = Rand
            elif mech_func[z] == "r":
                val = 0.5
            # i = inhibiting input
            elif mech_func[z] == "i":
                non_input_nodes = [n for n in node_indices if n not in input_nodes]
                input_weights = [
                    -input_modifier if state[n] == 0 else 1 for n in input_nodes
                ] * np.array([weights[z][n] for n in input_nodes])
                other_weights = [state[n] for n in non_input_nodes] * np.array(
                    [weights[z][n] for n in non_input_nodes]
                )
                weights_sum = sum(input_weights) + sum(other_weights)
                val = Gauss(weights_sum, mu, si)
            else:
                raise NameError("Mechanism function not recognized")

            tpm_line.append(val)

        tpm[s] = tuple(tpm_line)

    cm = np.array(
        [[np.float(1) if w else 0 for w in weights[n]] for n in range(len(weights))]
    )
    cm = cm.T
    network = pyphi.Network(tpm, cm, node_labels)

    if pickle_network:
        save_network(network, network_name)

    return network


def get_node_tpms(
    mech_func,
    weights,
    node_labels=None,
    state_domain=[0, 1],
):

    node_indices = [n for n in range(len(weights))]
    nodes_n = len(node_indices)

    if node_labels is None:
        node_labels = [string.ascii_uppercase[n] for n in range(len(weights))]

    states = list(pyphi.utils.all_states(nodes_n))

    tpm = np.zeros([2 ** nodes_n, nodes_n])

    for s in range(len(states)):
        if state_domain == [-1, 1]:
            state = [2 * v - 1 for v in states[s]]
        else:
            state = states[s]

        tpm_line = []

        for z in node_indices:
            if mech_func[z] == "l":
                val = LogFunc(
                    sum(state * np.array([weights[z][n] for n in node_indices])),
                    l,
                    k,
                    x0,
                )

            else:
                raise NameError("Mechanism function not recognized")

            tpm_line.append(val)

        tpm[s] = tuple(tpm_line)

    cm = np.array(
        [[np.float(1) if w else 0 for w in weights[n]] for n in range(len(weights))]
    )
    cm = cm.T
    network = pyphi.Network(tpm, cm, node_labels)

    return network


def get_toroidal_grid_network(
    n_nodes,
    determinism_value,
    weight_distribution,
    self_loop_value=None,
    weight_decay_value=None,
    state="all_off",
    node_labels=None,
    x0=0.5,
    state_domain=[0, 1],
):
    n = n_nodes
    k = determinism_value

    if weight_distribution == "L":
        s = self_loop_value
        l = get_toroidal_L_grid_weights(n, s)
        h = l
        m = h

    elif weight_distribution == "pareto":
        s, h, m = get_toroidal_pareto_grid_weights(n, weight_decay_value)

    else:
        s, h, m = weight_distribution

    if n == 3:
        weights_matrix = np.array(
            [
                [s, h, h],  # A
                [h, s, h],  # B
                [h, h, s],  # C
                # A, B, C
            ]
        )

    elif n == 4:
        weights_matrix = np.array(
            [
                [s, h, m, h],  # A
                [h, s, h, m],  # B
                [m, h, s, h],  # C
                [h, m, h, s],  # D
                # A, B, C, D
            ]
        )

    elif n == 5:
        weights_matrix = np.array(
            [
                [s, h, m, m, h],  # A
                [h, s, h, m, m],  # B
                [m, h, s, h, m],  # C
                [m, m, h, s, h],  # D
                [h, m, m, h, s],  # E
                # A, B, C, D, E
            ]
        )

    if node_labels is None:
        node_labels = [string.ascii_uppercase[n] for n in range(len(weights_matrix))]

    mech_func = ["l" for n in range(len(weights_matrix))]

    network = get_net(
        mech_func,
        weights_matrix,
        l=1,
        k=k,
        x0=x0,
        node_labels=node_labels,
        pickle_network=False,
        state_domain=state_domain,
    )

    return network


def get_BGC_grid_network(
    n_nodes,
    determinism_value,
    weight_distribution,
    self_loop_value=None,
    weight_decay_value=None,
    state="all_off",
    node_labels=None,
    x0=0.5,
    state_domain=[0, 1],
):
    u = 7
    n = n_nodes
    k = determinism_value

    if weight_distribution == "L":
        s = self_loop_value
        l = get_toroidal_L_grid_weights(u, s)
        h = l
        m = l
        w = l

    elif weight_distribution == "nearest_neighbor":
        s = self_loop_value
        l = (1 - self_loop_value) / 2
        h = l
        m = 0
        w = 0

    elif weight_distribution == "pareto":
        s, h, m, w = get_toroidal_pareto_grid_weights(u, weight_decay_value)

    weights_matrix = np.array(
        [
            [s, h, m, w, w, m, h],  # A
            [h, s, h, m, w, w, m],  # B
            [m, h, s, h, m, w, w],  # C
            [w, m, h, s, h, m, w],  # D
            [w, w, m, h, s, h, m],  # E
            [m, w, w, m, h, s, h],  # F
            [h, m, w, w, m, h, s],  # G
            # A, B, C, D, E, F, G
        ]
    )

    if node_labels is None:
        node_labels = [string.ascii_uppercase[n] for n in range(len(weights_matrix))]

    mech_func = ["l" for n in range(len(weights_matrix))]

    network = get_net(
        mech_func,
        weights_matrix,
        l=1,
        k=k,
        x0=x0,
        state_domain=state_domain,
        node_labels=node_labels,
        pickle_network=False,
    )

    return network


def get_toroidal_L_grid_weights(n_nodes, self_loop_value):
    l = (1 - self_loop_value) / (n_nodes - 1)
    return l


def get_nearest_neighbor_weights(self_loop_value):
    l = (1 - self_loop_value) / 2
    return l


def get_pareto_weights(n_nodes, g, normalize=True):
    weights = [1 / (u ** g) for u in range(1, n_nodes)]
    if normalize:
        weights = [w / sum(weights) for w in weights]
    return weights


def get_toroidal_pareto_grid_weights(n_nodes, g):
    weights = [1 / (u ** g) for u in range(1, n_nodes)]
    if n_nodes == 3:
        weights = [weights[i] for i in [0, 1, 1]]
        weights = [w / sum(weights) for w in weights]
        s, h = [weights[0], weights[1]]
        return s, h
    elif n_nodes == 4:
        weights = [weights[i] for i in [0, 1, 1, 2]]
        weights = [w / sum(weights) for w in weights]
        s, h, m = [weights[0], weights[1], weights[3]]
        return s, h, m
    elif n_nodes == 5:
        weights = [weights[i] for i in [0, 1, 1, 2, 2]]
        weights = [w / sum(weights) for w in weights]
        s, h, m = [weights[0], weights[1], weights[3]]
        return s, h, m
    elif n_nodes == 6:
        weights = [weights[i] for i in [0, 1, 1, 2, 2, 3]]
        weights = [w / sum(weights) for w in weights]
        s, h, m, n = [weights[0], weights[1], weights[3], weights[5]]
        return s, h, m, n
    elif n_nodes == 7:
        weights = [weights[i] for i in [0, 1, 1, 2, 2, 3, 3]]
        weights = [w / sum(weights) for w in weights]
        s, h, m, w = [weights[0], weights[1], weights[3], weights[5]]
        return s, h, m, w
    elif n_nodes == 8:
        weights = [weights[i] for i in [0, 1, 1, 2, 2, 3, 3, 4]]
        weights = [w / sum(weights) for w in weights]
        s, m, n, l, w = [weights[0], weights[1], weights[3], weights[5], weights[7]]
        return s, m, n, l, w


class ToroidalGrid:
    # Fix for NN
    def __init__(
        self,
        n_nodes,
        determinism_value,
        weight_distribution,
        self_loop_value=None,
        backgound_conditions=False,
        weight_decay_value=None,
        state="all_off",
        node_labels=None,
        x0=0.5,
        state_domain=[0, 1],
    ):
        self.determinism = determinism_value
        self.self_loop_value = self_loop_value

        if not backgound_conditions:
            if type(weight_distribution) is tuple:
                self.weights = weight_distribution

            elif weight_distribution == "L" and self_loop_value:
                self.weights = get_toroidal_L_grid_weights(n_nodes, self_loop_value)

            elif weight_distribution == "pareto" and weight_decay_value:
                self.weights = get_toroidal_pareto_grid_weights(
                    n_nodes, weight_decay_value
                )

            self.network = get_toroidal_grid_network(
                n_nodes,
                determinism_value,
                weight_distribution,
                self_loop_value,
                weight_decay_value,
                state="all_off",
                node_labels=None,
                x0=x0,
                state_domain=state_domain,
            )
            self.state = (0,) * n_nodes if state == "all_off" else state
            self.subsystem = pyphi.Subsystem(self.network, self.state)
        else:

            self.network = get_BGC_grid_network(
                n_nodes,
                determinism_value,
                weight_distribution,
                self_loop_value,
                weight_decay_value,
                state="all_off",
                node_labels=None,
                x0=x0,
                state_domain=state_domain,
            )
            if weight_distribution == "L":
                self.weights = get_toroidal_L_grid_weights(7, self_loop_value)
            elif weight_distribution == "pareto":
                self.weights = get_toroidal_pareto_grid_weights(7, weight_decay_value)
            elif weight_distribution == "nearest_neighbor":
                self.weights = get_nearest_neighbor_weights(self_loop_value)
            self.state = (0,) * self.network.size if state == "all_off" else state
            self.subsystem = pyphi.Subsystem(
                self.network, self.state, nodes=range(n_nodes)
            )


def plotLogFunc(l, k, x0, start=None, stop=None, num=101):
    if start is None:
        start = x0 - 1
    if stop is None:
        stop = x0 + 1
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(start, stop, num)
    y = LogFunc(x, l, k, x0)

    return ax.plot(x, y)


# Map vs grid logistic function
def MvsGLogFunc(x, th):
    y = 1 / (1 + np.exp(-x / th))
    return y


def plotGauss(mu, si, start=None, stop=None, num=101):
    if start is None:
        start = mu - 1
    if stop is None:
        stop = mu + 1
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(start, stop, num)
    y = Gauss(x, mu, si)
    return ax.plot(x, y)


def plotNR(exponent, threshold, start=None, stop=None, num=101):
    if start is None:
        start = 0
    if stop is None:
        stop = 2
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(start, stop, num)
    y = NR(x, exponent, threshold)

    return ax.plot(x, y)
