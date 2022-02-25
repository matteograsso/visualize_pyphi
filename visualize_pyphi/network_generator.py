import pyphi
import pickle
import numpy as np
from tqdm import tqdm_notebook
from pyphi.convert import nodes2indices as n2i
import scipy.io
from pathlib import Path
from tqdm import tqdm_notebook
import string
import glob
from scipy.stats import norm
from tqdm.auto import tqdm

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

    mechs_pset = list(pyphi.utils.powerset(range(nodes_n), nonempty=True))
    states = list(pyphi.utils.all_states(nodes_n))
    tpm = np.zeros([2 ** nodes_n, nodes_n])

    for s in range(len(states)):
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


def get_toroidal_L_grid_network(
    n_nodes, determinism_value, self_loop_value, state="all_off", node_labels=None
):
    n = n_nodes
    k = determinism_value
    s, l = get_toroidal_L_grid_weights(n, self_loop_value)

    if n == 4:
        weights_matrix = np.array(
            [
                [s, l, l, l],  # A
                [l, s, l, l],  # B
                [l, l, s, l],  # C
                [l, l, l, s],  # D
                # A, B, C, D
            ]
        )

    elif n == 5:
        weights_matrix = np.array(
            [
                [s, l, l, l, l],  # A
                [l, s, l, l, l],  # B
                [l, l, s, l, l],  # C
                [l, l, l, s, l],  # D
                [l, l, l, l, s],  # E
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
        x0=0.5,
        node_labels=node_labels,
        pickle_network=False,
    )

    return network


# def get_toroidal_grid_network(
#     n_nodes,
#     determinism_value,
#     self_loop_value,
#     state="all_off",
#     node_labels=None,
#     L_weight_distribution=True
# ):
#     n = n_nodes
#     k = determinism_value

#     if L_weight_distribution:
#         s, l = get_toroidal_L_grid_weights(n, self_loop_value)
#     elif:
#         ws = [1 / (u ** g) for u in range(1, n)]
#         ws = [ws[i] for i in [0, 1, 1, 1]]
#         ws = [w / sum(ws) for w in ws]

#         s = ws[0]  # self (strong)
#         h = ws[1]  # high
#         m = ws[3]  # medium
#         s, l = get_toroidal_L_grid_weights(n, self_loop_value)
#     if n == 4:
#         weights_matrix = np.array(
#             [
#                 [s, l, l, l],  # A
#                 [l, s, l, l],  # B
#                 [l, l, s, l],  # C
#                 [l, l, l, s],  # D
#                 # A, B, C, D
#             ]
#         )

#     elif n == 5:
#         weights_matrix = np.array(
#             [
#                 [s, l, l, l, l],  # A
#                 [l, s, l, l, l],  # B
#                 [l, l, s, l, l],  # C
#                 [l, l, l, s, l],  # D
#                 [l, l, l, l, s],  # E
#                 # A, B, C, D, E
#             ]
#         )

#     if node_labels is None:
#         node_labels = [string.ascii_uppercase[n] for n in range(len(weights_matrix))]

#     mech_func = ["l" for n in range(len(weights_matrix))]

#     network = get_net(
#         mech_func,
#         weights_matrix,
#         l=1,
#         k=k,
#         x0=0.5,
#         node_labels=node_labels,
#         pickle_network=False,
#     )

#     return network


def get_toroidal_L_grid_weights(n_nodes, self_loop_value):
    return [self_loop_value, (1 - self_loop_value) / (n_nodes - 1)]


class ToroidalLGrid:
    def __init__(
        self,
        n_nodes,
        determinism_value,
        self_loop_value,
        state="all_off",
        node_labels=None,
    ):
        self.determinism = determinism_value
        self.self_loop_value = self_loop_value
        self.weights = get_toroidal_L_grid_weights(n_nodes, self_loop_value) + [
            get_toroidal_L_grid_weights(n_nodes, self_loop_value)[1]
            for i in range(n_nodes - 2)
        ]
        self.network = get_toroidal_L_grid_network(
            n_nodes,
            determinism_value,
            self_loop_value,
            state="all_off",
            node_labels=None,
        )
        self.state = (0,) * n_nodes if state == "all_off" else state
        self.subsystem = pyphi.Subsystem(self.network, self.state)
