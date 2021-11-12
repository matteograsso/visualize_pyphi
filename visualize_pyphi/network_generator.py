import pyphi
import pickle
import numpy as np
from tqdm import tqdm_notebook


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
    y = 1 / (l + np.e ** (-k * (x - x0)))
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

    for s in tqdm_notebook(range(len(states))):
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

