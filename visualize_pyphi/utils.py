import pyphi
import pickle
import os
import numpy as np
from pyphi.convert import nodes2indices as n2i
from datetime import datetime
import scipy.io
import os
from itertools import *
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm_notebook
import pandas as pd
import string
import glob
import pyphi.relations as rel
from pyphi.models import (
    RepertoireIrreducibilityAnalysis,
    _null_ria,
    MaximallyIrreducibleCause,
    MaximallyIrreducibleEffect,
    MaximallyIrreducibleCauseOrEffect,
)
from pyphi.partition import mip_partitions
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.express as px
import itertools
import sympy
import random
from tqdm.auto import tqdm


pyphi.config.CACHE_REPERTOIRES = False
pd.options.display.max_rows = 4000

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

## Import up sound alert dependencies
from IPython.display import Audio, display


def sample_ces_per_order(network, ces, n_samples_per_order):
    ces_sample = []
    for n in range(network.size):
        order = n + 1
        sample_size = n_samples_per_order[n]
        ces_order = list(filter(lambda c: len(c.mechanism) == order, ces))
        rand_sample = random.sample(range(len(ces_order)), sample_size)
        rand_sample.sort()
        sorted_sample = [ces_order[i] for i in rand_sample]
        ces_sample.extend(sorted_sample)
    return ces_sample


def done(chime_file=None, autoplay=True):
    if chime_file is None:
        chime_file = "/home/mgrasso/projects/chime2.wav"
    display(Audio(filename=chime_file, autoplay=autoplay))


def pklthis(this, name):
    with open(name, "wb") as f:
        pickle.dump(this, f)


def loadpkl(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def load_ces_from_pickles(pickles):
    ces = []
    for file in pickles:
        with open(file, "rb") as f:
            c = pickle.load(f)
        ces.append(c)
    return ces


def load_ces(states, reducible=False):
    pickles = [
        [
            file
            for file in sorted(sorted(glob.glob(f"pickles/{strp(s)}/*.pkl")), key=len)
        ]
        for s in states
    ]

    all_ces = []
    for d in pickles:
        ces = []
        for file in d:

            with open(file, "rb") as f:
                C = pickle.load(f)

            if reducible:
                ces.append(C)
            else:
                if C.phi:
                    ces.append(C)
        all_ces.append(ces)

    return all_ces


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


def parall_compute_rels(subsystem, relatum):
    relation_label = [
        strip_punct(str(subsystem.indices2nodes(x))) for x in (relatum.mechanisms)
    ]
    #     print(f'Computing relation {relation_label[0]}-{relation_label[1]}')
    relation = rel.relation(relatum)
    return relation


def i2n(subsystem, mech):
    return strp(subsystem.indices2nodes(mech))


def ces2list(net_names, all_ces, subsystems):
    ces_list = [
        [
            (
                i2n(subsystems[s], c.mechanism),
                c.phi,
                i2n(subsystems[s], c.cause_purview),
                rel.maximal_state(c.cause)[0],
                i2n(subsystems[s], c.effect_purview),
                rel.maximal_state(c.effect)[0],
            )
            for c in all_ces[s]
            if c.phi
        ]
        for s in range(len(subsystems))
    ]
    return ces_list


def ces2df(net_names, all_ces, subsystems, save_df=False):

    ces_lists = [
        [
            (
                i2n(subsystems[s], c.mechanism),
                c.phi,
                i2n(subsystems[s], c.cause_purview),
                [rel.maximal_state(c.cause)[0][x] for x in c.cause_purview],
                i2n(subsystems[s], c.effect_purview),
                [rel.maximal_state(c.effect)[0][x] for x in c.effect_purview],
            )
            for c in all_ces[s]
            if c.phi
        ]
        for s in range(len(subsystems))
    ]

    dfs = [
        pd.DataFrame(
            [d[1:] for d in ces_list],
            index=[d[0] for d in ces_list],
            columns=["phi", "cause", "state", "effect", "state"],
        )
        for ces_list in ces_lists
    ]

    if save_df is True:
        for s, df in enumerate(dfs):
            df.to_csv(f"ces_{net_names[s]}.csv")
    return dfs


def sepces2df(sepces, subsystem, csv_name=None):
    s = subsystem
    ces_list = [
        (
            strp(i2n(m.mechanism, s)),
            m.direction.name,
            strp(i2n(m.purview, s)),
            strp([rel.maximal_state(m)[0][x] for x in m.purview]),
            m.phi,
        )
        for m in sepces
    ]

    df = pd.DataFrame(
        ces_list, columns=["mechanism", "direction", "purview", "state", "phi"]
    )

    if csv_name:
        df.to_csv(csv_name)
    return df


def pkls2df(net_name, subsystems, save_df=False):
    pickles = [
        [file for file in sorted(sorted(glob.glob(f"{d}/*.pkl")), key=len)]
        for d in sorted(glob.glob("pickles/*"))
    ]

    all_ces = []
    for d in pickles:
        ces = []
        for file in d:

            with open(file, "rb") as f:
                C = pickle.load(f)

            if C.phi:
                ces.append(C)
        all_ces.append(ces)
    print([len(ces) for ces in all_ces])

    ces_lists = [
        [
            (
                i2n(subsystems[s], c.mechanism),
                c.phi,
                i2n(subsystems[s], c.cause_purview),
                [rel.maximal_state(c.cause)[0][x] for x in c.cause_purview],
                i2n(subsystems[s], c.effect_purview),
                [rel.maximal_state(c.effect)[0][x] for x in c.effect_purview],
            )
            for c in all_ces[s]
            if c.phi
        ]
        for s in range(len(subsystems))
    ]

    dfs = [
        pd.DataFrame(
            [d[1:] for d in ces_list],
            index=[d[0] for d in ces_list],
            columns=["phi", "cause", "state", "effect", "state"],
        )
        for ces_list in ces_lists
    ]

    if save_df is True:
        for s, df in enumerate(dfs):
            df.to_csv(f"ces_{net_name}_{s}.csv")
    return dfs


def print_all_ces(subsystems, all_ces, output="list"):
    all_ces_lists = []
    all_ces_dfs = []
    for s, ces in enumerate(all_ces):
        sub = subsystems[s]
        ces_list = [
            (
                strip_punct(str(i2n(sub, c.mechanism))),
                strip_punct(str(i2n(sub, c.cause_purview))),
                strip_punct(
                    str([rel.maximal_state(c.cause)[0][x] for x in c.cause_purview])
                ),
                c.cause.phi,
                strip_punct(str(i2n(sub, c.effect_purview))),
                strip_punct(
                    str([rel.maximal_state(c.effect)[0][x] for x in c.effect_purview])
                ),
                c.effect.phi,
            )
            for c in ces
        ]
        ces_df = pd.DataFrame(
            np.array(ces_list)[:, 1:],
            index=[strip_punct(str(i2n(subsystems[s], c.mechanism))) for c in ces],
            columns=[
                "CAUSE",
                "CAUSE STATE",
                "CAUSE PHI",
                "EFFECT",
                "EFFECT STATE",
                "EFFECT PHI",
            ],
        ).rename_axis("MECHANISM", axis=1)
        all_ces_lists.append(ces_list)
        all_ces_dfs.append(ces_df)

    if output == "list":
        return all_ces_lists
    if output == "df":
        return all_ces_dfs


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


# Gaussian function (picky XOR)
def Gauss(x, mu, si):
    y = np.exp(-0.5 * (((x - mu) / si) ** 2))
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


def NR(x, exponent, threshold):
    x_exp = x ** exponent
    y = x_exp / (threshold + x_exp)
    return y


def plotNR(exponent, threshold, start=None, stop=None, num=101):
    if start is None:
        start = 0
    if stop is None:
        stop = 2
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(start, stop, num)
    y = NR(x, exponent, threshold)

    return ax.plot(x, y)


# Network generator functions
def Gauss_net(
    weights,
    mu,
    si,
    node_labels=None,
    network_name=None,
    pickle_network=True,
    plot_function=False,
):
    """
    Returns a pyphi network (Gaussian picky-XOR activation function)

    Args:
        weights: (numpy array) matrix of node by node weights (x sends to y)
        mu = mean
        si = variance
    """
    if plot_function:
        plotGauss(mu, si)

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
            val = Gauss(
                sum(state * np.array([weights[z][n] for n in node_indices])), mu, si
            )
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


def getNRGridTPM(threshold=1 / 4, lc=1 / 4, exponent=5, units=8):
    # THE RETURN OF NAKA-RUSHTON
    sc = 1
    weights = np.array(
        [
            [sc, lc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [lc, sc, lc, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, lc, sc, lc, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, lc, sc, lc, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, lc, sc, lc, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, lc, sc, lc, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, lc, sc, lc],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lc, sc],
        ]
    )
    weights = weights[0:units, 0:units]

    nN = len(weights)

    pset = pyphi.utils.powerset(np.arange(nN))

    newtpm = np.zeros([2 ** nN, nN])

    for inds in pset:
        istate = np.zeros(nN)
        for y in range(0, len(inds)):
            istate[inds[y]] = 1

        sw = np.zeros(nN, dtype="f")
        swN = np.zeros(nN)
        for z in range(0, len(weights)):
            inpt = istate

            sw[z] = sum(inpt * weights[z]) ** exponent

            swN[z] = sw[z] / (threshold + sw[z])

        V = 0
        for v in range(0, nN):
            V = V + istate[v] * 2 ** v
        newtpm[int(V)] = tuple(swN)

    cm = np.zeros(shape=weights.shape)
    for x in range(0, cm.shape[0]):
        for y in range(0, cm.shape[1]):
            cm[x, y] = int(abs(weights[y, x]) > 0)

    return newtpm, cm


def NR_net(
    weights,
    exponent,
    threshold,
    node_labels=None,
    network_name=None,
    pickle_network=True,
):

    """
    Returns a pyphi network (Naka-Rushton activation function)

    Args:
        weights: (numpy array) matrix of node by node weights (x sends to y)
        exponent: (float) exponent to be used in the Naka-Rushton function
        threshold: (float) threshold to be used in the Naka-Rushton function
    """
    weights = weights.T
    node_indices = [n for n in range(len(weights))]
    nodes_n = len(node_indices)

    if node_labels is None:
        node_labels = [string.ascii_uppercase[n] for n in range(len(weights))]

    mechs_pset = list(pyphi.utils.powerset(np.arange(nodes_n)))
    input_states = list(pyphi.utils.all_states(nodes_n))
    tpm = np.zeros([2 ** nodes_n, nodes_n])

    for i in tqdm_notebook(range(len(mechs_pset))):
        inds = mechs_pset[i]
        input_state = input_states[i]
        weights_sum = [
            sum(input_state * np.array([weights[z][n] for n in node_indices]))
            ** exponent
            for z in node_indices
        ]
        tpm_line = [
            weights_sum[z] / (threshold ** exponent + weights_sum[z])
            for z in range(nodes_n)
        ]
        tpm[i] = tuple(tpm_line)

    cm = np.array(
        [[np.float(1) if x else 0 for x in weights[i]] for i in range(len(weights))]
    )
    cm = cm.T
    network = pyphi.Network(tpm, cm, node_labels)

    if pickle_network:
        save_network(network, network_name)

    return network


def LogFunc_net(
    weights,
    l,
    k,
    x0,
    node_labels=None,
    network_name=None,
    pickle_network=True,
    plot_function=False,
):
    """
    Returns a pyphi network (Logistic activation function)

    Args:
        weights: (numpy array) matrix of node by node weights (x sends to y)
        x0 = midpoint value
        l = max value
        k = growth rate
    """
    if plot_function:
        plotLogFunc(l, k, x0)

    weights = weights.T
    node_indices = [n for n in range(len(weights))]
    nodes_n = len(node_indices)

    if node_labels is None:
        node_labels = [string.ascii_uppercase[n] for n in range(len(weights))]

    mechs_pset = list(pyphi.utils.powerset(range(nodes_n), nonempty=True))
    states = list(pyphi.utils.all_states(nodes_n))
    tpm = np.zeros([2 ** nodes_n, nodes_n])

    for i in tqdm_notebook(range(len(states))):
        state = states[i]
        tpm_line = []

        for z in node_indices:
            val = LogFunc(
                sum(state * np.array([weights[z][n] for n in node_indices])), l, k, x0
            )
            tpm_line.append(val)
        tpm[i] = tuple(tpm_line)

    cm = np.array(
        [[np.float(1) if x else 0 for x in weights[i]] for i in range(len(weights))]
    )
    cm = cm.T
    network = pyphi.Network(tpm, cm, node_labels)

    if pickle_network:
        save_network(network, network_name)

    return network


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


def check_self_rel(subsystem, distinction):
    selfrel = list(rel.relations(subsystem, [distinction]))
    if selfrel:
        sr_purv = selfrel[0].purview
        sr_state = tuple(rel.maximal_state(distinction.cause)[0, sr_purv])
        return (sr_purv, sr_state)


#########################


def parall_compute_save(mech, subsystem, skip_done=False, dispoutput=False):
    state_name = strip_punct(str(subsystem.state))
    if not os.path.isdir(Path(f"pickles/")):
        os.mkdir(Path(f"pickles/"))
    pickle_dir = Path(f"pickles/{state_name}/")
    filename = pickle_name(mech, subsystem)
    mech_name = strip_punct(str(subsystem.indices2nodes(mech)))

    if not os.path.isdir(pickle_dir):
        os.mkdir(pickle_dir)

    if skip_done:
        if os.path.exists(filename):
            #             if dispoutput:
            #                 print(f'Skipping {mech_name}')
            return None
    if dispoutput:
        print(f"Computing {mech_name}")
    C = subsystem.concept(mech)
    #     del C.subsystem
    if C.phi:
        print(
            f"M:{strip_punct(str(subsystem.indices2nodes(C.mechanism)))} C:{strip_punct(str(subsystem.indices2nodes(C.cause_purview)))} {C.cause.phi} E: {strip_punct(str(subsystem.indices2nodes(C.effect_purview)))} {C.effect.phi}"
        )

    save = save_concept(C, subsystem)
    #     print(f'Pickling {mech_name}')
    return C


def parall_compute(mech, subsystem, skip_done=False):
    C = subsystem.concept(mech)
    return C


def irreducible_purviews(network, direction, mechanism, dispoutput=True):
    purviews = pyphi.utils.powerset(network.node_indices, nonempty=True)
    cm = network.cm

    def reducible(purview):
        """Return ``True`` if purview is trivially reducible."""
        _from, to = direction.order(mechanism, purview)
        return pyphi.connectivity.block_reducible(cm, _from, to)

    return [purview for purview in purviews if not reducible(purview)]


def parall_compute_purvs(
    mech, purv, direction, subsystem, skip_done=False, dispoutput=False, save=True
):
    state_name = strip_punct(str(subsystem.state))
    if not os.path.isdir(Path(f"pickles/")):
        os.mkdir(Path(f"pickles/"))
    if not os.path.isdir(Path(f"pickles/{state_name}")):
        os.mkdir(Path(f"pickles/{state_name}"))
    if not os.path.isdir(Path(f"pickles/{state_name}/purvs")):
        os.mkdir(Path(f"pickles/{state_name}/purvs"))
    mech_name = strip_punct(str(subsystem.indices2nodes(mech)))
    purv_name = strip_punct(str(subsystem.indices2nodes(purv)))
    pickle_dir = Path(f"pickles/{state_name}/purvs/{mech_name}")
    filename = f"{mech_name}_{purv_name}_{str(direction)}.pkl"

    if not os.path.isdir(pickle_dir):
        os.mkdir(pickle_dir)

    if skip_done:
        if os.path.exists(filename):
            #             print(f'Skipping {mech_name}_{purv_name}_{str(direction)}')
            return None
    if dispoutput is True:
        print(f"Computing {mech_name}_{purv_name}_{str(direction)}")

    if direction == pyphi.Direction.CAUSE:
        P = subsystem.mic(mech, (purv,))
    if direction == pyphi.Direction.EFFECT:
        P = subsystem.mie(mech, (purv,))
    if save:
        with open(f"{pickle_dir}/{filename}", "wb") as f:
            pickle.dump(P, f)
    return P


def compute_parallmechs(mechs, subsystem, pickle_mechs=True, skip_done=True, n_jobs=-1):
    if pickle_mechs:
        ces = Parallel(n_jobs=n_jobs, verbose=10, backend="multiprocessing")(
            delayed(parall_compute_save)(mech, subsystem, skip_done)
            for mech in tqdm_notebook(mechs)
        )
    else:
        ces = Parallel(n_jobs=n_jobs, verbose=10, backend="multiprocessing")(
            delayed(parall_compute)(mech, subsystem) for mech in tqdm_notebook(mechs)
        )
    return ces


def compute_mechs_parallpurvs(
    network,
    subsystem,
    mechs,
    n_jobs=-1,
    skip_done_mech=True,
    skip_done_purvs=True,
    dispoutput=False,
    dispurvs=False,
):
    directions = [pyphi.Direction.CAUSE, pyphi.Direction.EFFECT]
    all_purviews = list(pyphi.utils.powerset(network.node_indices, nonempty=True))

    concepts = []

    for mech in tqdm_notebook(mechs):

        mech_label = strip_punct(str(i2n(mech, subsystem)))
        state_label = strip_punct(str(subsystem.state))

        if not skip_done_mech or not os.path.exists(
            f"pickles/{state_label}/{strip_punct(str(i2n(mech,subsystem)))}.pkl"
        ):
            print(f"Computing {mech_label}")
            irr_purvs_cause = irreducible_purviews(
                network, directions[0], mech, all_purviews
            )
            irr_purvs_effect = irreducible_purviews(
                network, directions[1], mech, all_purviews
            )

            if skip_done_purvs:
                irr_purvs_cause_todo = []
                irr_purvs_cause_done = []
                for purv in irr_purvs_cause:
                    mech_name = mech_label
                    state_name = state_label
                    purv_name = strip_punct(str(subsystem.indices2nodes(purv)))
                    pickle_purv = Path(
                        f"pickles/{state_name}/purvs/{mech_name}/{mech_name}_{purv_name}_{str(directions[0])}.pkl"
                    )
                    if not os.path.exists(pickle_purv):
                        irr_purvs_cause_todo.append(purv)
                    else:
                        with open(pickle_purv, "rb") as f:
                            done_purv = pickle.load(f)
                        irr_purvs_cause_done.append(done_purv)

                irr_purvs_effect_todo = []
                irr_purvs_effect_done = []
                for purv in irr_purvs_effect:
                    mech_name = mech_label
                    state_name = state_label
                    purv_name = strip_punct(str(subsystem.indices2nodes(purv)))
                    pickle_purv = Path(
                        f"pickles/{state_name}/purvs/{mech_name}/{mech_name}_{purv_name}_{str(directions[1])}.pkl"
                    )
                    if not os.path.exists(pickle_purv):
                        irr_purvs_effect_todo.append(purv)
                    else:
                        with open(pickle_purv, "rb") as f:
                            done_purv = pickle.load(f)
                        irr_purvs_effect_done.append(done_purv)

                irr_purvs_todo = [irr_purvs_cause_todo, irr_purvs_effect_todo]
                irr_purvs_done = [irr_purvs_cause_done, irr_purvs_effect_done]
                if dispurvs:
                    print(
                        f"Cause purviews already computed: {len(irr_purvs_cause_done)} of {len(irr_purvs_cause)}"
                    )
                    print(
                        f"Effect purviews already computed: {len(irr_purvs_effect_done)} of {len(irr_purvs_effect)}"
                    )

            else:
                irr_purvs_todo = [irr_purvs_cause, irr_purvs_effect]

            C = []

            for d, direction in enumerate(directions):
                if dispurvs:
                    print(
                        f"{direction}: {len(irr_purvs_todo[d])} candidate purviews to compute"
                    )
                    print(irr_purvs_todo[d])
                irr_purvs = Parallel(n_jobs, verbose=0, backend="multiprocessing")(
                    delayed(parall_compute_purvs)(mech, purv, direction, subsystem)
                    for purv in irr_purvs_todo[d]
                )
                try:
                    for purv in irr_purvs_done[d]:
                        irr_purvs.append(purv)
                except:
                    pass

                try:
                    C.append(max(irr_purvs))
                except:
                    pass

            reducible_purvs = [
                pyphi.models._null_ria(direction, mech, ())
                for direction in list(pyphi.Direction)[:2]
            ]
            assert len(C) < 3

            # if len(C)==0:
            #     concept = pyphi.models.mechanism.Concept(mech, reducible_purvs[0], reducible_purvs[1], subsystem)
            #     print(f'{strip_punct(str(i2n(mech,subsystem)))} is reducible')
            # elif len(C)==1:
            #     assert C[0].direction in list(pyphi.Direction)[:2]
            #     if C[0].direction == reducible_purvs[0]:
            #         concept = pyphi.models.mechanism.Concept(mech, C[0], reducible_purvs[1], subsystem)
            #     else:
            #         concept = pyphi.models.mechanism.Concept(mech, reducible_purvs[0], C[0], subsystem)

            #     print(f'{strip_punct(str(i2n(mech,subsystem)))} is reducible\n')

            if len(C) > 1:
                concept = pyphi.models.mechanism.Concept(
                    C[0].mechanism, C[0], C[1], subsystem
                )
                c_repr = (
                    i2n(concept.mechanism, subsystem),
                    concept.phi,
                    i2n(concept.cause_purview, subsystem),
                    [
                        rel.maximal_state(concept.cause)[0][x]
                        for x in concept.cause_purview
                    ],
                    i2n(concept.effect_purview, subsystem),
                    [
                        rel.maximal_state(concept.effect)[0][x]
                        for x in concept.effect_purview
                    ],
                )
                print(c_repr, "\n*****\n")
                concepts.append(concept)
            else:
                concept = pyphi.models.mechanism.Concept(
                    mech,
                    subsystem.null_concept.cause,
                    subsystem.null_concept.effect,
                    subsystem,
                )

            del concept.subsystem

            with open(f"pickles/{state_label}/{mech_label}.pkl", "wb") as f:
                pickle.dump(concept, f)
        else:
            print(f"{mech_label} already computed")

    return concepts


def compute_partition(subsystem, direction, mechanism, purview, partition):

    phi, partitioned_repertoire = subsystem.evaluate_partition(
        direction, mechanism, purview, partition
    )

    return phi


def chunk_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def compute_partition_chunk(subsystem, direction, mechanism, purview, chunk):

    phis = [
        subsystem.evaluate_partition(direction, mechanism, purview, partition)[0]
        for partition in chunk
    ]
    #     print(phis)
    min_i = np.argmin(phis)
    return phis[min_i], chunk[min_i]


def count_iterable(iterable, pred=bool):
    "Count how many times the predicate is true"
    return sum(map(pred, iterable))


def compute_purv_parallparts(
    subsystem,
    direction,
    mechanism,
    purview,
    n_jobs=-1,
    chunksize=1000,
    count_chunks=False,
    estimate_chunks=False,
    skip_done=True,
    smart_chunk=False,
):

    state_name = strip_punct(str(subsystem.state))
    mech_name = strip_punct(str(subsystem.indices2nodes(mechanism)))
    purv_name = strip_punct(str(subsystem.indices2nodes(purview)))

    pickle_dir = Path(f"pickles/{state_name}/purvs/{mech_name}")
    filename = f"{pickle_dir}/{mech_name}_{purv_name}_{str(direction)}.pkl"

    if not skip_done or not os.path.exists(filename):

        all_partitions = pyphi.partition.mip_partitions(
            mechanism, purview, subsystem.node_labels
        )
        n_chunks = None

        if smart_chunk:
            n = len(mechanism + purview)
            stirling_n = sum(
                sympy.functions.combinatorial.numbers.stirling(n, x) for x in range(4)
            )
            n_parts = int(stirling_n)
            n_chunks = 160
            chunksize = int(n_parts / n_chunks)
            print(
                f"Estimated partitions to compute: {n_parts}\nNumber of chunks: {n_chunks}\nSmart chunkzise set to: {chunksize}\n"
            )

        chunks = chunk_iterable(all_partitions, chunksize)

        if estimate_chunks and not count_chunks and not smart_chunk:
            n = len(mechanism + purview)
            stirling_n = sum(
                sympy.functions.combinatorial.numbers.stirling(n, x) for x in range(4)
            )
            n_parts = int(stirling_n)
            n_chunks = int(n_parts / chunksize)
            print(
                f"Estimated partitions to compute: {n_parts}\nNumber of chunks: {n_chunks}\nChunkzise set to: {chunksize}"
            )

        if count_chunks:
            chunks, chunks_copy = itertools.tee(chunks)
            print("Counting chunks...")
            n_chunks = count_iterable(chunks_copy)
            print(f"Number of chunks: {n_chunks}\nChunkzise set to: {chunksize}")

        parts = Parallel(n_jobs, verbose=10, backend="multiprocessing")(
            delayed(compute_partition_chunk)(
                subsystem, direction, mechanism, purview, chunk
            )
            for chunk in tqdm_notebook(chunks, total=n_chunks)
        )
        #     print(parts)
        phis = [part[0] for part in parts]
        mips = [part[1] for part in parts]

        mip_i = np.argmin(phis)
        phi = phis[mip_i]
        mip = mips[mip_i]

        uncut_repertoire = subsystem.repertoire(direction, mechanism, purview)
        cut_repertoire = subsystem.partitioned_repertoire(direction, mip)

        ria_data = [
            phi,
            direction,
            mechanism,
            purview,
            mip,
            uncut_repertoire,
            cut_repertoire,
            subsystem.node_labels,
        ]

        ria = RepertoireIrreducibilityAnalysis(*ria_data)

        irr_purv = MaximallyIrreducibleCauseOrEffect(ria)

        if not os.path.isdir(pickle_dir):
            os.makedirs(pickle_dir)

        with open(f"{filename}", "wb") as f:
            pickle.dump(irr_purv, f)

        return irr_purv

    else:
        print(f"Purview already computed: {filename}")


def stirling2(n, m):
    ## STIRLING2 computes the Stirling numbers of the second kind.
    #    S2(N,M) represents the number of distinct partitions of N elements
    #    into M nonempty sets.  For a fixed N, the sum of the Stirling
    #    numbers S2(N,M) is represented by B(N), called "Bell's number",
    #    and represents the number of distinct partitions of N elements.
    import numpy as np

    s2 = np.zeros((n, m))

    if n <= 0:
        return s2

    if m <= 0:
        return s2

    s2[0, 0] = 1
    for j in range(1, m):
        s2[0, j] = 0

    for i in range(1, n):

        s2[i, 0] = 1

        for j in range(1, m):
            s2[i, j] = (j + 1) * s2[i - 1, j] + s2[i - 1, j - 1]

    return s2


def compute_ks_relations(subsystem, mechanisms, ks):
    all_purviews = pyphi.relations.separate_ces(mechanisms)
    ks_relations = []
    for k in ks:
        relata = [
            pyphi.relations.Relata(subsystem, pair)
            for pair in itertools.combinations(all_purviews, k)
        ]
        k_relations = [pyphi.relations.relation(relatum) for relatum in relata]
        k_relations = list(filter(lambda r: r.phi > 0, k_relations))
        ks_relations.extend(k_relations)
    return ks_relations


def compute_k_relations_chunk(chunk):
    relata = chunk
    k_relations = [pyphi.relations.relation(relatum) for relatum in relata]
    k_relations = list(filter(lambda r: r.phi > 0, k_relations))
    return k_relations


def parallcompute_ks_relations(subsystem, mechanisms, ks, n_jobs=-1, chunk_size=5000):
    all_purviews = pyphi.relations.separate_ces(mechanisms)
    ks_relations = []
    for k in ks:
        relata = [
            pyphi.relations.Relata(subsystem, pair)
            for pair in itertools.combinations(all_purviews, k)
        ]
        chunks = chunk_iterable(relata, chunk_size)
        k_relations = Parallel(n_jobs=n_jobs, verbose=10, backend="multiprocessing")(
            delayed(compute_k_relations_chunk)(chunk) for chunk in tqdm_notebook(chunks)
        )
        k_relations_flat = list(itertools.chain.from_iterable(k_relations))
        ks_relations.extend(k_relations_flat)

    return ks_relations


def parallcompute_ks_relations_sep(
    subsystem, separated_ces, ks, n_jobs=-1, chunk_size=5000
):
    all_purviews = separated_ces
    ks_relations = []
    for k in ks:
        relata = [
            pyphi.relations.Relata(subsystem, pair)
            for pair in itertools.combinations(all_purviews, k)
        ]
        chunks = chunk_iterable(relata, chunk_size)
        k_relations = Parallel(n_jobs=n_jobs, verbose=10, backend="multiprocessing")(
            delayed(compute_k_relations_chunk)(chunk) for chunk in tqdm_notebook(chunks)
        )
        k_relations_flat = list(itertools.chain.from_iterable(k_relations))
        ks_relations.extend(k_relations_flat)

    return ks_relations


# def parallcompute_k_relations(subsystem, mechanisms, k, n_jobs=-1, chunk_size=5000):
#     all_purviews = pyphi.relations.separate_ces(mechanisms)
#     relata = [pyphi.relations.Relata(subsystem, pair) for pair in itertools.combinations(all_purviews, k)]
#     chunks = chunk_iterable(relata, chunk_size)
#     k_relations = Parallel(n_jobs=-1, verbose=10, backend='multiprocessing')(
#             delayed(compute_k_relations_chunk)(chunk) for chunk in tqdm_notebook(chunks)
#         )
#     k_relations_flat = list(itertools.chain.from_iterable(k_relations))

#     return k_relations_flat

# def sum_node_relphis_purvcount(subsystem, relations, return_df=False, state=None):
#     if state:
#         nodes_relphis = [
#         [r[0].phi for r in relations if node in flatten(r[0].purview) and r[1][node]==state]
#         for node in subsystem.node_indices
# ]
#     else:
#         nodes_relphis = [
#             [r[0].phi for r in relations if node in r[0].purview]
#             for node in subsystem.node_indices
# ]
#     totals = [sum(node_relphis) for node_relphis in nodes_relphis]
#     if return_df:
#         df = pd.DataFrame(totals, index=subsystem.node_labels, columns=['SUM(relation_phis)'])
#         return df
#     else:
#         return totals


def sum_node_relphis_purvcount(subsystem, relations, state, return_df=False):

    nodes_relphis = [
        [
            r[0].phi
            for r in relations
            if node in flatten(r[0].purview) and r[1][node] == state
        ]
        for node in subsystem.node_indices
    ]

    totals = [sum(node_relphis) for node_relphis in nodes_relphis]
    if return_df:
        df = pd.DataFrame(
            totals, index=subsystem.node_labels, columns=["SUM(relation_phis)"]
        )
        return df
    else:
        return totals


def sum_node_relphis_mechcount(subsystem, relations, return_df=False):

    nodes_relphis = [
        [r[0].phi for r in relations if node in flatten(r[0].mechanisms)]
        for node in subsystem.node_indices
    ]

    totals = [sum(node_relphis) for node_relphis in nodes_relphis]
    if return_df:
        df = pd.DataFrame(
            totals, index=subsystem.node_labels, columns=["SUM(relation_phis)"]
        )
        return df
    else:
        return totals


def plot_df(df, title=None):
    fig = px.bar(df, title=title)
    fig.show()


######Compositional state


def compositional_state_from_system_state(state):

    # if a single state is sent in, it assumes thae cause and effect states are the same
    if type(state) is tuple:
        cause_state = state
        effect_state = state
    else:
        cause_state = state[0]
        effect_state = state[1]

    # returns a dictionary with cause and effect compositional states as sub-dictionaries
    # the subdictionaries have purview indices as keys and their state as value
    return {
        direction: {
            subset_elements: tuple(state[i] for i in subset_elements)
            for subset_elements in pyphi.utils.powerset(
                range(len(state)), nonempty=True
            )
        }
        for state, direction in zip(
            [cause_state, effect_state],
            [pyphi.direction.Direction.CAUSE, pyphi.direction.Direction.EFFECT],
        )
    }


def filter_ces_by_compositional_state(ces, compositional_state):

    # first separate the ces into mices and define the directions
    c = pyphi.direction.Direction.CAUSE
    e = pyphi.direction.Direction.EFFECT

    # next we run through all the mices and append any mice that has a state corresponding to the compositional state
    mices_with_correct_state = [
        mice
        for mice in ces
        if (
            tuple([rel.maximal_state(mice)[0][i] for i in mice.purview])
            == compositional_state[mice.direction][mice.purview]
        )
    ]

    # next we find the set of purviews (congruent with the compositional state) specified by the system
    cause_purviews = set(
        [mice.purview for mice in mices_with_correct_state if mice.direction == c]
    )
    effect_purviews = set(
        [mice.purview for mice in mices_with_correct_state if mice.direction == e]
    )

    # the following two loops do the filtering, and are identical except the first does cause and the other effect
    # If the same purview is specified by multiple mechinisms, we only keep the one with max phi
    filtered_ces = []
    for purview in cause_purviews:
        mices = list(
            filter(
                lambda mice: mice.direction == c and mice.purview == purview,
                mices_with_correct_state,
            )
        )
        filtered_ces.append(mices[np.argmax([mice.phi for mice in mices])])

    for purview in effect_purviews:
        mices = list(
            filter(
                lambda mice: mice.direction == e and mice.purview == purview,
                mices_with_correct_state,
            )
        )
        filtered_ces.append(mices[np.argmax([mice.phi for mice in mices])])

    return pyphi.models.CauseEffectStructure(filtered_ces)


def filter_relations(relations, filtered_ces):
    return list(
        filter(
            lambda r: all([relatum in filtered_ces for relatum in r.relata]), relations
        )
    )


def get_all_compositional_states(ces):
    c = pyphi.direction.Direction.CAUSE
    e = pyphi.direction.Direction.EFFECT

    # now we make a nested dict that contains all the purviews with their related states, mechanisms, and phi values
    all_purviews = {c: dict(), e: dict()}

    # we loop through every mice in the separated CES
    for mice in ces:

        # define some variables for later use
        purview = mice.purview
        mechanism = mice.mechanism
        max_state = tuple([rel.maximal_state(mice)[0][element] for element in purview])
        direction = mice.direction
        phi = mice.phi

        # check if the purview is already in our dict
        if len(max_state) > 0:
            if purview in all_purviews[direction]:
                # check if the purview is already represented with this state
                if not max_state in all_purviews[direction][purview]:
                    all_purviews[direction][purview].append(max_state)
            else:
                all_purviews[direction][purview] = [max_state]

    cause_states = [
        {
            purv: purview_state
            for purv, purview_state in zip(all_purviews[c].keys(), purview_states)
        }
        for purview_states in product(*all_purviews[c].values())
    ]
    effect_states = [
        {
            purv: purview_state
            for purv, purview_state in zip(all_purviews[e].keys(), purview_states)
        }
        for purview_states in product(*all_purviews[e].values())
    ]

    all_compositional_states = [
        {c: c_state, e: e_state}
        for c_state, e_state in product(cause_states, effect_states,)
    ]
    return all_compositional_states


def filter_using_sum_of_phi(ces, relations, all_compositional_states):
    phis = {
        i: sum(
            [
                sum([c.phi for c in filter_ces_by_compositional_state(ces, comp)]),
                sum(
                    [
                        r.phi
                        for r in filter_relations(
                            relations, filter_ces_by_compositional_state(ces, comp)
                        )
                    ]
                ),
            ]
        )
        for i, comp in (
            enumerate(
                tqdm(all_compositional_states)
                if len(all_compositional_states) > 100
                else enumerate(all_compositional_states)
            )
        )
    }

    max_compositional_state = all_compositional_states[
        list(phis.values()).index(max(phis.values()))
    ]
    filtered_ces = filter_ces_by_compositional_state(ces, max_compositional_state)
    filtered_relations = filter_relations(relations, filtered_ces)
    return filtered_ces, filtered_relations, max_compositional_state


def get_filtered_ces(ces, state, system=None):

    if state == "actual":
        state = tuple(system.state[i] for i in system.node_indices)
        state = compositional_state_from_system_state(state)
        filtered_ces = filter_ces_by_compositional_state(ces, state)

    elif type(state) is tuple or type(state) is list:
        state = compositional_state_from_system_state(state)
        filtered_ces = filter_ces_by_compositional_state(ces, state)

    elif type(state) is dict:
        filtered_ces = filter_ces_by_compositional_state(ces, state)
        filtered_rels = filter_relations(relations, filtered_ces)

    elif state == "max_phi":
        all_compositional_states = get_all_compositional_states(ces)
        filtered_ces, filtered_rels, state = filter_using_sum_of_phi(
            ces, [], all_compositional_states
        )

    return (
        filtered_ces,
        state,
    )


def get_filtered_ces_and_rels(ces, relations, state, system=None):
    if state == "actual":
        state = tuple(system.state[i] for i in system.node_indices)
        state = compositional_state_from_system_state(state)
        filtered_ces = filter_ces_by_compositional_state(ces, state)
        filtered_rels = filter_relations(relations, filtered_ces)

    elif type(state) is tuple or type(state) is list:
        state = compositional_state_from_system_state(state)
        filtered_ces = filter_ces_by_compositional_state(ces, state)
        filtered_rels = filter_relations(relations, filtered_ces)

    elif type(state) is dict:
        filtered_ces = filter_ces_by_compositional_state(ces, state)
        filtered_rels = filter_relations(relations, filtered_ces)

    elif state == "max_phi":
        all_compositional_states = get_all_compositional_states(ces)
        filtered_ces, filtered_rels, state = filter_using_sum_of_phi(
            ces, relations, all_compositional_states
        )

    return (
        filtered_ces,
        filtered_rels,
        state,
    )


CAUSE = pyphi.direction.Direction(0)
EFFECT = pyphi.direction.Direction(1)


def distinction_touched(mice, part1, part2, direction):
    mechanism_cut = all(
        [any([m in part for m in mice.mechanism]) for part in [part1, part2]]
    )
    purview_cut = all(
        [any([p in part for p in mice.purview]) for part in [part1, part2]]
    )
    connection_cut = (
        all([m in part1 for m in mice.mechanism])
        and all([p in part2 for p in mice.purview])
        and mice.direction == direction
    )
    return mechanism_cut or purview_cut or connection_cut


def relation_untouched(untouched_ces, relation):
    relata_in_ces = all([relatum in untouched_ces for relatum in relation.relata])
    return relata_in_ces


def get_big_phi(ces, relations, indices):
    sum_of_small_phi = sum([mice.phi for mice in ces]) + sum([r.phi for r in relations])

    partitions = [
        part
        for part in pyphi.partition.bipartition(indices)
        if all([len(p) > 0 for p in part])
    ]

    min_phi = np.inf
    for parts in (
        tqdm(partitions, desc="System partitions")
        if len(partitions) > 100 and len(indices) > 4
        else partitions
    ):
        for p1, p2, direction in product(parts, parts, [CAUSE, EFFECT]):
            if not p1 == p2:
                untouched_ces = pyphi.models.CauseEffectStructure(
                    [
                        mice
                        for mice in ces
                        if not distinction_touched(mice, p1, p2, direction)
                    ]
                )
                untouched_relations = [
                    r for r in relations if relation_untouched(untouched_ces, r)
                ]

                sum_phi_untouched = sum([mice.phi for mice in untouched_ces]) + sum(
                    [r.phi for r in untouched_relations]
                )

                lost_phi = sum_of_small_phi - sum_phi_untouched

                if lost_phi < min_phi:
                    min_phi = lost_phi
                    min_cut = parts, p1, p2, direction

    big_phi = (sum_of_small_phi / 2 ** len(indices)) * (sum_of_small_phi - min_phi)
    return big_phi, min_cut


def compute_rels_and_ces_for_compositional_state(system, state, ces, max_k=3):

    if type(state) is tuple:
        state = compositional_state_from_system_state(state)
    filtered_ces = filter_ces_by_compositional_state(ces, state)
    relations = compute_relations(system, filtered_ces, max_k=max_k)

    return (
        filtered_ces,
        relations,
        state,
    )


def get_all_ces(system, ces=None):

    # unfold ces
    if ces == None:
        directions = [pyphi.direction.Direction.CAUSE, pyphi.direction.Direction.EFFECT]
        print("unfolding CES")
        ces = pyphi.models.subsystem.CauseEffectStructure(
            [
                system.find_mice(d, m)
                for d in directions
                for m in pyphi.utils.powerset(system.node_indices, nonempty=True)
            ],
            subsystem=system,
        )
        for m in ces:
            m.node_labels = system.node_labels

    all_ces = []
    compositional_states = get_all_compositional_states(ces)
    big_phi = 0
    for compositional_state in tqdm(compositional_states):
        (
            filtered_ces,
            filtered_relations,
            compositional_state,
        ) = compute_rels_and_ces_for_compositional_state(
            system, compositional_state, ces
        )
        phi, cut = get_big_phi(filtered_ces, filtered_relations, system.node_indices)

        all_ces.append(
            {
                "ces": filtered_ces,
                "relations": filtered_relations,
                "compositional_state": compositional_state,
                "big phi": phi,
            }
        )

    return all_ces


def get_maximal_ces(system, ces=None, max_k=3):

    # unfold ces
    if ces == None:
        directions = [pyphi.direction.Direction.CAUSE, pyphi.direction.Direction.EFFECT]
        print("unfolding CES")
        ces = pyphi.models.subsystem.CauseEffectStructure(
            [
                system.find_mice(d, m)
                for d in directions
                for m in pyphi.utils.powerset(system.node_indices, nonempty=True)
            ],
            subsystem=system,
        )
        for m in ces:
            m.node_labels = system.node_labels

    compositional_states = get_all_compositional_states(ces)
    big_phi = 0
    for compositional_state in tqdm(
        compositional_states, desc="Computing Big Phi for all compositional states"
    ):
        (
            filtered_ces,
            filtered_relations,
            compositional_state,
        ) = compute_rels_and_ces_for_compositional_state(
            system, compositional_state, ces, max_k
        )
        phi, cut = get_big_phi(filtered_ces, filtered_relations, system.node_indices)

        if phi > big_phi:
            maximal = {
                "ces": filtered_ces,
                "relations": filtered_relations,
                "compositional_state": compositional_state,
                "big phi": phi,
            }
            big_phi = phi

    return maximal


def i2n(indices, subsystem):
    return subsystem.indices2nodes(indices)


def ces_to_df(separated_ces, subsystem):
    s = subsystem
    ces_lists = [
        (
            i2n(m.mechanism, s),
            m.direction,
            i2n(m.purview, s),
            m.phi,
            [rel.maximal_state(m)[0][i] for i in m.purview],
        )
        for m in separated_ces
    ]
    return pd.DataFrame(
        ces_lists, columns=["mechanism", "direction", "purview", "phi", "state"]
    )


def add_node_labels(mice, system):
    mice.node_labels = tuplle()
    return mice


def unfold_separated_ces(system):
    CAUSE = pyphi.direction.Direction(0)
    EFFECT = pyphi.direction.Direction(1)
    purviews = tuple(pyphi.utils.powerset(system.node_indices, nonempty=True))

    mices = [
        system.find_mice(direction, mechanism, purviews=purviews)
        for mechanism in pyphi.utils.powerset(system.node_indices, nonempty=True)
        for direction in [CAUSE, EFFECT]
    ]

    return pyphi.models.CauseEffectStructure(mices)


def compute_relations(subsystem, ces, max_k=3):
    ks_relations = []
    for k in range(2, max_k + 1):
        relata = [
            rel.Relata(subsystem, mices)
            for mices in itertools.combinations(ces, k)
            if all([mice.phi > 0 for mice in mices])
        ]
        k_relations = [
            rel.relation(relatum)
            for relatum in (tqdm(relata) if len(relata) > 5000 else relata)
        ]
        k_relations = list(filter(lambda r: r.phi > 0, k_relations))
        ks_relations.extend(k_relations)
    return ks_relations


def all_relations(subsystem, sepces):
    """Return all relations, even those with zero phi."""
    # Relations can be over any combination of causes/effects in the CES, so we
    # get a flat list of all causes and effects
    #     ces = rel.separate_ces(ces)
    ces = sepces
    # Compute all relations
    return map(
        rel.relation,
        (
            rel.Relata(subsystem, subset)
            for subset in filter(
                lambda purviews: len(purviews) > 1,
                pyphi.utils.powerset(ces, nonempty=True),
            )
        ),
    )


def relations(subsystem, sepces):
    """Return the irreducible relations among the causes/effects in the CES."""
    return filter(None, all_relations(subsystem, sepces))

