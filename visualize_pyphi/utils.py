import pandas as pd
import string
import itertools

from pyrsistent import pset
import pyphi
from joblib import Parallel, delayed
import pickle
from tqdm.auto import tqdm
import ray
import glob
import toolz
import numpy as np
from visualize_pyphi import gist_bigphi, network_generator, visualize_ces
from pyphi.models.subsystem import FlatCauseEffectStructure as sep
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import igraph as ig
import sys
import plotly.express as px


def done():
    display(Audio(filename="/home/mgrasso/projects/chime2.wav", autoplay=True))


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


def i2n(mech, subsystem):
    return strp(subsystem.indices2nodes(mech))


def strip_punct(s):
    return str(
        s.translate(str.maketrans({key: None for key in string.punctuation})).replace(
            " ", ""
        )
    )


def strp(x):
    return strip_punct(str(x))


def sepces2df(sepces, subsystem, csv_name=None):
    s = subsystem
    ces_list = [
        (
            strp(i2n(m.mechanism, s)),
            m.direction.name,
            strp(i2n(m.purview, s)),
            strp(m.specified_state),
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


def ces2df(ces, state_as_lettercase=True):
    s = ces[0].subsystem

    if not state_as_lettercase:
        ces_list = [
            (
                strp(i2n(d.mechanism, s)),
                strp(d.mechanism_state),
                strp(i2n(d.cause_purview, s)),
                strp(d.cause.specified_state),
                d.cause.phi,
                strp(i2n(d.effect_purview, s)),
                strp(d.effect.specified_state),
                d.effect.phi,
            )
            for d in ces
        ]

        df = pd.DataFrame(
            ces_list,
            index=[n for n in range(1, len(ces_list) + 1)],
            columns=[
                "mechanism",
                "state",
                "cause",
                "state",
                "phi",
                "effect",
                "state",
                "phi",
            ],
        )
    else:
        ces_list = [
            (
                lettercase_state(
                    d.mechanism, node_labels=s.network.node_labels, state=s.state
                ),
                lettercase_state(
                    d.cause_purview,
                    node_labels=s.node_labels,
                    state=d.cause.specified_state.tolist()[0],
                ),
                lettercase_state(
                    d.effect_purview,
                    node_labels=s.node_labels,
                    state=d.effect.specified_state.tolist()[0],
                ),
                d.phi,
            )
            for d in ces
        ]

        df = pd.DataFrame(
            ces_list,
            index=[n for n in range(1, len(ces_list) + 1)],
            columns=[
                "mechanism",
                "cause",
                "effect",
                "phi",
            ],
        )

    return df


def ces2df_supertexts(ces, subsystem):
    s = subsystem
    ces_list = [
        (
            strp(i2n(d.mechanism, s)),
            strp(d.mechanism_state),
            strp(i2n(d.cause_purview, s)),
            strp(d.cause.specified_state),
            d.cause.phi,
            strp(i2n(d.effect_purview, s)),
            strp(d.effect.specified_state),
            d.effect.phi,
        )
        for d in ces
    ]
    ces_list = ces_list + [None]

    df = pd.DataFrame(
        ces_list,
        index=[n for n in range(1, len(ces_list) + 1)],
        columns=[
            "mechanism",
            "state",
            "cause",
            "state",
            "phi",
            "effect",
            "state",
            "phi",
        ],
    )

    return df


def chunk_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def compute_k_relations_chunk(chunk):
    relata = chunk
    k_relations = [pyphi.relations.relation(relatum) for relatum in relata]
    k_relations = list(filter(lambda r: r.phi > 0, k_relations))
    return k_relations


def parallcompute_ks_relations(
    subsystem,
    separated_ces,
    ks,
    n_jobs=-1,
    chunk_size=5000,
    verbose=5,
):
    all_purviews = separated_ces
    ks_relations = []
    for k in ks:
        relata = [
            pyphi.relations.Relata(subsystem, pair)
            for pair in itertools.combinations(all_purviews, k)
        ]
        chunks = chunk_iterable(relata, chunk_size)
        k_relations = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
            delayed(compute_k_relations_chunk)(chunk) for chunk in tqdm(chunks)
        )
        k_relations_flat = list(itertools.chain.from_iterable(k_relations))
        ks_relations.extend(k_relations_flat)

    return ks_relations


def pklthis(this, name):
    with open(name, "wb") as f:
        pickle.dump(this, f)


def jsonthis(this, name):
    with open(name, "wt") as f:
        pyphi.jsonify.dump(this, f)


def loadpkl(name):
    with open(name, "rb") as f:
        return pickle.load(f)


@ray.remote
def compute_mice(subsystem, direction, mechanism, purview):
    # Compute a single mice
    mice = subsystem.find_mice(direction, mechanism, (purview,))
    return mice


def parallcompute_distinction(subsystem, mechanism, dispoutput=False):
    # Compute all potential mices of a distinction in parallel
    potential_causes = subsystem.network.potential_purviews(
        pyphi.Direction.CAUSE, mechanism
    )
    potential_effects = subsystem.network.potential_purviews(
        pyphi.Direction.EFFECT, mechanism
    )
    if dispoutput:
        print(f"Evaluating {len(potential_causes)} causes...")
    futures_causes = [
        compute_mice.remote(subsystem, pyphi.Direction.CAUSE, mechanism, purview)
        for purview in potential_causes
    ]
    causes = ray.get(futures_causes)
    if causes:
        cause = max(causes)
        if dispoutput:
            print(cause)
            print(f"Evaluating {len(potential_effects)} effects...")
    else:
        return []

    futures_effects = [
        compute_mice.remote(subsystem, pyphi.Direction.EFFECT, mechanism, purview)
        for purview in potential_effects
    ]
    effects = ray.get(futures_effects)
    if effects:
        effect = max(effects)
    else:
        return []
    if dispoutput:
        print(effect)

    return pyphi.models.Concept(mechanism, cause, effect, subsystem)


def decompress_relations(subsystem, ces, pickles_path):
    pickles = sorted(sorted(glob.glob(pickles_path)), key=len)
    pickles = [loadpkl(p) for p in tqdm(pickles)]
    indirect_relations = list(toolz.concat(pickles))
    return [
        pyphi.relations.Relation.from_indirect_json(subsystem, sep(ces), r)
        for r in tqdm(indirect_relations)
    ]


def compute_relations_by_k(
    subsystem,
    ces,
    filepath_and_name,
    ks=None,
    missing_ks_only=True,
    batch_size=10000,
    verbose=5,
    n_jobs=160,
):
    relations_pickles = sorted(
        sorted(glob.glob(f"{filepath_and_name}_krels_*.pkl")), key=len
    )
    if missing_ks_only:
        missing_ks = [
            k
            for k in range(2, len(sep(ces)) + 1)
            if f"{filepath_and_name}_krels_{k}.pkl" not in relations_pickles
        ]
    for k in tqdm(missing_ks):
        print(f"Computing k={k}...")
        relations = list(
            pyphi.relations.relations(
                subsystem,
                sep(ces),
                min_degree=k,
                max_degree=k,
                parallel=True,
                parallel_kwargs=dict(
                    batch_size=batch_size, verbose=verbose, n_jobs=n_jobs
                ),
            )
        )
        compressed_relations = [r.to_indirect_json(sep(ces)) for r in relations]
        pklthis(compressed_relations, f"{filepath_and_name}_krels_{k}.pkl")
    print("Success!")


def plotLogFunc(l, k, x0, start=None, stop=None, num=101, title=None):
    if start is None:
        start = x0 - 1
    if stop is None:
        stop = x0 + 1
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(start, stop, num)
    y = network_generator.LogFunc(x, l, k, x0)
    fig.patch.set_facecolor("white")
    plt.ylabel("p")
    plt.xlabel("input")
    plt.suptitle(title)
    return ax.plot(x, y)


@ray.remote
def compute_relations(subsystem, ces):
    return pyphi.relations.relations(subsystem, ces)


def lettercase_state(node_indices, node_labels=None, state=False):

    node_labels = node_labels.indices2labels(node_indices)

    nl = []
    # capitalizing labels of mechs that are on
    if not len(state) == len(node_labels):
        for n, i in zip(node_labels, node_indices):
            if state[i] == 0:
                nl.append(n.lower() + "")
            else:
                nl.append(n.upper() + "")
    else:
        for s, l in zip(state, node_labels):
            if s == 0:
                nl.append(l.lower() + "")
            else:
                nl.append(l.upper() + "")

    node_labels = nl

    return "".join(node_labels)


# Check for ties
def find_state_ties(ces):
    return [
        c
        for c in ces
        if len(c.cause.specified_state) != 1 or len(c.effect.specified_state) != 1
    ]


def find_purview_ties(ces):
    return [
        c
        for c in ces
        if len(list(c.cause.ties())) != 1 or len(list(c.effect.ties())) != 1
    ]


def independence_number(
    distinctions: pyphi.models.subsystem.CauseEffectStructure,
) -> int:
    g, _ = pyphi.big_phi.conflict_graph(distinctions)
    g = ig.Graph.from_networkx(g)
    return g.independence_number()


def find_reflexive_distinctions(ces):
    reflexive_ds = []
    for d in ces:
        if d.mechanism == d.cause_purview and d.mechanism == d.effect_purview:
            reflexive_ds.append(d)
    return reflexive_ds


def make_label(node_indices, node_labels=None, state=False):

    if node_labels is None:
        node_labels = [string.ascii_uppercase[n] for n in node_indices]
    else:
        node_labels = node_labels.indices2labels(node_indices)

    if state:
        nl = []
        # capitalizing labels of mechs that are on
        if not len(state) == len(node_labels):
            for n, i in zip(node_labels, node_indices):
                if state[i] == 0:
                    nl.append(n.lower() + "")
                else:
                    nl.append(n.upper() + "")
        else:
            for s, l in zip(state, node_labels):
                if s == 0:
                    nl.append(l.lower() + "")
                else:
                    nl.append(l.upper() + "")

        node_labels = nl

    return "".join(node_labels)


def norm_inclusion(sia):
    n_nodes = sia.subsystem.size
    norm_inclusion = [
        len(p) / 2 ** n_nodes
        for ((o, s), p) in sia.phi_structure.distinctions.purview_inclusion()
        if len(o) == 1
    ]
    if not norm_inclusion:
        return [0, 0]
    else:
        return [min(norm_inclusion), max(norm_inclusion)]


def plot_subsystem_phis(phis, nodes, title=None, width=800, height=600):
    df = pd.DataFrame(phis, index=nodes, columns=["Φ"])
    fig = px.line(df, y="Φ", title=title)
    fig.update_layout(width=width, height=height)

    return fig


# def intrinsic_information(subsystem, mechanism, purview, direction):
#     repertoire = subsystem.repertoire(direction, mechanism, purview)
#     partition = pyphi.partition.complete_partition(mechanism, purview)

#     return max(
#         [
#             subsystem.evaluate_partition(
#                 direction,
#                 mechanism,
#                 purview,
#                 partition,
#                 repertoire=repertoire,
#                 state=state,
#             )[0]
#             for state in pyphi.utils.all_states(len(purview))
#         ]
#     )


def intrinsic_information(subsystem, mechanism, purview, direction):
    repertoire = subsystem.repertoire(direction, mechanism, purview)
    partition = pyphi.partition.complete_partition(mechanism, purview)

    return max(
        [
            subsystem.evaluate_partition(
                direction,
                mechanism,
                purview,
                partition,
                repertoire=repertoire,
                state=state,
            )[0]
            for state in pyphi.utils.all_states(len(purview))
        ]
    )


# def min_horizontal_cut(subsystem, normalization=False):
#     n_nodes = len(subsystem)
#     min_ii = n_nodes
#     min_cut = None
#     for part1 in map(set, pyphi.utils.powerset(range(n_nodes))):
#         if len(part1) == 0 or len(part1) == n_nodes:
#             continue

#         part2 = tuple(set(range(n_nodes)) - part1)
#         part1 = tuple(part1)

#         ii_e_across = intrinsic_information(
#             subsystem, part1, part2, pyphi.direction.Direction.EFFECT
#         )
#         ii_c_across = intrinsic_information(
#             subsystem, part1, part2, pyphi.direction.Direction.CAUSE
#         )
#         part_ii_across = min(ii_e_across, ii_c_across)

#         if normalization:
#             part_ii_across *= n_nodes / min(len(part1), len(part2))

#         if part_ii_across < min_ii:
#             min_ii = part_ii_across
#             min_cut = (part1, part2)
#     return min_ii, min_cut


# def system_integratedness(subsystem, direction, state, partition):
#     n_nodes = len(subsystem)
#     all_nodes = range(n_nodes)
#     repertoire = subsystem.repertoire(direction, mechanism=all_nodes, purview=all_nodes)
#     partitioned_repertoire = subsystem.partitioned_repertoire(
#         direction, partition=partition
#     )
#     return (
#         np.log2(repertoire[state] / partitioned_repertoire[state]).squeeze() ** 2,
#         partitioned_repertoire,
#     )


def system_integratedness(subsystem, direction, state, partition):
    n_nodes = len(subsystem)
    all_nodes = range(n_nodes)
    return subsystem.evaluate_partition(
        direction,
        mechanism=all_nodes,
        purview=all_nodes,
        partition=partition,
        state=state,
    )


def evaluate_system_partition(
    subsystem, sys_effect_state, sys_cause_state, partition, effect_only=False
):
    n_nodes = len(subsystem)
    all_nodes = range(n_nodes)

    if len(partition) == 1:
        cut = pyphi.partition.complete_partition(tuple(all_nodes), tuple(all_nodes))
        max_cut_info = n_nodes
    else:
        cut = pyphi.partition.KPartition(
            *[pyphi.partition.Part(tuple(part), tuple(part)) for part in partition]
        )
        max_cut_info = sum([len(part) * (n_nodes - len(part)) for part in partition])

    effect_info, _ = system_integratedness(
        subsystem,
        direction=pyphi.direction.Direction.EFFECT,
        state=sys_effect_state,
        partition=cut,
    )
    if effect_only:
        cut_info = effect_info
    else:
        cause_info, _ = system_integratedness(
            subsystem,
            direction=pyphi.direction.Direction.CAUSE,
            state=sys_cause_state,
            partition=cut,
        )

        cut_info = min(effect_info, cause_info)
        # print(effect_info, cause_info, max_cut_info, cut_info / max_cut_info)

    return cut_info, cut_info / max_cut_info


def min_vertical_cut(
    subsystem, normalization=True, effect_only=False, complete_partition=False
):
    n_nodes = len(subsystem)
    all_nodes = range(n_nodes)
    sys_effect_state = subsystem.find_maximal_state_under_complete_partition(
        pyphi.direction.Direction.EFFECT,
        mechanism=all_nodes,
        purview=all_nodes,
    )[0]

    sys_cause_state = subsystem.find_maximal_state_under_complete_partition(
        pyphi.direction.Direction.CAUSE,
        mechanism=all_nodes,
        purview=all_nodes,
    )[0]
    min_ii = n_nodes * n_nodes
    min_cut = None
    min_normalized_ii = 1.0

    partitions = pyphi.partition.partitions(all_nodes)
    if not complete_partition:
        partitions = list(partitions)[1:]

    for partition in partitions:
        cut_info, normalized_cut_info = evaluate_system_partition(
            subsystem, sys_effect_state, sys_cause_state, partition, effect_only
        )

        if normalization:
            if normalized_cut_info <= min_normalized_ii:
                min_ii = cut_info
                min_cut = partition
                min_normalized_ii = normalized_cut_info
        else:
            if cut_info <= min_ii:
                min_ii = cut_info
                min_cut = partition
                min_normalized_ii = min_ii

    return min_ii, min_normalized_ii, min_cut, sys_effect_state, sys_cause_state


def newbigphi(
    weights,
    k,
    state=None,
    normalization=True,
    verbose=False,
    cut_style=None,
    effect_only=False,
    all_subsystems=False,
    subsystem_nodes=None,
):
    phis, cuts, normalized_phis = [[], [], []]
    n_nodes = len(weights)
    node_labels = [string.ascii_uppercase[n] for n in range(n_nodes)]
    mech_func = ["l" for n in range(n_nodes)]
    if verbose:
        print("Creating network...")

    network = network_generator.get_net(
        mech_func,
        weights,
        k=k,
        node_labels=node_labels,
    )

    if state is None:
        state = (0,) * n_nodes

    if subsystem_nodes:
        subsystems = [
            pyphi.Subsystem(network, state, nodes) for nodes in subsystem_nodes
        ]

    elif all_subsystems:
        subsystems = [
            pyphi.Subsystem(network, state, nodes)
            for nodes in list(pyphi.utils.powerset(range(n_nodes), nonempty=True))
        ]
    else:
        subsystem_sizes = range(1, n_nodes + 1)
        subsystems = [
            pyphi.Subsystem(network, state, range(subsystem_size))
            for subsystem_size in subsystem_sizes
        ]

    if verbose:
        print("Computing Φ...")

    for subsystem in subsystems:
        if cut_style == "hybrid":
            phi, cut, _, _ = min_hybrid_cut(subsystem, normalization=normalization)
            # print(subsystem, phi, cut)
        elif cut_style == "horizontal":
            phi, cut = min_horizontal_cut(subsystem, normalization=normalization)
        elif cut_style == "vertical":
            phi, normalized_phi, cut, _, _ = min_vertical_cut(
                subsystem,
                normalization=normalization,
                effect_only=effect_only,
            )
            normalized_phis.append(normalized_phi)

        else:
            break

        phis.append(phi)
        cuts.append(cut)

        if verbose:
            print(
                make_label(subsystem.node_indices, subsystem.node_labels, state),
                phi,
                normalized_phi,
            )

    ray.shutdown()
    if cut_style == "vertical":
        return phis, normalizaed_phis, cuts, subsystems
    else:
        return phis, cuts, subsystems


# def newbigphi_nodetpms(
#     weights,
#     k,
#     state=None,
#     normalization=False,
#     verbose=False,
#     cut_style=None,
#     return_ces=False,
#     compute_ces_up_to_order=3,
#     effect_only=False,
#     remote=True,
# ):
#     phis, normalized_phis, cess, cuts = [[], [], [], []]
#     n_nodes = len(weights)
#     node_labels = [string.ascii_uppercase[n] for n in range(n_nodes)]
#     mech_func = ["l" for n in range(n_nodes)]
#     if verbose:
#         print("Creating network...")

#     node_tpms = get_node_tpms(lambda x: network_generator.LogFunc(x, 1, k, 0), weights)
#     network = pyphi.Network(node_tpms, cm=None)

#     if state is None:
#         state = (0,) * n_nodes

#     subsystem_sizes = range(1, n_nodes + 1)

#     if verbose:
#         print("Computing Φ...")
#     for subsystem_size in subsystem_sizes:
#         subsystem = pyphi.Subsystem(network, state, nodes=tuple(range(subsystem_size)))
#         ces = []
#         if cut_style == "horizontal":
#             phi, cut = min_horizontal_cut(subsystem, normalization=normalization)
#             phis.append(phi)
#             cuts.append(cut)
#         elif cut_style == "vertical":
#             phi, normalized_phi, cut, _, _ = min_vertical_cut(
#                 subsystem,
#                 normalization=normalization,
#                 effect_only=effect_only,
#                 remote=remote,
#             )
#             if return_ces:
#                 mechanisms = [
#                     tuple(range(x))
#                     for x in range(1, min(subsystem_size, compute_ces_up_to_order) + 1)
#                 ]
#                 ces = [
#                     parallcompute_distinction(subsystem, mechanism)
#                     for mechanism in mechanisms
#                 ]

#             phis.append(phi)
#             normalized_phis.append(normalized_phi)
#             cuts.append(cut)
#             cess.append(ces)
#         else:
#             break
#         if verbose:
#             print(
#                 make_label(subsystem.node_indices, subsystem.node_labels, state),
#                 phi,
#                 normalized_phi,
#             )
#             if ces:
#                 print("\n", ces2df(ces))

#     ray.shutdown()
#     return phis, normalized_phis, cess, cuts


def get_node_tpms(
    mech_func,
    weights,
    state_domain=[-1, 1],
):
    node_indices = [n for n in range(len(weights))]

    tpms = {}

    for i in node_indices:
        inputs = weights[:, i]
        input_indices = [n for n, value in enumerate(inputs) if value]
        input_states = pyphi.utils.all_states(len(input_indices))

        node_tpm = []
        for state in input_states:

            if state_domain == [-1, 1]:
                state = [2 * v - 1 for v in state]

            p = mech_func(
                sum(state * np.array([weights[:, i][n] for n in input_indices]))
            )
            node_tpm.append(p)

            tpms[tuple(input_indices + [i])] = node_tpm

    return tpms


def weights_matrix(input_indices, input_weights):
    """
        Returns an array of the connection weight between nodes.

    Args:
        - input_indices: list(tuple(ints)), each tuple specifies the inputs to the current node (whose index is its position in the list)
        - inpus_weights: list(list(floats)), each list specifies the input weights from each input node (defined in input_indices) to the current node (whose index is its position in the list)

    Example:
        input_indices = [(1,),(0,1,2),(0,2)]
        input_weights = [[1.0],[0.1,0.8,0.1],[0.2,0.8]]
        utils.weights_matrix(input_indices,input_weights)
    Returns:
        array([
            [0. , 0.1, 0.2],
            [1. , 0.8, 0. ],
            [0. , 0.1, 0.8]
            ])
    """
    inputs_weights = list(zip(input_indices, input_weights))
    n = len(inputs_weights)
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i in inputs_weights[j][0]:
                weights[i, j] = inputs_weights[j][1][inputs_weights[j][0].index(i)]
    return weights


def L_weights(n_nodes, s):
    l = (1 - s) / (n_nodes - 1)
    weights = np.full((n_nodes, n_nodes), l)
    for i in range(n_nodes):
        weights[i, i] = s
    return weights


def min_hybrid_cut(subsystem, normalization=False):
    # NOTE: Tie-breaking happens here when we access the first element.
    sys_effect_state = subsystem.find_maximal_state_under_complete_partition(
        pyphi.direction.Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )[0]
    sys_cause_state = subsystem.find_maximal_state_under_complete_partition(
        pyphi.direction.Direction.CAUSE,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )[0]

    min_phi = len(subsystem)
    min_cut = None
    for part in pyphi.utils.powerset(subsystem.node_indices, nonempty=True):
        if len(part) == len(subsystem):
            # partition = pyphi.partition.complete_partition(
            #     subsystem.node_indices,
            #     subsystem.node_indices,
            # )
            # max_phi = len(subsystem) ** 2
            continue
        else:
            partition = pyphi.partition.KPartition(
                pyphi.partition.Part(mechanism=part, purview=part),
                node_labels=subsystem.node_labels,
            )
            max_phi = len(part) * (len(subsystem) - len(part))

        # comparing pi(part|system) vs pi(part|part)
        phi_e, _ = subsystem.evaluate_partition(
            pyphi.direction.Direction.EFFECT,
            mechanism=subsystem.node_indices,
            purview=part,
            partition=partition,
            state=tuple(
                [sys_effect_state[subsystem.node_indices.index(node)] for node in part]
            ),
        )

        phi_c, _ = subsystem.evaluate_partition(
            pyphi.direction.Direction.CAUSE,
            mechanism=subsystem.node_indices,
            purview=part,
            partition=partition,
            state=tuple(
                [sys_cause_state[subsystem.node_indices.index(node)] for node in part]
            ),
        )

        phi = min(phi_e, phi_c)
        nphi = phi * len(subsystem) / max_phi
        # print(f"{partition} {phi} {nphi} \n")

        if normalization:
            phi *= len(subsystem) / max_phi

        if phi <= min_phi:
            min_phi = phi
            min_cut = (part,)
    return min_phi, min_cut, sys_effect_state, sys_cause_state



from itertools import product


def gen_all_deterministic_tpms(n_nodes):
    n_states = 2 ** n_nodes
    for comb in product(*[range(n_states) for _ in range(n_states)]):
        sbs = np.zeros((n_states, n_states))
        for i, j in enumerate(comb):
            sbs[i, j] = 1
        yield sbs


def L_weights(n_nodes, s):
    l = (1 - s) / (n_nodes - 1)
    weights = np.full((n_nodes, n_nodes), l)
    for i in range(n_nodes):
        weights[i, i] = s
    return weights


def copyloop_weights(n_nodes):
    weights = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        weights[i - 1, i] = 1
    return weights


def max_information_purview(subsystem, part1, part2, sys_effect_state, sys_cause_state):
    phi_e = max(
        [
            subsystem.evaluate_partition(
                pyphi.direction.Direction.EFFECT,
                mechanism=part1,
                purview=purview,
                partition=pyphi.partition.complete_partition(
                    mechanism=part1,
                    purview=purview,
                ),
                state=pyphi.utils.state_of(
                    [subsystem.node_indices.index(n) for n in purview],
                    sys_effect_state,
                ),
            )[0]
            for purview in pyphi.utils.powerset(part2, nonempty=True)
        ]
    )

    phi_c = max(
        [
            subsystem.evaluate_partition(
                pyphi.direction.Direction.CAUSE,
                mechanism=part1,
                purview=purview,
                partition=pyphi.partition.complete_partition(
                    mechanism=part1,
                    purview=purview,
                ),
                state=pyphi.utils.state_of(
                    [subsystem.node_indices.index(n) for n in purview],
                    sys_cause_state,
                ),
            )[0]
            for purview in pyphi.utils.powerset(part2, nonempty=True)
        ]
    )

    return min(phi_e, phi_c)


def min_horizontal_cut(subsystem, normalization=True):
    # NOTE: Tie-breaking happens here when we access the first element.
    sys_effect_state = subsystem.find_maximal_state_under_complete_partition(
        pyphi.direction.Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )[0]
    sys_cause_state = subsystem.find_maximal_state_under_complete_partition(
        pyphi.direction.Direction.CAUSE,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )[0]

    min_phi = len(subsystem)
    min_normalized_phi = 1.0
    min_cut = None
    for part1, part2 in pyphi.partition.directed_bipartition(
        subsystem.node_indices, nontrivial=True
    ):
        if len(part1) == len(subsystem) or len(part2) == len(subsystem):
            continue
        else:
            partition = pyphi.partition.complete_partition(
                mechanism=part2,
                purview=part1,
            )

        # comparing pi(part1|part2) vs pi(part1)
        phi_e, _ = subsystem.evaluate_partition(
            pyphi.direction.Direction.EFFECT,
            mechanism=part2,
            purview=part1,
            partition=partition,
            state=pyphi.utils.state_of(
                [subsystem.node_indices.index(n) for n in part1], sys_effect_state
            ),
        )

        phi_c, _ = subsystem.evaluate_partition(
            pyphi.direction.Direction.CAUSE,
            mechanism=part2,
            purview=part1,
            partition=partition,
            state=pyphi.utils.state_of(
                [subsystem.node_indices.index(n) for n in part1], sys_cause_state
            ),
        )

        phi = min(phi_e, phi_c)
        # normalized_phi = phi / len(part1)
        normalized_phi = phi / min(len(part1), len(part2))

        if normalization:
            if normalized_phi <= min_normalized_phi:
                min_phi = phi
                min_cut = (part1, part2)
                min_normalized_phi = normalized_phi
        else:
            if phi <= min_phi:
                min_phi = phi
                min_cut = (part1, part2)
                min_normalized_phi = normalized_phi

    return min_phi, min_normalized_phi, min_cut, sys_effect_state, sys_cause_state


def min_max_horizontal_cut(subsystem, normalization=True):
    (
        _,
        min_normalized_phi,
        min_cut,
        sys_effect_state,
        sys_cause_state,
    ) = min_horizontal_cut(subsystem, normalization=normalization)

    min_phi = min(
        max_information_purview(
            subsystem, min_cut[0], min_cut[1], sys_effect_state, sys_cause_state
        ),
        max_information_purview(
            subsystem, min_cut[1], min_cut[0], sys_effect_state, sys_cause_state
        ),
    )

    return min_phi, min_normalized_phi, min_cut, sys_effect_state, sys_cause_state


def plot(
    xs,
    ys,
    text=None,
    textposition="top center",
    title=None,
    xtitle=None,
    ytitle=None,
):

    df = pd.DataFrame(dict(x=xs, y=ys))
    fig = px.line(df, x="x", y="y", text=text, title=title)
    fig.update_traces(textposition=textposition)
    # fig.update_layout(autosize=True)

    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)

    return fig


def get_contiguous_nodes(n):
    return [tuple(range(i)) for i in range(1, n + 1)]


def print_sias(sias, act_val=[], mechs_func=[], weights=[], state=[], precision=5):
    if weights:
        print(f'weights:\n{weights}\n')
    if act_val:
        print(f'activation value: {act_val}')
    if mechs_func:
        print(f'mechs: {mechs_func}')
    if state:
        print(f'state: {state}')

    for sia in sias:
        print(
            f"""
Nodes: {make_label(sia.node_indices,state=sia.current_state)}
state =   {np.round(sia.current_state,precision)}
Φ =       {np.round(sia.phi,precision)}
Φn =      {np.round(sia.normalized_phi,precision)}
Φc =      {np.round(sia.phi_cause,precision)}
Φe =      {np.round(sia.phi_effect,precision)}
IIc =     {np.round(sia.system_state.intrinsic_information[0],precision)}
IIe =     {np.round(sia.system_state.intrinsic_information[1],precision)}
C-state = {sia.system_state.cause}
E-state = {sia.system_state.effect}

Partition:
{sia.partition}
""")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im#, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts

def plot_tpm(tpm,font_size=20, png_filename=''):
    tpm = pyphi.convert.state_by_node2state_by_state(tpm.squeeze())

    fig, ax = plt.subplots(1,1,figsize=(10,10))

    im = heatmap(tpm, [], [], ax=ax,
                       cmap="Greys")
    texts = annotate_heatmap(im)
    plt.rc('font', size=font_size)

    if png_filename:
        plt.savefig(f'{png_filename}.png')
    return fig