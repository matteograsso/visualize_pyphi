import pandas as pd
import string
import itertools
import pyphi
from joblib import Parallel, delayed
import pickle
from tqdm.auto import tqdm
import ray
import glob
import toolz
import numpy as np
from visualize_pyphi import network_generator, visualize_ces
from pyphi.models.subsystem import FlatCauseEffectStructure as sep
import matplotlib.pyplot as plt
from IPython.display import Audio, display


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


def parallcompute_distinction(subsystem, mechanism):
    # Compute all potential mices of a distinction in parallel
    potential_causes = subsystem.network.potential_purviews(
        pyphi.Direction.CAUSE, mechanism
    )
    potential_effects = subsystem.network.potential_purviews(
        pyphi.Direction.EFFECT, mechanism
    )
    print(f"Evaluating {len(potential_causes)} causes...")
    futures_causes = [
        compute_mice.remote(subsystem, pyphi.Direction.CAUSE, mechanism, purview)
        for purview in potential_causes
    ]
    causes = ray.get(futures_causes)
    cause = max(causes)
    print(f"Evaluating {len(potential_effects)} effects...")
    futures_effects = [
        compute_mice.remote(subsystem, pyphi.Direction.EFFECT, mechanism, purview)
        for purview in potential_effects
    ]
    effects = ray.get(futures_effects)
    effect = max(effects)

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
