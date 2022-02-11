import pandas as pd
import string
import itertools
import pyphi
from joblib import Parallel, delayed
import pickle
from tqdm.auto import tqdm
import ray

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


def ces2df(ces, subsystem, csv_name=None):
    s = subsystem
    ces_list = [
        (
            strp(i2n(d.mechanism, s)),
            d.phi,
            strp(i2n(d.cause.purview, s)),
            strp(d.cause.specified_state),
            d.cause.phi,
            strp(i2n(d.effect.purview, s)),
            strp(d.effect.specified_state),
            d.effect.phi,
        )
        for d in ces
    ]

    df = pd.DataFrame(
        ces_list, columns=["mechanism", "phi", "cause_purview", "cause_state", "cause_phi", "effect_purview", "effect_state", "effect_phi"]
    )

    if csv_name:
        df.to_csv(csv_name)
    return df


def sumarize_sia(sias):
    ces_list = [
        (
            sum(sia.phi_structure.distinctions.phis)
            + sum([rel.phi for rel in sia.phi_structure.relations]),
            sia.phi,
            sum(
                [
                    sum([len(d.cause.purview), len(d.effect.purview)])
                    for d in sia.phi_structure.distinctions
                ]
            ),
            len(sia.phi_structure.distinctions),
            sum(sia.phi_structure.distinctions.phis),
            len(sia.phi_structure.relations),
            sum([rel.phi for rel in sia.phi_structure.relations]),
            len(
                [d for d in sia.phi_structure.distinctions if len(d.cause.purview) == 1]
            ),
            len(
                [
                    d
                    for d in sia.phi_structure.distinctions
                    if len(d.effect.purview) == 1
                ]
            ),
            len(
                [d for d in sia.phi_structure.distinctions if len(d.cause.purview) == 2]
            ),
            len(
                [
                    d
                    for d in sia.phi_structure.distinctions
                    if len(d.effect.purview) == 2
                ]
            ),
            len(
                [d for d in sia.phi_structure.distinctions if len(d.cause.purview) == 3]
            ),
            len(
                [
                    d
                    for d in sia.phi_structure.distinctions
                    if len(d.effect.purview) == 3
                ]
            ),
            len(
                [d for d in sia.phi_structure.distinctions if len(d.cause.purview) == 4]
            ),
            len(
                [
                    d
                    for d in sia.phi_structure.distinctions
                    if len(d.effect.purview) == 4
                ]
            ),
        )
        for sia in sias
    ]
    df = pd.DataFrame(
        ces_list,
        columns=[
            "existence",
            "irreducibility",
            "total_purview_length",
            "n_distinctions",
            "distinction_phi",
            "n_relations",
            "relation_phi",
            "C: O=1",
            "E: O=1",
            "C: O=2",
            "E: O=2",
            "C: O=3",
            "E: O=3",
            "C: O=4",
            "E: O=4",
        ],
    )

    return df.sort_values(by="existence", ascending=False)



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
