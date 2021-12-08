import pandas as pd
import string
import itertools
import pyphi
from joblib import Parallel, delayed


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
