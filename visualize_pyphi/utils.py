import pandas as pd
import string
import itertools

# from pyrsistent import pset
import pyphi
from joblib import Parallel, delayed
import pickle
from tqdm.auto import tqdm
# import ray
import glob
import toolz
import numpy as np
from visualize_pyphi import network_generator, visualize_ces
from pyphi.models.subsystem import FlatCauseEffectStructure as sep
import matplotlib.pyplot as plt
from IPython.display import Audio, display
# import igraph as ig
import json
import os

# import sys
# import plotly.express as px
from pathlib import Path
import sys

lib_dir = Path("/home/mgrasso/projects/")
sys.path.append(str(lib_dir))
# from new_analytical_solution import sum_phi, num_relations


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


def ces2df(ces, state_as_lettercase=True, subsystem=None):
    if not subsystem:
        s = ces[0].subsystem
    else:
        s = subsystem
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
                    state=d.cause.specified_state.state,
                ),
                lettercase_state(
                    d.effect_purview,
                    node_labels=s.node_labels,
                    state=d.effect.specified_state.state,
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


def print_distinction(distinction, subsystem, decimals=3):
    print(
        lettercase_state(
            distinction.mechanism,
            node_labels=subsystem.network.node_labels,
            state=subsystem.state,
        ),
        lettercase_state(
            distinction.cause_purview,
            node_labels=subsystem.node_labels,
            state=distinction.cause.specified_state.state,
        ),
        lettercase_state(
            distinction.effect_purview,
            node_labels=subsystem.node_labels,
            state=distinction.effect.specified_state.state,
        ),
        np.round(distinction.phi, decimals=decimals),
    )





def pklthis(this, name):
    with open(name, "wb") as f:
        pickle.dump(this, f)


def jsonthis(this, name):
    with open(name, "wt") as f:
        pyphi.jsonify.dump(this, f)


def loadjson(name):
    with open(name, "rt") as f:
        return pyphi.jsonify.load(f)


def loadpkl(name):
    with open(name, "rb") as f:
        return pickle.load(f)





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




def get_contiguous_nodes(n):
    return [tuple(range(i)) for i in range(1, n + 1)]


def print_sias(sias, act_val=[], mechs_func=[], weights=[], state=[], precision=5):
    if weights:
        print(f"weights:\n{weights}\n")
    if act_val:
        print(f"activation value: {act_val}")
    if mechs_func:
        print(f"mechs: {mechs_func}")
    if state:
        print(f"state: {state}")

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
"""
        )


import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
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
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im  # , cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    precision="{:.1f}",
    **textkw,
):
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
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
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
            text = im.axes.text(j, i, precision.format(data[i, j], None), **kw)
            texts.append(text)
    return texts


def plot_tpm(
    tpm,
    font_size=20,
    filename="",
    sbs=True,
    text=True,
    precision="{:.1f}",
    dpi=200,
):
    if sbs:
        tpm = pyphi.convert.state_by_node2state_by_state(tpm.squeeze())

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    im = heatmap(tpm, [], [], ax=ax, cmap="Greys")
    if text:
        texts = annotate_heatmap(im, precision=precision)
        plt.rc("font", size=font_size)

    if filename:
        plt.savefig(f"{filename}", dpi=dpi)
    return fig


def print_repertoire(sia, subsystem, direction="cause", constrained=True):
    states = list(all_states(len(sia.node_indices)))
    if direction == "cause":
        if constrained:
            print("Cause repertoire:")
            print(
                pd.DataFrame(
                    [sia.repertoire_cause[state] for state in states],
                    index=states,
                    columns=["p"],
                )
            )
        else:
            print("Cause unconstrained repertoire:")
            print(
                pd.DataFrame(
                    [
                        subsystem.unconstrained_forward_cause_repertoire(
                            subsystem.node_indices, subsystem.node_indices
                        )[state]
                        for state in states
                    ],
                    index=states,
                    columns=["q"],
                )
            )

    if direction == "effect":
        if constrained:
            print("Effect repertoire:")
            print(
                pd.DataFrame(
                    [sia.repertoire_effect[state] for state in states],
                    index=states,
                    columns=["p"],
                )
            )
        else:
            print("Effect unconstrained repertoire:")
            print(
                pd.DataFrame(
                    [
                        subsystem.unconstrained_forward_effect_repertoire(
                            subsystem.node_indices, subsystem.node_indices
                        )[state]
                        for state in states
                    ],
                    index=states,
                    columns=["q"],
                )
            )


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            1.05 * height,
            "%d" % int(height),
            ha="center",
            va="bottom",
        )


def plot_repertoires(sia, subsystem, figsize=(15, 5)):
    states = list(pyphi.utils.all_states(len(sia.node_indices)))
    constrained = np.round(
        [
            [
                subsystem.forward_cause_repertoire(
                    subsystem.node_indices, subsystem.node_indices
                )[state]
                for state in states
            ],
            [
                subsystem.cause_repertoire(
                    subsystem.node_indices, subsystem.node_indices
                )[state]
                for state in states
            ],
            [
                subsystem.forward_effect_repertoire(
                    subsystem.node_indices, subsystem.node_indices
                )[state]
                for state in states
            ],
        ],
        2,
    )
    unconstrained = np.round(
        [
            [
                subsystem.unconstrained_forward_cause_repertoire(
                    subsystem.node_indices, subsystem.node_indices
                )[state]
                for state in states
            ],
            [
                subsystem.unconstrained_forward_cause_repertoire(
                    subsystem.node_indices, subsystem.node_indices
                )[state]
                for state in states
            ],
            [
                subsystem.unconstrained_forward_effect_repertoire(
                    subsystem.node_indices, subsystem.node_indices
                )[state]
                for state in states
            ],
        ],
        2,
    )
    p_colors = ["red", "red", "green"]
    states = [
        strip_punct(str(s)) for s in pyphi.utils.all_states(len(sia.node_indices))
    ]

    for ps, qs, p_color in zip(constrained, unconstrained, p_colors):
        # set width of bar
        barWidth = 0.5
        fig, ax = plt.subplots(figsize=figsize)

        # Set position of bar on X axis
        br1 = np.arange(len(ps))
        br2 = [x + barWidth / 2 for x in br1]

        # Make the plot
        bar1 = ax.bar(
            br1,
            ps,
            color=p_color,
            width=barWidth,
            edgecolor=p_color,
            label="Constrained repertoire",
        )

        bar2 = ax.bar(
            br2,
            qs,
            color="grey",
            width=barWidth,
            edgecolor="grey",
            label="Unconstrained repertoire",
        )

        # Adding Xticks
        plt.xlabel("states", fontsize=20)
        plt.ylabel("probability", fontsize=20)
        plt.xticks(
            [r + barWidth / 4 for r in range(len(ps))],
            states,
            rotation="vertical",
            fontdict=dict(fontsize=20),
        )
        plt.yticks(
            np.round(np.linspace(0, 1, 11), 1),
            np.round(np.linspace(0, 1, 11), 1),
            fontdict=dict(fontsize=20),
        )

        for i, repertoire, color in zip(
            [-barWidth / 2, barWidth / 2], [ps, qs], [p_color, "grey"]
        ):
            for index, data in enumerate(repertoire):
                plt.text(
                    x=index + i,
                    y=data,
                    s=f"{data}",
                    fontdict=dict(fontsize=15, color=color),
                )

        plt.tight_layout()
        plt.legend(prop={"size": 20})

        plt.show()


def print_sias_short(sias):
    for sia in sias:
        print(
            f"""
        nodes = {sia.node_indices}
        phi_s = {sia.phi}
        ii_c = {sia.system_state.intrinsic_information[pyphi.Direction.CAUSE]}
        ii_e = {sia.system_state.intrinsic_information[pyphi.Direction.EFFECT]}
        {sia.partition}
        """
        )


def normalize_values(min_size, max_size, values, min_val=None, max_val=None):
    if type(values) != np.ndarray:
        values = np.array(values)
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)
    if max_val == min_val:
        return [(min_size + max_size) / 2 for x in values]
    else:
        return min_size + (
            ((values - min_val) * (max_size - min_size)) / (max_val - min_val)
        )


def find_spots(ces):
    spots = []
    for d in ces:
        selfrel = set(d.cause_purview).intersection(set(d.effect_purview))
        if len(selfrel) > 1:
            spots.append(make_label(selfrel))
    return sorted(sorted(list(set(spots))), key=len)


def get_contiguous_mechs(n):
    return [
        x
        for x in sorted(
            sorted(
                set(
                    list(
                        toolz.concat(
                            list(
                                (
                                    tuple(tuple(range(i, j)) for i in range(n))
                                    for j in range(n + 1)
                                )
                            )
                        )
                    )
                )
            ),
            key=len,
        )
        if x
    ]



def repertoires_for_all_states(direction, subsystem, mechanism, purview):
    """Return a state-by-state 'sub-TPM'.

    Rows correspond to mechanims states, columns correspond to purview states,
    and each row is the forward repertoire corresponding to that state of the
    mechanism.
    """
    # Get subsystem states with all possible mechanism states
    mechanism_states = list(pyphi.utils.all_states(len(mechanism)))
    subsystem_states = np.tile(subsystem.state, (len(mechanism_states), 1))
    subsystem_states[:, list(mechanism)] = mechanism_states

    subsystems = [
        pyphi.Subsystem(subsystem.network, nodes=subsystem.node_indices, state=state)
        for state in subsystem_states
    ]
    return pyphi.ExplicitTPM(
        [
            pyphi.distribution.flatten(
                subsystem.forward_repertoire(direction, mechanism, purview)
            )
            for subsystem in subsystems
        ],
    )
