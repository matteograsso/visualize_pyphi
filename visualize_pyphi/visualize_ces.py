# TODO: - Update code to be in line with new relation module (e.g. maximal state will be part of MICE)

"""
This module plots a pyphi CES taking a subsystem, a CES, and a list of relations as arguments.
    subsystem: a pyphi subsystem
    ces: ces must be separated into cause and effect distinctions filtered by compositional state
    relations: only 2 and 3 relations will be plotted
"""

import itertools
import numpy as np
import pandas as pd
import plotly
from plotly import graph_objs as go
from tqdm.notebook import tqdm
import random
import pyphi
import pyphi.relations as rel
from pyphi.utils import powerset
import matplotlib.pyplot as plt
from scipy.special import comb
import math
from pyphi.models.subsystem import FlatCauseEffectStructure
import string

CAUSE = pyphi.Direction.CAUSE
EFFECT = pyphi.Direction.EFFECT


def flatten(iterable):
    return itertools.chain.from_iterable(iterable)


def phi_round(phi):
    return np.round(phi, 4)


def feature_matrix(ces, relations):
    """Return a matrix representing each cause and effect in the CES.

    .. note::
        Assumes that causes and effects have been separated.
    """
    N = len(ces)
    M = len(relations)
    # Create a mapping from causes and effects to indices in the feature matrix
    index_map = {distinction: i for i, distinction in enumerate(ces)}
    # Initialize the feature vector
    features = np.zeros([N, M])
    # Assign features
    for j, relation in enumerate(relations):
        indices = [index_map[relatum] for relatum in relation.relata]
        # Create the column corresponding to the relation
        relation_features = np.zeros(N)
        # Assign 1s where the cause/effect purview is involved in the relation
        relation_features[indices] = 1
        # Assign the feature column to the feature matrix
        features[:, j] = relation_features
    return features


def relation_vertex_indices(features, j):
    """Return the indices of the vertices for relation ``j``."""
    return features[:, j].nonzero()[0]


def make_label(node_indices, node_labels=None, bold=False, state=False):

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

        return "<i>" + "".join(node_labels) + "</i>"

    return "<b>" + "".join(node_labels) + "</b>" if bold else "".join(node_labels)


def label_mechanism(mice, bold=False, state=False):
    return make_label(
        mice.mechanism, node_labels=mice.ria.node_labels, bold=bold, state=state
    )


def label_purview(mice, state=False):
    return make_label(mice.purview, node_labels=mice.ria.node_labels, state=state)


def hovertext_purview(mice):
    return f"Distinction: {label_mechanism(mice)}<br>Direction: {mice.direction.name}<br>Purview: {label_purview(mice)}<br>φ = {phi_round(mice.phi)}<br>State: {[rel.specified_state(mice)[0][i] for i in mice.purview] if not hasattr(mice,'specified_state') else list(mice.specified_state.state)}"


def hovertext_relation(relation):
    relata = relation.relata

    relata_info = "".join(
        [
            f"<br>Distinction {n}: {label_mechanism(mice)}<br>Direction: {mice.direction.name}<br>Purview: {label_purview(mice)}<br>φ = {phi_round(mice.phi)}<br>State: {[rel.specified_state(mice)[0][i] for i in mice.purview] if not hasattr(mice,'specified_state') else list(mice.specified_state.state)}<br>"
            for n, mice in enumerate(relata)
        ]
    )

    relation_info = f"<br>Relation purview: {make_label(relation.purview, relation.subsystem.node_labels)}<br>Relation φ = {phi_round(relation.phi)}<br>"

    return f"<br>={len(relata)}-Relation=<br>" + relata_info + relation_info


def normalize_sizes(min_size, max_size, elements):
    phis = np.array([element.phi for element in elements])
    min_phi = phis.min()
    max_phi = phis.max()
    # Add exception in case all purviews have the same phi (e.g. monad case)
    if max_phi == min_phi:
        return [(min_size + max_size) / 2 for x in phis]
    else:
        return min_size + (
            ((phis - min_phi) * (max_size - min_size)) / (max_phi - min_phi)
        )


def normalize_values(bottom, top, values):
    min_val = values.min()
    max_val = values.max()
    # Add exception in case all purviews have the same phi (e.g. monad case)
    if max_val == min_val:
        return [(bottom + max_val) / 2 for x in values]
    else:
        return bottom + (((values - min_val) * (top - bottom)) / (max_val - min_val))


def chunk_list(my_list, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(my_list), n):
        yield my_list[i : i + n]


def regular_polygon(n, center=(0, 0), angle=0, z=0, radius=None, scale=1):
    if radius == None:
        radius = n / (2 * math.pi)

    radius = radius * scale

    if n == 1:
        return [[center[0], center[1], z]]
    else:
        angle -= math.pi / n
        coord_list = [
            [
                center[0] + radius * math.sin((2 * math.pi / n) * i - angle),
                center[1] + radius * math.cos((2 * math.pi / n) * i - angle),
                z,
            ]
            for i in range(n)
        ]
        return coord_list


def get_edge_color(relation, colorcode_2_relations):
    if colorcode_2_relations:
        purview0 = list(relation.relata.purviews)[0]
        purview1 = list(relation.relata.purviews)[1]
        relation_purview = tuple(relation.purview)
        # Isotext (mutual full-overlap)
        if purview0 == purview1 == relation_purview:
            return "fuchsia"
        # Sub/Supertext (inclusion / full-overlap)
        elif purview0 != purview1 and (
            all(n in purview1 for n in purview0) or all(n in purview0 for n in purview1)
        ):
            return "indigo"
        # Paratext (connection / partial-overlap)
        elif (purview0 == purview1 != relation_purview) or (
            any(n in purview1 for n in purview0)
            and not all(n in purview1 for n in purview0)
        ):
            return "cyan"
        else:
            raise ValueError(
                "Unexpected relation type, check function to cover all cases"
            )
    else:
        return "teal"


################
def plot_ces(
    subsystem,
    ces,
    relations,
    network_name="",
    floor_width_scale=1.5,
    floor_width_scales=[1.5, 2, 2.5, 2, 1],
    floor_height_scale=[2, 2, 1.5, 1.25, 1],
    cause_effect_distance=0.2,
    base_height_scale=2.7,
    base_z_offset=0,
    base_width_scale=0.8,
    base_width_scales=[1, 1, 1, 1, 1],
    base_opacity=0.90,
    base_rotation=math.pi,
    base_intensity=0.9,
    base_color="grey",
    user_mechanism_coordinates=None,
    user_purview_coordinates=None,
    mechanism_labels_size=30,
    purview_labels_size=30,
    mechanism_label_position="middle center",
    purview_label_position="middle center",
    edge_size_range=(1, 3),
    state_as_lettercase=True,
    link_width_range=(2, 6),
    transparent_edges=False,
    surface_size_range=(0.1, 0.99),
    surface_colorscale="Blues",
    surface_opacity=0.15,
    axes_range=None,
    eye_coordinates=(0, 25, 0.25),
    hovermode="x",
    plot_dimensions=(2440, 1440),
    png_resolution=None,
    save_plot_to_html=True,
    save_plot_to_png=False,
    show_mechanism_base=True,
    show_chains=True,
    show_links=True,
    show_mesh=True,
    show_edges=True,
    show_labels=True,
    show_mechanism_labels=True,
    show_purview_labels=True,
    colorcode_2_relations=True,
    show_legend=True,
    transparent_background=False,
    chain_width=3,
    fig=None,
    matteo_edge_color=True,
    purview_color=False,
    mesh_legendgroup="",
    edge_color="red",
    epicycle_radius=0.2,
    surface_color_range=[0, 1],
    png_scale=2,
    relate_distinctions=True,
):
    if not isinstance(ces, FlatCauseEffectStructure):
        raise ValueError(f"ces must be a FlatCauseEffectStructure; got {type(ces)}")

    # Initialize figure
    if fig == None:
        fig = go.Figure()
    else:
        # because of an issue with potentially lacking 1st order mechs
        print("Not redrawing base")
        show_mechanism_base = False

    # get components needed for plotting
    relations = list(filter(lambda r: len(r.relata) <= 3, relations))
    purviews = [mice.purview for mice in ces]
    mechanisms = [mice.mechanism for mice in ces]
    N_units = len(subsystem)
    node_labels = subsystem.node_labels
    node_indices = subsystem.node_indices

    ######
    # Define the coordinates for all purviews
    ######

    # generate floors for the purviews
    floors = [
        np.array(
            regular_polygon(
                int(comb(N_units, k + 1)),
                center=(0, 0),
                angle=0,
                z=k * floor_height_scale[k]
                if len(floor_height_scale) > 0
                else k * floor_height_scale,
                scale=floor_width_scales[k]
                if floor_width_scales
                else floor_width_scale,
            )
        )
        for k in range(N_units)
    ]
    floor_vertices = np.concatenate([f for f in floors])

    # getting a list of all possible purviews
    all_purviews = list(powerset(subsystem.node_indices, nonempty=True))

    # find number of times each purview appears
    vertex_purview = {
        p: fv for p, fv in zip(all_purviews, floor_vertices) if purviews.count(p) > 0
    }

    # Create positional coordinataes for each purview that exists in the ces
    num_purviews = [
        purviews.count(p) if purviews.count(p) > 0 else 0 for p in all_purviews
    ]
    epicycles = [
        regular_polygon(n, center=(e[0], e[1]), z=e[2], radius=epicycle_radius)
        for e, n in zip(floor_vertices, num_purviews)
        if n > 0
    ]

    # associating each purview with vertices in a regular polygon around the correct floor vertex
    purview_positions = [
        {v: e, "N": 0} for v, e in zip(vertex_purview.keys(), epicycles)
    ]

    # placing purview coordinates in the correct order
    purview_vertex_coordinates = []
    for purview in purviews:
        for purview_position in purview_positions:
            if purview in purview_position.keys():
                purview_vertex_coordinates.append(
                    purview_position[purview][purview_position["N"]]
                )
                purview_position["N"] += 1

    purview_coordinates = np.array(purview_vertex_coordinates)

    ######
    # Define the coordinates for all mechanisms
    ######
    if not base_width_scales:
        base_width_scales = [base_width_scale] * N_units

    base = [
        np.array(
            regular_polygon(
                int(comb(N_units, k)),
                center=(0, 0),
                z=((k / N_units) * base_height_scale) + base_z_offset,
                scale=base_width_scales[k - 1],
                angle=base_rotation,
            )
        )
        for k in range(1, N_units + 1)
    ]

    base_vertices = np.concatenate([f for f in base])

    base_coordinates = {
        subset: coordinates for subset, coordinates in zip(all_purviews, base_vertices)
    }

    # store all the coordinates of the mechanisms in lists
    x_mechanism = np.array([base_coordinates[mice.mechanism][0] for mice in ces])
    y_mechanism = np.array([base_coordinates[mice.mechanism][1] for mice in ces])
    z_mechanism = np.array([base_coordinates[mice.mechanism][2] for mice in ces])

    # print(x_mechanism)

    x_purview = purview_coordinates[:, 0]
    y_purview = purview_coordinates[:, 1]
    z_purview = purview_coordinates[:, 2]

    x_purview_rels = x_purview
    y_purview_rels = y_purview
    z_purview_rels = z_purview

    #
    if user_mechanism_coordinates is not None:
        x_mechanism = user_mechanism_coordinates[0, :]
        y_mechanism = user_mechanism_coordinates[1, :]
        z_mechanism = user_mechanism_coordinates[2, :]

    if user_purview_coordinates is not None:
        x_purview = user_purview_coordinates[0, :]
        y_purview = user_purview_coordinates[1, :]
        z_purview = user_purview_coordinates[2, :]

    if relate_distinctions:
        x_purview_rels = x_mechanism
        y_purview_rels = y_mechanism
        z_purview_rels = z_mechanism

    # create the link coordinates
    link_coordinates = (
        list(zip(x_purview, x_mechanism)),
        list(zip(y_purview, y_mechanism)),
        list(zip(z_purview, z_mechanism)),
    )

    ###
    # Create a matrix that connects distinctions to relations
    ###
    features = feature_matrix(ces, relations)

    # separate cause and effect purview coordinates
    causes_x = [x for i, x in enumerate(x_purview) if ces[i].direction == CAUSE]
    causes_y = [y for i, y in enumerate(y_purview) if ces[i].direction == CAUSE]
    causes_z = [z for i, z in enumerate(z_purview) if ces[i].direction == CAUSE]

    effects_x = [x for i, x in enumerate(x_purview) if ces[i].direction == EFFECT]
    effects_y = [y for i, y in enumerate(y_purview) if ces[i].direction == EFFECT]
    effects_z = [z for i, z in enumerate(z_purview) if ces[i].direction == EFFECT]

    # Get mechanism and purview labels
    mechanism_labels = [
        label_mechanism(
            mice,
            bold=False,
            state=subsystem.state if state_as_lettercase else False,
        )
        for mice in ces
    ]

    labels_mechanisms_trace = go.Scatter3d(
        visible=(show_labels or show_mechanism_labels),
        x=x_mechanism,
        y=y_mechanism,
        z=z_mechanism,
        mode="text",
        text=mechanism_labels,
        name="Mechanism Labels",
        showlegend=True,
        textfont=dict(
            size=mechanism_labels_size,
            color="black",
        ),
        textposition=mechanism_label_position,
        hoverinfo="text",
        hovertext=False,
        hoverlabel=dict(bgcolor="black", font_color="white"),
    )
    fig.add_trace(labels_mechanisms_trace)

    # Make mechanism base
    if show_mechanism_base:
        first_order_mechanisms = list(filter(lambda m: len(m) == 1, mechanisms))

        base_mechanisms = []
        base_pair_counter = 0
        for m1, mech1 in enumerate(mechanisms):
            for m2, mech2 in enumerate(first_order_mechanisms):
                if mech2[0] in mech1:
                    base_mechanisms.append((base_pair_counter, (m1, m2)))
                    base_pair_counter += 1

        base_mechanisms_pairs = [base_pair[1] for base_pair in base_mechanisms]

        base_mechanisms_triplets = []
        ss = []
        for a, b in itertools.combinations(base_mechanisms_pairs, 2):
            s = set(a).union(b)
            if len(s) == 3:
                base_mechanisms_triplets.append(tuple(sorted(s)))

        base_mechanisms_triplets = sorted(list(set(base_mechanisms_triplets)))

        base_mechanisms_triangles = np.array(
            [
                triplet
                for triplet in base_mechanisms_triplets
                if len(triplet) == 3
                and len(
                    list(
                        filter(
                            lambda mechanism_index: mechanism_index
                            > len(first_order_mechanisms) - 1,
                            triplet,
                        )
                    )
                )
                == 1
            ]
        )

        base_mesh = go.Mesh3d(
            visible=show_mechanism_base,
            legendgroup="base mesh",
            showlegend=True,
            x=x_mechanism,
            y=y_mechanism,
            z=z_mechanism,
            i=base_mechanisms_triangles[:, 0],
            j=base_mechanisms_triangles[:, 1],
            k=base_mechanisms_triangles[:, 2],
            name="Base",
            intensity=[base_intensity for x in x_mechanism],
            opacity=base_opacity,
            colorscale=[base_color for x in x_mechanism],
            showscale=False,
        )
        fig.add_trace(base_mesh)

    ####
    # Label purviews
    ####
    purview_labels = [
        label_purview(
            mice,
            state=list(rel.specified_state(mice)[0])
            if not hasattr(mice, "specified_state")
            else list(mice.specified_state.state)
            if state_as_lettercase
            else False,
        )
        for mice in ces
    ]

    cause_purview_labels = [
        x for i, x in enumerate(purview_labels) if ces[i].direction == CAUSE
    ]
    effect_purview_labels = [
        x for i, x in enumerate(purview_labels) if ces[i].direction == EFFECT
    ]

    vertices_hovertext = list(map(hovertext_purview, ces))
    causes_hovertext = [
        x for i, x in enumerate(vertices_hovertext) if ces[i].direction == CAUSE
    ]
    effects_hovertext = [
        x for i, x in enumerate(vertices_hovertext) if ces[i].direction == EFFECT
    ]

    # Create labels for cause purviews
    labels_cause_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=causes_x,
        y=causes_y,
        z=causes_z,
        mode="text",
        text=cause_purview_labels,
        textposition=purview_label_position,
        name="Cause Purview Labels",
        showlegend=True,
        textfont=dict(
            size=purview_labels_size,
            color=purview_color if purview_color else "red",
        ),
        hoverinfo="text",
        hovertext=causes_hovertext,
        hoverlabel=dict(bgcolor="red"),
    )
    fig.add_trace(labels_cause_purviews_trace)

    # effects
    labels_effect_purviews_trace = go.Scatter3d(
        visible=show_purview_labels,
        x=effects_x,
        y=effects_y,
        z=effects_z,
        mode="text",
        text=effect_purview_labels,
        textposition=purview_label_position,
        name="Effect Purview Labels",
        showlegend=True,
        textfont=dict(
            size=purview_labels_size,
            color=purview_color if purview_color else "green",
        ),
        hoverinfo="text",
        hovertext=effects_hovertext,
        hoverlabel=dict(bgcolor="green"),
    )
    fig.add_trace(labels_effect_purviews_trace)

    # plotting links
    links_widths = normalize_sizes(link_width_range[0], link_width_range[1], ces)
    links_widths = list(flatten(list(zip(links_widths, links_widths))))

    links_counter = 0
    for i, mice in enumerate(ces):
        link_trace = go.Scatter3d(
            visible=show_links,
            legendgroup="Links",
            showlegend=True if links_counter == 0 else False,
            x=link_coordinates[0][i],
            y=link_coordinates[1][i],
            z=link_coordinates[2][i],
            mode="lines",
            name="Links",
            line_width=links_widths[i],
            line_color=["orange"],  # ["red"] if mice.direction == CAUSE else ["green"],
            hoverinfo="skip",
        )
        links_counter += 1
        fig.add_trace(link_trace)

    # Make mechanism chains
    if show_chains:

        chained_mechanisms = []
        chain_counter = 0
        for m1, mech1 in enumerate(mechanisms):
            for m2, mech2 in enumerate(first_order_mechanisms):
                if mech2[0] in mech1:
                    chained_mechanisms.append((chain_counter, (m1, m2)))
                    chain_counter += 1

        chains_xs = [
            (x_mechanism[c[0]], x_mechanism[c[1]]) for i, c in chained_mechanisms
        ]
        chains_ys = [
            (y_mechanism[c[0]], y_mechanism[c[1]]) for i, c in chained_mechanisms
        ]
        chains_zs = [
            (z_mechanism[c[0]], z_mechanism[c[1]]) for i, c in chained_mechanisms
        ]

        for m, mechanism in chained_mechanisms:

            chains_trace = go.Scatter3d(
                visible=show_chains,
                legendgroup="Chains",
                showlegend=True if m == 0 else False,
                x=chains_xs[m],
                y=chains_ys[m],
                z=chains_zs[m],
                mode="lines",
                name="Chains",
                line={
                    "dash": "dash",
                    "color": "black",
                    "width": chain_width,
                },
                hoverinfo="skip",
            )
            fig.add_trace(chains_trace)

    # 2-relations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if show_edges:
        # Get edges from all relations
        edges = list(
            flatten(
                relation_vertex_indices(features, j)
                for j in range(features.shape[1])
                if features[:, j].sum() == 2
            )
        )
        if edges:
            # Convert to DataFrame
            edges = pd.DataFrame(
                dict(
                    x=x_purview_rels[edges],
                    y=y_purview_rels[edges],
                    z=z_purview_rels[edges],
                )
            )

            # Plot edges separately:
            two_relations = list(filter(lambda r: len(r.relata) == 2, relations))

            two_relations_sizes = normalize_sizes(
                edge_size_range[0], edge_size_range[1], two_relations
            )

            two_relations_coords = [
                list(chunk_list(list(edges["x"]), 2)),
                list(chunk_list(list(edges["y"]), 2)),
                list(chunk_list(list(edges["z"]), 2)),
            ]

            for r, relation in tqdm(
                enumerate(two_relations),
                desc="Computing edges",
                total=len(two_relations),
            ):
                relation_nodes = list(flatten(relation.mechanisms))
                relation_color = (
                    get_edge_color(relation, colorcode_2_relations)
                    if matteo_edge_color
                    else edge_color
                )

                legend_mechanisms = []

                # Make all 2-relations traces and legendgroup
                edge_two_relation_trace = go.Scatter3d(
                    visible=show_edges,
                    legendgroup="All 2-Relations",
                    showlegend=True if r == 0 else False,
                    x=two_relations_coords[0][r],
                    y=two_relations_coords[1][r],
                    z=two_relations_coords[2][r],
                    mode="lines",
                    name="All 2-Relations",
                    line_width=two_relations_sizes[r],
                    line_color="rgba(0,0,0,0)" if transparent_edges else relation_color,
                    hoverinfo="text",
                    hovertext=hovertext_relation(relation),
                )

                fig.add_trace(edge_two_relation_trace)

    # 3-relations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get triangles from all relations
    if show_mesh:
        triangles = [
            relation_vertex_indices(features, j)
            for j in range(features.shape[1])
            if features[:, j].sum() == 3
        ]

        if triangles:
            three_relations = list(filter(lambda r: len(r.relata) == 3, relations))
            # three_relation_orders = [len(r.purview) for r in three_relations]
            three_relation_orders = [
                np.mean([len(rr.purview) for rr in r.relata]) for r in three_relations
            ]
            # three_relation_orders = [
            #     o / max(three_relation_orders) for o in three_relation_orders
            # ]

            three_relation_orders = normalize_values(
                surface_size_range[0],
                surface_size_range[1],
                np.array(three_relation_orders),
            )

            # three_relations_sizes = normalize_sizes(
            #     surface_size_range[0] * surface_opacity,
            #     surface_size_range[1] * surface_opacity / 2,
            #     three_relations,
            # )

            # Extract triangle indices
            i, j, k = zip(*triangles)

            """triangle_three_relation_trace = go.Mesh3d(
                    visible=True,
                    legendgroup="All 3-Relations",
                    showlegend=False,
                    # x, y, and z are the coordinates of vertices
                    x=x_purview,
                    y=y_purview,
                    z=z_purview,
                    # i, j, and k are the vertices of triangles
                    i=i,
                    j=j,
                    k=k,
                    intensity=three_relations_sizes,
                    intensitymode='cell',
                    opacity=.1,
                    colorscale=surface_colorscale,
                    showscale=False,
                    name="All 3-Relations",
                    hoverinfo="text",
                    hovertext=[hovertext_relation(relation) for relation in three_relations],
                )
            fig.add_trace(triangle_three_relation_trace)"""

            cmap = plt.cm.get_cmap(surface_colorscale)
            color_picker = np.linspace(
                surface_color_range[0], surface_color_range[1], len(triangles)
            )

            random.Random(len(triangles)).shuffle(color_picker)

            for r, triangle in tqdm(
                enumerate(triangles), desc="Computing triangles", total=len(triangles)
            ):
                relation = three_relations[r]
                relation_nodes = list(flatten(relation.mechanisms))
                # print(three_relation_orders[r], "\n", three_relations[r].relata)

                legend_mechanisms = []

                triangle_three_relation_trace = go.Mesh3d(
                    autocolorscale=False,
                    visible=show_mesh,
                    legendgroup=mesh_legendgroup
                    if mesh_legendgroup
                    else "All 3-Relations",
                    showlegend=True if r == 0 else False,
                    # x, y, and z are the coordinates of vertices
                    x=x_purview_rels,
                    y=y_purview_rels,
                    z=z_purview_rels,
                    # i, j, and k are the vertices of triangles
                    i=[i[r]],
                    j=[j[r]],
                    k=[k[r]],
                    colorscale=[
                        [0, "red"],
                        [
                            0.5,
                            "rgb"
                            + str(
                                tuple(
                                    [int(c * 255) for c in cmap(color_picker[r])[:-1]]
                                )
                            ),
                        ],
                        [1, "blue"],
                    ],
                    intensity=[three_relation_orders[r]],
                    intensitymode="cell",
                    opacity=max(0, three_relation_orders[r]),
                    showscale=False,
                    name="All 3-Relations",
                    hoverinfo="text",
                    hovertext=hovertext_relation(relation),
                )
                # print(triangle_three_relation_trace)
                fig.add_trace(triangle_three_relation_trace)

    # Create figure
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if axes_range is None:
        axes_range = [
            (min(d) - 10, max(d) + 10)
            for d in (
                np.append(x_purview, x_mechanism),
                np.append(y_purview, y_mechanism),
                np.append(z_purview, z_mechanism),
            )
        ]

    axes = [
        dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            gridcolor="lightgray",
            showticklabels=False,
            showspikes=False,
            autorange=False,
            range=axes_range[dimension],
            backgroundcolor="white",
            title="",
        )
        for dimension in range(3)
    ]

    # convert eye_coordinates to cartesian
    # eye_coordinates is given in 3d polar coordinates:
    # (horizontal angle from x, vertical angle from xy-plane, distance) in degrees
    x_eye = (
        eye_coordinates[2]
        * np.sin((-eye_coordinates[1] + 90) * 2 * np.pi / 360)
        * np.cos(-eye_coordinates[0] * 2 * np.pi / 360)
    )
    y_eye = (
        eye_coordinates[2]
        * np.sin((-eye_coordinates[1] + 90) * 2 * np.pi / 360)
        * np.sin(-eye_coordinates[0] * 2 * np.pi / 360)
    )
    z_eye = eye_coordinates[2] * np.cos((-eye_coordinates[1] + 90) * 2 * np.pi / 360)

    layout = go.Layout(
        showlegend=show_legend,
        scene_xaxis=axes[0],
        scene_yaxis=axes[1],
        scene_zaxis=axes[2],
        scene_camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye)),
        hovermode=hovermode,
        title="",
        title_font_size=30,
        legend=dict(
            title=dict(
                text="Trace legend (click trace to show/hide):",
                font=dict(color="black", size=15),
            )
        ),
        autosize=True,
        width=plot_dimensions[0],
        height=plot_dimensions[1],
        paper_bgcolor="rgba(0,0,0,0)" if transparent_background else "white",
        plot_bgcolor="rgba(0,0,0,0)" if transparent_background else "white",
    )

    # Apply layout
    fig.layout = layout

    if save_plot_to_html is True:
        plotly.io.write_html(fig, f"{network_name}.html")
        print(f"Fig saved to {network_name}.html")
    elif type(save_plot_to_html) == str:
        plotly.io.write_html(fig, save_plot_to_html)
        print(f"Fig saved to {save_plot_to_html}")

    if save_plot_to_png is True:
        if not png_resolution:
            png_resolution = plot_dimensions
        fig.write_image(
            f"{network_name}.png",
            width=png_resolution[0],
            height=png_resolution[1],
            scale=1,
        )
        print(f"Fig saved to {network_name}.png")

    elif type(save_plot_to_png) == str:
        if not png_resolution:
            png_resolution = plot_dimensions
        fig.write_image(
            save_plot_to_png,
            width=png_resolution[0],
            height=png_resolution[1],
            scale=png_scale,
            #           transparent=transparent_background,
        )
        print(f"Fig saved to {save_plot_to_png}")

    return fig
