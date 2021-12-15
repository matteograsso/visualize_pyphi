from . import visualize_ces as viz
from . import compute
from pyphi.models import CauseEffectStructure
import inspect

# standard_kwargs inherits default values from plot_ces
signature = inspect.signature(viz.plot_ces)
standard_kwargs = {
    k: v.default
    for k, v in signature.parameters.items()
    if v.default is not inspect.Parameter.empty
}


def overlaid_ces_plot(system, cess, relations, nonstandard_kwargs):

    for i, inputs in enumerate(nonstandard_kwargs):
        fig_kwargs = standard_kwargs.copy()
        fig_kwargs.update(inputs)
        if i == 0:
            fig_kwargs["fig"] = None
        else:
            fig_kwargs["fig"] = fig

        fig = viz.plot_ces(system, cess[i], relations[i], **fig_kwargs)

    return fig


def plot_effect_of_MIP(
    system,
    ces,
    relations,
    figure_name,
    partitions=None,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict(), dict()],
):

    Phi, cut = compute.get_big_phi(ces, relations, system.node_indices, partitions)
    untouched_ces, untouched_relations = compute.get_untouced_ces_and_rels(
        ces, relations, cut
    )
    untouched_relations = [
        r for r in relations if compute.relation_untouched(untouched_ces, r)
    ]
    
    touched_ces = [mice for mice in ces if mice not in untouched_ces]
    touched_relations = [
        r for r in relations if r not in untouched_relations
    ]
    
    touched_ces_expanded = list(set([mice for relation in relations for mice in relation.relata]))

    cess = [ces, touched_ces_expanded, touched_ces]
    relations = [relations, touched_relations, []]

    default_common_kwargs = dict(
        network_name=figure_name, show_legend=False
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.3,
            show_labels=False,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.9,
            show_labels=False,
            show_chains=False,
            show_mechanism_base=False,
            matteo_edge_color=False,
            show_links=False,
        ),
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.01,
            show_edges=False,
            show_labels=True,
            show_chains=False,
            show_mechanism_base=False,
        ),
    ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)


def plot_component(
    system,
    ces,
    component_distinctions,
    figure_name,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict(), dict()],
    relations=None,
):

    if relations == None:
        relations = compute.compute_relations(system, ces)

    component_ces = CauseEffectStructure(
        [
            mice
            for mice, component_distinction in zip(ces, component_distinctions)
            if component_distinction
        ]
    )
    component_relations = [
        r for r in relations if compute.relation_untouched(component_ces, r)
    ]

    component_context_relations = compute.context_relations(
        relations,
        [
            mice
            for mice, distinction in zip(ces, component_distinctions)
            if distinction
        ],
    )
    component_context_ces = CauseEffectStructure(
        set(
            [
                mice
                for relation in component_context_relations
                for mice in relation.relata
            ]
        )
    )

    print(len(component_context_ces))
    cess = [ces, component_context_ces, component_ces]
    relations = [relations, component_context_relations, component_relations]

    default_common_kwargs = dict(
        network_name=figure_name,
        show_legend=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.1,
            show_labels=False,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Oranges",
            surface_opacity=0.15,
            show_labels=False,
            show_links=False,
            show_edges=True,
            save_plot_to_png=False,
            save_plot_to_html=False,
            show_mechanism_base=False,
            show_chains=False,
        ),
        dict(
            surface_colorscale="Purples",
            surface_opacity=0.9,
            show_mechanism_base=False,
            show_chains=False,
        ),
    ]

    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)

def compound_distinction(
    system,
    ces,
    relations,
    mechanism,
    figure_name,
    partitions=None,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict(), dict()],
):

    mices = [mice for mice in ces if mice.mechanism==mechanism]
    context_relations = [r for r in relations if any([relatum in mices for relatum in r.relata])]
    context_distinctions = list(set([mice for relation in context_relations for mice in relation.relata]))

    cess = [ces, context_distinctions, mices]
    relations = [relations, context_relations, []]

    default_common_kwargs = dict(
        network_name=figure_name, show_legend=False, show_labels=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.3,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Oranges",
            surface_opacity=0.9,
            show_chains=False,
            show_mechanism_base=False,
            matteo_edge_color=True,
            show_links=False,
            show_purview_labels=True,
        ),
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.01,
            show_edges=False,
            show_chains=False,
            show_mechanism_base=False,
            show_mechanism_labels=True,
        ),
    ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)


def compound_relation(
    system,
    ces,
    relations,
    mechanisms,
    figure_name,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict(), dict()],
):

    all_mices = [[mice for mice in ces if mice.mechanism==mechanism] for mechanism in mechanisms]
    
    all_context_relations = [[r for r in relations if any([relatum in mices for relatum in r.relata])] for mices in all_mices]
    all_context_distinctions = [list(set([mice for relation in context_relations for mice in relation.relata])) for context_relations in all_context_relations]

    # Getting the relations that have relata in exactly one of the contexts
    context_biding_relations = [r for r in relations 
                                if all([any([relatum in context_distinctions for relatum in r.relata]) for context_distinctions in all_context_distinctions]) 
                                and all([sum([relatum in context_distinctions for relatum in r.relata])==1 for context_distinctions in all_context_distinctions])]
    required_mices = list(set([mice for relation in context_biding_relations for mice in relation.relata]))
    
    cess = [ces, [mice for mices in all_mices for mice in mices], required_mices]
    relations = [relations, [], context_biding_relations]

    default_common_kwargs = dict(
        network_name=figure_name, show_legend=False, show_labels=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.1,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.01,
            show_edges=False,
            show_chains=False,
            show_mechanism_base=False,
            show_mechanism_labels=True,
            show_links=True,
        ),
        dict(
            surface_colorscale="Purples",
            surface_opacity=0.9,
            show_chains=False,
            show_mechanism_base=False,
            matteo_edge_color=False,
            show_links=False,
            show_edges=True,
            show_purview_labels=True,
        ),
    ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)
    
    
def mechanism_phi_fold(
    system,
    ces,
    relations,
    mechanism,
    figure_name,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict(), dict()],
):
    
    mices_specified_by_units = [mice for unit in mechanism for mice in ces if unit in mice.mechanism]
    relations_specified_by_units = [r for r in relations if any([relatum in mices_specified_by_units for relatum in r.relata])]
    
    required_distinctions = list(set([mice for relation in relations_specified_by_units for mice in relation.relata]))
    
    cess = [ces, required_distinctions, mices_specified_by_units]
    relations = [relations, [], relations_specified_by_units]

    default_common_kwargs = dict(
        network_name=figure_name, show_legend=False, show_labels=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.1,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.01,
            show_edges=False,
            show_chains=False,
            show_mechanism_base=False,
            show_mechanism_labels=True,
            show_links=True,
        ),
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.9,
            show_chains=False,
            show_mechanism_base=False,
            matteo_edge_color=False,
            show_links=False,
            show_edges=True,
            show_purview_labels=True,
        ),
    ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)
    

def purview_phi_fold(
    system,
    ces,
    relations,
    mechanisms,
    figure_name,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict(), dict()],
):

    all_mices = [[mice for mice in ces if mice.mechanism==mechanism] for mechanism in mechanisms]
    
    all_context_relations = [[r for r in relations if any([relatum in mices for relatum in r.relata])] for mices in all_mices]
    all_context_distinctions = [list(set([mice for relation in context_relations for mice in relation.relata])) for context_relations in all_context_relations]

    # Getting the relations that have relata in exactly one of the contexts
    context_biding_relations = [r for r in relations 
                                if all([any([relatum in context_distinctions for relatum in r.relata]) for context_distinctions in all_context_distinctions]) 
                                and all([sum([relatum in context_distinctions for relatum in r.relata])==1 for context_distinctions in all_context_distinctions])]
    required_mices = list(set([mice for relation in context_biding_relations for mice in relation.relata]))
    
    cess = [ces, [mice for mices in all_mices for mice in mices], required_mices]
    relations = [relations, [], context_biding_relations]

    default_common_kwargs = dict(
        network_name=figure_name, show_legend=False, show_labels=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.1,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.01,
            show_edges=False,
            show_chains=False,
            show_mechanism_base=False,
            show_mechanism_labels=True,
            show_links=True,
        ),
        dict(
            surface_colorscale="Purples",
            surface_opacity=0.9,
            show_chains=False,
            show_mechanism_base=False,
            matteo_edge_color=False,
            show_links=False,
            show_edges=True,
            show_purview_labels=True,
        ),
    ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)
    
    

def compound_phi_fold(
    system,
    ces,
    relations,
    mechanisms,
    figure_name,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict(), dict()],
):

    all_mices = [[mice for mice in ces if mice.mechanism==mechanism] for mechanism in mechanisms]
    
    all_context_relations = [[r for r in relations if any([relatum in mices for relatum in r.relata])] for mices in all_mices]
    all_context_distinctions = [list(set([mice for relation in context_relations for mice in relation.relata])) for context_relations in all_context_relations]

    # Getting the relations that have relata in exactly one of the contexts
    context_biding_relations = [r for r in relations 
                                if all([any([relatum in context_distinctions for relatum in r.relata]) for context_distinctions in all_context_distinctions]) 
                                and all([sum([relatum in context_distinctions for relatum in r.relata])==1 for context_distinctions in all_context_distinctions])]
    required_mices = list(set([mice for relation in context_biding_relations for mice in relation.relata]))
    
    cess = [ces, [mice for mices in all_mices for mice in mices], required_mices]
    relations = [relations, [], context_biding_relations]

    default_common_kwargs = dict(
        network_name=figure_name, show_legend=False, show_labels=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.1,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.01,
            show_edges=False,
            show_chains=False,
            show_mechanism_base=False,
            show_mechanism_labels=True,
            show_links=True,
        ),
        dict(
            surface_colorscale="Purples",
            surface_opacity=0.9,
            show_chains=False,
            show_mechanism_base=False,
            matteo_edge_color=False,
            show_links=False,
            show_edges=True,
            show_purview_labels=True,
        ),
    ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)
    
    

def content(
    system,
    ces,
    relations,
    mechanisms,
    figure_name,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict(), dict()],
):

    all_mices = [[mice for mice in ces if mice.mechanism==mechanism] for mechanism in mechanisms]
    
    all_context_relations = [[r for r in relations if any([relatum in mices for relatum in r.relata])] for mices in all_mices]
    all_context_distinctions = [list(set([mice for relation in context_relations for mice in relation.relata])) for context_relations in all_context_relations]

    # Getting the relations that have relata in exactly one of the contexts
    context_biding_relations = [r for r in relations 
                                if all([any([relatum in context_distinctions for relatum in r.relata]) for context_distinctions in all_context_distinctions]) 
                                and all([sum([relatum in context_distinctions for relatum in r.relata])==1 for context_distinctions in all_context_distinctions])]
    required_mices = list(set([mice for relation in context_biding_relations for mice in relation.relata]))
    
    cess = [ces, [mice for mices in all_mices for mice in mices], required_mices]
    relations = [relations, [], context_biding_relations]

    default_common_kwargs = dict(
        network_name=figure_name, show_legend=False, show_labels=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.1,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.01,
            show_edges=False,
            show_chains=False,
            show_mechanism_base=False,
            show_mechanism_labels=True,
            show_links=True,
        ),
        dict(
            surface_colorscale="Purples",
            surface_opacity=0.9,
            show_chains=False,
            show_mechanism_base=False,
            matteo_edge_color=False,
            show_links=False,
            show_edges=True,
            show_purview_labels=True,
        ),
    ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)
def compound_distinction_old(
    system,
    ces,
    units,
    figure_name,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict()],
    relations=None,
):

    if relations == None:
        relations = compute.compute_relations(system, ces)

    compound_ces = CauseEffectStructure(
        [
            mice
            for mice in ces
            if (
                any([unit in mice.mechanism for unit in units])
                or any([unit in mice.purview for unit in units])
            )
        ]
    )
    compound_relations = [
        r for r in relations if compute.relation_untouched(compound_ces, r)
    ]
    print(len(compound_ces))

    cess = [ces, compound_ces]
    relations = [relations, compound_relations]

    default_common_kwargs = dict(
        network_name=figure_name,
        show_legend=False,
        show_chains=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Greys",
            surface_opacity=0.4,
            show_labels=False,
            show_links=False,
            show_edges=False,
            save_plot_to_png=False,
            save_plot_to_html=False,
        ),
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.9,
            show_labels=True,
            show_links=True,
            show_edges=True,
            save_plot_to_png=False,
            save_plot_to_html=True,
        ),
    ]

    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)


def plot_perception(
    system,
    ces,
    triggered_distinctions,
    figure_name,
    common_kwargs=dict(),
    uncommon_kwargs=[dict(), dict()],
):

    relations = compute.compute_relations(system, ces)

    triggered_ces = CauseEffectStructure(
        [mice for mice, triggered in zip(ces, triggered_distinctions) if triggered]
    )
    triggered_relations = [
        r for r in relations if compute.relation_untouched(triggered_ces, r)
    ]

    cess = [ces, triggered_ces]
    relations = [relations, triggered_relations]

    default_common_kwargs = dict(
        network_name=figure_name,
        show_legend=False,
        show_chains=False,
        save_plot_to_html=False,
    )
    default_common_kwargs.update(common_kwargs)

    default_uncommon_kwargs = [
        dict(
            surface_colorscale="Blues",
            surface_opacity=0.1,
            show_labels=False,
            show_links=False,
            show_edges=False,
        ),
        dict(
            surface_colorscale="Greens",
            surface_opacity=1.0,
        ),
    ]

    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg)
        for default_uncommon_kwarg, uncommon_kwarg in zip(
            default_uncommon_kwargs, uncommon_kwargs
        )
    ]

    nonstandard_kwargs = [
        dict(**default_common_kwargs, **kwargs) for kwargs in default_uncommon_kwargs
    ]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)