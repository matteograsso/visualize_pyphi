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
        if i==0:
            fig_kwargs['fig']=None
        else:
            fig_kwargs['fig']=fig
            
        fig = viz.plot_ces(system, cess[i], relations[i], **fig_kwargs)
        
    return fig

def plot_effect_of_MIP(system,ces,relations,figure_name,partitions=None,common_kwargs=dict(),uncommon_kwargs=[dict(), dict()]):
    
    Phi, cut = compute.get_big_phi(ces,relations, system.node_indices,partitions)
    untouched_ces, untouched_relations = compute.get_untouced_ces_and_rels(
        ces, relations, cut
    )
    untouched_relations = [r for r in relations if compute.relation_untouched(untouched_ces, r)]
    
    cess = [ces, untouched_ces]
    relations = [relations, untouched_relations]

    default_common_kwargs = dict(
            network_name=figure_name,
            show_legend=False,
            show_chains=False
        )
    default_common_kwargs.update(common_kwargs)
    
    default_uncommon_kwargs = [
            dict(
                surface_colorscale='Greys',
                surface_opacity=0.001,
                show_labels=False,
                show_links=False,
                show_edges=False,
                show_legend=False,
                show_mechanism_base=False,
                show_chains=False,
                save_plot_to_png=False,
                save_plot_to_html=False
            ),
            dict(
                surface_colorscale='Blues',
                surface_opacity=1.0,
            ),
        ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg) 
        for default_uncommon_kwarg, uncommon_kwarg in zip(default_uncommon_kwargs, uncommon_kwargs)
        ]

    nonstandard_kwargs = [dict(**common_kwargs, **kwargs) for kwargs in uncommon_kwargs]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)
    


def plot_perception(system,ces,triggered_distinctions,figure_name,common_kwargs=dict(),uncommon_kwargs=[dict(), dict()]):
    
    relations = compute.compute_relations(system,ces)
    
    triggered_ces = CauseEffectStructure([mice for mice, triggered in zip(ces, triggered_distinctions) if triggered])
    triggered_relations = [r for r in relations if compute.relation_untouched(triggered_ces, r)]
    
    cess = [ces, triggered_ces]
    relations = [relations, triggered_relations]
    
    default_common_kwargs = dict(
        network_name=figure_name,
        show_legend=False,
    )
    default_common_kwargs.update(common_kwargs)
    
    default_uncommon_kwargs = [
        dict(
            surface_colorscale='Greys',
            surface_opacity=0.001,
            show_labels=False,
            show_links=False,
            show_edges=False,
        ),
        dict(
            surface_colorscale='Blues',
            surface_opacity=1.0,
        ),
    ]
    default_uncommon_kwargs = [
        dict(**default_uncommon_kwarg, **uncommon_kwarg) 
        for default_uncommon_kwarg, uncommon_kwarg in zip(default_uncommon_kwargs, uncommon_kwargs)
    ]

    nonstandard_kwargs = [dict(**common_kwargs, **kwargs) for kwargs in uncommon_kwargs]

    overlaid_ces_plot(system, cess, relations, nonstandard_kwargs)
    