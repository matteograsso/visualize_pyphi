from . import visualize_ces as viz

standard_kwargs = dict(
    network_name="",
    floor_width_scale=2,
    floor_height_scale=1,
    cause_effect_distance=0.1,   
    base_height_scale=1.5,
    base_z_offset=0.1,
    base_width_scale=1,
    base_opacity=0.9,
    base_color="white",
    base_intensity=0.5,
    user_mechanism_coordinates=None,
    user_purview_coordinates=None,
    mechanism_labels_size=15,
    mechanism_label_position="top center",
    purview_label_position="top center",
    edge_size_range=(1, 3),
    state_as_lettercase=True,
    purview_labels_size=16,
    link_width_range=(2, 6),
    transparent_edges=False,
    surface_size_range=(0.1,0.99),
    surface_colorscale='Purples',
    surface_opacity=0.2,
    axes_range=None,
    eye_coordinates=(-0.2, 0.2, -0.2),
    hovermode="x",
    plot_dimensions=(1200,1000),
    save_plot_to_html=True,
    save_plot_to_png=False,
    show_mechanism_base=True,    
    show_chains=True, 
    show_links=True,
    show_mesh=True,
    show_edges=True,
    show_labels=True,
    colorcode_2_relations=True,
    showlegend=True,
    transparent_background=True,
    chain_width=3,
    fig=None,
)

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