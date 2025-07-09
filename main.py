import colorsys
import copy
import re
from copy import deepcopy

import dash
import dash_cytoscape as cyto
import igraph as ig
import leidenalg
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Global variables for graphs
selected_graph = None
g_all = None
g_disease = None
g_gene = None

# Variables for clustering results (if needed later)
results_eb = []
results_ev = []

# Function to load graph from GML file
def upload_graph(gml_file):
    global g_all, g_disease, g_gene

    G = ig.Graph.Read_GML(gml_file)  # Load the graph from file

    # Assign the graph to the corresponding variable based on the file name
    if "all" in gml_file:
        g_all = G.copy()
        return get_elements_from_graph(g_all, mode=True)  # With automatic layout
    elif "disease" in gml_file:
        g_disease = G.copy()
        return get_elements_from_graph(g_disease, mode=False)  # Use layout from g_all
    elif "gene" in gml_file:
        g_gene = G.copy()
        return get_elements_from_graph(g_gene, mode=False)  # Use layout from g_all
    else:
        return []

# Function to convert igraph graph into Dash Cytoscape elements
def get_elements_from_graph(G, mode=True):
    elements = []

    global g_all_elements

    try:
        if mode:
            # Compute automatic layout (Kamada-Kawai)
            layout = G.layout("kk")
            x = [float(l[0]) for l in layout]
            y = [float(l[1]) for l in layout]
            G.vs["x"] = x
            G.vs["y"] = y
        else:
            # Retrieve node positions from g_all_elements
            x, y = [], []
            for v in G.vs['label']:
                for d in g_all_elements:
                    if "position" in d.keys() and v == d["data"]["label"]:
                        x.append(float(d["position"]["x"]) / 100)
                        y.append(float(d["position"]["y"]) / 100)
                        break
            G.vs["x"] = x
            G.vs["y"] = y

        # Mapping node IDs
        id_mapping = {}
        for v in G.vs:
            node_id = str(v["id"]) if "id" in v.attributes() else str(v["label"]) if "label" in v.attributes() else str(v.index)
            id_mapping[v.index] = node_id

        # Creating node elements
        for v in G.vs:
            node_id = id_mapping[v.index]
            node_data = {
                'id': node_id,
                'label': str(v["label"]) if "label" in v.attributes() else node_id,
                'color': v["color"] if "color" in v.attributes() else "#999",
                'category': v["category"] if "category" in v.attributes() else "unknown"
            }

            # Add any additional attributes
            for a in v.attributes():
                if a not in node_data:
                    node_data[a] = str(v[a])

            elements.append({
                'data': node_data,
                'position': {'x': v['x'] * 100, 'y': v['y'] * 100}
            })

        # Creating edge elements
        for e in G.es:
            elements.append({
                'data': {
                    'source': id_mapping[e.source],
                    'target': id_mapping[e.target]
                }
            })
    except Exception as e:
        print(f"Error in get_elements_from_graph: {e}")
        elements = []

    return elements

# Load initial graphs
g_all_elements = upload_graph("g_all.gml")
g_disease_elements = upload_graph("g_disease.gml")
g_gene_elements = upload_graph("g_gene.gml")

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Main app layout with tabs
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='graph_visualization', children=[
        dcc.Tab(label='Main Info', value='graph_visualization'),
        dcc.Tab(label='Clustering', value='page-2'),
    ]),
    html.Div(id='tabs-content')
])


# Tab content update callback
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'graph_visualization':
        return html.Div([
            dcc.Store(id='selected-categories', data=[]),
            html.Label('Select the graph you want to view and analyze:', style={'margin-right': '10px'}),
            dcc.RadioItems(
                id='graph-selector',
                options=[
                    {'label': 'No Graph', 'value': 'None'},
                    {'label': 'Graph with All Nodes', 'value': 'all'},
                    {'label': 'Graph with Diseases Only', 'value': 'disease'},
                    {'label': 'Graph with Genes Only', 'value': 'gene'}
                ],
                value='None',
                inputStyle={'margin-right': '10px'}
            ),
            html.Div([
                html.Label('Filter by properties:', style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='properties-method',
                    options=[
                        {'label': 'all degree centrality', 'value': 'alldegreecentrality'},
                        {'label': 'in degree centrality', 'value': 'indegreecentrality'},
                        {'label': 'out degree centrality', 'value': 'outdegreecentrality'},
                        {'label': 'betweenness centrality', 'value': 'betweennesscentrality'},
                        {'label': 'eigenvector centrality', 'value': 'eigenvectorcentrality'},
                        {'label': 'pagerank', 'value': 'pagerank'},
                        {'label': 'closeness', 'value': 'closeness'}
                    ],
                    placeholder="Select Properties",
                    value=None,
                    style={'width': '200px', 'display': 'inline-block'}
                ),
                html.Label('Select threshold:', style={'margin-left': '10px', 'margin-right': '10px'}),
                dcc.Input(
                    id='threshold-number',
                    type='number',
                    placeholder='Enter a number',
                    value=0,
                    style={'width': '150px'}
                ),
                html.Button("Apply", id='apply-threshold', n_clicks=0, disabled=True,
                            style={'margin-left': '10px', 'margin-right': '10px'}),
                html.Label('', id='minmax1', style={'margin-left': '10px', 'margin-right': '10px'}),
            ], id="t_div", style={'display': 'none', 'align-items': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}),
            html.Div([
                cyto.Cytoscape(
                    id='cytoscape-graph',
                    layout={'name': 'preset'},
                    style={'width': '70%', 'height': '1000px'},
                    elements=[],
                    stylesheet=[]
                ),
                html.Div([
                    html.H4("Legend (Click to Filter):"),
                    html.Div(id='legend', style={
                        'flex': '3',
                        'overflowY': 'auto',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'border': '1px solid #ccc',
                        'padding': '5px'
                    }),
                    html.Div(id='node-info', style={
                        'flex': '5',
                        'overflowY': 'auto',
                        'whiteSpace': 'pre-wrap',
                        'border': '1px solid #ccc',
                        'padding': '5px'
                    }),
                ], style={
                    'width': '30%',
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'height': '800px'
                })
            ], style={'display': 'flex'}),
            dcc.Graph(id='distribution', figure=go.Figure(), style={'display': 'none', 'height': '500px'}),
            dcc.Store(id='full-elements')
        ])
    elif tab == 'page-2':
        return html.Div([
            dcc.Store(id='c-full', data=[]),
            dcc.Store(id='c-selected-categories', data=[]),
            dcc.Store(id='c-selected-clusters', data=[]),
            dcc.Store(id='c-s-elements', data=[]),
            html.H3('Clustering'),
            html.Div([
                html.Div([
                    html.Label('Select Clustering Method:', style={'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='clustering-method',
                        options=[
                            {'label': 'Edge Betweenness (Girvan–Newman)', 'value': '1'},
                            {'label': 'Infomap', 'value': '2'},
                            {'label': 'Label Propagation', 'value': '3'},
                            {'label': 'Spinglass', 'value': '4'},
                            {'label': 'Walktrap', 'value': '5'},
                            {'label': 'Leading Eigenvector', 'value': '6'},
                            {'label': 'Fast Greedy', 'value': '7'},
                            {'label': 'Louvain', 'value': '8'},
                            {'label': 'Leiden', 'value': '9'}
                        ],
                        placeholder="Select Clustering Method",
                        value=None,
                        style={'width': '200px', 'display': 'inline-block'}
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),
                html.Div([
                    html.Label('Enter a Positive Number:', style={'margin-right': '10px'}),
                    dcc.Input(
                        id='cluster-number',
                        type='number',
                        min=1,
                        step=1,
                        placeholder='Enter a positive number',
                        style={'width': '150px'}
                    )
                ], id='cluster-number-container'),

                html.Button("Apply", id='apply-clustering', n_clicks=0, disabled=True, style={'margin-top': '10px'})
            ], id='clustering-container'),
            html.Div([
                html.Label('Filter by properties:', style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id='properties-c',
                    options=[
                        {'label': 'size', 'value': 'size'},
                        {'label': 'assortativity', 'value': 'assortativity'},
                        {'label': 'none', 'value': 'None'}
                    ],
                    placeholder="Select Properties",
                    value=None,
                    style={'width': '200px', 'display': 'inline-block'}
                ),
                html.Label('Select threshold:', style={'margin-left': '10px', 'margin-right': '10px'}),
                dcc.Input(
                    id='threshold-number-c',
                    type='number',
                    placeholder='Enter a number',
                    value=0,
                    style={'width': '150px'}
                ),
                html.Button("Apply", id='apply-threshold-c', n_clicks=0, disabled=True,
                            style={'margin-left': '10px', 'margin-right': '10px'}),
                html.Label('', id='minmax2', style={'margin-left': '10px', 'margin-right': '10px'}),
            ], id="c_div",
                style={'display': 'none', 'align-items': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}),
            html.H4("Partition Visualization"),
            html.Div([
                cyto.Cytoscape(
                    id='cytoscape-graph-c',
                    layout={'name': 'preset'},
                    style={'width': '70%', 'height': '1000px'},
                    elements=[],
                    stylesheet=[]
                ),
                html.Div([
                    html.H4("Legend (Click to Filter):"),
                    html.Div(id='legend-clusters', style={
                        'flex': '3',
                        'overflowY': 'auto',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'border': '1px solid #ccc',
                        'padding': '5px'
                    }),
                    html.H4("Legend (Click to Filter):"),
                    html.Div(id='legend-categories', style={
                        'flex': '3',
                        'overflowY': 'auto',
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'border': '1px solid #ccc',
                        'padding': '5px'
                    }),
                    html.Div(id='node-info-c', style={
                        'flex': '5',
                        'overflowY': 'auto',
                        'whiteSpace': 'pre-wrap',
                        'border': '1px solid #ccc',
                        'padding': '5px'
                    }),
                ], style={
                    'width': '30%',
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'height': '1000px'
                })
            ], style={'display': 'flex'}),
            html.H4("Metrics Progression"),
            dcc.Graph(id='metric-table', figure=go.Figure(), style={'height': '500px'}),
            dcc.Graph(id='metric-sizes', figure=go.Figure(), style={'height': '500px'}),
            dcc.Graph(id='metric-assortativity', figure=go.Figure(), style={'height': '500px'}),
            dcc.Graph(id='metric-graph', figure=go.Figure(), style={'height': '500px'}),
        ])
    return None

# Show/hide the input field for number of clusters based on method
@app.callback(
    Output('cluster-number-container', 'style'),
    Input('clustering-method', 'value')
)
def toggle_cluster_number_visibility(method):
    if method in ['1']:
        # Show input field
        return {'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}
    else:
        # Hide input field
        return {'display': 'none'}

# Enable/disable "Apply" button based on inputs
@app.callback(
    Output('apply-clustering', 'disabled'),
    [Input('clustering-method', 'value'),
     Input('cluster-number', 'value')]
)
def toggle_button(method, number):
    if method in ['1']:
        # Requires number of clusters
        return number is None or number < 1
    else:
        # No need for cluster number
        return False


# Enable/disable "Apply" button based on inputs
@app.callback(
    Output('apply-threshold', 'disabled'),
    Input('properties-method', 'value'),
    Input('graph-selector', 'value')
)
def toggle_button2(prop, selector):
    return (prop is None) or (selector is None) or (selector == 'None')

@app.callback(
    Output('apply-threshold-c', 'disabled'),
    Input('properties-c', 'value')
)
def toggle_button3(prop):
    return prop is None

# Enable/disable "Apply" button based on inputs
@app.callback(
    Output('t_div', 'style'),
    Output('properties-method', 'value'),
    Output('threshold-number', 'value'),
    Input('graph-selector', 'value')
)
def update_t_div(selector):
    if (selector is None) or (selector == 'None'):
        return {'display': 'none', 'align-items': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}, None, 0
    else:
        return {'display': 'flex', 'align-items': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}, None, 0


# Callback to update the graph visualization
@app.callback(
    Output('cytoscape-graph', 'elements'),
    Output('cytoscape-graph', 'stylesheet'),
    Output('legend', 'children'),
    Output('full-elements', 'data'),
    Output('selected-categories', 'data'),
    Output('distribution', 'figure'),
    Output('distribution', 'style'),
    Input('tabs', 'value'),
    Input('graph-selector', 'value'),
    Input('apply-threshold', 'n_clicks'),
    Input({'type': 'legend-button', 'index': dash.ALL}, 'n_clicks'),
    State('full-elements', 'data'),
    State('selected-categories', 'data'),
    State('properties-method', 'value'),
    State('threshold-number', 'value'),
    prevent_initial_call=True
)
def update_graph(tab, selected_values, n, legend_clicks, all_elements, selected_categories, prop, threshold):
    # Global variables to track selected graph and results
    global selected_graph
    global results_eb
    global results_ev
    global list_table
    global g_all_elements
    global g_disease_elements
    global g_gene_elements

    if tab != 'graph_visualization':
        raise dash.exceptions.PreventUpdate

    # Define graph stylesheet
    stylesheet = [
        {
            'selector': 'node',
            'style': {
                'background-color': 'data(color)',
                'color': 'data(color)',
                'font-size': '12px'
            }
        },
        get_edge_stylesheet(selected_values)
    ]

    trigger = ctx.triggered_id

    # Legend button click
    if isinstance(trigger, dict) and trigger.get('type') == 'legend-button' and all_elements:
        cat = trigger['index']
        if cat in selected_categories:
            selected_categories.remove(cat)
        else:
            selected_categories.append(cat)

        if selected_categories:
            selected_nodes = {int(float(el['data']['id'])) for el in all_elements if el['data'].get('category') in selected_categories}
            elements, stylesheet = get_elements_filtered(all_elements, selected_nodes, stylesheet)
        else:
            elements = deepcopy(all_elements)

        legend_items = get_category_legend(selected_categories, all_elements, "legend-button")

        return elements, stylesheet, legend_items, all_elements, selected_categories, dash.no_update, dash.no_update
    elif trigger == "graph-selector":
        # Graph selector or node tap
        if selected_values == 'all':
            selected_graph = ("all", g_all.copy())
            elements = g_all_elements
        elif selected_values == 'disease':
            selected_graph = ("disease", g_disease.copy())
            elements = g_disease_elements
        elif selected_values == 'gene':
            selected_graph = ("gene", g_gene.copy())
            elements = g_gene_elements
        else:
            selected_graph = None
            elements = []

        results_eb = []
        results_ev = []
        list_table = []

        legend_items = get_category_legend(selected_categories, elements, "legend-button")

        return elements, stylesheet, legend_items, elements, [], dash.no_update, {'display': 'none', 'height': '500px'}
    elif trigger == "apply-threshold":
        selected_nodes = {int(float(el['data']['id'])) for el in all_elements if prop in el['data'].keys() and float(el['data'].get(prop, "")) >= float(threshold)}

        elements, stylesheet = get_elements_filtered(all_elements, selected_nodes, stylesheet)

        boxplot = go.Figure(
            data=[
                go.Box(
                    y=sorted([float(el['data'][prop]) for el in all_elements if 'data' in el and ('source' not in el['data']) and ('target' not in el['data'])]),
                    name="",
                    boxmean=True
                )
            ]
        )

        boxplot.update_layout(
            title="Boxplot of " + prop + " distribution"
        )
        return elements, stylesheet, dash.no_update, dash.no_update, dash.no_update, boxplot, {'display': 'block', 'height': '500px'}
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

list_table = []

# Callback to apply clustering
@app.callback(
    Output('cytoscape-graph-c', 'elements'),
    Output('cytoscape-graph-c', 'stylesheet'),
    Output('metric-graph', 'figure'),
    Output('metric-table', 'figure'),
    Output('legend-categories', 'children'),
    Output('legend-clusters', 'children'),
    Output('c-selected-categories', 'data'),
    Output('c-selected-clusters', 'data'),
    Output('c-full', 'data'),
    Output('metric-graph', 'style'),
    Output('metric-sizes', 'figure'),
    Output('metric-assortativity', 'figure'),
    Output('c_div', 'style'),
    Input('apply-clustering', 'n_clicks'),
    Input('apply-threshold-c', 'n_clicks'),
    Input({'type': 'c-legend-cat', 'index': dash.ALL}, 'n_clicks'),
    Input({'type': 'c-legend-clu', 'index': dash.ALL}, 'n_clicks'),
    State('clustering-method', 'value'),
    State('cluster-number', 'value'),
    State('c-selected-categories', 'data'),
    State('c-selected-clusters', 'data'),
    State('c-full', 'data'),
    State('properties-c', 'value'),
    State('threshold-number-c', 'value'),
    prevent_initial_call=True
)
def apply_clustering(n_clicks, n_clicks_c, l1, l2, method, cluster_number, selected_categories, selected_clusters, all_elements, prop, threshold):
    trigger = ctx.triggered_id

    stylesheet = [{
        'selector': 'node',
        'style': {
            'background-color': 'data(fill_color)',
            'border-width': 4,
            'border-color': 'data(border_color)'
        }
    }, get_edge_stylesheet(selected_graph[0])]

    if trigger == "apply-clustering":
        if not selected_graph or selected_graph[0] == "gene":
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        g = selected_graph[1].copy()

        method_name = ""
        clustering = None
        style_graph = {'display': 'none', 'height': '500px'}

        # Clustering method selection
        if method == '1':
            method_name = 'Edge Betweenness'
            clustering = g.community_edge_betweenness(clusters=cluster_number).as_clustering(n=cluster_number)
        elif method == '2':
            method_name = 'Infomap'
            clustering = g.community_infomap()
        elif method == '3':
            method_name = 'Label Propagation'
            clustering = g.community_label_propagation()
        elif method == '4':
            method_name = 'Spinglass'
            clustering = g.community_spinglass()
        elif method == '5':
            method_name = 'Walktrap'
            clustering = g.community_walktrap().as_clustering()
        elif method == '6':
            method_name = 'Leading Eigenvector'
            if selected_graph[0] == "all":
                g = g.copy().as_undirected(mode="collapse")
                stylesheet[1] = get_edge_stylesheet("")

            max_retries = 100
            attempt = 0

            clustering = None

            while attempt < max_retries:
                try:
                    clustering = g.community_leading_eigenvector()
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    attempt += 1

        elif method == '7':
            method_name = 'Fast Greedy'
            if selected_graph[0] == "all":
                g = g.as_undirected(mode="collapse")
                stylesheet[1] = get_edge_stylesheet("")
            clustering = g.community_fastgreedy().as_clustering()
        elif method == '8':
            method_name = 'Louvain'
            if selected_graph[0] == "all":
                g = g.as_undirected(mode="collapse")
                stylesheet[1] = get_edge_stylesheet("")
            clustering = g.community_multilevel()
        elif method == '9':
            method_name = 'Leiden'
            clustering = leidenalg.find_partition(g_all, leidenalg.ModularityVertexPartition)

        if not clustering:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        g.vs['cluster'] = list(clustering.membership)
        g.vs['cluster_str'] = most_common_in_cluster(g, g.vs['cluster'])
        g.vs['size cluster'] = [list(clustering.sizes())[i] for i in g.vs['cluster']]

        # Metrics calculation
        NMI = normalized_mutual_info_score(g.vs['category'], g.vs['cluster'])
        ARI = adjusted_rand_score(g.vs['category'], g.vs['cluster'])
        NMI_LM = normalized_mutual_info_score(g.vs['category'], g.vs['cluster'])
        ARI_LM = adjusted_rand_score(g.vs['category'], g.vs['cluster'])

        modularity = clustering.modularity

        # Color nodes by cluster
        colors = generate_colors(len(set(clustering.membership)))

        g.vs['fill_color'] = [colors[c] for c in g.vs['cluster']]
        g.vs['border_color'] = g.vs['color']
        g.vs['assortativity cluster'] = [clustering.subgraph(i).assortativity_degree() for i in g.vs['cluster']]

        elements = get_elements_from_graph(g, mode=False)
        selected_categories = []
        selected_clusters = []

        global results_eb
        global list_table

        x = {
            'Clusters': len(clustering),
            'NMI': NMI,
            'ARI': ARI,
            'NMI LM': NMI_LM,
            'ARI LM': ARI_LM,
            'Modularity': modularity
        }

        if method == '1':
            results_eb.append(x)
            results = deepcopy(results_eb)
            results.sort(key=lambda x: x['Clusters'])
            style_graph = {'display': 'block', 'height': '500px'}
        else:
            results = []

        df = pd.DataFrame(results)
        fig_graph = go.Figure()
        for key in df.keys():
            if 'Clusters' not in key:
                fig_graph.add_trace(go.Scatter(x=df['Clusters'], y=df[key], mode='lines+markers', name=key))

        list_table.append({
            'Method': method_name,
            'Clusters': len(clustering),
            'NMI': round(NMI, 5),
            'ARI': round(ARI, 5),
            'NMI LM': round(NMI_LM, 5),
            'ARI LM': round(ARI_LM, 5),
            'Modularity': round(modularity, 5)
        })

        list_table = [dict(t) for t in {frozenset(d.items()) for d in list_table}]
        table = pd.DataFrame(list_table)
        table = table[['Method', 'Clusters', 'Modularity', 'NMI', 'ARI', 'NMI LM', 'ARI LM']]

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(table.columns)),
            cells=dict(values=[table[col] for col in table.columns])
        )])

        legend_category = get_category_legend([], elements, 'c-legend-cat')
        legend_clusters = get_cluster_legend([], elements, 'c-legend-clu')

        boxplot_s = go.Figure(
            data=[
                go.Box(
                    y=sorted(clustering.sizes()),
                    name="",
                    boxmean=True
                )
            ]
        )

        boxplot_s.update_layout(
            title="Boxplot cluster size distribution"
        )

        assortativity = [a for a in g.vs['assortativity cluster'] if a >= -1]
        global_assortativity = round(g.assortativity_degree(), 5)

        boxplot_a = go.Figure(
            data=[
                go.Box(
                    y=sorted(assortativity),
                    name="",
                    boxmean=True
                )
            ]
        )

        boxplot_a.add_shape(
            type="line",
            x0=-0.5,  # Va bene per singolo boxplot
            x1=0.5,
            y0=global_assortativity,
            y1=global_assortativity,
            line=dict(color="red", dash="dash")
        )

        boxplot_a.update_layout(
            title=(
                f"Boxplot assortativity distribution "
                f"({len(clustering) - len(list(set(assortativity)))} cluster with NaN value)\n"
                f"{global_assortativity} is the global assortativity degree"
            ),
            yaxis=dict(range=[-1.1, 1.1])
        )

        return elements, stylesheet, fig_graph, fig_table, legend_category, legend_clusters, selected_categories, selected_clusters, elements, style_graph, boxplot_s, boxplot_a, {'display': 'block', 'align-items': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}

    if trigger == "apply-threshold-c":
        if prop != "None":
            if prop == "assortativity":
                prop2 = "assortativity cluster"
            else:
                prop2 = "size cluster"

            selected_nodes = {int(float(el['data']['id'])) for el in all_elements if
                              prop2 in el['data'].keys() and float(el['data'].get(prop2, "")) >= float(threshold)}
        else:
            selected_nodes = {int(float(el['data']['id'])) for el in all_elements}

        elements, stylesheet = get_elements_filtered(all_elements, selected_nodes, stylesheet)

        return elements, stylesheet, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, all_elements, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if isinstance(trigger, dict) and (trigger.get('type') == 'c-legend-cat' or trigger.get('type') == 'c-legend-clu') and all_elements:
        to_do = False
        # Clic su Legenda
        if trigger.get('type') == 'c-legend-cat':
            cat = trigger['index']
            to_do = True
            if cat in selected_categories:
                selected_categories.remove(cat)
            else:
                selected_categories.append(cat)

        if trigger.get('type') == 'c-legend-clu':
            clu = trigger['index']
            to_do = True
            if clu in selected_clusters:
                selected_clusters.remove(clu)
            else:
                selected_clusters.append(clu)

        if to_do:
            if  (not selected_categories) and (not selected_clusters):
                elements = copy.deepcopy(all_elements)
            else:
                selected_nodes = {int(float(el['data']['id'])) for el in all_elements if
                                  ((el['data'].get('category') in selected_categories) or (el['data'].get('cluster') in selected_clusters))}

                elements, stylesheet = get_elements_filtered(all_elements, selected_nodes, stylesheet)

            legend_category = get_category_legend(selected_categories, all_elements, 'c-legend-cat')
            legend_clusters = get_cluster_legend(selected_clusters, all_elements, 'c-legend-clu')

            return elements, stylesheet, dash.no_update, dash.no_update, legend_category, legend_clusters, selected_categories, selected_clusters, all_elements, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            legend_category = get_category_legend(selected_categories, all_elements, 'c-legend-cat')
            legend_clusters = get_cluster_legend(selected_clusters, all_elements, 'c-legend-clu')

            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, legend_category, legend_clusters, selected_categories, selected_clusters, all_elements, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, selected_categories, selected_clusters, all_elements, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Utility function to generate distinct colors
def generate_colors(n):
    colors = []
    golden_ratio_conjugate = 0.61803398875
    hue = 0

    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        colors.append(hex_color)

    return colors


# Get most common category within a cluster
def most_common_in_cluster(g, membership):
    b = {}
    categories = sorted(set(g.vs["category"]))
    set_cluster = sorted(set(membership))

    for m in set_cluster:
        b[m] = [0] * len(categories)

    for i, m in enumerate(membership):
        b[m][categories.index(g.vs["category"][i])] += 1

    return [categories[b[m].index(max(b[m]))] for m in membership]


@app.callback(
    Output('node-info-c', 'children'),
    Input('cytoscape-graph-c', 'tapNodeData'),
    prevent_initial_call=True
)
def update_node_c(tapped_node):
    return get_node_info(tapped_node)


@app.callback(
    Output('node-info', 'children'),
    Input('cytoscape-graph', 'tapNodeData'),
    prevent_initial_call=True
)
def update_node_d(tapped_node):
    return get_node_info(tapped_node)


def get_node_info(tapped_node):
    node_info = ""
    
    if tapped_node:
        node_info = "Node Attribute:\n\n"
        for key, value in tapped_node.items():
            if key not in ["id", "color", "x", "y", "timeStamp", "fill_color", "border_color", "igraphdegree", "igraphbtw"]:
                labels = {
                    "alldegreecentrality": "all degree centrality",
                    "indegreecentrality": "in degree centrality",
                    "outdegreecentrality": "out degree centrality",
                    "betweennesscentrality": "betweenness centrality",
                    "eigenvectorcentrality": "eigenvector centrality",
                    "cluster_str": "most frequent category in the cluster"
                }
                new_key = labels.get(key, key)

                new_value = value
                if type(value) != str or len(re.findall(r'\d*\.\d+', value))>0:
                    new_value = round(float(value), 5)
                node_info += f"{new_key}: {new_value}\n"

    return node_info

def get_edge_stylesheet(selected_values):
    return {
        'selector': 'edge',
        'style': {
            'line-color': '#888',
            'width': 2,
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle' if selected_values == 'all' else 'none',
            'target-arrow-color': '#888'
        }
    }


def get_category_legend(selected_categories, all_elements, types):
    return [
        html.Button(cat, id={'type': types, 'index': cat}, n_clicks=0,
                    style={
                        'backgroundColor': color,
                        'border': '2px solid black' if cat in selected_categories else 'none',
                        'padding': '10px',
                        'margin': '5px',
                        'cursor': 'pointer',
                        'color': 'white'
                    })
        for cat, color in {
            el['data']['category']: el['data']['color']
            for el in all_elements if 'data' in el and 'category' in el['data']
        }.items()
    ]


def get_cluster_legend(selected_clusters, all_elements, types):
    return [
        html.Button("Cluster " + clu[0] + ": " + clu[1], id={'type': types, 'index': clu[0]}, n_clicks=0,
                    style={
                        'backgroundColor': color,
                        'border': '2px solid black' if clu[0] in selected_clusters else 'none',
                        'padding': '10px',
                        'margin': '5px',
                        'cursor': 'pointer',
                        'color': 'white'
                    })
        for color, clu in {
            el['data']['fill_color']: [el['data']['cluster'], el['data']['cluster_str']]
            for el in all_elements if 'data' in el and 'cluster' in el['data']
        }.items()
    ]

@app.callback(
    Output("minmax1", "children"),
    Input('properties-method', 'value')
)
def update_label1(prop):
    if selected_graph is None or len(selected_graph) != 2 or prop is None or prop == "None":
        return ""

    g = selected_graph[1].copy()
    prop_v = [round(float(p), 5) for p in g.vs[prop]]
    return f"Min: {min(prop_v)} Max: {max(prop_v)}"

@app.callback(
    Output("minmax2", "children"),
    Input('properties-c', 'value'),
    State('c-full', 'data'),
)
def update_label2(prop, all_elements):
    if selected_graph is None or len(selected_graph) != 2 or prop is None or prop == "None" or all_elements is None or all_elements == []:
        return ""

    if prop == "assortativity":
        prop2 = "assortativity cluster"
    else:
        prop2 = "size cluster"

    selected_value = [round(float(el['data'].get(prop2, "")), 5) for el in all_elements if "source" not in el['data'].keys()]

    return f"Min: {min(selected_value)} Max: {max(selected_value)}"


def get_elements_filtered(all_elements, selected_nodes, stylesheet):
    nodes = []
    for el in all_elements:
        if 'data' in el and 'source' not in el['data']:
            if float(el['data']['id']) in selected_nodes:
                nodes.append({'data': el['data'], 'classes': 'highlight'})
            else:
                nodes.append({'data': el['data'], 'classes': 'faded'})

    edges = []
    for el in all_elements:
        if 'data' in el and 'source' in el['data'] and 'target' in el['data']:
            if int(float(el['data']['source'])) in selected_nodes or int(float(el['data']['target'])) in selected_nodes:
                edges.append({'data': el['data'], 'classes': 'highlight'})
            else:
                edges.append({'data': el['data'], 'classes': 'faded'})

    elements = nodes + edges

    stylesheet.append(
        {
            'selector': '.highlight',
            'style': {
                'opacity': 1
            }
        }
    )

    stylesheet.append(
        {
            'selector': '.faded',
            'style': {
                'opacity': 0.2
            }
        }
    )

    return elements, stylesheet

if __name__ == '__main__':
    app.run_server(debug=True)