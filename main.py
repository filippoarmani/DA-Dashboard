import copy
import dash
import dash_cytoscape as cyto
import igraph as ig
from dash import dcc, html, Output, Input, ctx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import leidenalg
from copy import deepcopy


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
            html.H1("Dashboard"),
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
                cyto.Cytoscape(
                    id='cytoscape-graph',
                    layout={'name': 'preset'},
                    style={'width': '70%', 'height': '800px'},
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
                        value='1',
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
                html.Div(id='clustering-container2', children=[
                    # Selection is already here; add results below:
                    html.Div(id='clustering-result')
                ], style={'width': '70%', 'height': '1000px'}),
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
            ], style={'display': 'flex'})
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


# Callback to update the graph visualization
@app.callback(
    Output('cytoscape-graph', 'elements'),
    Output('cytoscape-graph', 'stylesheet'),
    Output('node-info', 'children'),
    Output('legend', 'children'),
    Output('full-elements', 'data'),
    Output('selected-categories', 'data'),
    Input('tabs', 'value'),
    Input('graph-selector', 'value'),
    Input('cytoscape-graph', 'tapNodeData'),
    Input({'type': 'legend-button', 'index': dash.ALL}, 'n_clicks'),
    State('full-elements', 'data'),
    State('selected-categories', 'data'),
    prevent_initial_call=True
)
def update_graph(tab, selected_values, tapped_node, legend_clicks, all_elements, selected_categories):
    # Global variables to track selected graph and results
    global selected_graph
    global results_eb
    global results_ev

    if tab != 'visualizzazione_grafo':
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
            selected_nodes = {el['data']['id'] for el in all_elements if el['data'].get('category') in selected_categories}
            filtered_elements = [el for el in all_elements if 'data' in el and el['data'].get('id') in selected_nodes]
            filtered_edges = [el for el in all_elements if 'data' in el and 'source' in el['data'] and
                              el['data']['source'] in selected_nodes and el['data']['target'] in selected_nodes]
            elements = filtered_elements + filtered_edges
        else:
            elements = deepcopy(all_elements)

        legend_items = get_category_legend(selected_categories, all_elements, "legend-button")

        return elements, stylesheet, "", legend_items, all_elements, selected_categories

    # Graph selector or node tap
    if selected_values == 'all':
        selected_graph = ("all", g_all.copy())
        elements = g_all_elements
        results_eb = []
        results_ev = []
    elif selected_values == 'disease':
        selected_graph = ("disease", g_disease.copy())
        elements = g_disease_elements
        results_eb = []
        results_ev = []
    elif selected_values == 'gene':
        selected_graph = ("gene", g_gene.copy())
        elements = g_gene_elements
        results_eb = []
        results_ev = []
    else:
        selected_graph = None
        elements = []

    node_info = get_node_info(tapped_node)
    legend_items = get_category_legend(selected_categories, elements, "legend-button")

    return elements, stylesheet, node_info, legend_items, elements, selected_categories

# Callback to apply clustering
@app.callback(
    Output('clustering-container2', 'children'),
    Input('apply-clustering', 'n_clicks'),
    State('clustering-method', 'value'),
    State('cluster-number', 'value'),
    prevent_initial_call=True
)
def apply_clustering(n_clicks, method, cluster_number):
    if not selected_graph:
        return html.Div("Please select a graph in the Graph Visualization tab first.")

    if selected_graph[0] == "gene":
        return html.Div("Not available.")

    g = selected_graph[1].copy()

    stylesheet = [{
        'selector': 'node',
        'style': {
            'background-color': 'data(fill_color)',
            'border-width': 4,
            'border-color': 'data(border_color)'
        }
    }, get_edge_stylesheet(selected_graph[0])]

    metrics = []
    clustering = None
    partitions = []
    cluster_param_methods = ['1']  # Edge Betweenness

    # Clustering method selection
    if method == '1':
        clustering = g.community_edge_betweenness(clusters=cluster_number).as_clustering(n=cluster_number)
    elif method == '2':
        clustering = g.community_infomap()
    elif method == '3':
        clustering = g.community_label_propagation()
    elif method == '4':
        clustering = g.community_spinglass()
    elif method == '5':
        clustering = g.community_walktrap().as_clustering()
    elif method == '6':
        if selected_graph[0] == "all":
            g = g.as_undirected(mode="collapse")
            stylesheet[1] = get_edge_stylesheet("")
        clustering = try_leading_eigenvector(g)
    elif method == '7':
        if selected_graph[0] == "all":
            g = g.as_undirected(mode="collapse")
            stylesheet[1] = get_edge_stylesheet("")
        clustering = g.community_fastgreedy().as_clustering()
    elif method == '8':
        if selected_graph[0] == "all":
            g = g.as_undirected(mode="collapse")
            stylesheet[1] = get_edge_stylesheet("")
        clustering = g.community_multilevel()
    elif method == '9':
        clustering = leidenalg.find_partition(g_all, leidenalg.ModularityVertexPartition)

    if not clustering:
        return html.Div("Error applying clustering.")

    g.vs['cluster'] = list(clustering.membership)
    g.vs['cluster_str'] = most_common_in_cluster(g, g.vs['cluster'])
    g.vs['size cluster'] = [list(clustering.sizes())[i] for i in g.vs['cluster']]

    # Average assortativity per cluster
    assortativity_clusters = []
    for cluster_nodes in clustering:
        if len(cluster_nodes) > 1:
            subgraph = g.subgraph(cluster_nodes)
            assort = subgraph.assortativity_degree(directed=g.is_directed())
            assortativity_clusters.append(assort)

    g.vs['assortativity cluster'] = [assortativity_clusters[i] for i in g.vs['cluster']]

    # Metrics calculation
    NMI = normalized_mutual_info_score(g.vs['category'], g.vs['cluster'])
    ARI = adjusted_rand_score(g.vs['category'], g.vs['cluster'])
    NMI_LM = normalized_mutual_info_score(g.vs['category'], g.vs['cluster'])
    ARI_LM = adjusted_rand_score(g.vs['category'], g.vs['cluster'])
    AVG_ASS = [a for a in assortativity_clusters if not np.isnan(a)]

    AVG_ASS = sum(AVG_ASS) / len(AVG_ASS) if AVG_ASS else None
    modularity = clustering.modularity

    # Color nodes by cluster
    colors = generate_colors(len(set(clustering.membership)))

    for idx, cluster in enumerate(clustering):
        color = colors[idx % len(colors)]
        for v in cluster:
            g.vs[v]['fill_color'] = color
            g.vs[v]['border_color'] = g.vs[v]['color']

    elements = get_elements_from_graph(g, mode=True)

    # Clustering output
    if method in cluster_param_methods:
        global results_eb, results_ev

        x = {
            'Clusters': cluster_number,
            'NMI': NMI,
            'ARI': ARI,
            'NMI LM': NMI_LM,
            'ARI LM': ARI_LM,
            'AVG Assortativity Clusters': AVG_ASS,
            'Modularity': modularity
        }

        if method == '1':
            results_eb.append(x)
            results = deepcopy(results_eb)
        else:
            results_ev.append(x)
            results = deepcopy(results_ev)

        results.sort(key=lambda x: x['Clusters'])

        df = pd.DataFrame(results)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Clusters'], y=df['NMI'], mode='lines+markers', name='NMI'))
        fig.add_trace(go.Scatter(x=df['Clusters'], y=df['ARI'], mode='lines+markers', name='ARI'))
        fig.add_trace(go.Scatter(x=df['Clusters'], y=df['NMI LM'], mode='lines+markers', name='NMI LM'))
        fig.add_trace(go.Scatter(x=df['Clusters'], y=df['ARI LM'], mode='lines+markers', name='ARI LM'))
        fig.add_trace(go.Scatter(x=df['Clusters'], y=df['Modularity'], mode='lines+markers', name='Modularity'))

        return html.Div([
            html.H4("Partition Visualization"),
            cyto.Cytoscape(
                id='cytoscape-graph-c',
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '600px'},
                elements=elements,
                stylesheet=stylesheet
            ),
            html.H4("Metrics Progression"),
            dcc.Graph(figure=fig)
        ])
    else:
        table = pd.DataFrame([{
            'NMI': NMI,
            'ARI': ARI,
            'NMI LM': NMI_LM,
            'ARI LM': ARI_LM,
            'AVG Assortativity Clusters': AVG_ASS,
            'Modularity': modularity
        }])

        return html.Div([
            html.H4("Partition Visualization"),
            cyto.Cytoscape(
                id='cytoscape-graph-c',
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '600px'},
                elements=elements,
                stylesheet=stylesheet
            ),
            html.H4("Clustering Metrics"),
            dcc.Graph(
                figure=go.Figure(data=[go.Table(
                    header=dict(values=list(table.columns)),
                    cells=dict(values=[table[col] for col in table.columns])
                )])
            )
        ])

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


def try_leading_eigenvector(g):
  max_retries = 100
  attempt = 0

  leading_eigenvector = None

  while attempt < max_retries:
      try:
          leading_eigenvector = g.community_leading_eigenvector()
          break
      except Exception as e:
          print(f"Attempt {attempt+1} failed: {e}")
          attempt += 1

  if not leading_eigenvector:
      print("Error: community_leading_eigenvector could not converge after several attempts.")

  return leading_eigenvector


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
    Output('c-full', 'data'),
    Output('legend-categories', 'children'),
    Output('legend-clusters', 'children'),
    Output('c-selected-categories', 'data'),
    Output('c-selected-clusters', 'data'),
    Output('cytoscape-graph-c', 'elements'),
    Input('cytoscape-graph-c', 'tapNodeData'),
    Input({'type': 'c-legend-cat', 'index': dash.ALL}, 'n_clicks'),
    Input({'type': 'c-legend-clu', 'index': dash.ALL}, 'n_clicks'),
    State('c-full', 'data'),
    State('c-selected-categories', 'data'),
    State('c-selected-clusters', 'data'),
    State('cytoscape-graph-c', 'elements'),
    prevent_initial_call=True
)
def update_node_c(tapped_node, l1, l2, all_elements, selected_categories, selected_clusters, elements):
    node_info = get_node_info(tapped_node)

    if not all_elements:
        all_elements = copy.deepcopy(elements)

    trigger = ctx.triggered_id
    to_do = False
    # Clic su Legenda
    if isinstance(trigger, dict) and trigger.get('type') == 'c-legend-cat' and all_elements:
        cat = trigger['index']
        to_do = True
        if cat in selected_categories:
            selected_categories.remove(cat)
        else:
            selected_categories.append(cat)

    if isinstance(trigger, dict) and trigger.get('type') == 'c-legend-clu' and all_elements:
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
            selected_nodes = {el['data']['id'] for el in all_elements if
                              ((el['data'].get('category') in selected_categories) or (el['data'].get('cluster') in selected_clusters))}
            filtered_elements = [el for el in all_elements if ('data' in el and el['data'].get('id') in selected_nodes)]
            filtered_edges = [el for el in all_elements if ('data' in el and 'source' in el['data'] and
                                                            el['data']['source'] in selected_nodes and
                                                            el['data']['target'] in selected_nodes)]
            elements = filtered_elements + filtered_edges

        legend_category = get_category_legend(selected_categories, all_elements, 'c-legend-cat')
        legend_clusters = get_cluster_legend(selected_clusters, all_elements, 'c-legend-clu')

        return node_info, all_elements, legend_category, legend_clusters, selected_categories, selected_clusters, elements

    legend_category = get_category_legend(selected_categories, all_elements, 'c-legend-cat')
    legend_clusters = get_cluster_legend(selected_clusters, all_elements, 'c-legend-clu')

    return node_info, all_elements, legend_category, legend_clusters, selected_categories, selected_clusters, elements


def get_node_info(tapped_node):
    node_info = ""
    if tapped_node:
        node_info = "Node Attribute:\n\n"
        for key, value in tapped_node.items():
            if key not in ["id", "color", "x", "y", "timeStamp", "fill_color", "border_color"]:
                labels = {
                    "alldegreecentrality": "all degree centrality",
                    "indegreecentrality": "in degree centrality",
                    "outdegreecentrality": "out degree centrality",
                    "betweennesscentrality": "betweenness centrality",
                    "eigenvectorcentrality": "eigenvector centrality",
                    "cluster_str": "most frequent category in the cluster"
                }
                new_key = labels.get(key, key)
                node_info += f"{new_key}: {value}\n"

    return node_info


def get_edge_stylesheet(selected_values):
    return {
        'selector': 'edge',
        'style': {
            'line-color': '#888',
            'width': 2,
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle' if selected_values == 'all' else 'none',
            'target-arrow-color': '#888' if selected_values == 'all' else '#888'
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


if __name__ == '__main__':
    app.run_server(debug=True)