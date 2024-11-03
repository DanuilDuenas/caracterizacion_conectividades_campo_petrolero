### HERRAMIENTA DE ANALISIS DE RESULTADOS SL

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State#, ClientsideFunction
from dash import no_update
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objects import Layout
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import plotly.colors as pc  
import plotly.graph_objects as go
import os
import dash_auth
from users import USERNAME_PASSWORD_PAIRS

app = dash.Dash(__name__, 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.title = "Caracterización Conectividad - Campo A"

# auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

server = app.server
app.config.suppress_callback_exceptions = True


#### CARGA DE INFORMACIÓN ####

### MAPAS ###

with open('Data/stats_df_GR_filt.pkl', 'rb') as file:
    stats_df_GR_filt = pickle.load(file)

with open('Data/stats_df_GR_allfm_filt.pkl', 'rb') as file:
    stats_df_GR_allfm_filt = pickle.load(file)

with open('Data/stats_df_DEN_filt.pkl', 'rb') as file:
    stats_df_DEN_filt = pickle.load(file)

with open('Data/stats_df_NEUT_filt.pkl', 'rb') as file:
    stats_df_NEUT_filt = pickle.load(file)

with open('Data/Mapas/map_GR_mean.pkl', 'rb') as file:
    mapa_GR_mean = pickle.load(file)

with open('Data/Mapas/map_GR_median.pkl', 'rb') as file:
    mapa_GR_median = pickle.load(file)

with open('Data/Mapas/map_GR_allfms_mean.pkl', 'rb') as file:
    mapa_GR_allfms_mean = pickle.load(file)

with open('Data/Mapas/map_GR_allfms_median.pkl', 'rb') as file:
    mapa_GR_allfms_median = pickle.load(file)

with open('Data/Mapas/map_DEN_mean.pkl', 'rb') as file:
    mapa_DEN_mean = pickle.load(file)

with open('Data/Mapas/map_DEN_median.pkl', 'rb') as file:
    mapa_DEN_median = pickle.load(file)

with open('Data/Mapas/map_NEUT_mean.pkl', 'rb') as file:
    mapa_NEUT_mean = pickle.load(file)

with open('Data/Mapas/map_NEUT_median.pkl', 'rb') as file:
    mapa_NEUT_median = pickle.load(file)

datos_mapa = {'2':[stats_df_GR_filt, mapa_GR_mean, 'GR (mean)'],
              '3':[stats_df_GR_filt, mapa_GR_median, 'GR (median)'],
              '4':[stats_df_GR_allfm_filt, mapa_GR_allfms_mean, 'GR_all (mean)'],
              '5':[stats_df_GR_allfm_filt, mapa_GR_allfms_mean, 'GR_all (median)'],
              '6':[stats_df_DEN_filt, mapa_DEN_mean, 'Densidad (mean)'],
              '7':[stats_df_DEN_filt, mapa_DEN_median, 'Densidad (median)'],
              '8':[stats_df_NEUT_filt, mapa_NEUT_mean, 'Neutrón (mean)'],
              '9':[stats_df_NEUT_filt, mapa_NEUT_median, 'Neutrón (median)'],
            }

#####
### DISTANCIAS ###

with open('Data/coord_filt.pkl', 'rb') as file:
    df_filt = pickle.load(file)

# Se generan las opciones para el selector de métricas
path = "Data/Distancias_DTW"
list_label = [{'label': 'Ninguno', 'value': '1'}]
dicc_dist = {}
for i, x in enumerate(os.listdir(path)):
    dicc_temp = {}
    if x.endswith(".pkl"):
        dicc_temp['label'] = os.path.splitext(x)[0]
        dicc_temp['value'] = str(i + 2)

        list_label.append(dicc_temp)

        with open(f"{path}/{x}", 'rb') as file:
            datos =pickle.load(file)
        
        dicc_dist[str(i+2)] = datos   


### RESULTADOS

with open('Data/CRM/casos_minimos.pkl', 'rb') as file:
    minimos = pickle.load(file)

with open('Data/CRM/resultados_corridas.pkl', 'rb') as file:
    resultados_caso = pickle.load(file)

with open('Data/CRM/conectividades_crm_por_caso_comb.pkl', 'rb') as file:
    conectividades_crm_por_caso = pickle.load(file)

with open('posicion_label_casos.pkl', 'rb') as file:
    posicion_label_caso = pickle.load(file)

with open('posicion_label_DTW.pkl', 'rb') as file:
    posicion_label_DTW = pickle.load(file)



############ FUNCIONES ####################


try:
    plt.style.use('seaborn-v0_8-white')
except:
    plt.style.use('seaborn-white')


def map_log(coord, datos, label, alpha=0.6):
    x = coord['X']
    y = coord['Y']
    gr = coord['Mean']
    pozo = coord['POZO']

    gridx, gridy, z_interp = datos

    # Mapas de contornos
    contour = go.Contour(
        x=gridx,     
        y=gridy,     
        z=z_interp,  
        colorscale="RdYlBu_r",
        opacity=alpha,
        colorbar=dict(
            title=label,
            title_side="right",
            # x=1.1
            ),
        contours=dict(showlines=False),
    )

    # Gráfico de pozos
    scatter = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(color=gr, colorscale="Greys", size=4), 
                    # colorbar=dict(
                    #     title="Mean(GR)",
                    #     x=1.2)),
        text=pozo,  
        hoverinfo='text',
    )

    # Se combinan los gráficos
    fig = go.Figure(data=[contour, scatter])


    # Set axis properties
    fig.update_layout(
        xaxis=dict(
            tickformat=".0f",
            tickmode="auto",
            showline=True,          
            linecolor="black",       
            linewidth=1,              
            mirror=True,
            ticks="inside"
        ),
        yaxis=dict(
            tickformat=".0f",
            tickmode="auto",
            showline=True,           
            linecolor="black",       
            linewidth=1,                
            mirror=True,
            ticks="inside"
        ),
        legend=dict(x=0.95, y=0.05,
                    xanchor='right', bgcolor='rgba(0,0,0,0)',
                    yanchor='bottom'),
        # width=800,
        # height=800,
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_scaleanchor='y',  
        yaxis_scaleanchor='x',  
        yaxis_constrain='domain',
    )

    config = {'toImageButtonOptions': {
        'format': 'jpeg',  # one of png, svg, jpeg, webp
        'height': None,
        'width': None,
        'scale': 15,  # Image quality
    }, 'scrollZoom': True, 'displaylogo': False}

    fig.update_layout(newshape_line_color='blue',
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath',
                                   'drawcircle', 'drawrect', 'eraseshape'])
    
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=30))

    return fig

def plot_map_plus(list_con, edge_cost=None, lab='GR', avg_prop_map=False, datos_mapear=None, alpha=1.0, edge_vmin=None, 
                  edge_vmax=None, edge_cmap='RdYlGn_r', normal_widths=5):
    
    # Define the graph
    G = nx.Graph()

    # Add nodes using the well names
    G.add_nodes_from(df_filt.POZO)

    # Create a dictionary for well types and set it as node attributes
    tipo = {w: t for w, t in zip(df_filt.POZO, df_filt.Tipo)}
    nx.set_node_attributes(G, tipo, name="type")

    # Get node positions from the dataframe
    layout = {w: p for w, p in zip(df_filt.POZO, df_filt.pos)}
    nx.set_node_attributes(G, layout, name="location")

    # Separate layouts for different well types
    layout_inyector = {w: p for w, p in layout.items() if G.nodes[w]['type'] == 'Inyector'}
    layout_producer = {w: p for w, p in layout.items() if G.nodes[w]['type'] == 'Productor'}

    # Define colors based on well type
    colores = {w: ('green' if p == 'Productor' else 'blue') for w, p in zip(df_filt.POZO, df_filt.Tipo)}

    # Extract nodes by type
    inyector_nodes = [n for (n, ty) in nx.get_node_attributes(G, 'type').items() if ty == 'Inyector']
    producer_nodes = [n for (n, ty) in nx.get_node_attributes(G, 'type').items() if ty == 'Productor']

    # Create a figure
    fig = go.Figure()

    # Average property map
    if avg_prop_map:

        gridx, gridy, z_interp = datos_mapear[1]

        fig.add_trace(go.Contour(
            x=gridx,
            y=gridy,
            z=z_interp,
            colorscale='RdYlBu_r',
            opacity=alpha,
            name='Log',
            colorbar=dict(title=datos_mapear[2], titleside='right'),
            contours=dict(showlines=False)
        ))

    if list_con:
        G.add_edges_from(list_con)

        if edge_cost:
            # Define edge costs as an attribute
            nx.set_edge_attributes(G, edge_cost, name='cost')
            cost_values = [G.edges[e]['cost'] for e in G.edges()]
            min_cost = min(cost_values)
            max_cost = max(cost_values)
            normalized_widths = [(cost - min_cost) / (max_cost - min_cost) * normal_widths + 0.5 for cost in cost_values]

            if edge_vmin is None:
                edge_vmin = min(cost_values)
            if edge_vmax is None:
                edge_vmax = max(cost_values)

            # Add edges to the figure
            for (u, v), width, cost in zip(G.edges(), normalized_widths, cost_values):
                # Set the color based on the normalized cost value
                cost_norm = (cost - edge_vmin) / (edge_vmax - edge_vmin)
                cost_norm = min(0.999, cost_norm)
                cost_norm = max(0.0001, cost_norm)
                color = pc.sample_colorscale(edge_cmap, cost_norm)

                fig.add_trace(go.Scatter(
                    x=[layout[u][0], layout[v][0]],
                    y=[layout[u][1], layout[v][1]],
                    mode='lines',
                    line=dict(
                        width=width,       
                        color=color[0]      
                    ),
                    showlegend=False
                ))

            # Add colorbar for edge costs
            # Create a dummy trace for the colorbar

            cost_values_cut = [min(c, edge_vmax) for c in cost_values]
            cost_values_cut = [max(c, edge_vmin) for c in cost_values_cut]

            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                line=dict(width=10, color='blue'),  # Example color for colorbar (change as needed)
                name='',
                hoverinfo='none',
                marker=dict(size=0,
                            color=cost_values_cut,
                    colorbar=dict(
                    title="Costo DTW",
                    titlefont = dict(
                                family="Arial",
                                size=12
                                ),
                    tickfont = dict(
                                family="Arial",
                                size=12
                                ),
                    thickness=25,
                    len=1,
                    xanchor='left',
                    titleside='right',
                    x=1.15,
                    y=0.5,
                    ),
                    colorscale = edge_cmap
                ), showlegend=False)

            )

    
    # Add nodes to the graph
    fig.add_trace(go.Scatter(
        x=[layout[n][0] for n in inyector_nodes],
        y=[layout[n][1] for n in inyector_nodes],
        text=list(layout_inyector.keys()),
        mode='markers',
        marker=dict(size=7, color='dodgerblue', symbol='triangle-up', line=dict(color='black', width=0.4)),
        name='Inyector'
    ))

    fig.add_trace(go.Scatter(
        x=[layout[n][0] for n in producer_nodes],
        y=[layout[n][1] for n in producer_nodes],
        text=list(layout_producer.keys()),
        mode='markers',
        marker=dict(size=4, color='whitesmoke', symbol='circle', line=dict(color='black', width=0.4)),
        name='Productor'
    ))

    # Add annotations for descriptive text
    textstr = f"Número de nodos= {G.number_of_nodes()}<br>Número de conexiones= {G.number_of_edges()}"
    fig.add_annotation(
        text=textstr,
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        showarrow=False,
        font=dict(size=10),
        align="left"
    )

    # # Set axis properties
    fig.update_layout(
    #     xaxis=dict(
    #         tickformat=".0f",
    #         tickmode="auto",
    #         showline=True,          
    #         linecolor="black",       
    #         linewidth=1,              
    #         mirror=True,
    #         ticks="inside"
    #     ),
    #     yaxis=dict(
    #         tickformat=".0f",
    #         tickmode="auto",
    #         showline=True,           
    #         linecolor="black",       
    #         linewidth=1,                
    #         mirror=True,
    #         ticks="inside"
    #     ),
        legend=dict(x=0.95, y=0.05,
                    xanchor='right', bgcolor='rgba(0,0,0,0)',
                    yanchor='bottom'),
    #     # width=800,
    #     # height=800,
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     xaxis_scaleanchor='y',  
    #     yaxis_scaleanchor='x',  
    #     xaxis_constrain='domain',
    )

    # fig.update_layout(
    # font_family="Arial",
    # # font_color="blue",
    # title_font_family="Arial",
    # # title_font_color="red",
    # # legend_title_font_color="green"
    # )
    fig.update_coloraxes(colorbar_title_font_family="Times New Roman")

    fig.update_yaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside", scaleanchor = "x", scaleratio = 1, tickfont = dict(size=10))
    fig.update_xaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside", tickfont = dict(size=10))

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=True, yaxis_title=None, xaxis_title=None, dragmode="pan", uirevision=True, autosize=True, 
            coloraxis_colorbar_x=-0.15,)
    
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=30))


    fig.update_layout(newshape_line_color='blue',
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath',
                                   'drawcircle', 'drawrect', 'eraseshape'])

    return fig

def filt_map_log(mapa_sel, met_sel, edge_min=None, edge_max=None, alpha=0.6):

    datos_registro = None
    prop_map=False
    if mapa_sel != '1': 
        datos_registro = datos_mapa[mapa_sel]
        prop_map=True

    if met_sel != '1':
        # mapa completo
        datos_dtw = dicc_dist[met_sel]

        valores = dicc_dist[met_sel].values()
        cmin = min(valores)
        cmax = max(valores)

        #conversión de escala
        vmin = cmin + (edge_min / 100) * (cmax - cmin)
        vmax = cmin + (edge_max / 100) * (cmax - cmin)

        return [plot_map_plus(list(datos_dtw.keys()), datos_dtw, avg_prop_map=prop_map, datos_mapear=datos_registro, alpha=alpha, normal_widths=5, edge_vmax=vmax, edge_vmin=vmin),
                [vmin, vmax]]
    else:
        if mapa_sel != '1':
            # solo mapa de registro
            return [map_log(datos_registro[0], datos_registro[1], datos_registro[2], alpha=alpha), [None, None]]
        else:
            return [None, [None, None]]

def plot_pattern(costo_filt, pozo, offset_title = 10, font_size = 8, normal_widths=5, edge_cmap='RdYlGn_r', edge_vmin=None, edge_vmax=None, label='Costo DTW', value_label_pos = None, filtro_nombres=True):

    costo_inj = {ip : c for ip, c in costo_filt.items() if ip[0] in pozo}

    nodos = [ip[1] for ip in costo_inj.keys()]
    nodos += pozo
    
    df_filt_patt = df_filt[df_filt.POZO.isin(nodos)]

    # Define the graph
    G = nx.Graph()

    # Add nodes using the well names
    G.add_nodes_from(df_filt_patt.POZO)

    # Create a dictionary for well types and set it as node attributes
    tipo = {w: t for w, t in zip(df_filt_patt.POZO, df_filt_patt.Tipo)}
    nx.set_node_attributes(G, tipo, name="type")

    # Get node positions from the dataframe
    layout = {w: p for w, p in zip(df_filt_patt.POZO, df_filt_patt.pos)}
    nx.set_node_attributes(G, layout, name="location")

    # Separate layouts for different well types
    layout_inyector = {w: p for w, p in layout.items() if G.nodes[w]['type'] == 'Inyector'}
    layout_producer = {w: p for w, p in layout.items() if G.nodes[w]['type'] == 'Productor'}

    # Define colors based on well type
    colores = {w: ('green' if p == 'Productor' else 'blue') for w, p in zip(df_filt_patt.POZO, df_filt_patt.Tipo)}
    colores = [c for c in colores.values()]

    # Extract nodes by type
    inyector_nodes = [n for (n, ty) in nx.get_node_attributes(G, 'type').items() if ty == 'Inyector']
    producer_nodes = [n for (n, ty) in nx.get_node_attributes(G, 'type').items() if ty == 'Productor']

    # Create a figure
    fig = go.Figure()

    # # Average property map
    # if avg_prop_map:

    #     gridx, gridy, z_interp = datos_mapear[1]

    #     fig.add_trace(go.Contour(
    #         x=gridx,
    #         y=gridy,
    #         z=z_interp,
    #         colorscale='RdYlBu_r',
    #         opacity=alpha,
    #         name='Log',
    #         colorbar=dict(title=datos_mapear[2], titleside='right'),
    #         contours=dict(showlines=False)
    #     ))

    G.add_edges_from(list(costo_inj.keys()))


    # Define edge costs as an attribute
    nx.set_edge_attributes(G, costo_inj, name='cost')
    cost_values = [G.edges[e]['cost'] for e in G.edges()]
    min_cost = min(cost_values)
    max_cost = max(cost_values)
    if min_cost < max_cost:
        normalized_widths = [(cost - min_cost) / (max_cost - min_cost) * normal_widths + 0.5 for cost in cost_values]
    else:
        normalized_widths = cost_values

    if edge_vmin is None:
        edge_vmin = min(cost_values)
    if edge_vmax is None:
        edge_vmax = max(cost_values)

    # Add edges to the figure
    for (u, v), width, cost in zip(G.edges(), normalized_widths, cost_values):
        # Set the color based on the normalized cost value
        cost_norm = (cost - edge_vmin) / (edge_vmax - edge_vmin)
        cost_norm = min(0.999, cost_norm)
        cost_norm = max(0.0001, cost_norm)
        color = pc.sample_colorscale(edge_cmap, cost_norm)

        fig.add_trace(go.Scatter(
            x=[layout[u][0], layout[v][0]],
            y=[layout[u][1], layout[v][1]],
            mode='lines',
            line=dict(
                width=width,       
                color=color[0]      
            ),
            showlegend=False
        ))

    # Add colorbar for edge costs
    # Create a dummy trace for the colorbar

    cost_values_cut = [min(c, edge_vmax) for c in cost_values]
    cost_values_cut = [max(c, edge_vmin) for c in cost_values_cut]

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        line=dict(width=10, color='blue'),  
        name='',
        hoverinfo='none',
        marker=dict(size=0,
                    color=cost_values_cut,
        colorbar=dict(
            title=label,
            titlefont = dict(
                family='Arial',
                size=10
            ),
            tickfont = dict(family="Arial",
                            size=10
                            ),
            thickness=10,
            len=1,
            xanchor='left',
            titleside='right',
            # x=1.15,
            # y=0.5,
            ),
            colorscale = edge_cmap,
            cmin=edge_vmin,
            cmax=edge_vmax
        ), showlegend=False)

    )

    fig.add_trace(go.Scatter(
        x=[layout[n][0] for n in inyector_nodes],
        y=[layout[n][1] for n in inyector_nodes],
        text=list(layout_inyector.keys()),
        mode='markers+text',
        marker=dict(size=7, color='dodgerblue', symbol='triangle-up', line=dict(color='black', width=0.4)),
        name='Inyector',
        textposition="bottom center",
        textfont=dict(
            family="Arial",
            size=9,
            color="black"
        )
    ))

    fig.add_trace(go.Scatter(
        x=[layout[n][0] for n in producer_nodes],
        y=[layout[n][1] for n in producer_nodes],
        text=list(layout_producer.keys()),
        mode='markers+text',
        marker=dict(size=5, color='whitesmoke', symbol='circle', line=dict(color='black', width=0.4)),
        name='Productor',
        textposition="bottom center",
        textfont=dict(
            family="Arial",
            size=9,
            color="black"
        )
    ))

    if value_label_pos is not None and filtro_nombres==True:
        label_pos_flt = {ip: datos for ip, datos in value_label_pos.items() if ip[0] in pozo}
        x_label_pos = [valor[0] for valor in label_pos_flt.values()]
        y_label_pos = [valor[1] for valor in label_pos_flt.values()]
        c_label = [float(valor[2]) for valor in label_pos_flt.values()]

        for x,y,c in zip(x_label_pos, y_label_pos, c_label):
            fig.add_annotation(x=x, y=y, text="{:.2f}".format(c), hovertext="{:.4f}".format(c),)
        fig.update_annotations(font=dict(color='black', size=9),bgcolor='white', showarrow=False,)

    # fig.update_layout(
    #     legend=dict(x=0.95, y=0.05, font=dict(size=8),
    #                 xanchor='right', bgcolor='rgba(0,0,0,0)',
    #                 yanchor='bottom'),
    # )

    fig.update_layout(
    legend=dict(
        x=0.95, y=0.95,  # Start in the middle of the plot
        font=dict(size=8),
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(0,0,0,0)',
        traceorder="normal",  
        itemclick="toggleothers"  
    ),
    )

    fig.update_yaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside", scaleratio = 1, tickfont = dict(size=8))
    fig.update_xaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside", scaleanchor = "y", tickfont = dict(size=8))

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=True, yaxis_title=None, xaxis_title=None, dragmode="pan", uirevision=True, autosize=True, 
            coloraxis_colorbar_x=-0.15,)


    fig.update_layout(margin=dict(l=30, r=30, t=30, b=30))

    fig.update_layout(newshape_line_color='blue',
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath',
                                   'drawcircle', 'drawrect', 'eraseshape'])

    return fig

def filt_map_patt(met_sel, pozo_sel, edge_min = None, edge_max = None, filtro_nombres=True):
        
        if met_sel != '1':
            datos_patt = dicc_dist[met_sel]

            valores = dicc_dist[met_sel].values()
            cmin = min(valores)
            cmax = max(valores)

            #conversión de escala
            vmin = cmin + (edge_min / 100) * (cmax - cmin)
            vmax = cmin + (edge_max / 100) * (cmax - cmin)

            value_DTW_pos = {par: posicion_label_DTW[par] + (metrica,) for par, metrica in datos_patt.items()}
            


            return [plot_pattern(datos_patt, pozo=pozo_sel, edge_vmin = vmin, edge_vmax=vmax, value_label_pos=value_DTW_pos, filtro_nombres=filtro_nombres), [vmin, vmax]]
        else:
            return [None, [None, None]]

def plot_sum_distances(summed_list, summed_distances_sqr, minimos, selected_case):
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=summed_distances_sqr,
        y=summed_list,
        mode='markers',
        marker=dict(color='white', size=4, line=dict(color='gray', width=0.1)),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=minimos[0], #corresponding_values
        y=minimos[1], #min_rmse
        mode='markers+lines',  
        marker=dict(color='white', size=8, line=dict(color='black', width=1)),  
        line=dict(color='black'),  
        name='Min RMSE',  
        showlegend=False
    ))

    x_sel = [minimos[0][int(selected_case)]]
    y_sel = [minimos[1][int(selected_case)]]

    fig.add_trace(go.Scatter(
        x=x_sel,
        y=y_sel,
        mode='markers',
        marker=dict(color='red', size=12, line=dict(color='black', width=1)),
        name='Regularización Seleccionada', 
        showlegend=True
    ))


    fig.update_yaxes(showline=True, linecolor='black', mirror="all",ticks="inside", tickfont = dict(size=10))#, scaleanchor = "x", scaleratio = 1, )
    fig.update_xaxes(tickformat = ".f", showline=True, linecolor='black', mirror="all",ticks="inside", tickfont = dict(size=10))

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=True, dragmode="pan", uirevision=True, autosize=True, 
            coloraxis_colorbar_x=-0.15,)
    
        # Customize axis labels and style
    fig.update_layout(
        xaxis_title="Σ(λᵢⱼ × DTWᵢⱼ)²",
        yaxis_title="Σ RMSEₚ",
        xaxis=dict(
            title_font=dict(size=10, family='Arial'), 
            tickfont=dict(size=8, family='Arial')  
        ),
        yaxis=dict(
            title_font=dict(size=10, family='Arial'), 
            tickfont=dict(size=8, family='Arial') 
        ),
        margin=dict(l=30, r=80, t=30, b=30),
        legend=dict(
        x=0.98, 
        y=0.98, 
        xanchor='right', 
        yanchor='top',  
        bgcolor='rgba(255, 255, 255, 0.5)',  
        bordercolor='white', 
        borderwidth=0, 
        font=dict(
            size=10,       # Font size for the legend
            family="Arial",  # Font family for the legend
            color="black"  # 
        )),
    )
    
    

    fig.update_layout(newshape_line_color='blue',
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath',
                                   'drawcircle', 'drawrect', 'eraseshape'])

    return fig

def filt_crm_patt(caso_sele, pozo, label, filtro_nombres):
    conectividades = conectividades_crm_por_caso[int(caso_sele)]
    
    return plot_pattern(conectividades, pozo, label=label, edge_cmap='gray_r', value_label_pos=posicion_label_caso[int(caso_sele)], edge_vmin=0, edge_vmax=1, filtro_nombres=filtro_nombres, normal_widths=3)


# def rgb(minimum, maximum, value, modo='positivo'):
#     try:
#         avg = (minimum+maximum)/2
#         if minimum==maximum:
#             r=0
#             g=255
#         else:
#             if modo=='positivo':
#                 if value <= avg:
#                     r=255
#                     g=int((value - minimum) * 255/(avg - minimum))
#                 else:
#                     g=255
#                     r=int((maximum - value) * 255 /(maximum - avg))
#             else:
#                 if value <= avg:
#                     g=255
#                     r=int((value - minimum) * 255/(avg - minimum))
#                 else:
#                     r=255
#                     g=int((maximum - value) * 255 /(maximum - avg))
#         b=0
#     except:
#         r=0
#         g=255
#     return r, g, b



# def cross_df(coor, df):
#     # Por productor
#     df = df.merge(coor.drop_duplicates(subset=['Pozo']),  how='left', left_on='Inyector', right_on='Pozo')
#     df.drop(columns='Pozo', inplace=True)
#     # Por inyector
#     df = df.merge(coor.drop_duplicates(subset=['Pozo']), how='left', left_on='Productor', right_on='Pozo', suffixes=('_Iny', '_Prod'))
#     df.drop(columns=['Pozo', 'Tipo_Iny', 'Tipo_Prod'], inplace=True)
    
#     # Factor para espesor en función del factor geologico
#     N=5
#     # Cálculo de factor de espesor
#     if df.Fc.min()==df.Fc.max():
#         df['width']=N
#     else:
#         df['width']=(df.Fc-df.Fc.min())/(df.Fc.max()-df.Fc.min())*N

#     # Posición del factor de conectividad en gráfica
#     df['X_med']=(df.X_Iny+df.X_Prod)/2
#     df['Y_med']=(df.Y_Iny+df.Y_Prod)/2
#     return df

# # Corregir fechas formato Sahara 
# def corregir_fecha(f):
#     meses = {'ene':'01', 'feb':'02', 'mar':'03', 'abr':'04', 'may':'05', 'jun':'06', 'jul':'07', 'ago':'08', 'sep':'09', 'oct':'10', 'nov':'11', 'dic':'12'}
#     for word, replacement in meses.items():
#         f = f.replace(word, replacement)
#     return f

# # Homologar nombres de pozos
# def pozo_pivot(x):
#     x = x.replace('Z1','')
#     x = x.replace('Z2','')
#     x = x.replace('Z3','')
#     x = x.replace('Z4','')
#     x = x.replace('Z5','')
#     x = x.replace('Z6','')
#     x = x.replace('Z7','')
#     x = x.replace('Z8','')
#     x = x.replace('I','')
#     x = x.replace('_','')
#     return x    

# # Función para procesar resultados
# def process_result(Res_SL, conex, coord,  Dic_zonas={}, cutoff=0.05,):
    
#     """
#     Esta función procesa los archivos de resultados.
#         i. Limpieza de datos y homologación de nombres
#         ii. Determinación de conexiones
#         iii. Combinación con archivo de conexiones
#         iv. Cálculo de métricas de comparación y de captura de conexiones
#         v. Generación de listados de conexiones en cada clasificación
#         vi. Generación de df de listado de conexiones con coordenadas
#         vii. Generación de df con pozos sin coordenadas
    
#     """
#     # Homologar nombres de pozos inyectores 
#     if Dic_zonas: 
#         try:
#             Res_SL['I_POZO']=Res_SL['I_WELL'].apply(lambda x: x.replace(x,Dic_zonas[x]))
#         except:
#             Res_SL['I_POZO']=Res_SL['I_WELL']
            
#     else:
#       Res_SL['I_POZO']=Res_SL['I_WELL'].apply(lambda x: pozo_pivot(x))

#     # Seleccionar solo columnas de interés
#     Res_SL=Res_SL[['I_POZO','P_WELL','I_WAF_S(rb/rb)', 'P_WAF_S(rb/rb)']]

#     # Identificación de conexiones
#     pd.options.mode.chained_assignment = None
#     Res_SL['Conectados_SL'] = np.where(Res_SL['I_WAF_S(rb/rb)'] >= cutoff, 1, 0)
#     pd.options.mode.chained_assignment = 'warn'

#     # Organizando los valores de mayor a menor I_WAF
#     Res_SL=Res_SL.sort_values(by='I_WAF_S(rb/rb)', ascending=False)

#     # Eliminar los pares de pozos duplicados, conservando aquel par con mayor I_WAF
#     Res_SL_proc = Res_SL.drop_duplicates(subset = ['I_POZO', 'P_WELL'], keep = 'first').reset_index(drop = True)

#     # Combinación de df resultados SL y conexiones empíricas
#     mdf= conex.merge(Res_SL_proc, how='left', left_on=['Inyector', 'Productor'], right_on=['I_POZO', 'P_WELL'])

#     # Remplazando valores nulos con cero
#     mdf.fillna(0, inplace=True)

#     # Cálculo de métricas de comparación

#     #Casos donde el modelo SL captura las conexiones empíricas
#     mdf['Ajuste'] = np.where((mdf.Conectados_SL == 1)&(mdf.Conectados == 'Si'), 1, 0)

#     #Casos donde la conexión se identifica solo en el modelo SL
#     mdf['SoloSL'] = np.where((mdf.Conectados_SL == 1)&(mdf.Conectados == 'No'), 1, 0)

#     #Casos donde el modelo SL NO captura las conexiones empíricas
#     mdf['SoloConEmp'] = np.where((mdf.Conectados_SL == 0)&(mdf.Conectados == 'Si'), 1, 0)

#     # Cálculo Métrica de Captura de Conexiones

#     Conex_Emp=mdf['Conectados'].str.count('Si').sum()
#     Conex_Cap=(mdf['Ajuste']==1).sum()
#     Met_capt = Conex_Cap/Conex_Emp

#     #Listado de conexiones capturadas
#     mdf_ajuste=mdf[mdf.Ajuste==1][['Inyector', 'Productor']]

#     #Listado de Conexiones no Capturadas
#     mdf_soloEmp=mdf[mdf.SoloConEmp==1][['Inyector', 'Productor']]

#     # Conexiones solo en SL y no en archivo de seguimiento
#     mdf_SL= conex.merge(Res_SL_proc, how='right', left_on=['Inyector', 'Productor'], right_on=['I_POZO', 'P_WELL'])

#     # Remplazando valores nulos con cero
#     mdf_SL.fillna(0, inplace=True)

#     # Conexiones solo en SL
#     mdf_SL['SoloSL'] = np.where((mdf_SL.Conectados_SL == 1) & (mdf_SL.Conectados == 0), 1, 0)

#     # Listado de Conexiones solo en SL
#     mdf_SL = mdf_SL[mdf_SL.SoloSL==1][['I_POZO', 'P_WELL']]


#     ### PROCESAMIENTO DE DFS
#     # Cruzar df con ajuste en conexiones
#     # Por inyector
#     mdf_ajuste1 = mdf_ajuste.merge(coord, how='left', left_on='Inyector', right_on='Pozo')
#     mdf_ajuste1.drop(columns='Pozo', inplace=True)
#     # Por productor
#     mdf_ajuste1 = mdf_ajuste1.merge(coord, how='left', left_on='Productor', right_on='Pozo', suffixes=('_Iny', '_Prod'))
#     mdf_ajuste1.drop(columns=['Pozo', 'Tipo_Iny', 'Tipo_Prod'], inplace=True)
#     mdf_ajuste1 = mdf_ajuste1.drop_duplicates(keep = 'first').reset_index(drop = True)

#     # Cruzar df con conexiones no capturadas
#     # Por inyector
#     mdf_soloEmp1 = mdf_soloEmp.merge(coord, left_on='Inyector', right_on='Pozo')
#     mdf_soloEmp1.drop(columns='Pozo', inplace=True)
#     # Por productor
#     mdf_soloEmp1 = mdf_soloEmp1.merge(coord, left_on='Productor', right_on='Pozo', suffixes=('_Iny', '_Prod'))
#     mdf_soloEmp1.drop(columns=['Pozo', 'Tipo_Iny', 'Tipo_Prod'], inplace=True)
#     mdf_soloEmp1 = mdf_soloEmp1.drop_duplicates(keep = 'first').reset_index(drop = True)

#     # Cruzar df con conexiones SOLO en SL
#     # Por inyector
#     mdf_SL1 = mdf_SL.merge(coord, how='left', left_on='I_POZO', right_on='Pozo')
#     mdf_SL1.drop(columns='Pozo', inplace=True)
#     # Por productor
#     mdf_SL1 = mdf_SL1.merge(coord, how='left', left_on='P_WELL', right_on='Pozo', suffixes=('_Iny', '_Prod'))
#     mdf_SL1.drop(columns=['Pozo', 'Tipo_Iny', 'Tipo_Prod'], inplace=True)
#     mdf_SL1 = mdf_SL1.drop_duplicates(keep = 'first').reset_index(drop = True)
    

#     #Pozos sin coordenadas
#     df_pozos_sin_coord = mdf_SL1[mdf_SL1['X_Iny'].isnull()][['I_POZO','X_Iny']].drop_duplicates(keep = 'first').reset_index(drop = True)
#     mdf_SL1.rename(columns = {'I_POZO':'Inyector', 'P_WELL':'Productor'}, inplace=True)

#     return [mdf_ajuste1, mdf_soloEmp1, mdf_SL1, Met_capt]


# # Función de mapeo

# def plot_comparison(mdf_ajuste1, mdf_soloEmp1, mdf_SL1, coord, mostrar_pozo=1, filtro=0, list_pozos=[], titulo="", filtro_capt=1, filtro_nocapt=1, filtro_adic=1, caso='', cutoff=0.05, f_line=0, line=None):

#     #Lista para filtrar coordenadas
#     inj_filt = mdf_ajuste1[(mdf_ajuste1['Inyector'].isin(list_pozos)) | (mdf_ajuste1['Productor'].isin(list_pozos))].Inyector.unique()
#     prd_filt = mdf_ajuste1[(mdf_ajuste1['Inyector'].isin(list_pozos)) | (mdf_ajuste1['Productor'].isin(list_pozos))].Productor.unique()
#     in_second_but_not_in_first = set(prd_filt) - set(inj_filt)
#     List_coord = list(inj_filt) + list(in_second_but_not_in_first)


#     inj_filt = mdf_soloEmp1[(mdf_soloEmp1['Inyector'].isin(list_pozos)) | (mdf_soloEmp1['Productor'].isin(list_pozos))].Inyector.unique()
#     prd_filt = mdf_soloEmp1[(mdf_soloEmp1['Inyector'].isin(list_pozos)) | (mdf_soloEmp1['Productor'].isin(list_pozos))].Productor.unique()
#     in_second_but_not_in_first = set(prd_filt) - set(inj_filt)
#     List_coord_1 = list(inj_filt) + list(in_second_but_not_in_first)

#     in_second_but_not_in_first = set(List_coord_1) - set(List_coord)
#     List_coord += list(in_second_but_not_in_first)


#     inj_filt = mdf_SL1[(mdf_SL1['Inyector'].isin(list_pozos)) | (mdf_SL1['Productor'].isin(list_pozos))].Inyector.unique()
#     prd_filt = mdf_SL1[(mdf_SL1['Inyector'].isin(list_pozos)) | (mdf_SL1['Productor'].isin(list_pozos))].Productor.unique()
#     in_second_but_not_in_first = set(prd_filt) - set(inj_filt)
#     List_coord_1 = list(inj_filt) + list(in_second_but_not_in_first)

#     in_second_but_not_in_first = set(List_coord_1) - set(List_coord)
#     List_coord += list(in_second_but_not_in_first)

#     #Filtrar por selección de pozos iny o prod
#     if filtro == 1:
#         # Filtro por lista de pozos
#         coordp = coord[coord['Pozo'].isin(List_coord)].reset_index(drop=True)
#         mdf_soloEmpp = mdf_soloEmp1[(mdf_soloEmp1['Inyector'].isin(list_pozos)) | (mdf_soloEmp1['Productor'].isin(list_pozos))].reset_index(drop=True)
#         mdf_SLp = mdf_SL1[(mdf_SL1['Inyector'].isin(list_pozos)) | (mdf_SL1['Productor'].isin(list_pozos))].reset_index(drop=True)
#         mdf_ajustep = mdf_ajuste1[(mdf_ajuste1['Inyector'].isin(list_pozos)) | (mdf_ajuste1['Productor'].isin(list_pozos))].reset_index(drop=True)
#     else:
#         coordp = coord.copy()
#         mdf_soloEmpp = mdf_soloEmp1.copy()
#         mdf_SLp = mdf_SL1.copy()
#         mdf_ajustep = mdf_ajuste1.copy()

#     # Plot Pozos
#     coordp = coordp.sort_values(by=['Tipo'], ascending=True)
#     fig = px.scatter(coordp, x="X", y="Y", color="Tipo", #width=1100, height=900,
#                      symbol='Tipo',color_discrete_sequence=["dodgerblue", "forestgreen"], 
#                      symbol_sequence=["triangle-up","circle"], hover_name='Pozo')
#     fig.update_traces(marker=dict(size=8), textposition="bottom center",showlegend=False)

#     # mdf_ajuste = Conexiones capturadas
#     # mdf_soloEmp = Conexiones No capturada
#     # mdf_SL = Conexiones solo en SL
    
    
#     def plot_conex(df, color, nombre, dash, espesor=1):
#         for par, pozo in enumerate(df.Inyector):
#             if par==0:
#                 fig.add_trace(go.Scatter(x=[df.X_Iny[par],df.X_Prod[par]], y=[df.Y_Iny[par],df.Y_Prod[par]], mode='lines',
#                                     line_color=color,line=dict(dash=dash), line_width=espesor, name=nombre, 
#                                     hoverinfo='skip'))
#             else:
#                 fig.add_trace(go.Scatter(x=[df.X_Iny[par],df.X_Prod[par]], y=[df.Y_Iny[par],df.Y_Prod[par]], mode='lines',
#                                     line_color=color,line=dict(dash=dash), line_width=espesor, name=None, 
#                                     hoverinfo='skip'))
    
    
#     #Plot filtra que conexiones se quieren ver 1- Solo las capturadas, 2- 1 + Las No Capturadas, 3- 2 + Solo SL
#     if filtro_nocapt==1:plot_conex(mdf_soloEmpp, 'red', 'No capturada por SL', 'dot', 2)
#     if filtro_adic==1:plot_conex(mdf_SLp, 'gray', 'Adicional por SL', 'dashdot', 1)
#     if filtro_capt==1:plot_conex(mdf_ajustep, 'lime', 'Capturada por SL', 'solid', 3)


#     if mostrar_pozo == 1:
#         texto_prod = coordp[coordp.Tipo=='PRODUCTOR'].Pozo
#         texto_iny =  coordp[coordp.Tipo=='INYECTOR'].Pozo
#     else:
#         texto_prod=None
#         texto_iny=None

#     fig.add_trace(go.Scatter(x = coordp[coordp.Tipo=='PRODUCTOR'].X, y = coordp[coordp.Tipo=='PRODUCTOR'].Y,
#                         marker=dict(size=10,
#                                     color="forestgreen",),
#                         text = texto_prod,
#                         mode="markers+text",hoverinfo='skip'))

#     fig.add_trace(go.Scatter(x = coordp[coordp.Tipo=='INYECTOR'].X, y = coordp[coordp.Tipo=='INYECTOR'].Y,
#                     marker=dict(size=10,
#                                 color="dodgerblue",
#                                 symbol="triangle-up",),
#                     text = texto_iny,
#                     mode="markers+text",hoverinfo='skip'))                    

#     fig.update_traces(marker=dict(size=10),textposition="bottom center")

#     # ver líneas
#     if f_line == 1:
#         # plot fallas
#         if line is not None:
#             for i in line['T'].unique():
#                 df = line[line['T']==i]
#             # if i ==1:
#             #     fig = px.line(df, x="X", y="Y", title='Life expectancy in Canada')
            
#             # else:
#                 fig.add_trace(go.Scatter(x=df["X"], y=df["Y"], mode='lines',
#                                                 line_color='black', #line=dict(dash='dot'), 
#                                                 line_width=1, 
#                                                 hoverinfo='skip'))



#     fig.update_annotations(font=dict(color='black', size=8),bgcolor='white', showarrow=False,)
#     fig.update_yaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside", scaleanchor = "x", scaleratio = 1,)
#     fig.update_xaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside",)

#     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
#                        showlegend=True, 
#                        yaxis_title=None, 
#                        xaxis_title=None, 
#                        dragmode="pan", 
#                        uirevision=True, 
#                        autosize=True, margin=dict(l=20, r=30, t=50, b=20),
#             coloraxis_colorbar_x=-0.15,)

#     fig.update_layout(legend=dict(yanchor="top",
#         y=0.99,
#         xanchor="left",
#         x=0.01,
#         bordercolor="Black",
#         borderwidth=1,),
#         legend_title_text='Conexión', 
#         legend_title_font_color="green",
#                      )

#     for trace in fig['data']: 
#         if(trace['name'] not in ['No capturada por SL','Adicional por SL','Capturada por SL']): trace['showlegend'] = False

#     # config = {'toImageButtonOptions': {
#     #     'format': 'jpeg', # one of png, svg, jpeg, webp
#     #     'filename': 'Comparación Resultados','height': None,
#     #     'width': None,'scale': 15, # Image quality
#     #     },    'scrollZoom': True,'displaylogo': False}

#     plot_title={'text': '<b>'+ caso + '</b><br>Porcentaje de Captura de Conexiones: {:.2%}'.format(titulo) + ' [Cutoff = {:.0%}]'.format(cutoff),
#                 'x':0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'}
    
    
#     fig.update_traces(marker=dict(size=10),textposition="bottom center")
#     fig.update_annotations(font=dict(color='black', size=8),bgcolor='white', showarrow=False,)
#     fig.update_yaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside", scaleanchor = "x", scaleratio = 1,)
#     fig.update_xaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside",)

#     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=True, yaxis_title=None, xaxis_title=None, dragmode="pan", uirevision=True, autosize=True, 
#             coloraxis_colorbar_x=-0.15,)

#     fig.update_layout(legend=dict(yanchor="top",
#         y=0.99,
#         xanchor="left",
#         x=0.01,
#         bordercolor="Black",
#         borderwidth=1,),
#         legend_title_text='Conexión', 
#         legend_title_font_color="green",
#         )


#     fig.update_layout(newshape_line_color='blue',
#                       title=plot_title,
#                     modebar_add=['drawline','drawopenpath','drawclosedpath',
#                     'drawcircle','drawrect','eraseshape'])

#     # fig.show(config=config)
#     return fig

# # FUNCIÓN PARA PROCESAR ARCHIVOS Y COMPARAR
# def process_result_to_cons(list_resultados, coord, Dic_zonas={} , cutoff=0.05, N=4):
    
#     #PROCESAR LOS DF DE RESULTADOS
#     list_resul_pro=[]
#     for Res_SL in list_resultados:
#         # Homologar nombres de pozos inyectores 
#         if Dic_zonas: 
#           Res_SL['I_POZO']=Res_SL['I_WELL'].apply(lambda x: x.replace(x,Dic_zonas[x]))
#         else:
#           Res_SL['I_POZO']=Res_SL['I_WELL'].apply(lambda x: pozo_pivot(x))

#         # Seleccionar solo columnas de interés
#         Res_SL=Res_SL[['I_POZO','P_WELL','I_WAF_S(rb/rb)', 'P_WAF_S(rb/rb)']]

#         # Identificación de conexiones
#         pd.options.mode.chained_assignment = None
#         Res_SL['Conectados_SL'] = np.where(Res_SL['I_WAF_S(rb/rb)'] >= cutoff, 1, 0)
#         pd.options.mode.chained_assignment = 'warn'

#         # Organizando los valores de mayor a menor I_WAF
#         Res_SL=Res_SL.sort_values(by='I_WAF_S(rb/rb)', ascending=False)

#         # Eliminar los pares de pozos duplicados, conservando aquel par con mayor I_WAF
#         Res_SL_proc = Res_SL.drop_duplicates(subset = ['I_POZO', 'P_WELL'], keep = 'first').reset_index(drop = True)
#         Res_SL_proc= Res_SL_proc[['I_POZO','P_WELL','Conectados_SL']]
        
#         # print(Res_SL_proc[Res_SL_proc.P_WELL=='AK-17'])
        
#         list_resul_pro.append(Res_SL_proc)

#     #COMBINAR RESULTADOS
    
#     for i,df in enumerate(list_resul_pro):
        
#         if i == 0:
#             mdf=list_resul_pro[0]
#             # print(mdf[mdf.I_POZO=='AK-29ST2'])
#         else:
#             # Combinación de df resultados SL
#             mdf= mdf.merge(df, how='outer', on=['I_POZO', 'P_WELL'], suffixes=('', '_'+str(i)))
#             # print(mdf[mdf.I_POZO=='AK-29ST2'])
#             # Remplazando valores nulos con cero
#             mdf.fillna(0, inplace=True)
#         # print(mdf[mdf.I_POZO=='AK-29ST2'])

#     # print(mdf[mdf.I_POZO=='AK-29ST2'])

#     # Cálculo de métricas de comparación
    
#     mdf['Total']=mdf.sum(axis=1, numeric_only=True)
#     mdf = mdf.drop(mdf[mdf['Total'] == 0].index)
#     mdf['P(Con)'] = mdf['Total']/(len(list(mdf.columns))-3)
#     mdf.rename(columns = {'I_POZO':'Inyector', 'P_WELL':'Productor'}, inplace=True)
    
#     met_similaridad = mdf.Total.sum()/(len(mdf.index)*(len(list(mdf.columns))-4))
    
#     ### PROCESAMIENTO DE DFS

#     # Cruzar df con coordenadas
#     # Por inyector
#     mdf = mdf.merge(coord, how='left', left_on='Inyector', right_on='Pozo')
#     mdf.drop(columns='Pozo', inplace=True)
#     # Por productor
#     mdf = mdf.merge(coord, how='left', left_on='Productor', right_on='Pozo', suffixes=('_Iny', '_Prod'))
#     mdf.drop(columns=['Pozo', 'Tipo_Iny', 'Tipo_Prod'], inplace=True)
#     mdf = mdf.drop_duplicates(keep = 'first').reset_index(drop = True)

#     #Pozos sin coordenadas
#     df_pozos_sin_coord = mdf[mdf['X_Iny'].isnull()][['Inyector','X_Iny']].drop_duplicates(keep = 'first').reset_index(drop = True)
    
#     # Cálculo de factor de espesor
#     mdf['width']=1 + (mdf['P(Con)']-mdf['P(Con)'].min())/(mdf['P(Con)'].max()-mdf['P(Con)'].min())*N
#     mdf['X_med']=(mdf.X_Iny+mdf.X_Prod)/2
#     mdf['Y_med']=(mdf.Y_Iny+mdf.Y_Prod)/2
    
    
#     return mdf, met_similaridad, df_pozos_sin_coord

# # FUNCIÓN PARA MAPEAR COMPARACIÓN DE RESULTADOS Y CÁLCULO DE PROBABILIDADES
# def plot_comparison_bet(df, coord, mostrar_pozo=1, filtro=0, list_pozos=[], inp_min=0, inp_max=1, met_sim=0, f_line=0, line=None):

#     #Lista para filtrar coordenadas
#     inj_filt=df[(df['Inyector'].isin(list_pozos)) | (df['Productor'].isin(list_pozos))].Inyector.unique()
#     prd_filt=df[(df['Inyector'].isin(list_pozos)) | (df['Productor'].isin(list_pozos))].Productor.unique()
#     in_second_but_not_in_first = set(prd_filt) - set(inj_filt)
#     List_coord = list(inj_filt) + list(in_second_but_not_in_first)
    
#     #Filtrar por selección de pozos iny o prod
#     if filtro == 1:
#         # Filtro por lista de pozos
#         coordp=coord[coord['Pozo'].isin(List_coord)].reset_index(drop=True)
#         df1=df[(df['Inyector'].isin(list_pozos)) | (df['Productor'].isin(list_pozos))].reset_index(drop=True)
#     else:
#         coordp=coord.copy()
#         df1=df.copy()
    
#     df1=df1[(df1['P(Con)']>=inp_min) & (df1['P(Con)']<=inp_max)].reset_index(drop=True)
        
#     # Plot Pozos
#     coordp = coordp.sort_values(by=['Tipo'], ascending=True)
#     fig = px.scatter(coordp, x="X", y="Y", color="Tipo",# width=1100, height=900,
#                      symbol='Tipo',color_discrete_sequence=["dodgerblue", "forestgreen"], 
#                      symbol_sequence=["triangle-up","circle"], hover_name='Pozo',)
#     fig.update_traces(marker=dict(size=8),textposition="bottom center",showlegend=False)
    
    
#     v_max=df1.width.max()
#     v_min=df1.width.min()
#     cmax=df1['P(Con)'].max()
#     cmin=df1['P(Con)'].min()
    


#     for par, pozo in enumerate(df1.Inyector):
#         fig.add_trace(go.Scatter(x=[df1.X_Iny[par],df1.X_Prod[par]], y=[df1.Y_Iny[par], df1.Y_Prod[par]], mode='lines',
#                                 line_width=df1.width[par], line_color='rgb'+ str(rgb(v_min,v_max,df1.width[par])),
#                                 hoverinfo='skip'))
# #         fig.add_annotation(x=df1.X_med[par], y=df1.Y_med[par], text="{:.2f}".format(df1['P(Con)'][par]), hovertext="{:.4f}".format(df['P(Con)'][par]),)
    
#     # titulo='Probabilidad de Conexión'
#     variable = '<b>Probabilidad de<br>Conexión'
#     values=df1['P(Con)']
    
#     if mostrar_pozo == 1:
#         texto_prod = coordp[coordp.Tipo=='PRODUCTOR'].Pozo
#         texto_iny =  coordp[coordp.Tipo=='INYECTOR'].Pozo
#     else:
#         texto_prod=None
#         texto_iny=None
        

#     fig.add_trace(go.Scatter(x = coordp[coordp.Tipo=='PRODUCTOR'].X, y = coordp[coordp.Tipo=='PRODUCTOR'].Y,
#                         marker=dict(size=10, color="forestgreen",),
#                         text = texto_prod,
#                         mode="markers+text",hoverinfo='skip'))

#     fig.add_trace(go.Scatter(x = coordp[coordp.Tipo=='INYECTOR'].X, y = coordp[coordp.Tipo=='INYECTOR'].Y,
#                     marker=dict(size=10,
#                                 color="dodgerblue",
#                                 symbol="triangle-up",),
#                     text = texto_iny,
#                     mode="markers+text",hoverinfo='skip'))   

#         # ver líneas
#     if f_line == 1:
#         # plot fallas
#         if line is not None:
#             for i in line['T'].unique():
#                 df = line[line['T']==i]
#             # if i ==1:
#             #     fig = px.line(df, x="X", y="Y", title='Life expectancy in Canada')
            
#             # else:
#                 fig.add_trace(go.Scatter(x=df["X"], y=df["Y"], mode='lines',
#                                                 line_color='black', #line=dict(dash='dot'), 
#                                                 line_width=1, 
#                                                 hoverinfo='skip'))                 

#     fig.update_traces(marker=dict(size=10),textposition="bottom center")
    
#     fig.add_trace(go.Scatter(x=df1.X_med,y=df1.Y_med,
#                 marker=dict(size=0.1,cmax=cmax, cmin=cmin,
#                 color=values,
#                 colorbar=dict(title=dict(text=variable, font=dict(size=12)), ticks="outside",
#                             thicknessmode="pixels", thickness=30, lenmode="pixels", len=200,
#                             yanchor="top", y=1, x=0.02),
#                 colorscale=[[0,"rgb(255, 0, 0)"],[0.5,"rgb(255, 255, 0)"],[1,"rgb(0, 255, 0)"]],),
#                 mode="markers",hovertemplate="%{marker.color:.4f}<extra></extra>"))
    
    
#     fig.update_annotations(font=dict(color='black', size=8),bgcolor='white', showarrow=False,)
#     fig.update_yaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside", scaleanchor = "x", scaleratio = 1,)
#     fig.update_xaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside",)

#     for trace in fig['data']: 
#         if(trace['name'] not in ['No capturada por SL','Adicional por SL','Capturada por SL']): trace['showlegend'] = False

#         plot_title={'text': '<b>Probabilidad de Conexión</b><br> Similitud: {:.2%}'.format(met_sim),
#                 'x':0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'}

#     fig.update_layout(newshape_line_color='blue',
#                      title=plot_title,
#                     modebar_add=['drawline','drawopenpath','drawclosedpath',
#                     'drawcircle','drawrect','eraseshape'])

#     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=True, yaxis_title=None, xaxis_title=None, dragmode="pan", uirevision=True, autosize=True,
#             coloraxis_colorbar_x=-0.15,)

#     # fig.update_layout(legend=dict(yanchor="top",
#     #     y=0.99,
#     #     xanchor="left",
#     #     x=0.01,
#     #     bordercolor="Black",
#     #     borderwidth=1,),
#     #     legend_title_text='Conexión', 
#     #     legend_title_font_color="green",
        
#     #                  )
    
#     # Set axis properties
#     fig.update_layout(
#         xaxis=dict(
#             tickformat=".0f",
#             tickmode="auto",
#             showline=True,          
#             linecolor="black",       
#             linewidth=1,              
#             mirror=True,
#             ticks="inside"
#         ),
#         yaxis=dict(
#             tickformat=".0f",
#             tickmode="auto",
#             showline=True,           
#             linecolor="black",       
#             linewidth=1,                
#             mirror=True,
#             ticks="inside"
#         ),
#         legend=dict(x=0.95, y=0.05,
#                     xanchor='right', bgcolor='rgba(0,0,0,0)',
#                     yanchor='bottom'),
#         width=800,
#         height=800,
#         plot_bgcolor="rgba(0,0,0,0)",
#         xaxis_scaleanchor='y',  
#         yaxis_scaleanchor='x',  
#         yaxis_constrain='domain',
#     )

#     config = {'toImageButtonOptions': {
#         'format': 'jpeg',  # one of png, svg, jpeg, webp
#         'height': None,
#         'width': None,
#         'scale': 15,  # Image quality
#     }, 'scrollZoom': True, 'displaylogo': False}

#     fig.update_layout(newshape_line_color='blue',
#                       modebar_add=['drawline', 'drawopenpath', 'drawclosedpath',
#                                    'drawcircle', 'drawrect', 'eraseshape'])


#     fig.show(config=config)
#     return fig




##--------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------- ESTRUCTURA DE DASH --------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------

def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.Br(), html.Br(),
            html.Br(), 
            html.Hr(),
            html.H5("Caracterización Mejorada de la Conectividad entre Pozos - Campo A"),
            html.H3("Visualización de resultados"),
            html.Div(
                id="intro",
                children="Esta herramienta permite visualizar los resultados de la metodología de caracterización.",
            ),html.Hr(),
        ],
    )


# def generate_input_card_coor():
#     """
#     :return: A Div containing input for data.
#     """
#     return html.Div(
#         id='carga-coor',
#         children=[
#             html.B("Mapeo Conectividades", style={'width': '100%','textAlign': 'left',} ),
#             html.Hr(),
#             html.B("Coordenadas de pozos"),
#             dcc.Upload(
#                 id='upload-coor',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '30px',
#                     'lineHeight': '30px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '5px',
#                     'display':'inline-block'
#                 },

#             ),
#             html.Div(id='output-data-upload-coor'),
#             dcc.Store(id='stored-data-1', data=None)
#             # html.Br(),
#     ], style={"width": "100%"} )

# def generate_input_card_conemp():
#     """
#     :return: A Div containing input for data.
#     """
#     return html.Div(
#         id='carga-con',
#         children=[
#             html.B("Conexiones Empíricas"),
#             dcc.Upload(
#                 id='upload_con_emp',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '30px',
#                     'lineHeight': '30px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '5px'
#                 },
#                 # Allow multiple files to be uploaded
#                 # multiple=True
#             ),
#             html.Div(id='output-data-upload-conemp'),
#             dcc.Store(id='stored-data-2-1', data=None),
#     ], style={"width": "100%"} )


# def generate_input_card_dic():
#     """
#     :return: A Div containing input for data.
#     """
#     return html.Div(id='carga-dic', 
#             children= [
#             html.B("Diccionario Zona:Pozo"),
#             dcc.Upload(
#                 id='upload-dic',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '30px',
#                     'lineHeight': '30px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '5px'
#                 },
#                 # Allow multiple files to be uploaded
#                 # multiple=True
#             ),
#             html.Div(id='output-data-upload-dic'),
#             # html.Hr(),
#             dcc.Store(id='stored-data-3-1', data=None),
#     ], style={"width": "100%"} )

# def generate_input_card_resultados():
#     """
#     :return: A Div containing input for data.
#     """
#     return html.Div(id='carga-resultados',
#             children=[
#             html.B("Resultados Streamlines"),
#             dcc.Upload(
#                 id='upload_resultados_sl',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '30px',
#                     'lineHeight': '30px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '5px'
#                 },
#                 # Allow multiple files to be uploaded
#                 multiple=True
#             ),
#             html.Div(id='output-data-upload-resultados'),
#             html.Br(),
#             html.Hr(),
#             dcc.Store(id='stored-data-4-1', data=None),
#             dcc.Store(id='stored-data-4-2', data=None),
#     ], style={"width": "100%"} )


 

def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div([
            html.Br(),
            html.B("Mapeo Conectividades", style={'width': '100%','textAlign': 'left',} ),
            html.Hr(),
            html.P("Seleccione el mapa de fondo a visualizar:"),
            dcc.Dropdown(
                    id="mapa-select",
                    multi=False,
                    options = [{'label': 'Ninguno', 'value': '1'},
                               {'label': 'GR (mean)', 'value': '2'},
                               {'label': 'GR (median)', 'value': '3'},
                               {'label': 'GR_all (mean)', 'value': '4'},
                               {'label': 'GR_all (median)', 'value': '5'},
                               {'label': 'Densidad (mean)', 'value': '6'},
                               {'label': 'Densidad (median)', 'value': '7'},
                               {'label': 'Neutrón (mean)', 'value': '8'},
                               {'label': 'Neutrón (median)', 'value': '9'}],
                    value='1',
                    ),
            html.Br(),
            html.Hr(),
            html.B("Métrica de Similitud", style={'width': '100%','textAlign': 'left',} ),
            html.Hr(),
            html.P("Seleccione la métrica para caracterizar la conexión:"),
            dcc.Dropdown(
                    id="metrica-select",
                    multi=False,
                    options = list_label,
                    value='28',
                    ),
            html.Br(),
            # html.Div([
            #     html.Br(),
            #     html.B("Cutoff Conexión, %:", style={'width': '50%','textAlign': 'left',} ),
            #     html.Hr(),
            #     html.Div([dcc.Input(
            #                     id="input_cutoff",
            #                     type='number',
            #                     style={'width':'30%'},
            #                     value=5,
            #                 ),
            #             ],style={'display':'table-cell', 'verticalAlign':'left'}),
            # ], style={"width": "100%", 'textAlign':'center','display': 'flex', 
            #           'align-items': 'center', 'justify-content': 'center'}),
            
            html.Br(),
            html.Button(id='map-btn', children='GENERAR MAPA', ),
            html.Br(),html.Br(),
            # html.Button(id='prob-btn', children='PROBABILIDAD DE CONEXIONES', ),
            
            html.Br(),html.Br(),
            html.Hr(),
            html.Br(),html.Br(),
            html.Br(),html.Br(),
            html.Br(),html.Br(),
            html.Br(),html.Br(),
            html.Br(),html.Br(),
            html.Br(),html.Br(),
            
            
            html.B("Análisis por Patrón", style={'width': '100%','textAlign': 'left',} ),
            html.Hr(),
            # Dropdown
            html.Div(
                children=[
                    html.Label("Seleccione el pozo a analizar:"),
                    dcc.Dropdown(
                        id="well-select",
                        value=None,  # Default selection as None
                        multi=True
                    ),
                ],
                style={'margin-bottom': '20px'}
            ),

            html.Br(),
            html.Button(id='patt-btn', children='MOSTRAR PATRÓN', ),
            html.Br(),html.Br(),

            html.Hr(),
                    html.Br(),
                    html.Div(children="Regularización DTW"),
                    html.Div(
                     id="slider-reg-id", children=[
                    dcc.Slider(0, 11, 1,
                               value = 5,
                               id='slider-reg',
                               marks={0:{'label': '0', 'style':{'font-size':'10px'}},
                                      1:{'label': '1E-4', 'style':{'font-size':'10px'}},
                                      2:{'label': '1E-3', 'style':{'font-size':'10px'}},
                                      3:{'label': '2E-3', 'style':{'font-size':'10px'}},
                                      4:{'label': '3E-3', 'style':{'font-size':'10px'}},
                                      5:{'label': '5E-3', 'style':{'font-size':'10px'}},
                                      6:{'label': '1E-2', 'style':{'font-size':'10px'}},
                                      7:{'label': '2E-2', 'style':{'font-size':'10px'}},
                                      8:{'label': '3E-2', 'style':{'font-size':'10px'}},
                                      9:{'label': '4E-2', 'style':{'font-size':'10px'}},
                                      10:{'label': '5E-2', 'style':{'font-size':'10px'}},
                                      11:{'label': '0.1', 'style':{'font-size':'10px'}},
                                    #   12:{'label': '0.1', 'style':{'font-size':'10px'}}
                                    },
                            #    tooltip={"placement": "bottom", "always_visible": True}
                               ),
                               html.Div(id='output-reg')
                    ]),
                    html.Br(),
            dcc.Checklist(id='filtro_nombres',
            options=[
                {'label': 'Mostrar valores', 'value': '1'},
                    ],value=['1'],
            ),
            
            # html.B("Gráfico de Eficiencia/Rendimiento"),
            # html.Hr(),
            # html.Br(),
            # html.P("Seleccione la métrica a visualizar:"),
            # dcc.Dropdown(
            #     id="rel-select",
            #     options=[{"label": 'Eficiencia de Inyección [BOD/BWID]', "value": 0},{"label": 'Rendimiento [BWID/BOD]', "value": 1},],
            #     value=0,
            # ),
            # html.Br(),
            # html.Button(id='effi-btn', children='Graficar', ),

            # html.Br(),html.Br(),
            # html.Hr(),
            # html.Br(),html.Br(),html.Br(),
            # html.Br(),html.Br(),html.Br(),
            # html.Br(),html.Br(),html.Br(),
            # html.Br(),html.Br(),html.Br(),
            # html.Br(),html.Br(),html.Br(),
            # html.Br(),html.Br(),html.Br(),
            # html.Br(),html.Br(),html.Br(),

            
            
            # html.B("Gráfico de Eficiencia/Rendimiento Histórico"),
            # html.Hr(),
            # html.Br(),
            # html.P("Selección de pozo:"),
            # dcc.Dropdown(
            #         id="well-select-1",
            #         multi=True,
            #         ),

            # html.Br(),
            # html.Button(id='effi-hist-btn', children='Graficar', ),

        ], style={"width": "100%", 'textAlign':'center', }
    )



# def parse_contents(contents, filename, date, n):
#     content_type, content_string = contents.split(',')
    
#     decoded = base64.b64decode(content_string)
#     today = dtt.today()
    
#     try:
#         skip=[0]
#         if 'Análisis_SL' in filename: skip=None
#         if 'csv' in filename:
#             # Assume that the user uploaded a CSV file
#             decoded=''.join(decoded.decode('utf-8').replace('\t', ','))
#             df = pd.read_csv(io.StringIO(decoded), decimal='.', skiprows=skip, engine="python")# parse_dates=['Fecha_init','Fecha_fin'] )
#             ef=None
#         elif 'txt' in filename:             
#             # Assume that the user uploaded a txt file
#             df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t',decimal='.',)# parse_dates=['Fecha_init','Fecha_fin'] )
#             ef=None
#         elif 'xls' in filename:
#             # Assume that the user uploaded an excel file
#             df = pd.read_excel(io.BytesIO(decoded), sheet_name='Nivel Iny-Prod', parse_dates=['FECHA'])
#             ef = pd.read_excel(io.BytesIO(decoded), sheet_name='Nivel Patrón (Total)', parse_dates=['FECHA'])
        
#         elif 'Fal' in filename:
#             # Assume that the user uploaded an fault file from sahara
#             df = pd.read_csv(io.BytesIO(decoded), sep='\t')
#             ef = None      
                 
#     except Exception as e:
#         print(e)


#     return [df,html.Div([
#                     html.P('Archivo cargado: ' + filename),
#                     # html.H6(datetime.datetime.fromtimestamp(date)),
#                     dcc.Store(id='stored-data-'+str(n), data=df.to_dict('records'))
#                      ]),html.Div([
#                     html.P('Archivo cargado: ' + filename),
#                     # html.H6(datetime.datetime.fromtimestamp(date)),
#                     dcc.Store(id='stored-data-a-'+str(n), data=df.to_dict('records'))
#                      ]), ef]




app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        # html.Div(
        #     id="banner",
        #     className="banner",
        #     # children=[html.Img(src=app.get_asset_url("rwa_logo.png"))],
        # ),
        # Left column
        html.Div(
            id="left-column",
            className="three columns",
            children=[description_card(), 
            #  generate_input_card_coor(),
            #  generate_input_card_conemp(),
            #  generate_input_card_dic(),
            #  generate_input_card_resultados(),
             generate_control_card()]

            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        

        html.Div(
            id="middle-column",
            className="seven columns",
            children=[
                # Row for the first main graph spanning both sub-columns
                html.Div(
                    id="map_card",
                    children=[
                        dcc.Graph(
                            id="map",
                            config={
                                'toImageButtonOptions': {
                                    'format': 'jpeg',
                                    'filename': 'Mapa',
                                    'scale': 15,
                                },
                                'scrollZoom': True,
                                'displaylogo': False,
                            },
                            style={'height': '95vh'}
                        )
                    ],
                    # style={'display': 'flex', 'width': '100%'}
                ),
                
                html.Br(),
                html.Br(),
                html.Hr(),
                html.Br(),
                html.Br(),
                # Row with two sub-columns for additional graphs
                html.Div(
                    style={'flex': '0.1', 'display': 'flex', 'width': '100%', 'padding': '0', 'margin': '0'},
                    children=[
                        # First sub-column for additional graph
                        html.Div(
                            dcc.Graph(
                                id="map_patt_1",
                                config={
                                'toImageButtonOptions': {
                                    'format': 'jpeg',
                                    'filename': 'Mapa',
                                    'scale': 15,
                                },
                                
                                'scrollZoom': True,
                                'displaylogo': False,
                                
                                },
                                style={'height': '100%', 'padding': '0', 'margin': '0'}
                            ),
                            className="six columns",
                            style={'height': '100%', 'padding': '0', 'margin': '0'}
                            
                        ),
                        
                        # Second sub-column for additional graph
                        html.Div(
                            dcc.Graph(
                                id="map_patt_2",
                                 config={
                                'toImageButtonOptions': {
                                    'format': 'jpeg',
                                    'filename': 'Mapa',
                                    'scale': 15,
                                },
                                'scrollZoom': True,
                                'displaylogo': False,
                                },
                                # style={'height': '45vh'}
                                style={'height': '100%', 'padding': '0', 'margin': '0'}
                            ),
                            className="six columns",
                            style={'height': '100%', 'padding': '0', 'margin': '0'}
                            
                        ),
                    ],
                    # style={'display': 'flex', 'width': '100%'}
                ),


                html.Div(
                    style={'flex': '0.1', 'display': 'flex', 'width': '100%', 'padding': '0', 'margin': '0'},
                    children=[
                        # First sub-column for additional graph
                        html.Div(
                            dcc.Graph(
                                id="map_patt_3",
                                config={
                                'toImageButtonOptions': {
                                    'format': 'jpeg',
                                    'filename': 'Mapa',
                                    'scale': 15,
                                },
                                
                                'scrollZoom': True,
                                'displaylogo': False,
                                
                                },
                                style={'height': '100%', 'padding': '0', 'margin': '0'}
                            ),
                            className="six columns",
                            style={'height': '100%', 'padding': '0', 'margin': '0'}
                            
                        ),
                        
                        # Second sub-column for additional graph
                        html.Div(
                            dcc.Graph(
                                id="map_patt_4",
                                 config={
                                'toImageButtonOptions': {
                                    'format': 'jpeg',
                                    'filename': 'Mapa',
                                    'scale': 15,
                                },
                                'scrollZoom': True,
                                'displaylogo': False,
                                },
                                # style={'height': '45vh'}
                                style={'height': '100%', 'padding': '0', 'margin': '0'}
                            ),
                            className="six columns",
                            style={'height': '100%', 'padding': '0', 'margin': '0'}
                            
                        ),
                    ],
                    # style={'display': 'flex', 'width': '100%'}
                ),




                # html.Br(),html.Br(),html.Br(),
                html.Hr(),
            ]
        ),




        # # Middle column
        # html.Div(
        #     id="middle-column",
        #     className="seven columns",
        #     children=[
        #         html.Div(
        #             id="map_card",
        #             children=[
        #                 # html.B("Map"),
        #                 html.Hr(),
        #                 dcc.Graph(id="map", config = {'toImageButtonOptions': {
        #                                             'format': 'jpeg', # one of png, svg, jpeg, webp
        #                                             'height': None, 'filename': 'Mapa',
        #                                             'width': None,'scale': 15,  # Image quality
        #                                             }, 'scrollZoom': True,'displaylogo': False}, style={'height' : '95vh',}# 'width' : '56vw' }
        #                         ),
                        
        #             ], 
        #             #style={"width": "100%"} 
        #         ),
        #         # Fecha
        #         # html.Div(
        #         #      id="slider", children=[
        #         #          dcc.Slider(
        #         #                     id='date-slider',)
        #         #      ],style={"width": "100%"}),
        #         # html.Div(
        #         #     id='eff_card',
        #         #     children=[html.Br(),
        #         #         html.Hr(),
        #         #         html.Br(),
        #         #     dcc.Graph(id="effi_plot", config = {'toImageButtonOptions': {
        #         #                                     'format': 'jpeg', # one of png, svg, jpeg, webp
        #         #                                     'height': None, 'filename': 'Effi_plot',
        #         #                                     'width': None,'scale': 15,  # Image quality
        #         #                                     },    'scrollZoom': True,'displaylogo': False}, style={'height' : '70vh',}#'width' : '56vw' }
        #         #             ),

        #         #     ]),
        #         # html.Div(
        #         #      id="slider-eff", children=[
        #         #     dcc.Slider(id='date-slider-eff',)]),
        #         # html.Div(
        #         #     id='eff_plot_card',
        #         #     children=[html.Br(),
        #         #         html.Hr(),
        #         #         html.Br(),
        #         #     dcc.Graph(id="effi_plot_hist", config = {'toImageButtonOptions': {
        #         #                                     'format': 'jpeg', # one of png, svg, jpeg, webp
        #         #                                     'height': None, 'filename': 'Effi_plot_hist',
        #         #                                     'width': None,'scale': 15,  # Image quality
        #         #                                     },    'scrollZoom': True,'displaylogo': False}, style={'height' : '60vh',}#'width' : '56vw' }
        #         #             ),

        #             # ]),

        #     ],
        # ),
        # Third column
         html.Div(
            id="right-column",
            className="two columns",
            children=[
                html.Div(
                     children=[
                         
                    html.Br(),html.Br(),html.Br(),
                    html.B("Escala de Color (Conexiones)", style={'width': '100%','textAlign': 'left',} ),
                    html.Hr(),
                    # dcc.RadioItems(id='ref_escala',
                    #     options=[
                    #         {'label': 'Relativa', 'value': 0},
                    #         {'label': 'Absoluta', 'value': 1},
                    #     ],
                    #     value=1,
                    #     labelStyle={'display': 'inline-block'}
                    # ),
                    html.Br(),
                    html.Div(children="Limites de Escala"),
                    html.Div(
                     id="slider-ajust-color", children=[
                    dcc.RangeSlider(id='slider-color',
                                    min=0, max=99,
                                    value=[0,99],
                                    marks=None,
                                    step=1,
                                    allowCross=False,)
                    ]),
                    html.Div(id='output-escala'),
                    html.Br(),


                    html.Br(),html.Br(),html.Br(),
                    html.B("Opacidad (Mapa)", style={'width': '100%','textAlign': 'left',} ),
                    html.Hr(),
                    html.Br(),
                    html.Div(children="Grado de Opacidad"),
                    html.Div(
                     id="slider-opacity-id", children=[
                    dcc.Slider(0, 1, 0.1,
                               value=0.6,
                               id='slider-opacity',
                               marks=None,
                               tooltip={"placement": "bottom", "always_visible": True}
                               ),
                               html.Div(id='output-opacity')
                    ]),
                    html.Br(),

                    
                    # html.Br(),
                    # html.B("Filtro de Pozos", style={'width': '100%','textAlign': 'left',} ),
                    # html.Hr(),
                    # dcc.Checklist(id='filtro_wells',
                    # options=[
                    #     {'label': 'Filtrar', 'value': '1'},
                    #         ],
                    # ),
                    
                    # dcc.Dropdown(
                    # id="well-select",
                    # multi=True,
                    # ),
                    # # html.Br(), 

                    # # html.B("Escala de Color (Mapa)", style={'width': '100%','textAlign': 'left',} ),
                    # html.Hr(),
                    # dcc.Checklist(id='filtro_nombres',
                    # options=[
                    #     {'label': 'Mostrar nombres', 'value': '1'},
                    #         ],value=['1'],
                    # ),
                    # html.Hr(),
                    # html.Br(),
                    # html.P("Seleccione las conexiones a visualizar:"),
                    # dcc.Checklist(id='filtro_captura',
                    # options=[
                    #     {'label': 'Conexiones Capturadas', 'value': '1'},
                    #         ],value=['1'],
                    # ),
                    # dcc.Checklist(id='filtro_nocaptura',
                    # options=[
                    #     {'label': 'Conexiones NO Capturadas', 'value': '1'},
                    #         ],value=['1'],
                    # ),
                    # dcc.Checklist(id='filtro_adicional',
                    # options=[
                    #     {'label': 'Conexiones adic. por SL', 'value': '1'},
                    #         ],value=['1'],
                    # ),
                    # html.Br(),
                    # html.Hr(),
                    # html.P("Opciones de descarga"),
                    # html.Div([
                    #     html.Div([html.Button(id='descargar-btn', children='DATA', title='Descargar la información del caso selecionado',style={'width':'100%'}),
                    #                 dcc.Download(id="download-case"),
                    #                 # html.Button(id='shape-btn', children='SHAPE FILES', title='Descargar los archivos shape del caso',style={'width':'100%'} ),
                    #                 # dcc.Download(id="download-shape"),
                    #             ],style={'display':'table-cell', 'verticalAlign':'middle'}),
                    # ], style={"width": "100%", 'textAlign':'center','display': 'flex', 
                    #         'align-items': 'center', 'justify-content': 'center'}),
                    # html.Br(),
                    # html.Hr(),
                    # html.Div(
                    #         id='carga_fallas',
                    #         children=[
                    #             html.B("Líneas"),
                    #             html.Hr(),
                    #             dcc.Checklist(id='filtro_lineas',
                    #             options=[
                    #                 {'label': 'Ver', 'value': '1'},
                    #                     ],value=['1'],
                    #             ),

                    #             dcc.Upload(
                    #                 id='upload_fallas',
                    #                 children=html.Div([
                    #                     'Drag and Drop or ',
                    #                     html.A('Select Files')
                    #                 ]),
                    #                 style={
                    #                     'width': '100%',
                    #                     'height': '30px',
                    #                     'lineHeight': '30px',
                    #                     'borderWidth': '1px',
                    #                     'borderStyle': 'dashed',
                    #                     'borderRadius': '5px',
                    #                     'textAlign': 'center',
                    #                     'margin': '5px'
                    #                 },
                    #                 # Allow multiple files to be uploaded
                    #                 multiple=True
                    #             ),
                    #             html.Div(id='output-data-upload-fallas'),
                    #             dcc.Store(id='stored-data-fallas', data=None),
                    #     ], style={"width": "100%"} ),

                    #     html.Br(),
                    #     html.P("Seleccione las líneas a mapear:"),
                    #     dcc.Dropdown(
                    #             id="caso-select-fallas",
                    #             multi=False,
                    #             ),
                    #     html.Br(),
                    # dcc.RadioItems(id='ref_escala',
                    #     options=[
                    #         {'label': 'Relativa', 'value': 0},
                    #         {'label': 'Absoluta', 'value': 1},
                    #     ],
                    #     value=1,
                    #     labelStyle={'display': 'inline-block'}
                    # ),
                    # html.Br(), 


                    # html.B("Nota:", style={'width': '100%','textAlign': 'left',} ),
                    # html.Hr(),
                    # html.Div(
                    #     id="nota",
                    #     children="Los valores de eficiencia de inyección y rendimiento son acotados en los rangos [0-1] y [0-50], respectivamente.",
                    # ),
                    html.Hr(),
                    # html.Br(),html.Br(),html.Br(),
                    # html.Br(),html.Br(),html.Br(),
                    # html.Br(),html.Br(),html.Br(),
                    html.Br(),html.Br(),html.Br(),
                    html.Br(),html.Br(),html.Br(),
                    html.Br(),html.Br(),html.Br(),
                    html.Br(),html.Br(),html.Br(),
                    html.Br(),html.Br(),html.Br(),
                    html.Hr(),
                    html.B("Acerca de", style={'width': '100%','textAlign': 'left',} ),
                    html.Hr(),
                    html.Div(children="Desarrollado por Grupo 7:"),
                    html.Div(children="Danuil Dueñas"),
                    html.Div(children="Nicolas Aldana"),
                    html.Div(children="Jose Salazar"),
                    html.Hr(),
                     ],
                )
           ]),
    ],
)


#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------CALLBACKS------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------



# ## CARGA DE COORDENADAS
# @app.callback([Output('output-data-upload-coor', 'children'),
#               Output('well-select', 'options'),], 
#               [Input('upload-coor', 'contents'),
#               State('upload-coor', 'filename'),
#               State('upload-coor', 'last_modified')])
# def update_output(contents, name_file, date):
#     global coord
#     if contents is not None:

#         # engine = sqlalchemy.create_engine('postgresql://postgres:skliscat@127.0.0.1:5432/rwa') 

#         # # Registrar ingreso
#         # datos = {
#         # 'login_date': [datetime.today()],
#         # }
#         # df_temp = pd.DataFrame(datos)
#         # df_temp.to_sql('registro_SL_ana', con=engine, if_exists='append', index=False)

#         children = [
#             parse_contents(contents, name_file, date, 1)[2] ]
#         df=parse_contents(contents, name_file, date, 1)[0]
#         coord = df
#         pozos_unicos = sorted(list(df.Pozo.unique()))
        
#         df1=df[df.Tipo=='INYECTOR']
#         pozos_iny_unicos = sorted(list(df1.Pozo.unique()))
#         return [children + [html.Div([                   
#                     dcc.Store(id='stored-data-1', data=coord.to_dict('records'))
#                      ])], [{'label': i, 'value': i} for i in pozos_unicos],] 
#                     #  [{'label': i, 'value': i} for i in pozos_iny_unicos]]
#     else:
#         return None, []


# ## CARGA DE CONEXIONES EMPIRICAS
# @app.callback(Output('output-data-upload-conemp', 'children'),
#               Input('upload_con_emp', 'contents'),
#               State('stored-data-1','data'),
#               State('upload_con_emp', 'filename'),
#               State('upload_con_emp', 'last_modified'),
#               )
# def update_output(contents, coorde, name_file, date):
#     global conemp, coord
#     if contents is not None:
#         pc = parse_contents(contents, name_file, date, 2)
#         children = [ pc[1] ]
        
#         # Leyendo información cargada
#         conemp= pc[ 0 ]
#         # # Convirtiendo la data almacenada a dataframe
#         # coord = pd.DataFrame.from_dict(coord)
#         # Combinando dataframes
        
#         # fcgeo = cross_df(coord, fcgeo)

#         return children + [html.Div([                   
#                     dcc.Store(id='stored-data-2-1', data=conemp.to_dict('records'))
#                      ])]


# # CARGA DE DICCIONARIO DE ZONAS
# @app.callback(Output('output-data-upload-dic', 'children'),
#               Input('upload-dic', 'contents'),
#               State('upload-dic', 'filename'),
#               State('upload-dic', 'last_modified'),
#               )
# def update_output(contents, name_file, date):
#     global Dic_zonas
#     if contents is not None:
#         pc = parse_contents(contents, name_file, date, 3)
#         children = [ pc[1] ]
#         # Leyendo información cargada
#         dic_zon=pc[0]
        
#         #Convertir df a diccionario
#         dic_zon.set_index('Zona', inplace=True)
#         Dic_zonas=dic_zon.to_dict()['Pozo']

#         return children + [html.Div([                   
#                     dcc.Store(id='stored-data-3-1', data=Dic_zonas)
#                      ])]


# # CARGA DE RESULTADOS SL
# @app.callback([Output('output-data-upload-resultados', 'children'),
#               Output('caso-select', 'options')],
#               Input('upload_resultados_sl', 'contents'),
#               State('upload_resultados_sl', 'filename'),
#               State('upload_resultados_sl', 'last_modified'),
#               )
# def update_output(contents, name_file, date):
#     global coord, res_Sl, conemp, dfs_procesados, list_resul, nombres_casos 
#     if contents is not None:
#         list_resul=[]
#         for i, cont in enumerate(contents):
#             pc = parse_contents(cont, name_file[i], date[i], 4)
#             children = [ pc[1] ]
#             # Leyendo información cargada
#             res_Sl=pc[0]
#             list_resul.append(res_Sl)
#             nombres_casos=name_file

#         return [[html.Div([                   
#                     dcc.Store(id='stored-data-4-1', data=list_resul[0].to_dict('records')), 
#                      ])], [{'label': nombre, 'value': i} for i, nombre in enumerate(name_file)],] 
#     else:
#         return [None, []]



# # CARGA DE FALLAS
# @app.callback([Output('output-data-upload-fallas', 'children'),
#               Output('caso-select-fallas', 'options')],
#               Input('upload_fallas', 'contents'),
#               State('upload_fallas', 'filename'),
#               State('upload_fallas', 'last_modified'),
#               )
# def update_output(contents, name_file, date):
#     global fallas, list_fallas, nombres_lineas
#     # global coord, fallas, conemp, dfs_procesados, list_fallas, nombres_lineas

#     if contents is not None:
#         list_fallas=[]
#         for i, cont in enumerate(contents):
#             pc = parse_contents(cont, name_file[i], date[i], 4)
#             children = [ pc[1] ]
#             # Leyendo información cargada
#             fallas=pc[0]
#             list_fallas.append(fallas)
#             nombres_lineas=name_file

#         return [[html.Div([                   
#                     dcc.Store(id='stored-data-fallas', data=list_fallas[0].to_dict('records')), 
#                      ])], [{'label': nombre, 'value': i} for i, nombre in enumerate(name_file)],] 
#     else:
#         return [None, []]



# MAPEAR REGISTRO
@app.callback([Output('map', 'figure', allow_duplicate=True),
              Output('output-escala', 'children', allow_duplicate=True),
              Output('well-select', 'options')
              ],
              Input('map-btn','n_clicks'),
              State('mapa-select', 'value'),
              State('metrica-select', 'value'),
              Input('slider-color', 'value'),
              Input('slider-opacity', 'value'),
              prevent_initial_call=True,
              )
def make_graphs(n, mapa_sele, met_sele, rango_escala, opacity):
    if n is None:
        return no_update
    else:
        if met_sele != '1': 
            lista_pozos = [{'label': pozo, 'value': pozo} for pozo in {par[0] for par in dicc_dist[met_sele]}]
        else:
            lista_pozos = []
       
        col_min, col_max = rango_escala[0], rango_escala[1]
    

        map_fig = filt_map_log(mapa_sele, met_sele, edge_min = col_min, edge_max=col_max, alpha=opacity)[0]
        cmin, cmax = filt_map_log(mapa_sele, met_sele, edge_min = col_min, edge_max=col_max, alpha=opacity)[1]
        
        if cmin is not None:
            return [map_fig, 'Min: {:0.2f}, Max: {:0.2f}'.format(cmin, cmax), lista_pozos]
        else:
            return [map_fig, None, lista_pozos]




# MAPEAR PATRÓN
@app.callback([Output('map_patt_1', 'figure', allow_duplicate=True),
            #   Output('output-escala', 'children', allow_duplicate=True)
              Output('map_patt_2', 'figure'),
              Output('map_patt_3', 'figure'),
              Output('map_patt_4', 'figure')
              ],
              Input('patt-btn','n_clicks'),
            #   State('mapa-select', 'value'),
              State('metrica-select', 'value'),
              Input('slider-color', 'value'),
            #   Input('slider-opacity', 'value'),
              State('well-select', 'value'),
              Input('slider-reg', 'value'),
              Input('filtro_nombres', 'value'),
              prevent_initial_call=True,
              )
def map_patterns(n, met_sele, rango_escala, pozo_sele, caso_sele, filtro_nombres):
    if n is None:
        return no_update
    else:
        col_min, col_max = rango_escala[0], rango_escala[1]
        
        filt_nombres = False
        if filtro_nombres == ['1']: filt_nombres = True

        map_fig = filt_map_patt(met_sele, pozo_sele, edge_min = col_min, edge_max=col_max, filtro_nombres=filt_nombres)[0]
        cmin, cmax = filt_map_patt(met_sele, pozo_sele, edge_min = col_min, edge_max=col_max, filtro_nombres=filt_nombres)[1]

        fig_reg = plot_sum_distances(resultados_caso[1], resultados_caso[0], minimos, caso_sele)
        
        fig_crm = filt_crm_patt(0, pozo_sele, label='Conectividades CRM', filtro_nombres=filt_nombres)

        fig_crm_reg = filt_crm_patt(caso_sele, pozo_sele, label='Conectividades CRM con Regularización DTW', filtro_nombres=filt_nombres)


        if cmin is not None:
            return map_fig, fig_crm, fig_reg, fig_crm_reg#, 'Min: {:0.2f}, Max: {:0.2f}'.format(cmin, cmax)]
        else:
            return map_fig, None, None, None






# ACTUALIZAR LISTADO DE POZOS

# @app.callback(
#     Output("dropdown-list", "options"),
#     Input("radio-button", "value")
# )
# def update_dropdown_options(selected_option):
#     if selected_option == 'option1':
#         return lista_pozos_productores
#     else:
#         return lista_pozos_inyectores


# # MAPEAR CAPTURA
# @app.callback(Output('map', 'figure', allow_duplicate=True),
#               Input('map-btn','n_clicks'),
#               State('filtro_wells', 'value'),
#               State('filtro_captura','value'),   
#               State('filtro_nocaptura','value'), 
#               State('filtro_adicional','value'), 
#               State('filtro_nombres','value'),
#               State('filtro_lineas','value'),
               
#               State('well-select', 'value'),
#               State('caso-select', 'value'),
#               State('input_cutoff', 'value'),
#               State('caso-select-fallas', 'value'),
#               prevent_initial_call=True,
#               )
# def make_graphs(n, filt_well, filtro_cap, filtro_nocap, filtro_adi, filtro_name, filtro_line, wells, caso_sel, cut, lin_sel):
#     global dfs_procesados, coord, list_resul, nombres_casos, list_fallas
#     if n is None:
#         return no_update
#     else:
#         dfs_procesados = process_result(list_resul[caso_sel], conemp, coord, Dic_zonas, cutoff=float(cut)/100)
        
#         lineas = None
#         if lin_sel is not None:
#             lineas = list_fallas[lin_sel]

#         filt_line = 0
#         if filtro_line == ['1']:
#             filt_line = 1

#         filtro = 0
#         if filt_well==['1']:
#             filtro = 1
        
#         # Lista de pozos a filtrar
#         if wells is None:
#             wells=[]
        
#         # Conexiones capturadas
#         filtro_capt = 0
#         if filtro_cap==['1']:
#             filtro_capt=1
        
#         # Conexiones NO capturadas
#         filtro_nocapt = 0
#         if filtro_nocap==['1']:
#             filtro_nocapt=1
        
#         # Conexiones Adicionales
#         filtro_adic = 0
#         if filtro_adi==['1']:
#             filtro_adic=1

#         # Conexiones Nombres
#         filtro_nombre = 0
#         if filtro_name==['1']:
#             filtro_nombre=1

#         map_fig = plot_comparison(dfs_procesados[0], dfs_procesados[1], dfs_procesados[2], coord, filtro_nombre, filtro, wells, 
#                                   dfs_procesados[3], filtro_capt, filtro_nocapt, filtro_adic, nombres_casos[caso_sel], cutoff=float(cut)/100, f_line = filt_line, line=lineas)
        
#         return map_fig 
    

    
# # MAPA DE PROBABILIDAD DE CONEXIONES
# @app.callback(Output('map', 'figure', allow_duplicate=True),
#               Input('prob-btn','n_clicks'),
#               State('filtro_wells', 'value'),
#               State('filtro_nombres','value'), 
#               State('well-select', 'value'),
#               State('input_cutoff', 'value'),
#               State('filtro_lineas','value'),
#               State('caso-select-fallas', 'value'),
#               prevent_initial_call=True,
#               )
# def prob_plot(n, filt_well, filtro_name, wells, cut, filtro_line, lin_sel):
#     if n is None:
#         return no_update
#     else:
#         global list_resul, coord, df_prob, list_fallas

#         lineas = None
#         if lin_sel is not None:
#             lineas = list_fallas[lin_sel]

#         filt_line = 0
#         if filtro_line == ['1']:
#             filt_line = 1

#         filtro = 0
#         if filt_well==['1']:
#             filtro = 1

#         # Conexiones Nombres
#         filtro_nombre = 0
#         if filtro_name==['1']:
#             filtro_nombre=1

#          # Lista de pozos a filtrar
#         if wells is None:
#             wells=[]

#         inp_min=0
#         inp_max=1
        
        
#         df_prob=process_result_to_cons(list_resul, coord, Dic_zonas , float(cut)/100)
#         met_simil=df_prob[1]
#         df_prob=df_prob[0]

        

#         prob_fig=plot_comparison_bet(df_prob, coord, filtro_nombre, filtro, wells, inp_min, inp_max, met_simil, f_line = filt_line, line=lineas)

#         return prob_fig



# #-----------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------
# #----------------------------------------DESCARGA DE ARCHIVOS-----------------------------------------------
# #-----------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------------------------------

# @app.callback(Output('download-case', 'data'),
#     Input('descargar-btn', 'n_clicks'),
#     prevent_initial_call=True,)
# def func(n):
#     global dfs_procesados, df_prob

#     nombre_file = 'Análisis_SL.xlsx' 

#     df_p = df_prob[['Inyector', 'Productor', 'P(Con)']]
#     df_p=df_p.rename(columns = {'Inyector':'I_WELL', 'Productor':'P_WELL', 'P(Con)':'I_WAF_S(rb/rb)'})
#     df_p['P_WAF_S(rb/rb)']=df_p['I_WAF_S(rb/rb)']
#     data={'Conexiones Capturadas':dfs_procesados[0][['Inyector', 'Productor']],
#           'Conexiones No Capturadas':dfs_procesados[1][['Inyector', 'Productor']],
#           'Conexion Adicionales SL':dfs_procesados[2][['Inyector', 'Productor']],
#           'Probabilidad de Conexión':df_p
#            }
#     writer = pd.ExcelWriter(nombre_file, engine='xlsxwriter')
#     for nombre, df in data.items():
#         df.to_excel(writer,sheet_name=nombre, index=False )
#     writer.close()

#     df_p.to_csv('Análisis_SL.csv',sep=',',index=False)

#     nombre_file = 'Análisis_SL'
#     nombre_zip = 'Análisis_SL.zip' 

#     with ZipFile(nombre_zip, 'w') as zipObj2:
#     # Add multiple files to the zip
#         zipObj2.write(nombre_file+'.csv')
#         zipObj2.write(nombre_file+'.xlsx')


#     return dcc.send_file(nombre_zip)


# @app.callback(Output('download-shape', 'data'),
#     [Input('shape-btn', 'n_clicks'),
#      State('field-select', 'value')],
#     prevent_initial_call=True,)
# def func(n, nombre_caso):
#     global rwa_cum, effi_cum, rwa, effi

#     nombre_file = nombre_caso + '_RWA.shp' 
#     nombre_zip = nombre_caso + '_RWA.zip'
#     schema = {
#     'geometry':'LineString',
#     'properties':[('Name','str'), ('Winy_ip','float'),('Np_ip','float'),
#                                 ('tiny','float'),('Winy_i','float'),
#                                 ('Np_p','float'),('WAF_p_cum','float'),
#                                 ('WAF_i_cum','float'),('Eff_avg','float'),
#                                 ('Util_avg','float'),('Qiny_avg','float')]
#             }

#     lineShp = fiona.open(nombre_file, mode='w',driver='ESRI Shapefile',
#                         schema = schema,)# crs = "EPSG:4326")
    
#     lineDf = rwa_cum.copy()
#     #get list of lines
#     line_list=[]
#     rowName = []
    
#     Winy_ip = []
#     Np_ip = []
#     tiny = []
#     Winy_i = []
#     Np_p = []
#     WAF_p_cum = []
#     WAF_i_cum = []
#     Eff_avg = []
#     Util_avg = []
#     Qiny_avg = []

#     for index, row in lineDf.iterrows():
#         xyList = []
#         xyList.append((row.X_Prod, row.Y_Prod))
#         xyList.append((row.X_Iny, row.Y_Iny))
#         line_list.append(xyList)
#         rowName.append(row.INJECTOR +'-'+ row.PRODUCER)
#         Qiny_avg.append(row.Qiny_avg)
#         Winy_ip.append(row['Qiny(i,p)'])
#         Np_ip.append(row['Qo_inc_aso (i,p)'])
#         tiny.append(row.t_iny)
#         Winy_i.append(row.Qiny_total)
#         Np_p.append(row.Qo_Aso_total)
#         WAF_p_cum.append(row.WAF_p_cum)
#         WAF_i_cum.append(row.WAF_i_cum)
#         Eff_avg.append(row.Eff_avg)
#         Util_avg.append(row.Util_avg)


#     for i, line in enumerate(line_list):
#         rowDict = {
#                 'geometry' : {'type':'LineString',
#                                 'coordinates': line},
#                 'properties': {'Name' : rowName[i], 
#                                 'Winy_ip':Winy_ip[i],
#                                 'Np_ip':Np_ip[i],
#                                 'tiny':tiny[i],
#                                 'Winy_i':Winy_i[i],
#                                 'Np_p':Np_p[i],
#                                 'WAF_p_cum':WAF_p_cum[i],
#                                 'WAF_i_cum':WAF_i_cum[i],
#                                 'Eff_avg':Eff_avg[i],
#                                 'Util_avg':Util_avg[i],
#                                 'Qiny_avg':Qiny_avg[i],
#                                 },
                
#                 }
#         lineShp.write(rowDict)

#     lineShp.close()

#     with ZipFile(nombre_zip, 'w') as zipObj2:
#         # Add multiple files to the zip
#         zipObj2.write(nombre_caso+'_RWA.dbf')
#         zipObj2.write(nombre_caso+'_RWA.cpg')
#         zipObj2.write(nombre_caso+'_RWA.shp')
#         zipObj2.write(nombre_caso+'_RWA.shx')

#     return dcc.send_file(nombre_zip)


# # Run the server
# if __name__ == "__main__":

#     from waitress import serve
#     serve(app.server, host="0.0.0.0", port="8054")


# Run the server
if __name__ == "__main__":
    # app.run_server()
    app.run_server(debug=True)
