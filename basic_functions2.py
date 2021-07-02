import pandas as pd
import networkx as nx
import numpy as np

def get_all_nodes(df):
    
    '''Input: dataframe containing numperso_referencia, numperso_asociado, coincidencias, dias_coincidencias and comercios_coincidencia
       Output: list containing all nodes for the graph
       
       This function returns a list containing the nodes from the input raw data'''
    
    numper_ref_list = pd.Series.tolist(df.numperso_referencia)
    numper_asc_list = pd.Series.tolist(df.numperso_asociado)
    numper_ref_list.extend(numper_asc_list)
    return numper_ref_list #obtener elementos unicos aka nodos

def filter_hubs(df, degree_filt):
    
    '''Input: dataframe containing numperso_referencia, numperso_asociado, coincidencias, dias_coincidencias and comercios_coincidencia
              and integer corresponding to the number of coincidencias
       Output: dataframe cointaining only the rows which have coincidencias > degree_filt
      
       This function filters the input datafram to contain only the rows where the two numperso have coincided
       more than degree_filt times'''
    
    numper_ref_list = get_all_nodes(df)
    counter=Counter(numper_ref_list) #contador de ocurrencias de cada nodo (degree)
    index_hub=[ind for ind,deg in counter.most_common() if deg>degree_filt] #numperso de los nodos con degree > 50
    com_coin_gtr2_bis = com_coin_gtr2[(~com_coin_gtr2.numperso_asociado.isin(index_hub)) & (~com_coin_gtr2.numperso_referencia.isin(index_hub))]
    #eliminamos los nodos que tienen degree > 50
    return com_coin_gtr2_bis

def sort_dict(dictionary,desc=True):
    
    '''Input: dictionary and flag to indicate the sorting order
       Output: list containing tuples corresponding to the items of the dictionary
       
       This function sorts a dictionary in ascending or descending order by the value of each key
       and returns a list of tuples containing (key,value)'''
    
    return sorted(dictionary.items(), key=lambda kv: kv[1],reverse=desc)

def avg_degree(graph):
    
    '''Input: grap object from NetworkX library
       Output: integer of the average of the degree of the nodes in the graph'''
    
    degree_sequence = [d for n, d in graph.degree()]
    return np.mean(np.array(degree_sequence))

def draw_component(graph,node,n_size):
    
    '''Input: graph from NetworkX library and integer corresponding to the node that want to be inspected
       Output: draws graph object of one component of the whole graph 
       
       This function is used to draw a graph of the component of the node of interest and this node
       will be highlighted in green'''
    
    for G in nx.connected_component_subgraphs(graph):
        if node in G.nodes():
            pos = nx.layout.spring_layout(G)

            node_color = []

            M = G.number_of_edges()
            edge_colors = range(2, M + 2)
            edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
            node_list=nx.nodes(G)
            node_colors=['green' if i==node else 'blue' for i in node_list]

            nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=n_size)
            edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors,edge_cmap=plt.cm.Blues, width=2)

            plt.show()
            
def katz_bon_cent(graph,alpha_pct,beta):
    
    '''Input: NetworkX graph object, percentage of the inverse of the highest eigenvalue used as alpha,
              beta parameter
       Output: dictionary containing the normalized K-B centrality for each node
       
       This fnction compues the highest eigenvalue and multiplies it by alpha_pct and this will
       be used as th alpha parameter of the K-B centrality'''
    
    eigenvalg_graph=nx.adjacency_spectrum(graph,weight='weight')
    max_eigenva_inv=1/abs(max(eigenvalg_graph))
    return nx.katz_centrality(graph,alpha=alpha_pct*max_eigenva_inv,beta=beta,normalized=True,weight='weight')

def get_community_cluster_id(generator_list,overlapping=False):
    
    '''Input: list(generator_object) obtained from a clustering function from NetworkX
       Output: dictionary with node id as key and community assigned as value
           
       This function returns a dictionary with each node and the id of the community
       to which it belongs to. Useful to draw graphs with each community with different
       colors'''  
    
    if overlapping == False:
        community_id=0
        nodes_cluster_id={}
        for node_set in list(generator_list):
            for node in list(node_set):
                nodes_cluster_id[node]=community_id
            community_id += 1
        
    else:
        nodes_cluster_id=[]
        community_id=0
        for node_set in list(generator_list):
            for node in list(node_set):
                nodes_cluster_id.append((node,community_id))
            community_id += 1
        
    return nodes_cluster_id

def measures_GV(graph,level):
    gir_new=list(nx.algorithms.community.centrality.girvan_newman(graph))
    coverage = nx.algorithms.community.quality.coverage(graph,gir_new[level])
    performance = nx.algorithms.community.quality.performance(graph,gir_new[level])
    
    sum_conductance = []
    
    for clust in list(np.arange(len(gir_new[level]))):
        S, T = nodes_rest(clust,level)
        sum_conductance.append(nx.algorithms.cuts.conductance(graph,S,T))
    
    return ('coverage', coverage), ('performance',performance),('conductance', sum_conductance)

def nodes_rest(i,level):
    S = list(gir_new[level][i])
    T=[]
    not_i = list(np.arange(len(gir_new[level])))
    not_i.remove(i)
    for j in not_i:
        T.extend(list(gir_new[level][j]))
    return S,T

def draw_community(graph,label_dict):
    '''Draws a graph using networkx draw tool and colors the node depending on the community they belong
    Input: nx graph object, dictionary with numperso as key and id of the community as value (integer)'''
    vals = [label_dict.get(node) for node in graph.nodes()]
    nx.draw_spring(graph, cmap = plt.get_cmap('flag'), node_color = vals,labels=label_dict, with_labels=True) 


def LoL_to_dict(list_of_lists):
    
    '''Transform list of lists to dictionary of clusters
    Example [[1,2,3,4],[2,8,7,9],[8,1,3]] --> {0: [1, 2, 3, 4], 1: [2, 8, 7, 9], 2: [8, 1, 3]}
    Input: list
    Output: dictionary'''
    
    return {clust: item[0:] for clust,item in enumerate(list_of_lists)}

def modification(a,b):
    '''Helper function that removes element in b if it not appears in a, otherwise pass'''
    for x in b:
        try:
            a.remove(x)
        except ValueError:
            pass
    return a

def get_first_nei(data,list_node):
    '''This function returns direct paths from a given node
    Input: dataframe with columns numperso_referencia and numpero_asociado defining the nodes and node/s in 
    list form
    Output: list of nodes reachable directly from the given node'''
    
    as_ref =  list(data.numperso_asociado.loc[(data['numperso_referencia'].isin(list_node))])
    as_aso =  list(data.numperso_referencia.loc[(data['numperso_asociado'].isin(list_node))])
    as_ref.extend(as_aso)
    return list(set(as_ref))

def get_N_nei(data,base_list_nodes,N):
    '''This function returns neighbours of distance N from a given node
    Input: dataframe with columns numperso_referencia and numpero_asociado defining the nodes
    Output: list of nodes reachable after N steps from the given node'''
    nodes_visited = list(base_list_nodes)
    nodes_tovisit = list(base_list_nodes)
    for i in list(range(1,N+1)):
        new_nodes = get_first_nei(data,nodes_tovisit)
        nodes_tovisit = new_nodes
        nodes_visited.extend(new_nodes)
    return list(set(nodes_visited))

def subset_df_for_node(data,node_of_interest,N_nei):
    '''This function subsets the original dataframe into one containing only the desired node and its N neighbours
    Input: dataframe with columns numperso_referencia and numpero_asociado defining the nodes
    Output: subset of the original dataframe'''
    list_of_nodes = get_N_nei(data,node_of_interest,N_nei)
    return data.loc[(data['numperso_referencia'].isin(list_of_nodes)) | (data['numperso_asociado'].isin(list_of_nodes))]

def construct_graph_from_df (df):
    '''This function takes a dataframe and transforms it to a graph object
    Input: a dataframe containing the following columns: numperso_ref, numperso_associado,
             coincidencias,dias_coincidencia, comercios_coincidencia, meses_coincidencia
       Output: a networkx graph object
       '''
    
    #obtenemos los nodos (unicos) de la lista de todos los elementos
    nodes = set(get_all_nodes(df))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    nx.set_node_attributes(graph, name='numperso',values=dict(zip(nodes,nodes)))
    #creamos una lista de 3-tuples con(nodo,nodo,weight) para añadir edges
    edge_weight=list(df.iloc[:,0:3].itertuples(index=False, name=None))
    graph.add_weighted_edges_from(edge_weight,weight='weight')
    
    return graph

def get_friendship_df (clusters,node_int): #all the friends of node_int
    '''This function returns a dataframe with three columns: a column with the node_int, a column with all the friends of node_int and a column with the community id of all the friends
    Input: list of lists containing the clusters, node for which the df will be built
    Output: pandas dataframe
    '''
    node_amigo=[]
    node_interes=[]
    lista_com_id=[]
    for com_id,cluster in enumerate(clusters):
        if node_int in cluster:
            cluster_temp=list(cluster)
            cluster_temp.remove(node_int)
            node_amigo.extend(cluster_temp)
            node_interes.extend(np.ones(len(cluster_temp))*node_int)
            lista_com_id.extend(np.ones(len(cluster_temp))*(com_id+1))
    friendship_df=pd.DataFrame(data={'numperso_referencia':node_interes,'numperso_amigo':node_amigo,'id_comunidad':lista_com_id},dtype=int)
    return friendship_df

def subgraph(self, nbunch):
    '''Creates a subgraph (shallow copy) from a bunch of nodes. Use: graph.subgraph(nodes_in_subgraph)
    Input:list of nodes that wants to be in the subgraph
    Output: network x graph object'''
    g = super().subgraph(nbunch)
    g.sequence_src = self.sequence_src

    return g
    
def get_all_n_neigh (graph,node_int,n_neigh):
    '''Returns a set with all the neighbours up to distance n_neigh from node_int
    Input: a graph object from network x, list of nodes to search neighbours up to distance n, distance to search
    Output: set of neighbour nodes up to distance n_neigh from node_int, set of nodes reached at the last iteration'''
    neigh_list=list(node_int)
    for i in range(n_neigh):
        nodes_to_visit = []
        for j in node_int:
            nodes_temp = list(graph[j])
            nodes_temp_2=[node for node in nodes_temp if node not in neigh_list]
            nodes_to_visit.extend(nodes_temp_2)
            neigh_list.extend(nodes_temp)
        node_int = list(nodes_to_visit)
    return set(neigh_list), set(node_int)
	
def sorting(x):
#Auxiliar function for sorting
	x.sort()
	return x

