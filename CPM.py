from collections import defaultdict 
import networkx as nx
 
class CPM:
    def __init__(self,graph,k_clique):
        self.graph=graph
        self.k_clique=k_clique
        
    def get_community_cluster_id(self,generator_list,overlapping=False): 
    
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

    def get_adjacent_cliques(self,clique, membership_dict): 
        adjacent_cliques = set() 
        for n in clique: 
            for adj_clique in membership_dict[n]: 
                if clique != adj_clique: 
                    adjacent_cliques.add(adj_clique) 
        return adjacent_cliques 
    
    def get_percolated_cliques(self,G, k): 
        perc_graph = nx.Graph() 
        cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= k] 
        perc_graph.add_nodes_from(cliques) 

    # First index which nodes are in which cliques 
        membership_dict = defaultdict(list) 
        for clique in cliques: 
            for node in clique: 
                 membership_dict[node].append(clique) 

        # For each clique, see which adjacent cliques percolate 
        for clique in cliques: 
            for adj_clique in self.get_adjacent_cliques(clique, membership_dict): 
                if len(clique.intersection(adj_clique)) >= (k - 1): 
                    perc_graph.add_edge(clique, adj_clique) 

        # Connected components of clique graph with perc edges 
         # are the percolated cliques 
        for component in nx.connected_components(perc_graph): 
            yield(frozenset.union(*component)) 
    

    def deploy_cpm(self):
        cpm=self.get_percolated_cliques(self.graph,self.k_clique)
        communities=[]

        for i in cpm:
            communities.append(list(i))

        #cmp_list=self.get_community_cluster_id(communities,overlapping=True)

        label_dict={}
        #cmp_list.sort()
 
        return communities

