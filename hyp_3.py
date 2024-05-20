import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm 
from typing import Dict

from graph_utils import (
    get_repaired_interactions, USER_COLNAME, ITEM_COLNAME,
    GOWALLA_DATASET_NAME, YELP_2018_DATASET_NAME, AMAZON_BOOK_DATASET_NAME,
    save_dataset,
    get_khop_neighbors_dict,
    duplicate_configs_for_transformed_datasets
)

def get_intersect_card(s1: set, s2: set):
    return len(s1 & s2)

def get_union_card(s1: set, s2: set):
    return len(s1 | s2)

def get_improved_jaccard_coefficient(s1: set, s2: set, norm: float = None):
    card_union = get_union_card(s1, s2)
    if norm is None:
        return get_intersect_card(s1, s2) / card_union
    else:
        return get_intersect_card(s1, s2) / card_union * (norm / card_union)

def common_hop_nodes_score(u_k_to_nbrs: Dict[int, set], v_k_to_nbrs: Dict[int, set]):
    """Improved Jacard Coefficient"""
    k = len(u_k_to_nbrs)

    s = get_improved_jaccard_coefficient(u_k_to_nbrs[1], v_k_to_nbrs[1])
    union_card_lv1 = get_union_card(u_k_to_nbrs[1], v_k_to_nbrs[1])
    for i in range(2, k + 1):
        s += get_improved_jaccard_coefficient(u_k_to_nbrs[i], v_k_to_nbrs[i], union_card_lv1 / i)

    return s

def get_colname_to_nodes_mapping(G: nx.Graph):
    user_nodes, item_nodes = nx.bipartite.sets(G)
    colname_to_nodes = {
        USER_COLNAME: user_nodes,
        ITEM_COLNAME: item_nodes,
    }

    return colname_to_nodes

def get_untop_primary_nodes(G: nx.Graph, q: float, primary_colname: str, colname_to_nodes: Dict[str, set] = None):
    if colname_to_nodes is None:
        colname_to_nodes = get_colname_to_nodes_mapping(G)
    
    primary_node_degrees = np.array([
        [primary_node, G.degree[primary_node]]
        for primary_node in colname_to_nodes[primary_colname]
    ])
    primary_degree_threshold = np.quantile(primary_node_degrees[:, 1], q)
    print(f"{primary_degree_threshold=}")
    top_primary_nodes = primary_node_degrees[primary_node_degrees[:, 1] <= primary_degree_threshold][:, 0]

    return top_primary_nodes

def get_secondary_node_to_khop_dict_mapping(G: nx.Graph, secondary_colname: str, k: int, colname_to_nodes: Dict[str, set] = None):
    print("Precompute khop dicts for secondary nodes...", flush=True)
    if colname_to_nodes is None:
        colname_to_nodes = get_colname_to_nodes_mapping(G)
    
    return {secondary_node: get_khop_neighbors_dict(G, secondary_node, k) for secondary_node in tqdm(colname_to_nodes[secondary_colname])}

def smart_choice(a, portion, rng):
    size = max(1, int(len(a) * portion))
    return rng.choice(list(a), size=size, replace=False)

def get_most_similar_and_dissimilar_secondary_nodes(
        G: nx.Graph,
        node: int,
        secondary_colname: str,
        colname_to_nodes: Dict[str, set],
        secondary_node_to_khop_dict_mapping: Dict[str, Dict[int, set]],
        chns_cache_dict: Dict[tuple, float],
        seed: int = 42,
        n_share: float = 1.0,
        nn_share: float = 1.0,
        ):
    drng = np.random.default_rng(seed)

    neighbors = set(G.neighbors(node))
    not_neighbors = colname_to_nodes[secondary_colname] - neighbors

    rand_neighbors = smart_choice(neighbors, n_share, drng)
    rand_not_neighbors = smart_choice(not_neighbors, nn_share, drng)

    most_similar_node = -1
    most_dissimilar_node = -1

    most_similar_value = 0
    most_dissimilar_value = np.inf

    for not_neighbor in rand_not_neighbors:
        sum_chns = 0
        for neighbor in rand_neighbors:
            cache_key = (min(neighbor, not_neighbor), max(neighbor, not_neighbor))

            if cache_key not in chns_cache_dict:
                n_k_to_nbrs = secondary_node_to_khop_dict_mapping[neighbor]
                nn_k_to_nbrs = secondary_node_to_khop_dict_mapping[not_neighbor]
                chns_cache_dict[cache_key] = common_hop_nodes_score(n_k_to_nbrs, nn_k_to_nbrs)
            sum_chns += chns_cache_dict[cache_key]

        if sum_chns > most_similar_value:
            most_similar_value = sum_chns
            most_similar_node = not_neighbor
        elif sum_chns < most_dissimilar_value:
            most_dissimilar_value = sum_chns
            most_dissimilar_node = not_neighbor
    
    return most_similar_node, most_dissimilar_node

def pre_exploration_chns(
        name,
        primary_colname,
        secondary_colname,
        q,
        return_name_only=False,
        ):
    new_name_soft = f"{name}--soft-chns--{primary_colname}--{q=}"
    new_name_hard = f"{name}--hard-chns--{primary_colname}--{q=}"
    if return_name_only:
        print("skip all calculation and return name", flush=True)
        return new_name_soft, new_name_hard

    G, df_i, item_index_shift = get_repaired_interactions(name)

    colname_to_nodes = get_colname_to_nodes_mapping(G)
    untop_primary_nodes = get_untop_primary_nodes(G, q, primary_colname, colname_to_nodes)
    secondary_node_to_khop_dict_mapping = get_secondary_node_to_khop_dict_mapping(G, secondary_colname, k=2, colname_to_nodes=colname_to_nodes)
    chns_cache_dict = {}

    soft_pairs = []
    hard_pairs = []
    print("Obtaining similar objects for top primary nodes...", flush=True)
    for untop_primary_node in tqdm(untop_primary_nodes):
        most_similar_secondary_node, most_dissimilar_secondary_node = get_most_similar_and_dissimilar_secondary_nodes(
            G,
            untop_primary_node,
            secondary_colname,
            colname_to_nodes,
            secondary_node_to_khop_dict_mapping,
            chns_cache_dict,
            nn_share=0.1,
            n_share=0.1,
        )

        soft_pairs.append((most_similar_secondary_node, untop_primary_node))
        hard_pairs.append((most_dissimilar_secondary_node, untop_primary_node))

    df_soft_bc = pd.DataFrame(soft_pairs, columns=[secondary_colname, primary_colname])
    df_soft_final = pd.concat((df_i, df_soft_bc), axis=0).sort_values([USER_COLNAME, ITEM_COLNAME])
    save_dataset(new_name_soft, name, df_soft_final, item_index_shift)

    df_hard_bc = pd.DataFrame(hard_pairs, columns=[secondary_colname, primary_colname])
    df_hard_final = pd.concat((df_i, df_hard_bc), axis=0).sort_values([USER_COLNAME, ITEM_COLNAME])
    save_dataset(new_name_hard, name, df_hard_final, item_index_shift)

    return new_name_soft, new_name_hard

q = float(input("q="))
primary_colname = input("primary colname: ")
secondary_colname = input("secondary colname: ")
assert (
    primary_colname in [USER_COLNAME, ITEM_COLNAME] and \
    secondary_colname in [USER_COLNAME, ITEM_COLNAME] and \
    primary_colname != secondary_colname 
)

dataset_name = input("dataset name: ")
assert dataset_name in [GOWALLA_DATASET_NAME, YELP_2018_DATASET_NAME, AMAZON_BOOK_DATASET_NAME]

new_dataset_names = pre_exploration_chns(
    name=dataset_name,
    primary_colname=primary_colname,
    secondary_colname=secondary_colname,
    q=q
)

duplicate_configs_for_transformed_datasets(
    dataset_name,
    new_dataset_names
)
