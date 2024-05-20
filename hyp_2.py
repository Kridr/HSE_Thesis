import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import nx_parallel as nxp
from tqdm import tqdm 
import os
from typing import Dict

from graph_utils import (
    get_repaired_interactions, USER_COLNAME, ITEM_COLNAME,
    GOWALLA_DATASET_NAME, YELP_2018_DATASET_NAME, AMAZON_BOOK_DATASET_NAME,
    save_dataset,
    duplicate_configs_for_transformed_datasets
)

from hyp_2_bc_prep import NODE_COLNAME, BC_COLNAME, BC_DIRNAME

def connect_top_betweenness(
        dataset_name: str,
        bc_path: str,
        primary_colname: str,
        secondary_colname: str,
        q: float,
        return_name_only: bool = False,
        ):
    new_name_closest = f"{dataset_name}--c-bc--{primary_colname}--{q=}"
    new_name_farthest = f"{dataset_name}--f-bc--{primary_colname}--{q=}"
    if return_name_only:
        print("skip all calculation and return name")
        return new_name_closest, new_name_farthest
    G, df_i, item_index_shift = get_repaired_interactions(dataset_name)
    df_bc = pd.read_csv(bc_path).rename(columns={NODE_COLNAME: primary_colname})


    df_joined = df_i.merge(df_bc, on=primary_colname, how="inner")
    bc_th = (
        df_joined
        .drop_duplicates([primary_colname, BC_COLNAME])
        [BC_COLNAME]
        .quantile(q=q)
    )

    top_bc_primary_nodes = set(df_joined[df_joined[BC_COLNAME] >= bc_th][primary_colname].values)
    nodes_nc_w_tb = []
    for node in df_i[secondary_colname].unique():
        if len(set(G.neighbors(node)) & top_bc_primary_nodes) == 0:
            nodes_nc_w_tb.append(node)
    print(f"count of top bc nodes = {len(top_bc_primary_nodes)}")
    print(f"count of nodes not connected with top bc nodes = {len(nodes_nc_w_tb)}")

    closest_pairs = []
    farthest_pairs = []
    for nc_node in tqdm(nodes_nc_w_tb):
        closest_node = -1
        closest_distance = np.inf
        farthest_node = -1
        farthest_distance = 0
        for top_bc_primary_node in top_bc_primary_nodes:
            distance = nx.shortest_path_length(G, nc_node, top_bc_primary_node)
            if distance < closest_distance:
                closest_distance = distance
                closest_node = top_bc_primary_node
            if distance > farthest_distance:
                    farthest_distance = distance
                    farthest_node = top_bc_primary_node
        closest_pairs.append((nc_node, closest_node))
        farthest_pairs.append((nc_node, farthest_node))

    df_closest_bc = pd.DataFrame(closest_pairs, columns=[secondary_colname, primary_colname])
    df_closest_final = pd.concat((df_i, df_closest_bc), axis=0).sort_values([USER_COLNAME, ITEM_COLNAME])
    save_dataset(new_name_closest, dataset_name, df_closest_final, item_index_shift)

    df_farthest_bc = pd.DataFrame(farthest_pairs, columns=[secondary_colname, primary_colname])
    df_farhest_final = pd.concat((df_i, df_farthest_bc), axis=0).sort_values([USER_COLNAME, ITEM_COLNAME])
    save_dataset(new_name_farthest, dataset_name, df_farhest_final, item_index_shift)

    return new_name_closest, new_name_farthest

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

new_dataset_names = connect_top_betweenness(
    dataset_name=dataset_name,
    bc_path=f"{BC_DIRNAME}/{dataset_name}.csv",
    primary_colname=primary_colname,
    secondary_colname=secondary_colname,
    q=q
)

duplicate_configs_for_transformed_datasets(
    dataset_name,
    new_dataset_names
)