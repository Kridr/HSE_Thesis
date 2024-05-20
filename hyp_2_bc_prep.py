import pandas as pd
import nx_parallel as nxp
import os
from typing import Dict

from graph_utils import (
    get_repaired_interactions,
    GOWALLA_DATASET_NAME, YELP_2018_DATASET_NAME, AMAZON_BOOK_DATASET_NAME,
)

BC_COLNAME = "bc"
NODE_COLNAME = "node"
BC_DIRNAME = "betweenness"


def calculate_betweennes_and_save(name, path, portion=0.1, seed=42):
    assert portion >= 0 and portion <= 1
    G, _, _ = get_repaired_interactions(name)

    H = nxp.ParallelGraph(G)

    sample_nodes_count = int(portion * len(G))
    node_bc_dict = nxp.betweenness_centrality(H, k=sample_nodes_count, seed=seed)

    (
        pd.DataFrame.from_dict(node_bc_dict, orient="index", columns=[BC_COLNAME])
        .reset_index(drop=False, names=NODE_COLNAME)
        .to_csv(path, index=False)
    )

    print(f"`{name}` betweenness is saved!")

portion = float(input("portion of data: "))
assert portion > 0 and portion <= 1
seed = input("random seed (press enter for skip): ")
if not seed:
    seed = 42

dataset_name = input("dataset name: ")
assert dataset_name in [GOWALLA_DATASET_NAME, YELP_2018_DATASET_NAME, AMAZON_BOOK_DATASET_NAME]

if not os.path.isdir(BC_DIRNAME):
    os.mkdir(BC_DIRNAME)

calculate_betweennes_and_save(
    dataset_name,
    f"{BC_DIRNAME}/{dataset_name}.csv"
)
