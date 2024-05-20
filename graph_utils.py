import pandas as pd
import networkx as nx

from typing import Tuple, List

import os
import shutil

USER_COLNAME = "user"
ITEM_COLNAME = "item"

GOWALLA_DATASET_NAME = "gowalla"
YELP_2018_DATASET_NAME = "yelp-2018"
AMAZON_BOOK_DATASET_NAME = "amazon-book"

BASE_FRAMEWORK_DIR = "../Graph-Demo"
BASE_DATASET_DIR = f"{BASE_FRAMEWORK_DIR}/data"
BASE_CONFIG_DIR = f"{BASE_FRAMEWORK_DIR}/config_files"

def get_path_to_data(name: str, ttype="train"):
    if ttype is None:
        return f"{BASE_DATASET_DIR}/{name}"
    else:
        return f"{BASE_DATASET_DIR}/{name}/{ttype}.tsv"
    
def get_repaired_interactions_df(name, ttype="train") -> Tuple[pd.DataFrame, int]:
    df = pd.read_csv(get_path_to_data(name, ttype), sep="\t")
    df.columns = [USER_COLNAME, ITEM_COLNAME]
    item_index_shift = df[USER_COLNAME].max() + 1

    df[ITEM_COLNAME] = df[ITEM_COLNAME] + item_index_shift
    return df, item_index_shift

def get_repaired_interactions(name: str, ttype="train") -> Tuple[nx.Graph, pd.DataFrame, int]:
    df, item_index_shift = get_repaired_interactions_df(name, ttype)
    G = nx.Graph()
    G.add_edges_from(df.values)
    assert nx.bipartite.is_bipartite(G)

    return G, df, item_index_shift

def save_dataset(new_name: str, old_name: str, df: pd.DataFrame, item_index_shift=None):
    if item_index_shift is not None:
        df[ITEM_COLNAME] = df[ITEM_COLNAME] - item_index_shift
    if os.path.exists(get_path_to_data(new_name, None)):
        shutil.rmtree(get_path_to_data(new_name, None))
    os.mkdir(get_path_to_data(new_name, None))

    df.to_csv(
        get_path_to_data(new_name, "train"),
        index=False, header=None, sep="\t"
    )
    shutil.copy2(
        get_path_to_data(old_name, "test"),
        get_path_to_data(new_name, "test"),
    )


def get_khop_neighbors_dict(G: nx.Graph, start: int, k: int):
    nbrs = set([start])
    all_nbrs = nbrs.copy()
    k_to_nbrs = {}
    for i in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G.neighbors(n))) - all_nbrs
        k_to_nbrs[i + 1] = nbrs
        all_nbrs.update(nbrs)
    return k_to_nbrs

def df_table_to_latex(df: pd.DataFrame, index=True, caption=""):
    return (
        "\\begin{table}[H]\n\centering\n" +
        df.to_latex(index=index) +
        f"\caption{{{caption}}}\n" +
        "\\end{table}"
    )

def duplicate_configs_for_transformed_datasets(
        dataset_name: str,
        new_dataset_names: List[str],
        models: List[str] = None
        ):
    if models is None:
        models = ["ngcf", "lightgcn", "dgcf", "sgl", "ultragcn", "gfcf"]
    for new_dataset_name in new_dataset_names:
        for model in models:
            base_config_path = f"{BASE_CONFIG_DIR}/{model}_{dataset_name}.yml"
            new_config_path = f"{BASE_CONFIG_DIR}/{model}_{new_dataset_name}.yml"

            with open(base_config_path, "r") as f:
                file_str = f.read()

            file_str = file_str.replace(dataset_name, new_dataset_name)

            with open(new_config_path, "w") as f:
                f.write(file_str)