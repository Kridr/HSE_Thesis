from graph_utils import (
    get_repaired_interactions_df, USER_COLNAME, ITEM_COLNAME,
    GOWALLA_DATASET_NAME, YELP_2018_DATASET_NAME, AMAZON_BOOK_DATASET_NAME,
    save_dataset,
    duplicate_configs_for_transformed_datasets
)

def prune_neighborhoods(
        dataset_name: str,
        primary_colname: str,
        secondary_colname: str,
        q: float = 0.95,
        return_name_only: bool = False
        ):
    new_dataset_name = f"{dataset_name}--{primary_colname}--{q=}"
    if return_name_only:
        return new_dataset_name

    df_edges, item_index_shift = get_repaired_interactions_df(dataset_name)

    secondary_popularity_col = secondary_colname + "_popularity"
    secondary_popularity_rank_col = secondary_colname + "_popularity_rank"

    df_edges[secondary_popularity_col] = df_edges.groupby(secondary_colname).transform("count")
    df_edges[secondary_popularity_rank_col] = (
        df_edges.groupby(primary_colname)[secondary_popularity_col].rank(method="max")
    )

    threshold = df_edges.groupby(primary_colname)[secondary_colname].agg("count").quantile(q)
    df_edges = df_edges[df_edges[secondary_popularity_rank_col] < threshold]
    df_edges = df_edges.reset_index(drop=True)[[USER_COLNAME, ITEM_COLNAME]]

    save_dataset(new_dataset_name, dataset_name, df_edges, item_index_shift)
    return new_dataset_name

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

new_dataset_name = prune_neighborhoods(
    name=dataset_name,
    primary_colname=primary_colname,
    secondary_colname=secondary_colname,
    q=q
)

duplicate_configs_for_transformed_datasets(
    dataset_name,
    [new_dataset_name]
)
