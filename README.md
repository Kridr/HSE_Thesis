# HSE Thesis

This repository is devoted to my code part of my Master's Thesis at HSE where I research some initial transformation of bipartite graph of user-item interactions for increasing metrics of GNN-based RecSys algorithms.
[TBD link to this work]

It heavily depends on [Graph-Utils](https://github.com/sisinflab/Graph-Demo) repository. Their repository have some mistakes, but generally it is perfect. If you have troubles debugging their project, please write me in DM using Telegram in bio.

How to use this repository?
1) Download all files to some folder
2) The neighbor folder should be Graph-Demo repository (you should also mount Graph-Demo in Docker compose file). You should already download datasets.
3) Set up conda environment using `python 3.10` and provided `requirements.txt` file
4) Start using files!

Description of files
- python files with prefix `h_` are files to transform graph by different means. Some of the may take time to complete, so I recommend using `tmux`
  - for second hypothesis please run `hyp_2_bc_prep.py` beforehand    
- `graph_utils.py` is file with helpful functions that unify all the code in the repository
- `graph_analysis.ipynb` is devoted to basic analysis of graphs
- `post_analysis.ipynb` helps to analyze why my approaches failed
- `result_getter.ipynb` is needed to easily gain results from several runs of `Graph-Demo`
