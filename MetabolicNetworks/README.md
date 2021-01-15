In this directory you will find the 5610 directed metabolic networks derived from various KEGG entries from our paper.

They are divided into two .zip files. You will find an adjacency matrix per species in .npy format. All matrices have the same shape: we took the union of all nodes for all species, so the matrices are of size length of this list x length of the list.

The name of the nodes can be found in *all_nodes.txt*.

Adjacency matrices were generated with Python 2.
