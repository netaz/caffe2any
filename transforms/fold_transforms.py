""" This trasnform looks for the pattern [node1_type]==>[node2_type]
and for all matched subgraphs, it removes node2.
We call this "folding".
"""
def fold_pair(tplgy, node1_type, node2_type):
    pairs = tplgy.find_subgraph_pair(node1_type, node2_type)
    if len(pairs) > 0:
        bn_nodes = [pair[1] for pair in pairs]
        tplgy.remove_nodes(bn_nodes)
