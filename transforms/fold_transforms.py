def fold_pair(tplgy, node1_type, node2_type):
    """ This transform looks for the pattern [node1_type]==>[node2_type]
    and for all matched subgraphs, it removes node2.
    We call this "folding".
    """
    pairs = tplgy.find_subgraph_pair(node1_type, node2_type)
    if len(pairs) > 0:
        bn_nodes = [pair[1] for pair in pairs]
        tplgy.remove_nodes(bn_nodes)

def __concat_removal(node, nodes):
    if node.type == "Concat":
        nodes.append(node)

def concat_removal(tplgy):
    """ This transform removes Concat layers.
    Incoming edges to the Concat layer all produce into a
    single tensor.
    """
    nodes = []
    tplgy.traverse(lambda node: __concat_removal(node, nodes))
    for node in nodes:
        incoming_edges = tplgy.find_incoming_edges(node)
        outgoing_edges = tplgy.find_outgoing_edges(node)
        assert len(incoming_edges)>1 and len(outgoing_edges)>0
        parent_blob = outgoing_edges[0].blob
        for edge in incoming_edges:
            edge.blob.parent = parent_blob
