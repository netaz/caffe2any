def fold_pair(tplgy, op1_type, op2_type):
    """ This transform looks for the pattern [op1_type] ==> [tensor] ==> [op2_type]
    and for all matched subgraphs, it removes op2.
    We call this "folding".
    """
    subgraph = tplgy.find_type_pattern(op1_type, 'Tensor', op2_type)
    if len(subgraph) > 0:
        assert False, 'this is broken'
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
    concat_nodes = []
    tplgy.traverse(lambda node: __concat_removal(node, concat_nodes))
    for node in concat_nodes:
        incoming_edges = tplgy.find_incoming_edges(node)
        outgoing_edges = tplgy.find_outgoing_edges(node)
        assert len(incoming_edges)>1 and len(outgoing_edges)==1
        for edge in incoming_edges:
            tplgy.add_edge(edge.src, outgoing_edges[0].dst)
            tplgy.del_edge(edge)
        tplgy.del_op(node)
        tplgy.del_edge(outgoing_edges[0])
        #parent_blob = outgoing_edges[0].blob
        #for edge in incoming_edges:
        #    edge.blob.parent = parent_blob
