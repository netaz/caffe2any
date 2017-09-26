def fold_pair(tplgy, op1_type, op2_type):
    """ This transform looks for the pattern [op1_type] ==> [tensor] ==> [op2_type]
    and for all matched subgraphs, it removes op2.
    We call this "folding".
    """
    subgraphs = tplgy.find_type_pattern(op1_type, 'Tensor', op2_type)
    if len(subgraphs)>0:
        op2_list = [subgraph[2] for subgraph in subgraphs]
        tplgy.remove_ops(op2_list)
    #else:
    #    print("fold_pair: Did not find {}==>Tensor==>{}".format(op1_type, op2_type))

def __concat_removal(node, nodes):
    if node.type == "Concat":
        nodes.append(node)

def concat_removal(tplgy):
    """ This transform removes Concat layers.
    Incoming edges to the Concat layer all produce into a
    single tensor.  Therefore, the tensors on the incoming edges
    are called "virtual tensors", or "tensor views".
    See Torch Tensor view documentation: 
    http://jucor.github.io/torch-doc-template/tensor.html#toc_28
    """
    concat_nodes = []
    tplgy.traverse(lambda node: __concat_removal(node, concat_nodes))
    for node in concat_nodes:
        incoming_edges = tplgy.find_incoming_edges(node)
        outgoing_edges = tplgy.find_outgoing_edges(node)
        assert len(incoming_edges)>1 and len(outgoing_edges)==1
        for edge in incoming_edges:
            tplgy.add_edge(edge.src, outgoing_edges[0].dst)
            edge.src.parent = outgoing_edges[0].dst
            tplgy.del_edge(edge)
        tplgy.del_nodes([node])
        tplgy.del_edge(outgoing_edges[0])
