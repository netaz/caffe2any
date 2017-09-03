"""This transformer updates the sizes of the BLOBs based on the operation
of each of the nodes.

"""
import copy

def __update_blobs_sizes(tplgy, node):
    #print('updating node:' + node.name)
    in_edges = tplgy.find_incoming_edges(node)
    out_edges = tplgy.find_outgoing_edges(node)
    if node.type == 'Convolution':
        assert len(in_edges)==1 and len(out_edges)==1, (node.name, len(in_edges), len(out_edges), str(out_edges[0]))
        if in_edges[0].blob.shape != None:
            out_edges[0].blob.shape = node.transform_ifm(in_edges[0].blob.shape)
    elif node.type == 'InnerProduct':
        assert len(in_edges)==1 and len(out_edges)==1, node.name
        if in_edges[0].blob.shape != None:
            out_edges[0].blob.shape = node.transform_ifm(in_edges[0].blob.shape)
    elif node.type == 'ReLU':
        assert len(in_edges)==1, node.name
        if in_edges[0].blob.shape != None:
            for edge in out_edges:
                edge.blob.shape = in_edges[0].blob.shape
    elif node.type == 'Pooling':
        assert len(in_edges)==1 and len(out_edges)>0, node.name
        if in_edges[0].blob.shape != None:
            for edge in out_edges:
                edge.blob.shape = node.transform_ifm(in_edges[0].blob.shape)
    elif node.type == 'Concat':
        assert len(in_edges)>0 and len(out_edges)>0, node.name
        for in_edge in in_edges:
            if in_edge.blob.shape != None:
                representative_in_edge_shape = in_edge.blob.shape
            else:
                return False
        if representative_in_edge_shape != None:
            for out_edge in out_edges:
                out_edge.blob.shape = copy.deepcopy(representative_in_edge_shape)

            # concat on the channel dimension
            ch_dim_size = 0
            for in_edge in in_edges:
                ch_dim_size += in_edge.blob.shape[1]

            for out_edge in out_edges:
                out_edge.blob.shape[1] = ch_dim_size
    elif node.type == 'Deconvolution':
        assert len(in_edges) == 1 and len(out_edges) == 1, (node.name, len(in_edges), len(out_edges), str(out_edges[0]))
        if in_edges[0].blob.shape != None:
            out_edges[0].blob.shape = node.transform_ifm(in_edges[0].blob.shape)
    elif node.type == 'ROIPooling':
        assert len(in_edges)==2 and len(out_edges)==1, node.name
        #print(in_edges[0].blob.shape)
        if in_edges[0].blob.shape != None:
            out_edges[0].blob.shape = in_edges[0].blob.shape
    elif node.type == 'Eltwise':
        assert len(in_edges)==2 and len(out_edges)==1, node.name
        #if in_edges[0].blob.shape != None:
        # NETA: second edge was not evaluated yet
        out_edges[0].blob.shape = in_edges[0].blob.shape
    elif node.type == 'Python':
        pass # Don't know how to handle this
    elif node.type == 'Crop':
        pass # Don't know how to handle this
    elif node.type == 'Input':
        assert len(out_edges)==1, node.name
    elif node.type == 'Dropout':
        assert len(in_edges)==1, node.name
        if in_edges[0].blob.shape != None:
            out_edges[0].blob.shape = in_edges[0].blob.shape
    elif node.type == 'Reshape':
        assert len(in_edges) == 1 and len(out_edges) == 1, node.name
        out_edges[0].blob.shape = node.transform_ifm(in_edges[0].blob.shape)
    else:
        #assert len(in_edges)==1 and len(out_edges)==1, node.name
        if in_edges[0].blob.shape != None:
            out_edges[0].blob.shape = in_edges[0].blob.shape

    return True

def update_blobs_sizes(tplgy, node):
    tplgy.traverse(lambda node: __update_blobs_sizes(tplgy, node))
