"""This transformer updates the sizes of the BLOBs based on the operation
of each of the nodes.

"""
import copy
import topology
import logging
logger = None

def log():
    global logger
    if logger == None:
        logger = logging.getLogger('transformers')
    return logger

def __update_blobs_sizes(tplgy, node):
    if type(node) == topology.BLOB:
        return True
    log().debug('updating blob size of node:' + node.name)
    in_edges = tplgy.find_incoming_edges(node)
    out_edges = tplgy.find_outgoing_edges(node)
    if (node.type == 'Convolution') or (node.type == 'Convolution_ReLU'):
        assert len(in_edges)==1 and len(out_edges)==1, (node.name, len(in_edges), len(out_edges), str(out_edges[0]))
        assert in_edges[0].src.shape != None, "src is " + str(in_edges[0].src)
        if in_edges[0].src.shape != None:#in_edges[0].blob.shape != None:
            out_edges[0].dst.shape = node.transform_ifm(in_edges[0].src.shape)
            log().debug('shape of {} is now {}'.format(str(out_edges[0].dst), str(out_edges[0].dst.shape)))
    elif (node.type == 'InnerProduct') or (node.type == 'InnerProduct_ReLU'):
        assert len(in_edges)==1 and len(out_edges)==1, node.name
        if in_edges[0].src.shape != None:
            out_edges[0].dst.shape = node.transform_ifm(in_edges[0].src.shape)
            log().debug('shape of {} is now {}'.format(str(out_edges[0].dst), str(out_edges[0].dst.shape)))
    elif node.type == 'ReLU':
        assert len(in_edges)==1, node.name
        if in_edges[0].src.shape != None:
            for edge in out_edges:
                edge.dst.shape = in_edges[0].src.shape
                log().debug('shape of {} is now {}'.format(str(edge.dst), str(edge.dst.shape)))
    elif node.type == 'Pooling':
        assert len(in_edges)==1 and len(out_edges)>0, node.name
        if in_edges[0].src.shape != None:
            for edge in out_edges:
                edge.dst.shape = node.transform_ifm(in_edges[0].src.shape)
                log().debug('shape of {} is now {}'.format(str(edge.dst), str(edge.dst.shape)))
    elif node.type == 'Concat':
        assert len(in_edges)>0 and len(out_edges)>0, node.name
        ''' Concat input edges on the channel dimension
        (BEWARE: THIS IS A NON-GENERIC ASSUMPTION WHICH WILL FAIL ONE DAY!) '''

        ch_dim_size = 0
        for in_edge in in_edges:
            assert in_edge.src is not None
            assert type(in_edge.src)==topology.BLOB is not None
            #print(in_edge.src)
            assert in_edge.src.shape is not None
            ch_dim_size += in_edge.src.shape[1]

        for out_edge in out_edges:
            out_edge.dst.shape = copy.deepcopy(in_edge.src.shape)
            out_edge.dst.shape[1] = ch_dim_size
            log().debug('shape of {} is now {}'.format(str(out_edge.src), str(out_edge.src.shape)))
    elif node.type == 'Deconvolution':
        assert len(in_edges) == 1 and len(out_edges) == 1, (node.name, len(in_edges), len(out_edges), str(out_edges[0]))
        if in_edges[0].src.shape != None:
            out_edges[0].dst.shape = node.transform_ifm(in_edges[0].src.shape)
            log().debug('shape of {} is now {}'.format(str(out_edges[0].dst), str(out_edges[0].dst.shape)))
    elif node.type == 'ROIPooling':
        assert len(in_edges)==2 and len(out_edges)==1, node.name
        #log().debug(in_edges[0].blob.shape)
        if in_edges[0].src.shape != None:
            out_edges[0].dst.shape = in_edges[0].src.shape
            log().debug('shape of {} is now {}'.format(str(out_edges[0].dst), str(out_edges[0].dst.shape)))
    elif node.type == 'Eltwise':
        assert len(in_edges)==2 and len(out_edges)==1, node.name
        #if in_edges[0].blob.shape != None:
        # NETA: second edge was not evaluated yet
        out_edges[0].dst.shape = in_edges[0].src.shape
        log().debug('shape of {} is now {}'.format(str(out_edges[0].dst), str(out_edges[0].dst.shape)))
    elif node.type == 'Python':
        pass # Don't know how to handle this
    elif node.type == 'Crop':
        pass # Don't know how to handle this
    elif node.type == 'LRN':
        #assert False, node.name
        assert len(in_edges)==1, node.name
        if in_edges[0].src.shape != None:
            out_edges[0].dst.shape = in_edges[0].src.shape
            log().debug('shape of {} is now {}'.format(str(out_edges[0].dst), str(out_edges[0].dst.shape)))
        else:
            assert False, in_edges[0].src
    elif node.type == 'Reshape':
        assert len(in_edges) == 1 and len(out_edges) == 1, node.name
        out_edges[0].dst.shape = node.transform_ifm(in_edges[0].src.shape)
        log().debug('shape of {} is now {}'.format(str(out_edges[0].dst), str(out_edges[0].dst.shape)))
    else:
        if (in_edges[0].src.shape != None) and (out_edges[0].src != None):
            out_edges[0].dst.shape = in_edges[0].src.shape
            log().debug('shape of {} is now {}'.format(str(out_edges[0].dst), str(out_edges[0].dst.shape)))

    return True

def update_blobs_sizes(tplgy):
    tplgy.traverse(lambda node: __update_blobs_sizes(tplgy, node))
