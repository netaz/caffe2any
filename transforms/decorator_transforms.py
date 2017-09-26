"""This module contains decorator transforms, which "decorate" the
nodes with some attribute/annotation.

Martin Folwer wrote (https://martinfowler.com/bliki/Annotation.html):
"An annotation on a program element (commonly a class, method, or field) is a piece of meta-data added
to that program element which can be used to embellish that element with extra code."

For example, the ifm_size decorator transformer, annotates nodes
with the total size of a node's IFMs.
"""

import topology

def __get_ifm_size(node, tplgy):
    if not issubclass(type(node), topology.Op):
        return 0

    edges = tplgy.find_incoming_edges(node)
    if node.type in ['Convolution', 'Convolution_ReLU', 'Pooling', 'LRN', 'Softmax']:
        assert len(edges) == 1
        ifm = edges[0].src #edges[0].blob
        assert type(ifm)==topology.BLOB
        return ifm.size()
    elif node.type in ['Eltwise']:
        # Eltwise has two inputs of equal dimensions
        assert len(edges) == 2
        ifm = edges[0].src
        assert type(ifm)==topology.BLOB
        return ifm.size() * 2
    elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
        assert len(edges) == 1
        ifm = edges[0].src
        assert type(ifm)==topology.BLOB
        ifm_shape = ifm.shape
        return (ifm_shape[1] * ifm_shape[2] * ifm_shape[3])
    else:
        return 0

def __get_weight_size(node, tplgy):
    if not issubclass(type(node), topology.Op):
        return 0

    edges = tplgy.find_incoming_edges(node)
    if node.type in ['Convolution', 'Convolution_ReLU']:
        assert len(edges) == 1
        num_ifms = edges[0].src.shape[1]
        if node.type == 'Convolution_ReLU':
            node = node.node1
        return node.kernel_size * node.kernel_size * node.num_output * num_ifms
    elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
        assert len(edges) == 1
        ifm_size = node.get_attr('ifm_size')
        if node.type == 'InnerProduct_ReLU':
            #return (self.get_ifm_size(node, tplgy) * node.node1.num_output)
            return ifm_size * node.node1.num_output
        else:
            #return (self.get_ifm_size(node, tplgy) * node.num_output)
            return ifm_size * node.num_output
    else:
        return 0

def __get_bias_size(node):
    if node.type in ['Convolution', 'InnerProduct']:
        return node.num_output
    if node.type in ['Convolution_ReLU', 'InnerProduct_ReLU']:
        return node.node1.num_output
    else:
        return 0

def __get_ofm_shape(node, tplgy):
    ofm_shape = None
    edges = tplgy.find_outgoing_edges(node)
    if issubclass(type(node), topology.Op) and len(edges)==1:
        ''' Currently, only handle this simple, but prominent, case of one output edge'''
        assert type(edges[0].dst)==topology.BLOB
        ofm_shape = edges[0].dst.shape
    return ofm_shape

def __get_ofm_size(node, tplgy):
    ofm_shape = __get_ofm_shape(node, tplgy)
    if ofm_shape is None:
        return None
    return topology.BLOB.sizeof(ofm_shape)

def __add_size_annotations(node, tplgy, done_nodes):
    # If we've printed the contribution of this BLOB, then we skip it.
    # This will naturally filter out ReLU nodes, because they share their
    # BLOB with either Convolution or InnerProduct
    if node in done_nodes:
        #print("skipping BLOB: %s from edge %s" % (edge.blob, str(edge)))
        return
    # We don't want to see 'modifier' nodes (e.g. Concat) it in the CSV, since
    # they contribute no data transfer information
    #if edge.src_node.role == 'Modifier':
    #    return
    done_nodes.append(node)
    if type(node) is not topology.BLOB:
        node.set_attr('ifm_size', __get_ifm_size(node, tplgy))
        node.set_attr('ofm_size', __get_ofm_size(node, tplgy))
        node.set_attr('weights_size', __get_weight_size(node, tplgy))
        node.set_attr('bias_size', __get_bias_size(node))


def add_size_annotations(tplgy):
    done_blobs = []
    tplgy.traverse(lambda node: __add_size_annotations(node, tplgy, done_blobs))


# todo: move this to Topology (per Node class)
def __get_MACs(node, ofms_shape, tplgy):
    if not issubclass(type(node), topology.Op):
        return 0

    if node.type in ['Convolution', 'Convolution_ReLU']:
        edges = tplgy.find_incoming_edges(node)
        assert (len(edges) == 1)
        num_ifms = edges[0].src.shape[1]
        if node.type == 'Convolution_ReLU':
            node = node.node1
        return node.get_MACs(ofms_shape, num_ifms)
    elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
        return __get_weight_size(node, tplgy)
    else:
        return node.get_MACs()

def __add_macs_annotations(node, tplgy):
    if not issubclass(type(node), topology.Op):
        return
    macs = __get_MACs(node, __get_ofm_shape(node, tplgy), tplgy)
    node.set_attr('macs',macs)

    bw = node.get_attr('weights_size') if node.get_attr('weights_size') is not None else 0
    bw += node.get_attr('bias_size') if node.get_attr('bias_size') is not None else 0
    bw += node.get_attr('ifm_size') if node.get_attr('ifm_size') is not None else 0
    bw += node.get_attr('ofm_size') if node.get_attr('ofm_size') is not None else 0
    node.set_attr('bw', bw)
    node.set_attr('macs/bw', macs/bw if bw>0 else 0)

    # I'd like to add for Convolutions:
    # (OFM_H * OFM_W) / (K*K)
    # To show weights reuse - i.e. how many times a weight is used
    #print(edge.blob.shape[2] * edge.blob.shape[3] / ())

    # Also show input reuse
    # (OFM_H * OFM_W) / IFM_SIZE
    # All inputs are used

def add_macs_annotations(tplgy):
    done_blobs = []
    tplgy.traverse(lambda node: __add_macs_annotations(node, tplgy))

def __filter_edge(edges, edge, blob):
    if (edge.blob == blob and
        edge.dst_node is not None and
        edge.dst_node.type == 'Convolution_ReLU'): edges.append(edge)

def __horizontal_fusion(blob, tplgy):
    # Find all the edge that have their source in blob
    edges = []
    #tplgy.traverse(None, lambda edge: [edge.append(edge) if (edge.blob == blob)])
    #tplgy.traverse(None, map(lambda edge : edge, filter(lambda edge, blob : edge.blob == blob, )))
    tplgy.traverse(None, lambda edge: __filter_edge(edges, edge, blob))
    print(blob.name, len(edges))

def horizontal_fusion(tplgy):
    tplgy.traverse_blobs(lambda blob: __horizontal_fusion(blob, tplgy))
