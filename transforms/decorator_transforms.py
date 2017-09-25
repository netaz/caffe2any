"""This module contains decorator transforms, which "decorate" the
nodes with some attribute/annotation.

Martin Folwer wrote (https://martinfowler.com/bliki/Annotation.html):
"An annotation on a program element (commonly a class, method, or field) is a piece of meta-data added
to that program element which can be used to embellish that element with extra code."

For example, the ifm_size decorator transformer, annotates nodes
with the total size of a node's IFMs.
"""

def __get_ifm_size(node, tplgy):
    edges = tplgy.find_incoming_edges(node)
    if node.type in ['Convolution', 'Convolution_ReLU', 'Pooling', 'LRN', 'Softmax']:
        assert len(edges) == 1
        ifm = edges[0].src #edges[0].blob
        assert type(ifm)==BLOB
        return ifm.size()
    elif node.type in ['Eltwise']:
        # Eltwise has two inputs of equal dimensions
        assert len(edges) == 2
        ifm = edges[0].src #edges[0].blob
        assert type(ifm)==BLOB
        return ifm.size() * 2
    elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
        assert len(edges) == 1
        ifm = edges[0].src
        assert type(ifm)==BLOB
        ifm_shape = ifm.shape #edges[0].blob.shape
        return (ifm_shape[1] * ifm_shape[2] * ifm_shape[3])
    else:
        return 0

def __get_weight_size(node, tplgy):
    edges = tplgy.find_incoming_edges(node)
    if node.type in ['Convolution', 'Convolution_ReLU']:
        assert len(edges) == 1
        num_ifms = edges[0].src.shape[1] #edges[0].blob.shape[1]
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

def __get_ofm_size(edge):
    ofm_size = 0
    #if edge.blob.shape and edge.src_node.role != "Modifier":
    #    ofm_size = edge.blob.size()
    if edge.src.shape: #and edge.src_node.role != "Modifier":
        ofm_size = edge.src.size()
    return ofm_size


def __add_size_annotations(edge, tplgy, done_blobs):
    # If we've printed the contribution of this BLOB, then we skip it.
    # This will naturally filter out ReLU nodes, because they share their
    # BLOB with either Convolution or InnerProduct
    if edge.blob in done_blobs:
        #print("skipping BLOB: %s from edge %s" % (edge.blob, str(edge)))
        return
    # We don't want to see 'modifier' nodes (e.g. Concat) it in the CSV, since
    # they contribute no data transfer information
    if edge.src_node.role == 'Modifier':
        return
    done_blobs.append(edge.blob)
    edge.src_node.set_attr('ifm_size', __get_ifm_size(edge.src_node, tplgy))
    edge.src_node.set_attr('weights_size', __get_weight_size(edge.src_node, tplgy))
    edge.src_node.set_attr('ofm_size', __get_ofm_size(edge))
    edge.src_node.set_attr('bias_size', __get_bias_size(edge.src_node))


def add_size_annotations(tplgy):
    done_blobs = []
    tplgy.traverse(None, lambda edge: __add_size_annotations(edge, tplgy, done_blobs))


# todo: move this to Topology (per Node class)
def __get_MACs(node, ofms_descriptor, tplgy):
    if node.type in ['Convolution', 'Convolution_ReLU']:
        edges = tplgy.find_incoming_edges(node)
        assert (len(edges) == 1)
        num_ifms = edges[0].blob.shape[1]
        if node.type == 'Convolution_ReLU':
            node = node.node1
        return node.get_MACs(ofms_descriptor, num_ifms)
    elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
        return __get_weight_size(node, tplgy)
    else:
        return node.get_MACs()#(ofms_descriptor, num_ifms)

def __add_macs_annotations(edge, tplgy):
    macs = __get_MACs(edge.src_node, edge.blob.shape, tplgy)
    edge.src_node.set_attr('macs',macs)

    node = edge.src_node
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
    tplgy.traverse(None, lambda edge: __add_macs_annotations(edge, tplgy))

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
