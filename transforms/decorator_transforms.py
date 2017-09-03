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
        assert (len(edges) == 1)
        ifm = edges[0].blob
        return ifm.size()
    elif node.type in ['Eltwise']:
        # Eltwise has two inputs of equal dimensions
        assert (len(edges) == 2)
        ifm = edges[0].blob
        return ifm.size() * 2
    elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
        if len(edges) != 1:
            print("node %s (%s) has an unexpected number of edges (%d edges)" % (node.name, node.get_type(), len(edges)))
        assert (len(edges) == 1)
        ifm_shape = edges[0].blob.shape
        return (ifm_shape[1] * ifm_shape[2] * ifm_shape[3])
    else:
        return ''

def __get_weight_size(node, tplgy):
    edges = tplgy.find_incoming_edges(node)
    if node.type in ['Convolution', 'Convolution_ReLU']:
        assert (len(edges) == 1)
        num_ifms = edges[0].blob.shape[1]
        if node.type == 'Convolution_ReLU':
            node = node.node1
        return node.kernel_size * node.kernel_size * node.num_output * num_ifms
    elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
        assert (len(edges) == 1)
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
    if edge.blob.shape and edge.src_node.role != "Modifier":
        ofm_size = edge.blob.size()
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
    edge.src_node.set_attr('macs', __get_MACs(edge.src_node, edge.blob.shape, tplgy))

def add_macs_annotations(tplgy):
    done_blobs = []
    tplgy.traverse(None, lambda edge: __add_macs_annotations(edge, tplgy))
