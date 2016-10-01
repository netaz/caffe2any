"""
http://www.bogotobogo.com/python/python_graph_data_structures.php
A set of classes to model a DL topology

Todo: add find_input_blobs
Todo: remove Node.layer
"""
from collections import OrderedDict, Counter, deque
import math
import copy

DEBUG = False
DEBUG_TRANSFORM = False


def debug(str):
    if DEBUG:
        print (str)


def debug_tr(str):
    if DEBUG_TRANSFORM:
        print (str)


class Node:
    def __init__(self, name, type, role):
        self.name = name
        self.type = type
        self.role = role

    def __str__(self):
        return self.name + '(' + self.type + ')'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name

    def is_same(this, other):
        return True


class PoolingNode(Node):
    def __init__(self, name, type, layer):
        Node.__init__(self, name, type, 'Producer')
        param = layer.pooling_param
        self.kernel_size = param.kernel_size
        self.stride = param.stride
        self.pad = param.pad
        self.pool_type = param.pool

    def is_same(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.kernel_size, self.stride, self.pad, self.pool_type) == (
        other.kernel_size, other.stride, other.pad, other.pool_type)

    def transform_ifm(self, ifm_shape):
        ofm_shape = copy.deepcopy(ifm_shape)
        ifmh = ifm_shape[2]
        ofmw = (ifmh - self.kernel_size + 2.0 * self.pad) / self.stride + 1
        ofmh = math.ceil((ifmh - self.kernel_size + 2.0 * self.pad) / self.stride) + 1
        # The OFM is square, but I calculate the edges with different rounding strategies.
        # If the edges have different values, then we need to use the "ceiling"/"same" method
        self.ceiling = (ofmw != ofmh)
        ofm_shape[2] = int(ofmh)
        ofm_shape[3] = int(ofmh)
        debug_tr(str(ifm_shape) + '--> ' + str(ofm_shape))
        return ofm_shape


class ConvolutionNode(Node):
    def __init__(self, name, type, layer):
        Node.__init__(self, name, type, 'Producer')
        param = layer.convolution_param
        self.kernel_size = param.kernel_size
        self.stride = param.stride
        self.pad = param.pad
        self.num_output = param.num_output

    def is_same(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.kernel_size, self.stride, self.pad) == (other.kernel_size, other.stride, other.pad)

    def transform_ifm(self, ifm_shape):
        ofm_shape = copy.deepcopy(ifm_shape)
        ofm_shape[1] = self.num_output
        ifmh = ifm_shape[2]
        ofmh = (ifmh - self.kernel_size + 2.0 * self.pad) / self.stride + 1
        ofm_shape[2] = int(ofmh)
        ofm_shape[3] = int(ofmh)
        debug_tr(str(ifm_shape) + '--> ' + str(ofm_shape))
        return ofm_shape


class DeconvolutionNode(Node):
    def __init__(self, name, type, layer):
        Node.__init__(self, name, type, 'Producer')
        param = layer.convolution_param
        self.kernel_size = param.kernel_size
        self.stride = param.stride
        self.pad = param.pad
        self.num_output = param.num_output

    def is_same(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.kernel_size, self.stride, self.pad) == (other.kernel_size, other.stride, other.pad)


class InnerProductNode(Node):
    def __init__(self, name, type, layer):
        Node.__init__(self, name, type, 'Producer')
        self.num_output = layer.inner_product_param.num_output

    def transform_ifm(self, ifm_shape):
        ofm_shape = copy.deepcopy(ifm_shape)
        ofm_shape[3] = self.num_output  # ifm_shape[1] * ifm_shape[2] * ifm_shape[3]
        ofm_shape[1] = ofm_shape[2] = 1
        debug_tr(str(ifm_shape) + '--> ' + str(ofm_shape))
        return ofm_shape


class LRNNode(Node):
    def __init__(self, name, type, layer):
        Node.__init__(self, name, type, 'Producer')
        param = layer.lrn_param
        self.norm_region = layer.lrn_param.norm_region
        self.local_size = layer.lrn_param.local_size
        self.alpha = layer.lrn_param.alpha  # default = 1.
        self.beta = layer.lrn_param.beta  # default = 0.75

    def is_same(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.norm_region, self.alpha, self.beta, self.local_size) == (
        other.norm_region, other.alpha, other.beta, other.local_size)


def node_factory(name, type, layer, role):
    if type == "Pooling":
        new_node = PoolingNode(name, type, layer)
    elif type == "Convolution":
        new_node = ConvolutionNode(name, type, layer)
    elif type == "InnerProduct":
        new_node = InnerProductNode(name, type, layer)
    elif type == "LRN":
        new_node = LRNNode(name, type, layer)
    elif type == "Deconvolution":
        new_node = DeconvolutionNode(name, type, layer)
    else:
        new_node = Node(name, type, role)
    return new_node


class BLOB:
    def __init__(self, name, shape, producer):
        self.name = name
        self.shape = shape
        self.producer = producer

    def __str__(self):
        if self.shape != None:
            return 'BLOB [' + self.name + ': shape=' + str(self.shape) + ']'
        else:
            return 'BLOB [' + self.name + ': shape=None]'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name

    def size(self):
        if self.shape is None:
            return 0
        # shape[0] is the batch dimension, so don't count it
        return self.shape[1] * self.shape[2] * self.shape[3]


class Edge:
    def __init__(self, src_node, dst_node, blob):
        self.src_node = src_node
        self.dst_node = dst_node
        self.blob = blob

    def __str__(self):
        return ((self.src_node.name if self.src_node else 'None') + ' ==> ' +
                str(self.blob) + ' ==> ' +
                (self.dst_node.name if self.dst_node else 'None') + ']')


class Topology:
    def __init__(self):
        """
        Keep the the vertices ordered by insertion, so that we have 
        a starting point
        """
        self.nodes = OrderedDict()
        self.blobs = {}
        self.edges = []

    def add_node(self, name, type, layer, role):
        new_node = node_factory(name, type, layer, role)
        self.nodes[name] = new_node
        debug('created Node:' + name)
        return new_node

    def add_blob(self, name, shape, producer):
        new_blob = BLOB(name, shape, producer)
        self.blobs[name] = new_blob
        debug('created:' + str(new_blob))
        return new_blob

    def add_edge(self, src, dst, blob):
        new_edge = Edge(src, dst, blob)
        self.edges.append(new_edge)
        debug('created:' + str(new_edge))
        return new_edge

    def del_edge(self, edge_to_del):
        for edge in self.edges:
            if edge == edge_to_del:
                self.edges.remove(edge)
                return

    def nodes_count(self):
        node_cnt = []
        self.traverse(lambda node: node_cnt.append(node.type))
        return Counter(node_cnt)

    def get_start_node(self):
        return self.nodes.values()[0]

    def find_blob_by_name(self, name):
        if name not in self.blobs:
            return None
        return self.blobs[name]

    def find_outgoing_edges(self, node):
        edges = []
        for edge in self.edges:
            if (edge.src_node != None) and (edge.src_node.name == node.name):
                edges.append(edge)
        return edges

    def find_incoming_edges(self, node):
        edges = []
        for edge in self.edges:
            if (edge.dst_node != None) and (edge.dst_node.name == node.name):
                edges.append(edge)
        return edges

    # Output BLOBs have no consumer and therefore they don't appear on an edge.
    # We scan all blobs, checking which blobs don't appear on an edge
    # TODO: THIS HAS A BUG (Works only the first time!!!!)
    def find_output_blobs(self):
        blobs = []
        for blob in self.blobs:
            blob_has_consumer = False
            for edge in self.edges:
                if edge.blob.name == blob:
                    blob_has_consumer = True
                    continue
            if blob_has_consumer == False:
                blobs.append(blob)
        return blobs

    def traverse_blobs(self, blob_cb):
        done = []
        for blob in self.blobs:
            if blob in done:
                continue
            blob_cb(self.blobs[blob])

    def traverse(self, node_cb, edge_cb=None):
        """
        BFS traversal of the topology graph
        """
        pending = deque([self.get_start_node()])
        done = []
        while len(pending) > 0:
            node = pending.popleft()
            done.append(node)

            if node_cb != None:
                # Node callback can indicate failure, in which case we try again later
                cb_handled = node_cb(node)
                if cb_handled == False:
                    pending.append(node)
                    continue

            outgoing_edges = self.find_outgoing_edges(node)
            for edge in outgoing_edges:
                if edge_cb != None: edge_cb(edge)
                if (edge.dst_node != None) and (edge.dst_node not in done):
                    pending.append(edge.dst_node)


# parse_caffe_net
def parse_caffe_net(caffe_net):
    """
    Create and populate a Topology object, based on a given Caffe protobuf network object
    Todo: fix Input assignment
    """
    graph = Topology()

    # Input BLOBs
    for i in range(len(caffe_net.input)):
        if len(caffe_net.input_shape) > 0:
            graph.add_blob(caffe_net.input[i], caffe_net.input_shape[i].dim, None)
        elif len(caffe_net.input_dim) > 0:
            # graph.add_blob(caffe_net.input[i], caffe_net.input_dim[i], None)
            graph.add_blob(caffe_net.input[i], caffe_net.input_dim, None)

    if len(caffe_net.layer) < 1:
        exit("Something went wrong - the parser can't find any layers in the network")

    for layer in caffe_net.layer:
        debug('evaluating layer: ' + layer.name)

        # filter away layers used only in training phase
        phase = 1  # caffe_pb2.Phase.TEST
        if phase is not None:
            included = False
            if len(layer.include) == 0:
                included = True
            if len(layer.include) > 0 and len(layer.exclude) > 0:
                raise ValueError('layer ' + layer.name + ' has both include '
                                                         'and exclude specified.')
            for layer_phase in layer.include:
                included = included or layer_phase.phase == phase
            for layer_phase in layer.exclude:
                included = included and not layer_phase.phase == phase
            if not included:
                continue

        node_role = 'Producer'
        if (len(layer.bottom) == 1 and len(layer.top) == 1 and
                    layer.bottom[0] == layer.top[0]):
            # We have an in-place neuron layer.
            node_role = 'Modifier'

        new_node = graph.add_node(layer.name, layer.type, layer, node_role)

        # Iterate over BLOBs consumed by this layer and create edges to them
        for caffe_bottom_blob in layer.bottom:
            blob = graph.find_blob_by_name(caffe_bottom_blob)
            if blob == None:
                raise ValueError(layer.name + ' - could not find BLOB:' + caffe_bottom_blob)

            edge = graph.add_edge(src=blob.producer, dst=new_node, blob=blob)

            # Add the BLOBs produced by this layer to the topology
        for caffe_top_blob in layer.top:
            if new_node.type == "Input":
                graph.add_blob(caffe_top_blob, layer.input_param.shape[0].dim, producer=new_node)
            else:
                graph.add_blob(caffe_top_blob, None, producer=new_node)

    # Add fake output edges
    output_blobs = graph.find_output_blobs()
    for blob_name in output_blobs:
        blob = graph.find_blob_by_name(blob_name)
        graph.add_edge(src=blob.producer, dst=None, blob=blob)

    return graph
