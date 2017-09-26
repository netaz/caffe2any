"""
http://www.bogotobogo.com/python/python_graph_data_structures.php
A set of classes to model a DL topology

Todo: add find_input_blobs
Todo: remove Node.layer
"""
from collections import OrderedDict, deque
import math
import copy
import logging
import sys, traceback
logger = None

def log():
    global logger
    if logger == None:
        logger = logging.getLogger(__name__)
    #print(__name__, logger)
    return logger

class Op:
    def __init__(self, name, type, role):
        self.name = name
        self.type = type
        self.role = role
        self.attributes = {}

    def __str__(self):
        return self.name + '(' + self.type + ')'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name

    def is_same(this, other):
        return True

    def get_MACs(self):
            return 0

    def get_attr(self, name):
        try:
            return self.attributes[name]
        except KeyError:
            return None

    def set_attr(self, name, val):
        self.attributes[name] = val

class PoolingNode(Op):
    def __init__(self, name, type, layer):
        Op.__init__(self, name, type, 'Producer')
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
        ifmw = ifm_shape[3]
        ofmw = math.ceil((ifmw - self.kernel_size + 2.0 * self.pad) / self.stride) + 1
        ofmh = math.ceil((ifmh - self.kernel_size + 2.0 * self.pad) / self.stride) + 1
        ofmh_noceil = (ifmh - self.kernel_size + 2.0 * self.pad) / self.stride + 1
        # The OFM is square, but I calculate the edges with different rounding strategies.
        # If the edges have different values, then we need to use the "ceiling"/"same" method
        self.ceiling = (ofmh_noceil != ofmh)
        ofm_shape[2] = int(ofmh)
        ofm_shape[3] = int(ofmw)
        log().debug(str(ifm_shape) + '--> ' + str(ofm_shape))
        return ofm_shape


class ConvolutionNode(Op):
    def __init__(self, name, type, layer):
        Op.__init__(self, name, type, 'Producer')
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
        ifmw = ifm_shape[3]
        ofmh = (ifmh - self.kernel_size + 2.0 * self.pad) / self.stride + 1
        ofmw = (ifmw - self.kernel_size + 2.0 * self.pad) / self.stride + 1
        ofm_shape[2] = int(ofmh)
        ofm_shape[3] = int(ofmw)
        log().debug(str(ifm_shape) + '--> ' + str(ofm_shape))
        return ofm_shape

    def get_MACs(self, ofms_shape, num_ifms):
        # macs = #OFMs*OFM_X*OFM_Y*#IFMs*K_X*K_Y
        num_ofms = ofms_shape[1]
        ofm_x = ofms_shape[2]
        ofm_y = ofms_shape[3]
        MACs = num_ofms * ofm_x * ofm_y * num_ifms * self.kernel_size * self.kernel_size
        return MACs


class PairNode(Op):
    ''' Container of two Operations '''
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        name = node1.name + "  ++  " + node2.name
        type = node1.type + '_' + node2.type
        #type = new_type if new_type is not None else node1.type + '_' + node2.type
        Op.__init__(self, name, type, node1.role)

    def transform_ifm(self, ifm_shape):
        return self.node1.transform_ifm(ifm_shape)

    def is_same(self, other):
        return self.node1.is_same(other.node1) and self.node2.is_same(other.node2)

class DeconvolutionNode(Op):
    def __init__(self, name, type, layer):
        Op.__init__(self, name, type, 'Producer')
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
        ifmw = ifm_shape[3]
        # s*(W-1) + k - 2*P
        ofmh = self.stride * (ifmh-1) + self.kernel_size - 2 * self.pad
        ofmw = self.stride * (ifmw - 1) + self.kernel_size - 2 * self.pad
        ofm_shape[2] = int(ofmh)
        ofm_shape[3] = int(ofmw)
        log().debug(str(ifm_shape) + '--> ' + str(ofm_shape))
        return ofm_shape


class InnerProductNode(Op):
    def __init__(self, name, type, layer):
        Op.__init__(self, name, type, 'Producer')
        self.num_output = layer.inner_product_param.num_output

    def transform_ifm(self, ifm_shape):
        ofm_shape = copy.deepcopy(ifm_shape)
        ofm_shape[3] = self.num_output  # ifm_shape[1] * ifm_shape[2] * ifm_shape[3]
        ofm_shape[1] = ofm_shape[2] = 1
        log().debug(str(ifm_shape) + '--> ' + str(ofm_shape))
        return ofm_shape


class LRNNode(Op):
    def __init__(self, name, type, layer):
        Op.__init__(self, name, type, 'Producer')
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

class ReshapeNode(Op):
    def __init__(self, name, type, layer):
        Op.__init__(self, name, type, 'Modifier')
        param = layer.reshape_param
        self.reshape_param = param.shape

    def transform_ifm(self, ifm_shape):
        ofm_shape = copy.deepcopy(ifm_shape)
        # Calculate the IFM size; to be used to calculate the inferred dimension
        ifm_size = ifm_shape[0] * ifm_shape[1] * ifm_shape[2] * ifm_shape[3]
        infer = -1 # the index of the inferred dimension
        for i in xrange(4):
            if self.reshape_param.dim[i] > 0:
                # Positive numbers are used directly, setting the corresponding dimension
                # of the output blob. In addition, two special values are accepted for any
                # of the target dimension values:
                ofm_shape[i] = self.reshape_param.dim[i]
                ifm_size /= ofm_shape[i]
            elif self.reshape_param.dim[i] == 0:
                # 0 means 'copy the respective dimension of the bottom layer'. That is,
                # if the bottom has 2 as its 1st dimension, the top will have 2 as its
                # 1st dimension as well, given dim: 0 as the 1st target dimension.
                ofm_shape[i] = ifm_shape[i]
                ifm_size /= ofm_shape[i]
            elif self.reshape_param.dim[i] == -1:
                # -1 stands for 'infer this from the other dimensions'. This
                # dimension is calculated to keep the overall element count the same as in
                # the bottom layer. At most one -1 can be used in a reshape operation.
                infer = i
            if infer>0:
                ofm_shape[infer] = ifm_size
        return ofm_shape

class EltwiseNode(Op):
    def __init__(self, name, type, layer):
        Op.__init__(self, name, type, 'Producer')
        self.operation = layer.eltwise_param.operation

class ConcatNode(Op):
    def __init__(self, name, type, layer):
        Op.__init__(self, name, type, 'Modifier')

def op_factory(name, type, layer, role):
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
    elif type == "Reshape":
        new_node = ReshapeNode(name, type, layer)
    elif type == "Eltwise":
        new_node = EltwiseNode(name, type, layer)
    elif type == "Concat":
        new_node = ConcatNode(name, type, layer)
    else:
        new_node = Op(name, type, role)
    return new_node


class BLOB:
    def __init__(self, name, shape, producer):
        self.name = name
        self.shape = shape
        self.producer = producer
         # A BLOB's parent is the physical BLOB to which this BLOB is mapped.
         # Another way to look at it: if a BLOB has a parent, it is actually a view
         # into another BLOB and does not occupy an independent physical space
        self.parent = None
        self.type = 'Tensor'

    def __str__(self):
        if self.shape != None:
            return 'BLOB [' + self.name + ': shape=' + str(self.shape) + ']'
        else:
            return 'BLOB [' + self.name + ': shape=None]'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name

    @staticmethod
    def sizeof(shape):
        if shape is None:
            return 0
        # shape[0] is the batch dimension, so don't count it
        return shape[1] * shape[2] * shape[3]

    def size(self):
        return self.sizeof(self.shape)

class Edge:
    '''    def __init__(self, src_node, dst_node, blob):
        self.src_node = src_node
        self.dst_node = dst_node
        self.blob = blob
        self.is_deleted = False
    '''
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.is_deleted = False

    @staticmethod
    def __print_vertex(vertex):
        if vertex is None:
            return 'None'
        if type(vertex) == BLOB:
            return '[' + vertex.name + ']'
        return vertex.name

    def __str__(self):
        desc = self.__print_vertex(self.src) + ' ==> ' + self.__print_vertex(self.dst)
        if self.is_deleted:
            desc += " IS DELETED!!"
        return desc

class SubGraph:
    ''' A sub-graph is a specific view of the graph.
    (loosely following: https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.graph_editor/module_subgraph)
    It has incoming edges, outgoing edges, and a set of nodes.
    '''
    def __init__(self, nodes=None, in_edges=None, out_edges=None):
        self.nodes = nodes
        self.in_edges = in_edges
        self.out_edges = out_edges

class Topology:
    def __init__(self):
        """
        Keep the the vertices ordered by insertion, so that we have
        a starting point
        """
        self.__ops = OrderedDict()
        self.__blobs = {}
        self.__edges = []
        self.__first_node = None

    def dump_edges(self):
        print('Dumping edges')
        print('-----------------------------------------')
        for edge in self.__edges:
            print(str(edge))

    def dump_blobs(self):
        print('Dumping blobs')
        print('-----------------------------------------')
        for name, blob in self.__blobs.items():
            print(name + ': ' + str(blob))

    def add_op2(self, new_op):
        assert issubclass(type(new_op), Op)
        self.__ops[new_op.name] = new_op
        if self.__first_node is None:
            self.__first_node = new_op
        log().debug('created Op:' + new_op.name)
        return new_op

    def add_op(self, name, type, layer, role):
        new_op = op_factory(name, type, layer, role)
        return self.add_op2(new_op)

    def add_ops(self, ops_to_add):
        for op in ops_to_add:
            assert issubclass(type(op), Op)
            self.__ops[op.name] = op
            if self.__first_node is None:
                self.__first_node = op
            log().debug('created Op:' + op.name)

    def del_nodes(self, nodes_to_del):
        for node in nodes_to_del:
            if type(node) == BLOB:
                del self.__blobs[node.name]
            else:
                self.__del_op(node)

    def __del_op(self, op):
        assert issubclass(type(op), Op)
        # remove all edges which enter/exit this node
        incoming_edges = self.find_incoming_edges(op)
        outgoing_edges = self.find_outgoing_edges(op)
        for edge in incoming_edges:
            self.del_edge(edge)
        for edge in outgoing_edges:
            self.del_edge(edge)
        # Fix the first_node pointer
        if self.__first_node == op:
            self.__first_node = None
        # Finally, delete the node and change its name (for debug)
        del self.__ops[op.name]
        op.name = op.name + "[DELETED]"

    def remove_ops(self, ops):
        [self.remove_op(op) for op in ops]

    # The difference between del_op and remove_op?
    # remove_node will del_node and also reconnect the edge around
    # the node that was removed
    def remove_op(self, op):
        '''
        +--------+     +++++++++++     +----+     +++++++++++     +-------+
        | pre_op |---->| pre_data|---->| op |---->|post_data|---->|post_op|
        +--------+     +++++++++++     +----+     +++++++++++     +-------+

        +--------+     +++++++++++                                +-------+
        | pre_op |---->| pre_data|------------------------------->|post_op|
        +--------+     +++++++++++                                +-------+
        '''
        assert issubclass(type(op), Op)
        incoming_edges = self.find_incoming_edges(op)
        outgoing_edges = self.find_outgoing_edges(op)
        for incoming_edge in incoming_edges:
            pre_data = incoming_edge.src
            for outgoing_edge in outgoing_edges:
                post_data_2_post_op_edges = self.find_outgoing_edges(outgoing_edge.dst)
                for post_data_2_post_op_edge in post_data_2_post_op_edges:
                    new_edge = self.add_edge(pre_data, post_data_2_post_op_edge.dst)
                    #print("adding edge", new_edge)
                    self.del_edge(post_data_2_post_op_edge)

                # Delete post_data
                #self.del_nodes([outgoing_edge.dst])
                # Delete edge
                self.del_edge(outgoing_edge)
        self.__del_op(op)

    def remove_op_by_type(self, type_to_remove):
        done = False
        while not done:
            done = True
            for op_name in list(self.__ops.keys()):
                op = self.__ops[op_name]
                if op.type != type_to_remove:
                    continue
                self.remove_op(op)
                done = False

    def add_blob(self, name, shape, producer):
        new_blob = BLOB(name, shape, producer)
        #new_blob = BLOB("b_"+name, shape, producer)
        assert name not in self.__blobs, 'BLOB ' + name + ' already exists'
        #self.__blobs[new_blob.name] = new_blob
        self.__blobs[name] = new_blob
        log().debug('created:' + str(new_blob))
        if self.__first_node is None:
            self.__first_node = new_blob
        return new_blob

    def add_blob2(self, new_blob):
        assert type(new_blob)==BLOB
        #new_blob.name = "b_" + new_blob.name
        assert new_blob.name not in self.__blobs, "{} already a BLOB".format(new_blob.name)
        self.__blobs[new_blob.name] = new_blob
        log().debug('created:' + str(new_blob))
        if self.__first_node is None:
            self.__first_node = new_blob
        return new_blob

    '''
    def add_edge(self, src, dst, blob):
        new_edge = Edge(src, dst, blob)
        self.__edges.append(new_edge)
        log().debug('created edge:' + str(new_edge))
        return new_edge
    '''
    def add_edge(self, src, dst):
        new_edge = Edge(src, dst)
        self.__edges.append(new_edge)
        log().debug('created edge:' + str(new_edge))
        return new_edge

    def del_edge(self, edge_to_del):
        for edge in self.__edges:
            if edge == edge_to_del:
                log().debug("deleted edge: " + str(edge))
                self.__edges.remove(edge)
                return

    def get_start_node(self):
        #return self.__nodes.values()[0]
        log().debug("Start node: " + str(self.__first_node))
        return self.__first_node

    def find_blob_by_name(self, name):
        if name not in self.__blobs:
            return None
        return self.__blobs[name]

    def find_op_by_name(self, name):
        return self.__ops[name]

    def find_edge(self, src, dst):
        for edge in self.__edges:
            if edge.src==src and edge.dst==dst:
                return edge
        return None

    def find_outgoing_edges(self, node):
        edges = []
        for edge in self.__edges:
            if ((edge.is_deleted is False) and
               (edge.src != None) and
               (edge.src.name == node.name) and
               type(edge.src) == type(node)):
                edges.append(edge)
        return edges

    def find_incoming_edges(self, node):
        edges = []
        for edge in self.__edges:
            if ((edge.is_deleted is False) and
                (edge.dst != None) and
                (edge.dst.name == node.name) and
                type(edge.dst) == type(node)):
                edges.append(edge)
        return edges

    # Output BLOBs have no consumer and therefore they don't appear on an edge.
    # We scan all blobs, checking which blobs don't appear on an edge
    # TODO: THIS HAS A BUG (Works only the first time!!!!)
    def find_output_blobs(self):
        blobs = []
        for blob in self.__blobs:
            blob_has_consumer = False
            for edge in self.__edges:
                if edge.blob.name == blob:
                    blob_has_consumer = True
                    continue
            if blob_has_consumer is False:
                blobs.append(blob)
        return blobs
    '''
    def find_subgraph_pair(self, node1_type, node2_type):
        pairs = []
        for node_name in self.__nodes:
            # Search for a matching pair of nodes, by node types
            node1 = self.__nodes[node_name]
            if node1.type != node1_type:
                continue
            outgoing_edges = self.find_outgoing_edges(node1)
            #assert len(outgoing_edges) == 1
            out_edge = outgoing_edges[0]
            if out_edge.dst is None: ## or out_edge.dst.type != node2_type:
                continue

            # Found a match
            node2 = out_edge.dst
            pairs.append([node1, node2])
        return pairs
    '''
    def find_type_pattern(self, node1_type, node2_type, node3_type):
        ''' This is a very specific pattern matcher which looks for nodes
        having the pattern [type1] ==> [type2] ==> [type3]
        '''
        #list(self.__nodes.keys())
        found = []
        nodes1 = [op for op in list(self.__ops.values()) if op.type == node1_type]
        for node1 in nodes1:
            nodes2 = [edge.dst for edge in self.find_outgoing_edges(node1) if edge.dst.type == node2_type]
            for node2 in nodes2:
                nodes3 = [edge.dst for edge in self.find_outgoing_edges(node2) if edge.dst.type == node3_type]
                for node3 in nodes3:
                    #print(node1, node2, node3)
                    found.append((node1, node2, node3))
        return found

    def merge_ops(self, op1_type, op2_type):
        ''' Merge two Ops together
        '''
        log().debug('[merge_ops] looking for nodes: {} ==> Tensor ==> {}'.format(op1_type, op2_type))
        found = self.find_type_pattern(op1_type, 'Tensor', op2_type)
        for (node1, node2, node3) in found:
            new_node = PairNode(copy.deepcopy(node1), copy.deepcopy(node3))
            node3_outgoing_edges = self.find_outgoing_edges(node3)
            node1_incoming_edges = self.find_incoming_edges(node1)
            for node1_incoming_edge in node1_incoming_edges:
                self.add_edge(node1_incoming_edge.src, new_node)
            for node3_out_edge in node3_outgoing_edges:
                self.add_edge(new_node, node3_out_edge.dst)

            assert node2.name in self.__blobs,  node2.name + ' not found'

            log().debug('[merge_ops] deleting nodes: {}, {}, {}'.format(node1, node2, node3))
            self.del_nodes([node1, node2, node3])
            self.add_ops([new_node])
        if len(found)==0:
            log().debug('[merge_ops] didn`t find candidates for types {}, {}'.format(op1_type, op2_type))
            #print('[merge_ops] didn`t find candidates for types {}, {}'.format(op1_type, op2_type))

    '''
    def traverse_blobs(self, blob_cb):
        done = []
        for blob in self.__blobs:
            if blob in done:
                continue
            blob_cb(self.__blobs[blob])
    '''

    def traverse(self, node_cb, edge_cb=None):
        ''' BFS (with modifications) traversal of the topology graph.
        Essentially this is a topological sort with callbacks.
        For each node (Operation or BLOB) the node_cb is invoked, if it is not None.
        For each Edge, the edge_cb is invoked, if it is not None.
        '''
        pending = deque([self.get_start_node()])    # The list of nodes waiting to be processed
        done = []                                   # The list of nodes we've already processed
        log().debug('BFS: Starting traversal with node %s' % self.get_start_node())
        while len(pending) > 0:
            node = pending.popleft()

            # This is a modification of BFS: we mandate that all incoming edges
            # have been processed before processing the node to ensure processing order satisfies data dependency
            incoming_edges = self.find_incoming_edges(node)
            log().debug('BFS: processing node: {} ({})'.format(node.name, len(incoming_edges)))
            all_in_edges_were_processed = True
            for edge in incoming_edges:
                if edge.src and (edge.src not in done): #and (type(edge.src) != BLOB):
                    all_in_edges_were_processed = False
                    log().debug("BFS: %s is waiting for %s" % (node.name, edge.src.name))
            if all_in_edges_were_processed is False:
                continue

            done.append(node)
            log().debug("BFS: done with %s" % node.name)
            if node_cb is not None:# and type(node) != BLOB:
                # TODO: this can probably be removed after adding the data-dependency constraint
                # Node callback can indicate failure, in which case we try again later
                cb_handled = node_cb(node)
                if cb_handled is False:
                    pending.append(node)
                    continue

            outgoing_edges = self.find_outgoing_edges(node)
            # Invoke the edge callback
            for edge in outgoing_edges:
                if edge_cb is not None:
                    exit = edge_cb(edge)
                    if exit:
                        log().debug("BFS: abrupt traversal exit requested by edge", str(edge))
                        return

            # Add new nodes to visit.  We do this as a separate step from the edge-callbacks,
            # because the edge-callbacks might alter the graph
            # outgoing_edges = self.find_outgoing_edges(node)
            outgoing_edges = self.find_outgoing_edges(node)
            for edge in outgoing_edges:
                if (edge.dst is not None) and (edge.dst not in done) and edge.dst not in pending:
                    pending.append(edge.dst)
                    log().debug('BFS: adding node: %s' % edge.dst.name)
                elif edge.dst is not None:
                    log().debug('BFS: ignoring  node: %s' % edge.dst.name)
        log().debug("BFS: traversal completed")
        #for line in traceback.format_stack():
        #    print(line.strip())
