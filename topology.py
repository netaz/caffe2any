"""
http://www.bogotobogo.com/python/python_graph_data_structures.php
A set of classes to model a DL topology

"""
from collections import OrderedDict, Counter
DEBUG = False

def debug(str):
    if DEBUG: 
        print (str)

class Node:
    def __init__(self, name, type, layer):
        self.type = type
        self.name = name
        self.layer = layer

class BLOB:
    def __init__(self, name, tensor, producer):
        self.name = name
        self.tensor = tensor
        self.producer = producer

class Edge:
    def __init__(self, src_node, dst_node, blob):
        self.src_node = src_node
        self.dst_node = dst_node
        self.blob = blob

    def __str__(self):
        return 'Edge [' + self.blob.name +  ': ' + (self.src_node.name if self.src_node else 'None') + ' ==> ' + self.dst_node.name +  ']'

class Topology:
    def __init__(self):
        """
        Keep the the vertices ordered by insertion, so that we have 
        a starting point
        """
        self.nodes = OrderedDict()
        self.blobs = {}
        self.edges = []

    def add_node(self, name, type, layer):
        new_node = Node(name, type, layer)
        self.nodes[name] = new_node
        debug('created Node:' + name)
        return new_node

    def add_blob(self, name, tensor, producer):
        new_blob = BLOB(name, tensor, producer)
        self.blobs[name] = new_blob
        debug('created BLOB:' + name)
        return new_blob

    def add_edge(self, src, dst, blob):
        new_edge = Edge(src, dst, blob)
        self.edges.append(new_edge)

    def get_start_node(self):
        return self.nodes.values()[0]

    def find_blob(self, name):
        if name not in self.blobs:
            return None
        return self.blobs[name]

    def find_outgoing_edges(self, node):
        edges = []
        for edge in self.edges:
            #assert edge.src_node != None , str(edge) + " has no src "
            if (edge.src_node != None) and (edge.src_node.name == node.name):
                edges.append(edge)
        return edges

    # Output BLOBs have no consumer and therefore they don't appear on an edge.
    # We scan all blobs, checking which blobs don't appear on an edge
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

    def traverse(self, node_cb, edge_cb=None):
        pending = [ self.get_start_node() ]
        while len(pending)>0:
            node = pending.pop()
            if node_cb != None: node_cb(node)
            outgoing_edges = self.find_outgoing_edges(node)
            for edge in outgoing_edges:
                if edge_cb != None: edge_cb(edge)
                pending.append(edge.dst_node)

def populate(caffe_net):
    """
    Create and populate a Topology object, based on a given Caffe protobuf network object
    Todo: fix Input assignment
    """
    graph = Topology()

    # Input BLOBs
    for i in range(len(caffe_net.input)):
        graph.add_blob(caffe_net.input[i], caffe_net.input_shape[i].dim, None)

    for layer in caffe_net.layer:
        debug('evaluating layer: ' + layer.name)

        # filter away layers used only in training phase
        phase = 1 #caffe_pb2.Phase.TEST
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
        
        # Some prototxt files don't set the 'layer.exclude/layer.include' attributes.
        # Therefore, manually filter training-only layers
        if layer.type in ['Dropout']:
            continue

        new_node = graph.add_node(layer.name, layer.type, layer)

        # Iterate over BLOBs consumed by this layer and create edges to them
        for bottom_blob in layer.bottom:
            blob = graph.find_blob(bottom_blob)
            if blob == None:
                raise ValueError('could not find BLOB:' + bottom_blob)

            edge = graph.add_edge(src=blob.producer, dst=new_node, blob=blob)  

        # Add the BLOBs produced by this layer to the topology
        for top_blob in layer.top:
            graph.add_blob(top_blob, None, producer = new_node)

    return graph
