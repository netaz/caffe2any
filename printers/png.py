from __future__ import print_function
import caffe_pb2 as caffe

from globals import get_pooling_types_dict

"""
pydot is not supported under python 3 and pydot2 doesn't work properly.
pydotplus works nicely (pip install pydotplus)
"""
try:
    # Try to load pydotplus
    import pydotplus as pydot
except ImportError:
    import pydot

# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record',
                       'fillcolor': '#6495ED',
                       'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record',
                      'fillcolor': '#90EE90',
                      'style': 'filled'}
BLOB_STYLE = {'shape': 'box3d',
              'fillcolor': '#E0E0E0',
              'style': 'filled'}



# optional - caffe color scheme
def choose_style_by_layertype(layertype):
    layer_style = LAYER_STYLE_DEFAULT
    layer_style['fillcolor'] = '#6495ED'  # Default
    if layertype == 'Convolution' or layertype == 'Deconvolution':
        layer_style['fillcolor'] = '#FF5050'
    elif layertype == 'Pooling':
        layer_style['fillcolor'] = '#FF9900'
        #layer_style['shape'] = 'invtrapezium'
    elif layertype == 'InnerProduct':
        layer_style['fillcolor'] = '#CC33FF'

    if layertype == "Concat":
        layer_style = {'shape': 'box3d',
                       'fillcolor': 'gray',
                       'style': 'filled'}

    return layer_style


class PngPrinter(object):
    """The printer prints to PNG files"""

    def __init__(self, args, net):
        self.output_image_file = filename = args.infile + '.png'
        self.caffe_net = net
        self.pydot_nodes = {}
        self.pydot_edges = []

        print('Drawing net to %s' % self.output_image_file)

    def get_layer_label(self, layer, rankdir, verbose):
        """Define node label based on layer type.

        Parameters
        ----------
        layer : ?
        rankdir : {'LR', 'TB', 'BT'}
            Direction of graph layout.

        Returns
        -------
        string :
            A label for the current layer
        """

        if verbose==False:
            return layer.name

        if rankdir in ('TB', 'BT'):
            # If graph orientation is vertical, horizontal space is free and
            # vertical space is not; separate words with spaces
            separator = ' '
        else:
            # If graph orientation is horizontal, vertical space is free and
            # horizontal space is not; separate words with newlines
            separator = '\\n'

        if layer.type == 'Convolution' or layer.type == 'Deconvolution':
            # Outer double quotes needed or else colon characters don't parse
            # properly
            node_label = '"%s%s(%s)%skernel size: %d%sstride: %d%spad: %d"' %\
                         (layer.name,
                          separator,
                          layer.type,
                          separator,
                          layer.kernel_size,
                          separator,
                          layer.stride,
                          separator,
                          layer.pad)
        elif layer.type == 'Pooling':
            #pooling_types_dict = get_pooling_types_dict()
            node_label = self.print_pool(layer, separator)
            """
            node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' %\
                         (layer.name,
                          separator,
                          pooling_types_dict[layer.pool_type],
                          layer.type,
                          separator,
                          layer.kernel_size,
                          separator,
                          layer.stride,
                          separator,
                          layer.pad)
                          """
        else:
            node_label = '"%s%s(%s)"' % (layer.name, separator, layer.type)
        return node_label


    def print_pool(self, node, separator):
        #optional: compact/verbose
#        desc = pooling_type[node.pool_type] + ', k=' + str(node.kernel_size) + 'x' + str(
#            node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
#        if node.ceiling:
#            desc += ' ceiling'
#        return '"%s%s(%s)"' % (desc, separator, 'Pooling')
        pooling_types_dict = get_pooling_types_dict()
        node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' % \
                     (node.name,
                      separator,
                      pooling_types_dict[node.pool_type],
                      node.type,
                      separator,
                      node.kernel_size,
                      separator,
                      node.stride,
                      separator,
                      node.pad)
        return node_label

    def merge_nodes(self, src_node, dst_node):
        self.nodes_to_merge[src_node] = {dst_node}

    def add_pydot_node(self, node, tplgy, rankdir):
        #node_name = "%s_%s" % (node.name, node.type)
        #self.pydot_nodes[node.name] = pydot.Node(node.name,
                                    #    **NEURON_LAYER_STYLE)
        #optional
        layer_style = choose_style_by_layertype(node.type)
        # optional: verbosity
        node_label = self.get_layer_label(node, rankdir, verbose=True)
        self.pydot_nodes[node.name] = pydot.Node(node_label, **layer_style)

    def add_pydot_edge(self, edge, tplgy):
        if (edge.src_node is None) or (edge.dst_node is None):
            return
        #optional
        label_edges = True
        if label_edges and edge.blob != None:
            edge_label = str(edge.blob.shape) #get_edge_label(edge.src_node)
        else:
            edge_label = '""'

        src_name = edge.src_node.name
        self.pydot_edges.append({'src': src_name,
                                'dst': edge.dst_node.name,
                                'label': edge_label})

    def filter_relu_node(self, node, tplgy):
        if node.type == "ReLU":
            incoming_edges = tplgy.find_incoming_edges(node)
            outgoing_edges = tplgy.find_outgoing_edges(node)
            for incoming_edge in incoming_edges:
                src = incoming_edge.src_node
                src.name += "  +  " + node.name + "\n"
                for outgoing_edge in outgoing_edges:
                    tplgy.add_edge(src, outgoing_edge.dst_node, incoming_edge.blob)
                    #print("adding " + str(src) + "->" + str(outgoing_edge.dst_node))
                tplgy.del_edge(incoming_edge)

    def draw_net(self, caffe_net, rankdir, tplgy):
        pydot_graph = pydot.Dot(self.caffe_net.name if self.caffe_net.name else 'Net',
                                graph_type='digraph',
                                rankdir=rankdir)

        # optional: collapse ReLU nodes
        collapse_relu = True
        if collapse_relu:
            tplgy.traverse(lambda node: self.filter_relu_node(node, tplgy))

        tplgy.traverse(lambda node: self.add_pydot_node(node, tplgy, rankdir),
                       lambda edge: self.add_pydot_edge(edge, tplgy))


        # add the nodes and edges to the graph.
        for node in self.pydot_nodes.values():
            pydot_graph.add_node(node)

        for edge in self.pydot_edges:
            pydot_graph.add_edge(
                pydot.Edge(self.pydot_nodes[edge['src']],
                           self.pydot_nodes[edge['dst']],
                           label=edge['label']))

        print("number of nodes:", len(self.pydot_nodes))
        return pydot_graph.create_png()

    def print_bfs(self, tplgy):
        self.done_blobs = []

        rankdir = 'TB'  # {'LR', 'TB', 'BT'}

        with open(self.output_image_file, 'wb') as fid:
            fid.write(self.draw_net(self.caffe_net, rankdir, tplgy))
