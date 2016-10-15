from __future__ import print_function
from globals import get_pooling_types_dict, lrn_type
import copy
"""
pydot is not supported under python 3 and pydot2 doesn't work properly.
pydotplus works nicely (pip install pydotplus)
"""
try:
    # Try to load pydotplus
    import pydotplus as pydot
except ImportError:
    import pydot

# options
options = {
    # Merges Convolution and ReLU nodes. This makes for a more compact and readable graph.
    'merge_conv_relu': True,
    # For Test/Inference networks, Dropout nodes are not interesting and can be removed for readability
    'remove_dropout': True,
    'verbose': True,
    # The node label refers to the text that is inside each node in the graph
    'node_label': 'custom', # {'custom', 'caffe', 'minimal'}
    # Annotate the edges with the BLOB sizes
    'label_edges': True,
    # Graph drawing direction: left-right, top-bottom, bottom-top
    'rankdir': 'LR',  # {'LR', 'TB', 'BT'}
}

# Themes
CAFFE_THEME = {
    'layer_default': {'shape': 'record',
                      'fillcolor': '#6495ED',
                      'style': 'filled'},

    'Convolution':  {'shape': 'record',
                     'fillcolor': '#FF5050',
                     'style': 'filled'},

    'Pooling':      {'shape': 'record',
                     'fillcolor': '#FF9900',
                     'style': 'filled'},

    'InnerProduct': {'shape': 'record',
                     'fillcolor': '#CC33FF',
                     'style': 'filled'},
}

SOFT_THEME = {
    'layer_default': {'shape': 'record',
                      'fillcolor': '#6495ED',
                      'style': 'rounded, filled'},

    'Convolution':   {'shape': 'record',
                      'fillcolor': '#FF5050',
                      'style': 'rounded, filled'},

    'Pooling':       {'shape': 'record',
                      'fillcolor': '#FF9900',
                      'style': 'rounded, filled'},

    'InnerProduct':  {'shape': 'record',
                      'fillcolor': '#CC33FF',
                      'style': 'rounded, filled'},

    'Concat':        {'shape': 'box3d',
                      'fillcolor': 'gray',
                      'style': 'filled'},

    'Softmax':        {'shape': 'record',
                      'fillcolor': 'yellow',
                      'style': 'rounded, filled'},
}

# theme = CAFFE_THEME
theme = SOFT_THEME


def choose_style_by_layertype(layertype):
    try:
        layer_style = theme[layertype]
    except:
        layer_style = theme['layer_default']
        layer_style['fillcolor'] = '#6495ED'  # Default

    return layer_style


class PngPrinter(object):
    """The printer prints to PNG files"""

    def __init__(self, args, net):
        self.output_image_file = args.infile + '.png'
        self.caffe_net = net
        self.pydot_nodes = {}
        self.pydot_edges = []

        print('Drawing net to %s' % self.output_image_file)

    def get_layer_label(self, layer, rankdir, format):
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

        if format == "minimal":
            return layer.name

        if rankdir in ('TB', 'BT'):
            # If graph orientation is vertical, horizontal space is free and
            # vertical space is not; separate words with spaces
            separator = ' '
        else:
            # If graph orientation is horizontal, vertical space is free and
            # horizontal space is not; separate words with newlines
            separator = '\\n'

        layers = {
            'Convolution': self.print_conv,
            'Deconvolution': self.print_conv,
            'Pooling': self.print_pool,
            'LRN': self.print_lrn,
            'Reshape': self.print_reshape,
        }
        printer = layers.get(layer.type, None)
        if printer is None:
            node_label = '"%s%s(%s)"' % (layer.name, separator, layer.type)
        else:
            node_label = printer(layer, separator, options['node_label'])
        return node_label

    def print_conv(self, node, separator, format):
        if format == 'caffe':
            # Outer double quotes needed or else colon characters don't parse
            # properly
            node_label = '"%s%s(%s)%skernel size: %d%sstride: %d%spad: %d"' %\
                         (node.name, separator,
                          node.type, separator,
                          node.kernel_size, separator,
                          node.stride, separator, node.pad)
        elif format == 'custom':
            node_label = node.name + separator + 'k=' + str(node.kernel_size) + 'x' + str(
                node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
            node_label = '"%s%s(%s)"' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    def print_pool(self, node, separator, format):
        pooling_type = get_pooling_types_dict()
        if format=='caffe':
            node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' % \
                         (node.name, separator, pooling_type[node.pool_type],
                          node.type, separator,
                          node.kernel_size, separator,
                          node.stride, separator, node.pad)
        elif format == 'custom':
            node_label = node.name + separator + pooling_type[node.pool_type] + ', k=' + str(node.kernel_size) + 'x' + str(
                         node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
            if node.ceiling:
                node_label += ' ceiling'
            node_label = '"%s%s(%s)"' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    def print_lrn(self, node, separator, format):
        if format == 'caffe':
            node_label = node.name
        elif format == 'custom':
            node_label = node.name + separator + lrn_type[node.norm_region] + \
                          ' size=' + str(node.local_size) + ' alpha=' + str(
                          node.alpha) + ' beta=' + str(node.beta)
            node_label = '"%s%s(%s)"' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    def print_reshape(self, node, separator, format):
        if format == 'caffe':
            node_label = node.name
        elif format == 'custom':
            node_label = '%s%s[%s,%s,%s,%s]' % \
                         (node.name, separator, node.reshape_param.dim[0], node.reshape_param.dim[1], node.reshape_param.dim[2], node.reshape_param.dim[3])
            node_label = '"%s%s(%s)"' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    """
    def print_eltwise(selfself, node):
        op_lookup = get_eltwise_op_dict()
        return 'Eltwise,' + op_lookup[node.operation]

        if format == 'caffe':
            node_label = node.name
        elif format == 'custom':
            node_label = '%s%s[%s,%s,%s,%s]' % \
                         (node.name, separator, node.reshape_param.dim[0], node.reshape_param.dim[1],
                          node.reshape_param.dim[2], node.reshape_param.dim[3])
            node_label = '"%s%s(%s)"' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label
    """
    def add_pydot_node(self, node, tplgy, rankdir):
        # node_name = "%s_%s" % (node.name, node.type)
        # self.pydot_nodes[node.name] = pydot.Node(node.name,
                                    #    **NEURON_LAYER_STYLE)
        layer_style = choose_style_by_layertype(node.type)
        node_label = self.get_layer_label(node, rankdir, options['node_label'])
        self.pydot_nodes[node.name] = pydot.Node(node_label, **layer_style)

    def add_pydot_edge(self, edge, tplgy):
        if (edge.src_node is None) or (edge.dst_node is None):
            return

        if options['label_edges'] and edge.blob != None:
            edge_label = str(edge.blob.shape) #get_edge_label(edge.src_node)
        else:
            edge_label = '""'

        src_name = edge.src_node.name
        self.pydot_edges.append({'src': src_name,
                                'dst': edge.dst_node.name,
                                'label': edge_label})

    @staticmethod
    def remove_dropout_node(node, tplgy):
        tplgy.del_node_by_type(node, "Dropout")

    @staticmethod
    # Search and replace
    def merge_conv_relu_nodes(tplgy):
        done = False
        while not done:
            done = True
            for node_name in tplgy.nodes:
                node = tplgy.nodes[node_name]
                if node.type == 'Convolution':
                    outgoing_edges = tplgy.find_outgoing_edges(node)
                    assert len(outgoing_edges) == 1
                    out_edge = outgoing_edges[0]

                    if out_edge.dst_node.type == 'ReLU':
                        # Found a match
                        new_node = copy.deepcopy(node)
                        new_node.name += "  ++  " + node.name

                        relu_node = out_edge.dst_node
                        relu_outgoing_edges = tplgy.find_outgoing_edges(relu_node)
                        assert len(relu_outgoing_edges) == 1
                        relu_out_edge = relu_outgoing_edges[0]

                        conv_incoming_edges = tplgy.find_incoming_edges(node)
                        assert len(conv_incoming_edges) == 1
                        conv_incoming_edge = conv_incoming_edges[0]

                        tplgy.add_edge(conv_incoming_edge.src_node, new_node, copy.deepcopy(conv_incoming_edge.blob))
                        tplgy.add_edge(new_node, relu_out_edge.dst_node, copy.deepcopy(relu_out_edge.blob))

                        tplgy.del_edge(conv_incoming_edge)
                        tplgy.del_edge(relu_out_edge)
                        if tplgy.get_start_node == node:
                            # about to remove the first node
                            new_node
                        tplgy.del_nodes([node, relu_node])
                        tplgy.add_nodes([new_node])
                        done = False
                        break

    def draw_net(self, caffe_net, rankdir, tplgy):
        pydot_graph = pydot.Dot(self.caffe_net.name if self.caffe_net.name else 'Net',
                                graph_type='digraph',
                                rankdir=rankdir)

        # optional: collapse ReLU nodes
        if options['merge_conv_relu']:
            self.merge_conv_relu_nodes(tplgy)

        tplgy.dump_edges()
        if options['remove_dropout']:
            tplgy.traverse(lambda node: self.remove_dropout_node(node, tplgy))
        tplgy.dump_edges()
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
        with open(self.output_image_file, 'wb') as fid:
            fid.write(self.draw_net(self.caffe_net, options['rankdir'], tplgy))
