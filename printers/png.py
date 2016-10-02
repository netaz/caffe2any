from __future__ import print_function
from globals import get_pooling_types_dict, lrn_type

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
    'collapse_relu': True,
    'remove_dropout': True,
    'verbose': True,
    'node_label': 'custom', # {'custom', 'caffe', 'minimal'}
    'label_edges': True,
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
                      'style': 'filled'}
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

        if layer.type == 'Convolution' or layer.type == 'Deconvolution':
            node_label = self.print_conv(layer, separator, options['node_label'])
        elif layer.type == 'Pooling':
            node_label = self.print_pool(layer, separator, options['node_label'])
        elif layer.type == 'LRN':
            node_label = self.print_lrn(layer, separator, options['node_label'])
        elif layer.type == 'Reshape':
            node_label = self.print_reshape(layer, separator, options['node_label'])
        else:
            node_label = '"%s%s(%s)"' % (layer.name, separator, layer.type)
        # print (node_label)
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
        if node.type != "Dropout":
            return
        incoming_edges = tplgy.find_incoming_edges(node)
        outgoing_edges = tplgy.find_outgoing_edges(node)
        for incoming_edge in incoming_edges:
            src = incoming_edge.src_node
            for outgoing_edge in outgoing_edges:
                tplgy.add_edge(src, outgoing_edge.dst_node, incoming_edge.blob)
                # print("adding " + str(src) + "->" + str(outgoing_edge.dst_node))
            tplgy.del_edge(incoming_edge)

    @staticmethod
    def collapse_relu_node(node, tplgy):
        if node.type != "ReLU":
            return
        incoming_edges = tplgy.find_incoming_edges(node)
        outgoing_edges = tplgy.find_outgoing_edges(node)
        for incoming_edge in incoming_edges:
            src = incoming_edge.src_node
            src.name += "  ==>  " + node.name
            for outgoing_edge in outgoing_edges:
                tplgy.add_edge(src, outgoing_edge.dst_node, incoming_edge.blob)
                #print("adding " + str(src) + "->" + str(outgoing_edge.dst_node))
            tplgy.del_edge(incoming_edge)

    def draw_net(self, caffe_net, rankdir, tplgy):
        pydot_graph = pydot.Dot(self.caffe_net.name if self.caffe_net.name else 'Net',
                                graph_type='digraph',
                                rankdir=rankdir)

        # optional: collapse ReLU nodes
        if options['collapse_relu']:
            tplgy.traverse(lambda node: self.collapse_relu_node(node, tplgy))
        if options['remove_dropout']:
            tplgy.traverse(lambda node: self.remove_dropout_node(node, tplgy))

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
