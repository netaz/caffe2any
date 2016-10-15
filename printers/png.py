from __future__ import print_function
from globals import get_pooling_types_dict, lrn_type
import copy
import topology
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
    # Merges Convolution, ReLU, and Pooling nodes.
    'merge_conv_relu_pooling': True,
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

    'Convolution_ReLU':{'shape': 'record',
                      'fillcolor': 'coral3',
                      'style': 'rounded, filled'},

    'Convolution_ReLU_Pooling':
                      {'shape': 'record',
                       'fillcolor': 'darkslategray',
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

    @staticmethod
    def get_node_label(node, separator, format):
        """Define node label based on layer type.
        """

        if format == "minimal":
            return node.name

        layers = {
            'Convolution': PngPrinter.print_conv,
            'Deconvolution': PngPrinter.print_conv,
            'Pooling': PngPrinter.print_pool,
            'LRN': PngPrinter.print_lrn,
            'Reshape': PngPrinter.print_reshape,
            #'PairContainer': PngPrinter.print_container,
            'Convolution_ReLU': PngPrinter.print_container,
            'Convolution_ReLU_Pooling': PngPrinter.print_container,
        }

        printer = layers.get(node.type, PngPrinter.print_default)
        node_label = printer(node, separator, options['node_label'])
        return node_label

    @staticmethod
    def print_default(node, separator, format):
        node_label = '%s%s(%s)' % (node.name, separator, node.type)
        return node_label

    @staticmethod
    def print_container(node, separator, format):
        node1 = PngPrinter.get_node_label(node.node1, separator, format)
        node2 = PngPrinter.get_node_label(node.node2, separator, format)

        node1_list = node1.split(separator)
        node2_list = node2.split(separator)
        #node_label = '%s%s%s' % (node1_list[0], separator, node2_list[0])
        node_label = node1_list[0]          # first node's name
        for desc in node1_list[1:]:
            node_label += separator + desc
        for desc in node2_list:
            node_label += separator + desc
        return node_label

    @staticmethod
    def print_conv(node, separator, format):
        if format == 'caffe':
            # Outer double quotes needed or else colon characters don't parse
            # properly
            node_label = '%s%s(%s)%skernel size: %d%sstride: %d%spad: %d' %\
                         (node.name, separator,
                          node.type, separator,
                          node.kernel_size, separator,
                          node.stride, separator, node.pad)
        elif format == 'custom':
            node_label = node.name + separator + 'k=' + str(node.kernel_size) + 'x' + str(
                node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    @staticmethod
    def print_pool(node, separator, format):
        pooling_type = get_pooling_types_dict()
        if format=='caffe':
            node_label = '%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d' % \
                         (node.name, separator, pooling_type[node.pool_type],
                          node.type, separator,
                          node.kernel_size, separator,
                          node.stride, separator, node.pad)
        elif format == 'custom':
            node_label = node.name + separator + pooling_type[node.pool_type] + ', k=' + str(node.kernel_size) + 'x' + str(
                         node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
            if node.ceiling:
                node_label += ' ceiling'
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    @staticmethod
    def print_lrn(node, separator, format):
        if format == 'caffe':
            node_label = node.name
        elif format == 'custom':
            node_label = node.name + separator + lrn_type[node.norm_region] + \
                          ' size=' + str(node.local_size) + ' alpha=' + str(
                          node.alpha) + ' beta=' + str(node.beta)
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    @staticmethod
    def print_reshape(node, separator, format):
        if format == 'caffe':
            node_label = node.name
        elif format == 'custom':
            node_label = '%s%s[%s,%s,%s,%s]' % \
                         (node.name, separator, node.reshape_param.dim[0], node.reshape_param.dim[1], node.reshape_param.dim[2], node.reshape_param.dim[3])
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    def add_pydot_node(self, node, tplgy, rankdir):
        # node_name = "%s_%s" % (node.name, node.type)
        # self.pydot_nodes[node.name] = pydot.Node(node.name,
                                    #    **NEURON_LAYER_STYLE)
        layer_style = choose_style_by_layertype(node.type)
        if rankdir in ('TB', 'BT'):
            # If graph orientation is vertical, horizontal space is free and
            # vertical space is not; separate words with spaces
            separator = ' '
        else:
            # If graph orientation is horizontal, vertical space is free and
            # horizontal space is not; separate words with newlines
            separator = '\\n'

        node_label = self.get_node_label(node, separator, options['node_label'])
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

    def draw_net(self, caffe_net, rankdir, tplgy):
        pydot_graph = pydot.Dot(self.caffe_net.name if self.caffe_net.name else 'Net',
                                graph_type='digraph',
                                rankdir=rankdir)

        # optional: collapse ReLU nodes
        if options['merge_conv_relu']:
            tplgy.merge_nodes('Convolution', 'ReLU')
        if options['merge_conv_relu_pooling']:
            #tplgy.merge_nodes('PairContainer', 'Pooling')
            tplgy.merge_nodes('Convolution_ReLU', 'Pooling')

        # tplgy.dump_edges()
        if options['remove_dropout']:
            tplgy.remove_node_by_type('Dropout')

        # tplgy.dump_edges()
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
