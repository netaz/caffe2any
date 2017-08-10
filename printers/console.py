from __future__ import print_function
from .globals import *
pooling_type = get_pooling_types_dict()

class ConsolePrinter:
    """A simple console printer"""

    def __init__(self):
        pass

    def print_pool(self, node):
        desc = pooling_type[node.pool_type] + ', k=' + str(node.kernel_size) + 'x' + str(
            node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
        if node.ceiling:
            desc += ' ceiling'
        return 'Pooling', desc

    def print_deconv(self, node):
        return 'Deconvolution', \
               'k=' + str(node.kernel_size) + "x" + str(node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(
                   node.pad)

    def print_conv(self, node):
        return 'Convolution', \
               'k=' + str(node.kernel_size) + "x" + str(node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(
                   node.pad)

    def print_conv_relu(self, merged_node):
        node = merged_node.node1
        return 'Convolution/ReLU', \
               'k=' + str(node.kernel_size) + "x" + str(node.kernel_size) + '/s=' + str(
                    node.stride) + ' pad=' + str(node.pad)

    def print_lrn(self, node):
        return 'LRN', \
               lrn_type[node.norm_region] + ' size=' + str(node.local_size) + ' alpha=' + str(
                   node.alpha) + ' beta=' + str(node.beta)

    def print_unknown(self, node):
        return node.type, ""

    def print_node(self, node, count):
        print_fn = {
            "Pooling": self.print_pool,
            "Convolution": self.print_conv,
            "Convolution_ReLU": self.print_conv_relu,
            "Deconvolution": self.print_deconv,
            "LRN": self.print_lrn,
        }.get(node.type, self.print_unknown)
        row_format = "{:<20} {:<45} {:<40}"  # 3 is the number of cols
        # print('\t%-20s%-3s Count=%-10d' % print_fn(layer, count))
        print(row_format.format(*(print_fn(node) + (count,))))

    def print_inventory(self, inventory):
        print("Inventory:\n----------")
        cnt = 0
        for type in inventory:
            print('\t%-20s%-3i' % (type, inventory[type]))
            cnt += inventory[type]
        print("Total=", cnt)
        print("")

    def print_unique(self, unique_layers_list):
        for node in unique_layers_list:
            self.print_node(node[0], node[1])  # print node, count

    def print_unique_all(self, unique_layers_dict):
        print("Unique:\n--------")
        for type_name in unique_layers_dict:
            self.print_unique(unique_layers_dict[type_name])
        print("")

    def print_bfs(self, tplgy):
        tplgy.traverse(lambda node: print(str(node)),
                       lambda edge: print('\t' + str(edge)))
