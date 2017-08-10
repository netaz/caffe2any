"""This is the Inventory transformer, which is a type of reduction transformation.

"""

def is_unique(node, unique_list):
    unique = True
    for unique_layer in unique_list:
        if unique_layer[0].is_same(node):
            unique = False
            unique_list.remove(unique_layer)
            unique_list.append((node, unique_layer[1]+1))
            break
    return unique


def add_unique(node, unique_layers):
    if unique_layers.get(node.type)==None:
        unique_layers[node.type] = []
    if is_unique(node, unique_layers[node.type]):
        unique_layers[node.type].append((node, 1))
