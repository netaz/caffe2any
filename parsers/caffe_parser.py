# TODO: Remove node_role
from topology import Topology
import copy
import logging
logger = None

def log():
    global logger
    if logger == None:
        logger = logging.getLogger('parsers')
    return logger

def parse_caffe_net(caffe_net):
    """
    Create and populate a Topology object, based on a given Caffe protobuf network object
    Todo: fix Input assignment
    """
    graph = Topology()

    modifiers = []

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
        log().debug('evaluating layer: ' + layer.name)
        #print('evaluating layer: ' + layer.name)

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
            #if layer.type == 'Input':
            #    included = False
            if not included:
                continue

        node_role = 'Producer'
        if (len(layer.bottom) == 1 and len(layer.top) == 1 and
                    layer.bottom[0] == layer.top[0]):
            # We have an in-place neuron layer.
            node_role = 'Modifier'

        new_node = graph.add_node(layer.name, layer.type, layer, node_role)

        if node_role == 'Modifier':
            modifiers.append({ 'blob': layer.bottom[0],
                               'src': layer.bottom[0],
                               'modifier': new_node })
            '''
            blob = graph.find_blob_by_name(layer.bottom[0])
            replicated_blob = copy.deepcopy(blob)
            replicated_blob.name += '2'
            src = graph.find_node_by_name(layer.bottom[0])
            graph.add_edge(src=src, dst=replicated_blob)
            graph.add_edge(src=replicated_blob, dst=new_node)
            graph.add_edge(src=new_node, dst=blob)
            edge_to_remove = graph.find_edge(src, blob)
            graph.del_edge(edge_to_remove)
            '''
            continue

        # Iterate over BLOBs consumed by this layer and create edges to them
        for caffe_bottom_blob in layer.bottom:
            #blob = graph.find_blob_by_name('b_' + caffe_bottom_blob)
            blob = graph.find_blob_by_name(caffe_bottom_blob)
            if blob == None:
                raise ValueError(layer.name + ' - could not find BLOB:' + caffe_bottom_blob)

            edge = graph.add_edge(src=blob, dst=new_node)

        # Add the BLOBs produced by this layer to the topology
        for caffe_top_blob in layer.top:
            if new_node.type == "Input":
                new_blob = graph.add_blob(caffe_top_blob, layer.input_param.shape[0].dim, producer=new_node)
            else:
                new_blob = graph.add_blob(caffe_top_blob, None, producer=new_node)

            # Add producer edges
            edge = graph.add_edge(src=new_node, dst=new_blob)

    for mod in modifiers:
        blob = graph.find_blob_by_name(mod['blob'])
        src = graph.find_node_by_name(mod['src'])

        replicated_blob = graph.find_blob_by_name(blob.name + '_replica')
        if replicated_blob is None:
            replicated_blob = copy.deepcopy(blob)
            replicated_blob.name += '_replica'
            graph.add_blob2(replicated_blob)
            graph.add_edge(src=src, dst=replicated_blob)

        #if graph.find_blob_by_name(replicated_blob.name) is not None:


        graph.add_edge(src=replicated_blob, dst=mod['modifier'])
        graph.add_edge(src=mod['modifier'], dst=blob)
        edge_to_remove = graph.find_edge(src, blob)
        graph.del_edge(edge_to_remove)


    # Add fake output edges
    '''
    output_blobs = graph.find_output_blobs()
    for blob_name in output_blobs:
        blob = graph.find_blob_by_name(blob_name)
        graph.add_edge(src=blob.producer, dst=None, blob=blob)
    '''
    return graph
