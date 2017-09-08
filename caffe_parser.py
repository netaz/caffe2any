from topology import Topology
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
