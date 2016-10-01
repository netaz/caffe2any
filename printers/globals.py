import caffe_pb2 as caffe


def get_pooling_types_dict():
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d


def get_lrn_types_dict():
    desc = caffe.LRNParameter.NormRegion.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d

lrn_type = get_lrn_types_dict()
#lrn_type = {0: 'ACROSS_CHANNELS', 1: 'WITHIN_CHANNEL'}