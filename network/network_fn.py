from __future__ import print_function, division


def define_network(network_name, **kwargs):
    if network_name == "dtdnnss_voxceleb_base_v1":
        from network.dtdnnss_searched import DtdnnssBase_v1
        num_class = kwargs["config"].num_class
        in_channels = kwargs["config"].in_channels
        mid_channels = kwargs["config"].mid_channels
        reduction_channels = kwargs["config"].reduction_channels
        feature_dim = kwargs["config"].feature_dim
        chromosome = kwargs["config"].chromosome
        network = DtdnnssBase_v1(num_class=num_class, in_channels=in_channels, feature_dim=feature_dim,
                                 chromosome=chromosome, mid_channels=mid_channels, reduction_channels=reduction_channels)
    else:
        raise Exception('Unknown Network Name!')
    return network
