class ModelConfig(object):
    def __init__(self, conv_layers_num, fc_layers_num, config_dict):
        self._weights = config_dict['weights']
        self._biases = config_dict['biases']
        self._fc_layers_with_dropout = []
        self._conv_layers_num = conv_layers_num
        self._fc_layers_num = fc_layers_num
        self._config_dict = config_dict

    def get_fc_weights(self):
        return self._weights[self._conv_layers_num:]

    def get_fc_biases(self):
        return self._biases[self._conv_layers_num:]

    def get_conv_weights(self):
        return self._weights[0:self._conv_layers_num]

    def get_conv_biases(self):
        return self._biases[0:self._conv_layers_num]

    def get_strides(self):
        return self._config_dict['strides']

    def get_pool_strides(self):
        return self._config_dict['pool_strides']

    def get_pool_windows(self):
        return self._config_dict['pool_windows']
    
    def has_fc_dropout(self, index):
        return index in self._fc_layers_with_dropout

    def has_dropout_after_convolutions(self):
        return False

    def with_l2_norm(self):
        return False


class BaselineConfig(ModelConfig):
    from model_definition.baseline import baseline_config

    def __init__(self, conv_layers_num=3, fc_layers_num=2, 
                 config_dict=baseline_config):
        super(BaselineConfig, self).__init__(conv_layers_num,
                                             fc_layers_num,
                                             config_dict)


class NoRegularizationConfig(ModelConfig):
    from model_definition.additional_layers import additional_layers_config

    def __init__(self, conv_layers_num=4, fc_layers_num=3, 
                 config_dict=additional_layers_config):
        super(NoRegularizationConfig, self).__init__(
            conv_layers_num, fc_layers_num, config_dict)


# Three different regularization options used for
# building the network configuration
class OneDropoutRegularizationConfig(NoRegularizationConfig):
    def __init__(self, conv_layers_num, 
                 fc_layers_num, 
                 config_dict):
        super(OneDropoutRegularizationConfig, self).__init__(
            conv_layers_num, fc_layers_num, config_dict)
        # Dropout on second fully connected layers, zero based indexing
        self._fc_layers_with_dropout = [1]


class DropoutAfterConvolutionsConfig(OneDropoutRegularizationConfig):
    def __init__(self, conv_layers_num,
                 fc_layers_num, 
                 config_dict):
        super(DropoutAfterConvolutionsConfig, self).__init__(
            conv_layers_num, fc_layers_num, config_dict)

    def has_dropout_after_convolutions(self):
        return True


class DropoutsWithL2RegularizationConfig(DropoutAfterConvolutionsConfig):
    def __init__(self, conv_layers_num,
                 fc_layers_num, 
                 config_dict):
        super(DropoutsWithL2RegularizationConfig, self).__init__(
            conv_layers_num, fc_layers_num, config_dict)

    def with_l2_norm(self):
        return True


# With regularization - 2 dropouts and L2 norm
# filters and strides adopted to handle more slices with the same
# network depth
class DefaultConfig(ModelConfig):
    from model_definition.default import default_config

    def __init__(self, conv_layers_num=4, fc_layers_num=3, 
                 config_dict=default_config):
        super(DefaultConfig, self).__init__(conv_layers_num,
                                            fc_layers_num,
                                            config_dict)
        self._fc_layers_with_dropout = [1]

    def has_dropout_after_convolutions(self):
        return True

    def with_l2_norm(self):
        return True