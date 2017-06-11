from model_definition import default_config


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


class DefaultConfig(ModelConfig):
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
        

class BaselineConfig(ModelConfig):
    def __init__(self, conv_layers_num=3, fc_layers_num=2, 
                 config_dict=None):
        super(BaselineConfig, self).__init__(conv_layers_num,
                                             fc_layers_num,
                                             config_dict)