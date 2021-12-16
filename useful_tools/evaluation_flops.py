#!/usr/bin/python
# -*- coding:utf-8 -*-
from _collections import OrderedDict


class CNN_network(object):
    def __init__(self):
        self.name_stack = []  # [ [bottleneck, 0], [conv, 1] ]
        self.operation_statistics_dict = {}  # { key: [in_shape, out_shape, param_shape, FLOPs, param_num] }
        self.operation_tree_dict = OrderedDict()  # name of operations contains structure

        self.structure_len = 0  # length of structure that printed before statistics
        self.structure_other_len = 89  # length of statistics that be printed, set manually

        self.outprint_name = 'No name!'
        self.title_print_format = '| {:^15s} | {:^15s} | {:^20s} | {:^11s} | {:^11s} |'
        self.detail_print_format = '| {:>15s} | {:>15s} | {:>20s} | {:>10.2f}M | {:>10.2f}M |'

    # Below are operations for normal neural network

    def add_conv2D(self, shape_in, output_channel_num, kernel=[3, 3], stride=1, padding=1, group=1, name='conv'):
        name = self.add_name(name)

        K1, K2 = kernel
        Hin, Win, Cin = shape_in

        Cout = output_channel_num
        Hout = (Hin + padding * 2 - (K1 - 1)) // stride
        Wout = (Win + padding * 2 - (K2 - 1)) // stride

        # ['conv2D', Hin, Win, Cin, Hout, Wout, Cout, K1, K2, group]
        propertites = ['conv2D', Hin, Win, Cin, Hout, Wout, Cout, K1, K2, group]
        self.operation_tree_dict[name] = propertites

        return [Hout, Wout, Cout]

    def add_fullconnect(self, shape_of_input, len_of_output, name='fullyconnect'):
        name = self.add_name(name)

        len_of_input = 1
        for shape_i in shape_of_input:
            len_of_input *= shape_i

        # ['fullyconnect', len_of_input, len_of_output]
        propertites = ['fullyconnect', len_of_input, len_of_output]
        self.operation_tree_dict[name] = propertites

        return [len_of_output]

    def add_maxpool(self, shape_in, kernel=[3, 3], stride=1, padding=1, name='maxpool'):
        name = self.add_name(name)

        K1, K2 = kernel
        Hin, Win, Cin = shape_in

        Cout = Cin
        Hout = (Hin + padding * 2 - (K1 - 1)) // stride
        Wout = (Win + padding * 2 - (K2 - 1)) // stride

        # ['conv2D', Hin, Win, Cin, Hout, Wout, Cout, K1, K2]
        propertites = ['maxpool', Hin, Win, Cin, Hout, Wout, Cout, K1, K2]
        self.operation_tree_dict[name] = propertites

        return [Hout, Wout, Cout]

    def add_avgpool(self, shape_in, kernel=[3, 3], stride=1, padding=1, name='avgpool'):
        name = self.add_name(name)

        K1, K2 = kernel
        Hin, Win, Cin = shape_in

        Cout = Cin
        Hout = (Hin + padding * 2 - (K1 - 1)) // stride
        Wout = (Win + padding * 2 - (K2 - 1)) // stride

        # ['conv2D', Hin, Win, Cin, Hout, Wout, Cout, K1, K2]
        propertites = ['avgpool', Hin, Win, Cin, Hout, Wout, Cout, K1, K2]
        self.operation_tree_dict[name] = propertites

        return [Hout, Wout, Cout]

    @staticmethod
    def _filter_specific_prefix(key_list, specific_prefix, prefix_place=0):
        '''
        :function: filter names with a specific name in a specific place
        :param key_list: a list that contains all the key names, split with '.' 
        :param specific_prefic: a list that contains all the prefix acceptable
        :param prefix_place: 0, 1, 2...
        :return filtered_key: keys with specific prefix
        '''
        filtered_key = []
        for key in key_list:
            for prefix in specific_prefix:
                if key.split('.')[prefix_place] == prefix:
                    filtered_key.append(key)
                    break
        return filtered_key

    def add_name(self, new_name, new_module_name=False):
        '''
        :param new_name: Name of newly added operation
        :param new_module_name: Is this name a name for a module?
        '''
        # get keys with same prefix
        same_prefix = list(self.operation_tree_dict.keys())
        for i, (name_i, name_counter_i) in enumerate(self.name_stack):
            same_prefix = self._filter_specific_prefix(same_prefix, [name_i + '_' + str(name_counter_i)],
                                                       prefix_place=i)

        # get max counter on this prefix with the same name
        same_name_on_same_prefix = [x.split('.')[len(self.name_stack)] for x in same_prefix if
                                    new_name in x.split('.')[len(self.name_stack)]]

        name_ncounters = [int(x.split('_')[-1]) for x in same_name_on_same_prefix]
        name_ncounters.append(-1)
        max_name_counter = max(name_ncounters)

        # add new name and generate actural name        
        self.name_stack.append([new_name, max_name_counter + 1])
        your_actural_name = ''
        for name_i, name_counter_i in self.name_stack:
            your_actural_name += (name_i + '_' + str(name_counter_i) + '.')
        if len(your_actural_name) > 0:
            your_actural_name = your_actural_name[:-1]

        # If this name is a name for a new module, then keep it. Otherwise, don't keep.
        if not new_module_name:
            self.name_stack.pop()
        return your_actural_name

    # Below are functions for print   
    def print_structure(self, print_method='expand'):
        tree_stack_print = []
        len_of_structure_str = {}
        # record max length of string of each depth 
        for index, (name, properity) in enumerate(self.operation_tree_dict.items()):
            for i, knot in enumerate(name.split('.')):
                if len_of_structure_str.get(i):
                    len_of_structure_str[i] = max(len(knot), len_of_structure_str[i])
                else:
                    len_of_structure_str[i] = len(knot)

        self.structure_len = 0
        for k in len_of_structure_str.keys():
            len_of_structure_str[k] += 2  # add some white blanks
            self.structure_len += len_of_structure_str[k]

        # print structure
        def print_tight_model(name_in):
            string2print = ''
            for i, knot in enumerate(name_in.split('.')):
                if i < len(tree_stack_print):
                    if knot == tree_stack_print[i]:
                        string2print += '|'.center(len_of_structure_str[i])
                    else:
                        string2print += knot.center(len_of_structure_str[i])
                        while len(tree_stack_print) > i:
                            tree_stack_print.pop()
                        tree_stack_print.append(knot)
                else:
                    string2print += knot.center(len_of_structure_str[i])
                    tree_stack_print.append(knot)

            in_shape, out_shape, param_shape, FLOPs, param_num = self.operation_statistics_dict[name_in]
            print(string2print.ljust(self.structure_len), self.detail_print_format.format(
                str(in_shape), str(out_shape), str(param_shape), FLOPs / 1e6, param_num / 1e6))

        def print_expand_model(name_in):
            string2print = ''
            break_flag = False
            for i, knot in enumerate(name_in.split('.')):
                if i < len(tree_stack_print):
                    if knot == tree_stack_print[i]:
                        string2print += '|'.center(len_of_structure_str[i])
                    else:
                        string2print += knot.center(len_of_structure_str[i])
                        while len(tree_stack_print) > i:
                            tree_stack_print.pop()
                        tree_stack_print.append(knot)

                        break_flag = True
                else:
                    string2print += knot.center(len_of_structure_str[i])
                    tree_stack_print.append(knot)

                    break_flag = True

                if break_flag and i < len(name_in.split('.')) - 1:
                    break
                else:
                    break_flag = False

            if break_flag:
                print(string2print.ljust(self.structure_len), self.title_print_format.format('', '', '', '', ''))
                print_expand_model(name_in)
            else:
                in_shape, out_shape, param_shape, FLOPs, param_num = self.operation_statistics_dict[name_in]
                print(string2print.ljust(self.structure_len),
                      self.detail_print_format.format(str(in_shape), str(out_shape), str(param_shape), FLOPs / 1e6,
                                                      param_num / 1e6))

        # print name title
        print('*' * (self.structure_len + self.structure_other_len))
        print(self.outprint_name.center(self.structure_len + self.structure_other_len))

        # print title
        print('-' * (self.structure_len + self.structure_other_len))
        print('Operations'.center(self.structure_len),
              self.title_print_format.format('In shape', 'Out shape', 'Param shape', 'FLOPs', 'Param Num'))
        print('-' * (self.structure_len + self.structure_other_len))

        # print details
        for index, (name, properity) in enumerate(self.operation_tree_dict.items()):
            if print_method == 'tight':
                print_tight_model(name)
            elif print_method == 'expand':
                print_expand_model(name)
            elif print_method == 'none':
                pass
            else:
                print('Unsupported print method!')
                exit(0)

        # print sum up
        import numpy as np
        FLOPs_sum, param_sum = np.sum(
            np.array([[x, y] for _, _, _, x, y in self.operation_statistics_dict.values()], np.float64), axis=0)
        print('-' * (self.structure_len + self.structure_other_len))
        print(''.center(self.structure_len),
              self.detail_print_format.format('', '', '', FLOPs_sum / 1e6, param_sum / 1e6))
        print('-' * (self.structure_len + self.structure_other_len))

    def calculate_statistics(self, criteria=1):
        for key_i, operations_i in self.operation_tree_dict.items():
            in_shape = None
            out_shape = None
            param_shape = None
            cal_FLOPs = None
            param_num = None
            operation_name = operations_i[0]
            if operation_name == 'conv2D':
                _, Hin, Win, Cin, Hout, Wout, Cout, K1, K2, group = operations_i
                if criteria == 1:  # from the paper if NVIDIA, with bias
                    cal_FLOPs = 2 * Hout * Wout * (Cin * K1 * K2 + 1) * Cout / group
                elif criteria == 2:  # from the paper if NVIDIA, without bias
                    cal_FLOPs = 2 * Hout * Wout * (Cin * K1 * K2 + 0) * Cout / group
                elif criteria == 3:  # form Internet, with bias
                    cal_FLOPs = Hout * Wout * (Cin * K1 * K2 + 1) * Cout / group
                elif criteria == 4:  # form Internet, simplified
                    cal_FLOPs = Hout * Wout * Cin * K1 * K2 * Cout / group

                if criteria == 1 or 3:  # with bias
                    param_num = Cin * K1 * K2 * Cout / group + Cout
                if criteria == 2 or 4:  # without bias
                    param_num = Cin * K1 * K2 * Cout / group

                in_shape = (Hin, Win, Cin)
                out_shape = (Hout, Wout, Cout)
                param_shape = (Cin, Cout // group, K1, K2)
            elif operation_name == 'maxpool' or operation_name == 'avgpool':
                _, Hin, Win, Cin, Hout, Wout, Cout, K1, K2 = operations_i
                cal_FLOPs = 0
                param_num = 0

                in_shape = (Hin, Win, Cin)
                out_shape = (Hout, Wout, Cout)
                param_shape = ()
            elif operation_name == 'fullyconnect':
                _, I, O = operations_i
                if criteria == 1 or 2:  # from the paper if NVIDIA
                    cal_FLOPs = (2 * I - 1) * O
                if criteria == 3 or 4:  # from Internet
                    cal_FLOPs = I * O
                param_num = I * O

                in_shape = (I)
                out_shape = (O)
                param_shape = (I, O)

            self.operation_statistics_dict[key_i] = [in_shape, out_shape, param_shape, cal_FLOPs, param_num]

    # Below are forward function for debug
    def forward(self, shape_in, **kwargs):
        '''
        :param shape_in: list that indicates shape of input, in format [H, W, C]. e.g. [224, 224, 3]
        '''
        x = self.add_conv2D(shape_in, 90)


class VGGNet(CNN_network):
    def __init__(self, depth=16, criteria=4):
        super().__init__()
        self.outprint_name = 'VGGNet-{:d}'.format(depth)

        if depth == 16:
            assign_channel_num = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]
        elif depth == 19:
            assign_channel_num = [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]]

        self.forward([224, 224, 3], assign_channel_num)
        self.calculate_statistics(criteria=criteria)

    def vgg_block(self, shape_in, channel_nums, last_conv_1x1=False, name='vgg block'):
        self.add_name(name, new_module_name=True)
        x = shape_in

        for i in range(len(channel_nums)):
            if i == len(channel_nums)-1:
                break
            x = self.add_conv2D(x, channel_nums[i], kernel=[3, 3], stride=1, padding=1, group=1)

        if last_conv_1x1:
            x = self.add_conv2D(x, channel_nums[-1], kernel=[1, 1], stride=1, padding=1, group=1)
        else:
            x = self.add_conv2D(x, channel_nums[-1], kernel=[3, 3], stride=1, padding=1, group=1)

        x = self.add_maxpool(x, kernel=[3, 3], stride=2, padding=1)

        self.name_stack.pop()
        return x

    def forward(self, shape_in, assign_channel):
        x = shape_in
        for channel_nums in assign_channel:
            x = self.vgg_block(x, channel_nums)

        x = self.add_fullconnect(x, 4096)
        x = self.add_fullconnect(x, 4096)
        x = self.add_fullconnect(x, 1000)

        return x


class VGGNet_cifar10(CNN_network):
    def __init__(self, depth=16, criteria=4):
        super().__init__()
        self.outprint_name = 'VGGNet cifar10 -{:d}'.format(depth)

        # 1. using fc(512) instead of fc(4069)
        # 2. using 2 fc layers instead of 3
        if depth == 16:
            self.forward = self.forward_16
        elif depth == 19:
            self.forward = self.forward_19

        self.forward([32, 32, 3])
        self.calculate_statistics(criteria=criteria)

    def vgg_block(self, shape_in, channel_num, repeat_time, last_conv_1x1=False, name='vgg block'):
        self.add_name(name, new_module_name=True)
        x = shape_in

        for _ in range(repeat_time-1):
            x = self.add_conv2D(x, channel_num, kernel=[3, 3], stride=1, padding=1, group=1)

        if last_conv_1x1:
            x = self.add_conv2D(x, channel_num, kernel=[1, 1], stride=1, padding=1, group=1)
        else:
            x = self.add_conv2D(x, channel_num, kernel=[3, 3], stride=1, padding=1, group=1)

        x = self.add_maxpool(x, kernel=[3, 3], stride=2, padding=1)

        self.name_stack.pop()
        return x

    def forward_16(self, shape_in):
        x = shape_in
        x = self.vgg_block(x, 64, 2)
        x = self.vgg_block(x, 128, 2)
        x = self.vgg_block(x, 256, 3)
        x = self.vgg_block(x, 512, 3)
        x = self.vgg_block(x, 512, 3)

        x = self.add_fullconnect(x, 512)
        x = self.add_fullconnect(x, 10)

        return x

    def forward_19(self, shape_in):
        x = shape_in
        x = self.vgg_block(x, 64, 2)
        x = self.vgg_block(x, 128, 2)
        x = self.vgg_block(x, 256, 4)
        x = self.vgg_block(x, 512, 4)
        x = self.vgg_block(x, 512, 4)

        x = self.add_fullconnect(x, 512)
        x = self.add_fullconnect(x, 10)

        return x


class ResNet(CNN_network):
    def __init__(self, depth=18, criteria=4):
        super().__init__()
        self.outprint_name = 'ResNet-{:d}'.format(depth)

        if depth == 18:
            self.forward([224, 224, 3], [2, 2, 2, 2], make_layer='BasciBlock')
        elif depth == 34:
            self.forward([224, 224, 3], [3, 4, 6, 3], make_layer='BasciBlock')
        elif depth == 50:
            self.forward([224, 224, 3], [3, 4, 6, 3], make_layer='Bottleneck')
        elif depth == 101:
            self.forward([224, 224, 3], [3, 4, 23, 3], make_layer='Bottleneck')

        self.calculate_statistics(criteria=criteria)

    def add_BasicBlock(self, shape_in, assign_1, channel_out, stride=1, downsample=False, name='basicblock'):
        name = self.add_name(name, new_module_name=True)

        x = shape_in
        x = self.add_conv2D(x, assign_1, kernel=[3, 3], stride=stride, padding=1, group=1)
        x = self.add_conv2D(x, channel_out, kernel=[3, 3], stride=1, padding=1, group=1)
        if downsample:
            residual = shape_in
            x1 = self.add_conv2D(residual, channel_out, kernel=[1, 1], stride=stride, padding=0, group=1,
                                 name='downsample')
            assert x == x1

        self.name_stack.pop()

        return x

    def add_Bottleneck(self, shape_in, assign_1, assign_2, channel_out, stride=1, downsample=False, name='bottleneck'):
        name = self.add_name(name, new_module_name=True)

        x = shape_in
        x = self.add_conv2D(x, assign_1, kernel=[1, 1], stride=stride, padding=0, group=1)
        x = self.add_conv2D(x, assign_2, kernel=[3, 3], stride=1, padding=1, group=1)
        x = self.add_conv2D(x, channel_out, kernel=[1, 1], stride=1, padding=0, group=1)
        if downsample:
            residual = shape_in
            x1 = self.add_conv2D(residual, channel_out, kernel=[0, 0], stride=stride, padding=0, group=1,
                                 name='downsample')
            assert x == x1

        self.name_stack.pop()

        return x

    def _make_layer_BasicBlock(self, x, in_channel, repeat, stride_in=2):
        name = self.add_name('layer', new_module_name=True)

        for i in range(repeat):
            stride = 1
            downsample = False
            if i == 0 and stride_in == 2:
                stride = 2
                downsample = True
            x = self.add_BasicBlock(x, in_channel, in_channel, stride=stride, downsample=downsample)

        self.name_stack.pop()
        return x

    def _make_layer_Bottleneck(self, x, in_channel, repeat, stride_in=2):
        name = self.add_name('layer', new_module_name=True)

        for i in range(repeat):
            stride = 1
            downsample = False
            if i == 0 and stride_in == 2:
                stride = 2
                downsample = True
            x = self.add_Bottleneck(x, in_channel, in_channel, in_channel * 4, stride=stride, downsample=downsample)

        self.name_stack.pop()
        return x

    def forward(self, shape_in, repeat_time, make_layer='BasciBlock'):
        if make_layer == 'BasciBlock':
            _make_layer = self._make_layer_BasicBlock
        elif make_layer == 'Bottleneck':
            _make_layer = self._make_layer_Bottleneck

        x = shape_in
        x = self.add_conv2D(x, 64, kernel=[7, 7], stride=2, padding=3, group=1)
        x = self.add_maxpool(x, kernel=[3, 3], stride=2, padding=1)

        x = _make_layer(x, 64, repeat_time[0], stride_in=1)
        x = _make_layer(x, 128, repeat_time[1], stride_in=2)
        x = _make_layer(x, 256, repeat_time[2], stride_in=2)
        x = _make_layer(x, 512, repeat_time[3], stride_in=2)

        x = self.add_avgpool(x, kernel=[7, 7], stride=1, padding=0)
        x = self.add_fullconnect(x, 1000)

        return x


class ResNet_cifar10(CNN_network):
    def __init__(self, depth=20, criteria=4):
        super().__init__()
        self.outprint_name = 'ResNet_cifar10-{:d}'.format(depth)

        if depth == 20:
            self.forward([32, 32, 3], [3, 3, 3], make_layer='BasciBlock')
        elif depth == 32:
            self.forward([32, 32, 3], [5, 5, 5], make_layer='BasciBlock')
        elif depth == 56:
            self.forward([32, 32, 3], [9, 9, 9], make_layer='Bottleneck')
        elif depth == 110:
            self.forward([32, 32, 3], [18, 18, 18], make_layer='Bottleneck')

        self.calculate_statistics(criteria=criteria)

    def add_BasicBlock(self, shape_in, assign_1, channel_out, stride=1, downsample=False, name='basicblock'):
        name = self.add_name(name, new_module_name=True)

        x = shape_in
        x = self.add_conv2D(x, assign_1, kernel=[3, 3], stride=stride, padding=1, group=1)
        x = self.add_conv2D(x, channel_out, kernel=[3, 3], stride=1, padding=1, group=1)
        if downsample:
            residual = shape_in
            x1 = self.add_conv2D(residual, channel_out, kernel=[1, 1], stride=stride, padding=0, group=1,
                                 name='downsample')
            assert x == x1

        self.name_stack.pop()

        return x

    def add_Bottleneck(self, shape_in, assign_1, assign_2, channel_out, stride=1, downsample=False, name='bottleneck'):
        name = self.add_name(name, new_module_name=True)

        x = shape_in
        x = self.add_conv2D(x, assign_1, kernel=[1, 1], stride=stride, padding=0, group=1)
        x = self.add_conv2D(x, assign_2, kernel=[3, 3], stride=1, padding=1, group=1)
        x = self.add_conv2D(x, channel_out, kernel=[1, 1], stride=1, padding=0, group=1)
        if downsample:
            residual = shape_in
            x1 = self.add_conv2D(residual, channel_out, kernel=[0, 0], stride=stride, padding=0, group=1,
                                 name='downsample')
            assert x == x1

        self.name_stack.pop()

        return x

    def _make_layer_BasicBlock(self, x, in_channel, repeat, stride_in=2):
        name = self.add_name('layer', new_module_name=True)

        for i in range(repeat):
            stride = 1
            downsample = False
            if i == 0 and stride_in == 2:
                stride = 2
                downsample = True
            x = self.add_BasicBlock(x, in_channel, in_channel, stride=stride, downsample=downsample)

        self.name_stack.pop()
        return x

    def _make_layer_Bottleneck(self, x, in_channel, repeat, stride_in=2):
        name = self.add_name('layer', new_module_name=True)

        for i in range(repeat):
            stride = 1
            downsample = False
            if i == 0 and stride_in == 2:
                stride = 2
                downsample = True
            x = self.add_Bottleneck(x, in_channel, in_channel, in_channel * 4, stride=stride, downsample=downsample)

        self.name_stack.pop()
        return x

    def forward(self, shape_in, repeat_time, make_layer='BasciBlock'):
        if make_layer == 'BasciBlock':
            _make_layer = self._make_layer_BasicBlock
        elif make_layer == 'Bottleneck':
            _make_layer = self._make_layer_Bottleneck

        x = shape_in
        x = self.add_conv2D(x, 16, kernel=[3, 3], stride=1, padding=1, group=1)

        x = _make_layer(x, 16, repeat_time[0], stride_in=1)
        x = _make_layer(x, 32, repeat_time[1], stride_in=2)
        x = _make_layer(x, 64, repeat_time[2], stride_in=2)

        x = self.add_avgpool(x, kernel=[x[0], x[1]], stride=1, padding=0)
        x = self.add_fullconnect(x, 10)

        return x


class MobileNetV2(CNN_network):
    def __init__(self, depth=18, criteria=4):
        super().__init__()
        self.outprint_name = 'MobileNetV2'

        self.forward([224, 224, 3])

        self.calculate_statistics(criteria=criteria)

    def add_inv_bottleneck(self, shape_in, assign_1, assign_2, channel_out, stride=1, skip_conv=False,
                           name='Inv_bottleneck'):
        self.add_name(name, new_module_name=True)

        x = shape_in
        if not skip_conv:
            x = self.add_conv2D(x, assign_1, kernel=[1, 1], stride=1, padding=0, group=1)
        x = self.add_conv2D(x, assign_2, kernel=[3, 3], stride=1, padding=1, group=x[2])
        x = self.add_conv2D(x, channel_out, kernel=[1, 1], stride=stride, padding=0, group=1)

        self.name_stack.pop()
        return x

    def forward(self, shape_in):
        def _make_layer(shape_in, stride_first_block, repeat_time, channe_out, expand_ratio):
            self.add_name('block', new_module_name=True)

            if expand_ratio == 1:
                skip_conv = True
            else:
                skip_conv = False

            x = shape_in

            for i in range(repeat_time):
                if i == 0:
                    stride = stride_first_block
                else:
                    stride = 1

                x = self.add_inv_bottleneck(x,
                                            assign_1=x[2] * expand_ratio,
                                            assign_2=x[2] * expand_ratio,
                                            channel_out=channe_out,
                                            stride=stride, skip_conv=skip_conv)

            self.name_stack.pop()
            return x

        x = shape_in
        x = self.add_conv2D(x, 32, kernel=[3, 3], stride=2, padding=1, group=1)
        x = _make_layer(x, 1, 1, 16, 1)
        x = _make_layer(x, 2, 2, 24, 6)
        x = _make_layer(x, 2, 3, 32, 6)
        x = _make_layer(x, 2, 4, 64, 6)
        x = _make_layer(x, 1, 3, 96, 6)
        x = _make_layer(x, 2, 3, 160, 6)
        x = _make_layer(x, 1, 1, 320, 6)
        x = self.add_conv2D(x, 1280, kernel=[1, 1], stride=1, padding=0, group=1)
        x = self.add_avgpool(x, kernel=[7, 7], stride=1, padding=0)
        x = self.add_conv2D(x, 1000, kernel=[1, 1], stride=1, padding=0, group=1)

        return x


class VGGNet_pruned(CNN_network):
    def __init__(self, depth=16, criteria=4):
        super().__init__()
        self.outprint_name = 'VGGNet-{:d}'.format(depth)

        if depth == 16:
            assign_channel_num = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]
        elif depth == 19:
            assign_channel_num = [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]]

        self.forward([224, 224, 3], assign_channel_num)
        self.calculate_statistics(criteria=criteria)

    def vgg_block(self, shape_in, channel_nums, last_conv_1x1=False, name='vgg block'):
        self.add_name(name, new_module_name=True)
        x = shape_in

        for i in range(len(channel_nums)):
            if i == len(channel_nums)-1:
                break
            x = self.add_conv2D(x, channel_nums[i], kernel=[3, 3], stride=1, padding=1, group=1)

        if last_conv_1x1:
            x = self.add_conv2D(x, channel_nums[-1], kernel=[1, 1], stride=1, padding=1, group=1)
        else:
            x = self.add_conv2D(x, channel_nums[-1], kernel=[3, 3], stride=1, padding=1, group=1)

        x = self.add_maxpool(x, kernel=[3, 3], stride=2, padding=1)

        self.name_stack.pop()
        return x

    def forward(self, shape_in, assign_channel):
        x = shape_in
        for channel_nums in assign_channel:
            x = self.vgg_block(x, channel_nums)

        x = self.add_fullconnect(x, 4096)
        x = self.add_fullconnect(x, 4096)
        x = self.add_fullconnect(x, 1000)

        return x


class VGGNet_cifar10_pruned(CNN_network):
    def __init__(self, depth=16, criteria=4):
        super().__init__()
        self.outprint_name = 'VGGNet cifar10 pruned -{:d}'.format(depth)

        # 1. using fc(512) instead of fc(4069)
        # 2. using 2 fc layers instead of 3
        if depth == 16:
            assign_channel_num = [[64]*2, [128]*2, [256]*3, [512]*3, [512]*3]
            self.forward = self.forward_16
        elif depth == 19:
            assign_channel_num = [[64] * 2, [128] * 2, [256] * 4, [512] * 4, [512] * 4]
            self.forward = self.forward_19

        self.forward([32, 32, 3], assign_channel_num)
        self.calculate_statistics(criteria=criteria)

    def vgg_block(self, shape_in, assign_num, last_conv_1x1=False, name='vgg block'):
        self.add_name(name, new_module_name=True)
        x = shape_in

        for i in range(len(assign_num)-1):
            x = self.add_conv2D(x, assign_num[i], kernel=[3, 3], stride=1, padding=1, group=1)

        if last_conv_1x1:
            x = self.add_conv2D(x, assign_num[-1], kernel=[1, 1], stride=1, padding=1, group=1)
        else:
            x = self.add_conv2D(x, assign_num[-1], kernel=[3, 3], stride=1, padding=1, group=1)

        x = self.add_maxpool(x, kernel=[3, 3], stride=2, padding=1)

        self.name_stack.pop()
        return x

    def forward_16(self, shape_in, assign_channel_num):
        x = shape_in
        for assign_num in assign_channel_num:
            x = self.vgg_block(x, assign_num)
        x = self.add_fullconnect(x, 512)
        x = self.add_fullconnect(x, 10)

        return x

    def forward_19(self, shape_in, assign_channel_num):
        x = shape_in
        for assign_num in assign_channel_num:
            x = self.vgg_block(x, assign_num)
        x = self.add_fullconnect(x, 512)
        x = self.add_fullconnect(x, 10)

        return x


class ResNet_pruned(CNN_network):
    def __init__(self, depth=18, criteria=4):
        super().__init__()
        self.outprint_name = 'ResNet-{:d} Pruned'.format(depth)

        if depth == 18:
            assign_channel_num = [[39, 39], [77, 77], [154, 154], [308, 308]]
            self.forward([224, 224, 3], [2, 2, 2, 2], assign_channel_num, make_layer='BasciBlock')
        elif depth == 34:
            assign_channel_num = [[46, 46, 46], [91, 91, 91, 91], [180, 180, 180, 180, 180, 180], [359, 359, 359]]
            self.forward([224, 224, 3], [3, 4, 6, 3], assign_channel_num, make_layer='BasciBlock')
        elif depth == 50:
            assign_channel_num = [[39, 39, 39, 39, 39, 39], [77, 77, 77, 77, 77, 77, 77, 77], [154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154], [308, 308, 308, 308, 308, 308]]
            self.forward([224, 224, 3], [3, 4, 6, 3], assign_channel_num, make_layer='Bottleneck')

        self.calculate_statistics(criteria=criteria)

    def add_BasicBlock(self, shape_in, assign_1, channel_out, stride=1, downsample=False, name='basicblock'):
        name = self.add_name(name, new_module_name=True)

        x = shape_in
        x = self.add_conv2D(x, assign_1, kernel=[3, 3], stride=stride, padding=1, group=1)
        x = self.add_conv2D(x, channel_out, kernel=[3, 3], stride=1, padding=1, group=1)
        if downsample:
            residual = shape_in
            x1 = self.add_conv2D(residual, channel_out, kernel=[1, 1], stride=stride, padding=0, group=1,
                                 name='downsample')
            assert x == x1

        self.name_stack.pop()

        return x

    def add_Bottleneck(self, shape_in, assign_1, assign_2, channel_out, stride=1, downsample=False, name='bottleneck'):
        name = self.add_name(name, new_module_name=True)

        x = shape_in
        x = self.add_conv2D(x, assign_1, kernel=[1, 1], stride=stride, padding=0, group=1)
        x = self.add_conv2D(x, assign_2, kernel=[3, 3], stride=1, padding=1, group=1)
        x = self.add_conv2D(x, channel_out, kernel=[1, 1], stride=1, padding=0, group=1)
        if downsample:
            residual = shape_in
            x1 = self.add_conv2D(residual, channel_out, kernel=[1, 1], stride=stride, padding=0, group=1,
                                 name='downsample')
            assert x == x1

        self.name_stack.pop()

        return x

    def _make_layer_BasicBlock(self, x, in_channel, repeat, assign_channel_num, stride_in=2):
        self.add_name('layer', new_module_name=True)

        for i in range(repeat):
            stride = 1
            downsample = False
            if i == 0 and stride_in == 2:
                stride = 2
                downsample = True
            x = self.add_BasicBlock(x, assign_channel_num[i], in_channel, stride=stride, downsample=downsample)

        self.name_stack.pop()
        return x

    def _make_layer_Bottleneck(self, x, in_channel, repeat, assign_channel_num, stride_in=2):
        self.add_name('layer', new_module_name=True)

        for i in range(repeat):
            stride = 1
            downsample = False
            if i == 0 and stride_in == 2:
                stride = 2
                downsample = True
            x = self.add_Bottleneck(x, assign_channel_num[2 * i], assign_channel_num[2 * i + 1], in_channel * 4,
                                    stride=stride, downsample=downsample)

        self.name_stack.pop()
        return x

    def forward(self, shape_in, repeat_time, assign_channel_num, make_layer='BasciBlock'):
        if make_layer == 'BasciBlock':
            _make_layer = self._make_layer_BasicBlock
        elif make_layer == 'Bottleneck':
            _make_layer = self._make_layer_Bottleneck

        x = shape_in
        x = self.add_conv2D(x, 64, kernel=[7, 7], stride=2, padding=3, group=1)
        x = self.add_maxpool(x, kernel=[3, 3], stride=2, padding=1)

        x = _make_layer(x, 64, repeat_time[0], assign_channel_num[0], stride_in=1)
        x = _make_layer(x, 128, repeat_time[1], assign_channel_num[1], stride_in=2)
        x = _make_layer(x, 256, repeat_time[2], assign_channel_num[2], stride_in=2)
        x = _make_layer(x, 512, repeat_time[3], assign_channel_num[3], stride_in=2)

        x = self.add_avgpool(x, kernel=[7, 7], stride=1, padding=0)
        x = self.add_fullconnect(x, 1000)

        return x


class ResNet_cifar10_pruned(CNN_network):
    def __init__(self, depth=20, criteria=4):
        super().__init__()
        self.outprint_name = 'ResNet_cifar-{:d} Pruned'.format(depth)

        assign_layer_channel = [16, 32, 64]
        if depth == 20:
            assign_channel_num = [[10, 10, 10], [20, 20, 20], [39, 39, 39]]
            self.forward([32, 32, 3], [3, 3, 3], assign_layer_channel, assign_channel_num, make_layer='BasciBlock')
        elif depth == 32:
            assign_channel_num = [[9, 9, 9, 9, 9], [19, 19, 19, 19, 19], [38, 38, 38, 38, 38]]
            self.forward([32, 32, 3], [5, 5, 5], assign_layer_channel, assign_channel_num, make_layer='BasciBlock')
        elif depth == 56:
            assign_channel_num = [[9, 8, 6, 6, 6, 6, 5, 5, 7, 10, 7, 7, 6, 8, 6, 7, 9, 9], [20, 25, 11, 15, 9, 10, 10, 15, 10, 12, 11, 15, 11, 14, 14, 24, 19, 24], [43, 45,27, 46, 26, 37, 25, 32, 27, 36, 30, 41, 36, 46, 47, 49, 49, 49]]
            self.forward([32, 32, 3], [9, 9, 9], assign_layer_channel, assign_channel_num, make_layer='Bottleneck')
        elif depth == 110:
            assign_channel_num = [[10] * 36, [20] * 36, [39] * 36]
            self.forward([32, 32, 3], [18, 18, 18], assign_layer_channel, assign_channel_num, make_layer='Bottleneck')

        self.calculate_statistics(criteria=criteria)

    def add_BasicBlock(self, shape_in, assign_1, channel_out, stride=1, downsample=False, name='basicblock'):
        name = self.add_name(name, new_module_name=True)

        x = shape_in
        x = self.add_conv2D(x, assign_1, kernel=[3, 3], stride=stride, padding=1, group=1)
        x = self.add_conv2D(x, channel_out, kernel=[3, 3], stride=1, padding=1, group=1)
        if downsample:
            residual = shape_in
            x1 = self.add_conv2D(residual, channel_out, kernel=[1, 1], stride=stride, padding=0, group=1,
                                 name='downsample')
            assert x == x1

        self.name_stack.pop()

        return x

    def add_Bottleneck(self, shape_in, assign_1, assign_2, channel_out, stride=1, downsample=False, name='bottleneck'):
        name = self.add_name(name, new_module_name=True)

        x = shape_in
        x = self.add_conv2D(x, assign_1, kernel=[1, 1], stride=stride, padding=0, group=1)
        x = self.add_conv2D(x, assign_2, kernel=[3, 3], stride=1, padding=1, group=1)
        x = self.add_conv2D(x, channel_out, kernel=[1, 1], stride=1, padding=0, group=1)
        if downsample:
            residual = shape_in
            x1 = self.add_conv2D(residual, channel_out, kernel=[1, 1], stride=stride, padding=0, group=1,
                                 name='downsample')
            assert x == x1

        self.name_stack.pop()

        return x

    def _make_layer_BasicBlock(self, x, in_channel, repeat, assign_channel_num, stride_in=2):
        self.add_name('layer', new_module_name=True)

        for i in range(repeat):
            stride = 1
            downsample = False
            if i == 0 and stride_in == 2:
                stride = 2
                downsample = True
            x = self.add_BasicBlock(x, assign_channel_num[i], in_channel, stride=stride, downsample=downsample)

        self.name_stack.pop()
        return x

    def _make_layer_Bottleneck(self, x, in_channel, repeat, assign_channel_num, stride_in=2):
        self.add_name('layer', new_module_name=True)

        for i in range(repeat):
            stride = 1
            downsample = False
            if i == 0 and stride_in == 2:
                stride = 2
                downsample = True
            x = self.add_Bottleneck(x, assign_channel_num[i*2], assign_channel_num[i*2+1], in_channel * 4, stride=stride, downsample=downsample)

        self.name_stack.pop()
        return x

    def forward(self, shape_in, repeat_time, assign_layer_channel, assign_channel_num, make_layer='BasciBlock'):
        if make_layer == 'BasciBlock':
            _make_layer = self._make_layer_BasicBlock
        elif make_layer == 'Bottleneck':
            _make_layer = self._make_layer_Bottleneck

        x = shape_in
        x = self.add_conv2D(x, 16, kernel=[3, 3], stride=1, padding=1, group=1)

        x = _make_layer(x, assign_layer_channel[0], repeat_time[0], assign_channel_num[0], stride_in=1)
        x = _make_layer(x, assign_layer_channel[1], repeat_time[1], assign_channel_num[1], stride_in=2)
        x = _make_layer(x, assign_layer_channel[2], repeat_time[2], assign_channel_num[2], stride_in=2)

        x = self.add_avgpool(x, kernel=[x[0], x[1]], stride=1, padding=0)
        x = self.add_fullconnect(x, 10)

        return x
    

class MobileNetV2_pruned(CNN_network):
    def __init__(self, depth=18, criteria=4):
        super().__init__()
        self.outprint_name = 'MobileNetV2'

        assign_layer_channel = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        assign_channel_num = [[31], [94, 136], [138, 157, 154], [176, 292, 285, 290], [326, 413, 418], [494, 792, 758], [839]]
        self.forward([224, 224, 3], assign_layer_channel, assign_channel_num)

        self.calculate_statistics(criteria=criteria)

    def add_inv_bottleneck(self, shape_in, assign_1, channel_out, stride=1, skip_conv=False,
                           name='Inv_bottleneck'):
        self.add_name(name, new_module_name=True)

        assign_2 = assign_1

        x = shape_in
        if not skip_conv:
            x = self.add_conv2D(x, assign_1, kernel=[1, 1], stride=1, padding=0, group=1)
        x = self.add_conv2D(x, assign_2, kernel=[3, 3], stride=1, padding=1, group=x[2])
        x = self.add_conv2D(x, channel_out, kernel=[1, 1], stride=stride, padding=0, group=1)

        self.name_stack.pop()
        return x

    def forward(self, shape_in, assign_layer_channel, assign_channel_num):
        def _make_layer(shape_in, stride_first_block, repeat_time, channe_out, expand_ratio, assign_num):
            self.add_name('block', new_module_name=True)

            if expand_ratio == 1:
                skip_conv = True
            else:
                skip_conv = False

            x = shape_in

            for i in range(repeat_time):
                if i == 0:
                    stride = stride_first_block
                else:
                    stride = 1

                x = self.add_inv_bottleneck(x,
                                            assign_1=assign_num[i],
                                            channel_out=channe_out,
                                            stride=stride, skip_conv=skip_conv)

            self.name_stack.pop()
            return x

        x = shape_in
        x = self.add_conv2D(x, assign_layer_channel[0], kernel=[3, 3], stride=2, padding=1, group=1)
        x = _make_layer(x, 1, 1, assign_layer_channel[1], 1, assign_channel_num[0])
        x = _make_layer(x, 2, 2, assign_layer_channel[2], 6, assign_channel_num[1])
        x = _make_layer(x, 2, 3, assign_layer_channel[3], 6, assign_channel_num[2])
        x = _make_layer(x, 2, 4, assign_layer_channel[4], 6, assign_channel_num[3])
        x = _make_layer(x, 1, 3, assign_layer_channel[5], 6, assign_channel_num[4])
        x = _make_layer(x, 2, 3, assign_layer_channel[6], 6, assign_channel_num[5])
        x = _make_layer(x, 1, 1, assign_layer_channel[7], 6, assign_channel_num[6])
        x = self.add_conv2D(x, assign_layer_channel[8], kernel=[1, 1], stride=1, padding=0, group=1)
        x = self.add_avgpool(x, kernel=[7, 7], stride=1, padding=0)
        x = self.add_conv2D(x, 1000, kernel=[1, 1], stride=1, padding=0, group=1)

        return x


def main():
    global_criteria = 4

    # Below are models
    VGG_16 = VGGNet(depth=16, criteria=global_criteria)
    ResNet_18 = ResNet(depth=18, criteria=global_criteria)
    ResNet_34 = ResNet(depth=34, criteria=global_criteria)
    ResNet_50 = ResNet(depth=50, criteria=global_criteria)
    MobileNet_V2 = MobileNetV2(criteria=global_criteria)

    VGG_16_cifar10 = VGGNet_cifar10(depth=16, criteria=global_criteria)
    ResNet_20_cifar10 = ResNet_cifar10(depth=20, criteria=global_criteria)
    ResNet_32_cifar10 = ResNet_cifar10(depth=32, criteria=global_criteria)
    ResNet_56_cifar10 = ResNet_cifar10(depth=56, criteria=global_criteria)
    ResNet_110_cifar10 = ResNet_cifar10(depth=110, criteria=global_criteria)

    # Below are pruned models
    VGG_16_pruned = VGGNet_pruned(depth=16, criteria=global_criteria)
    ResNet_18_pruned = ResNet_pruned(depth=18, criteria=global_criteria)
    ResNet_34_pruned = ResNet_pruned(depth=34, criteria=global_criteria)
    ResNet_50_pruned = ResNet_pruned(depth=50, criteria=global_criteria)
    MobileNet_V2_pruned = MobileNetV2_pruned(criteria=global_criteria)

    VGG_16_cifar10_pruned = VGGNet_cifar10_pruned(depth=16, criteria=global_criteria)
    VGG_19_cifar10_pruned = VGGNet_cifar10_pruned(depth=19, criteria=global_criteria)

    ResNet_20_cifar10_pruned = ResNet_cifar10_pruned(depth=20, criteria=global_criteria)
    ResNet_32_cifar10_pruned = ResNet_cifar10_pruned(depth=32, criteria=global_criteria)
    ResNet_56_cifar10_pruned = ResNet_cifar10_pruned(depth=56, criteria=global_criteria)
    ResNet_110_cifar10_pruned = ResNet_cifar10_pruned(depth=110, criteria=global_criteria)

    # Print model structure
    VGG_16_pruned.print_structure(print_method='expand')


if __name__ == '__main__':
    main()
