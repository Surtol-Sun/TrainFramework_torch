import os
import re
import argparse
from collections import OrderedDict
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Training curve visualization')
parser.add_argument('--dir', type=str, help='Path to the root of files', default=r'./')
parser.add_argument('--file', type=str, help="'all' or file name", default=r'all')
parser.add_argument('--protype', type=str, help="'Epoch' or 'Iteration'", default=r'Iteration')
args = parser.parse_args()


def main():
    log_file_dir = args.dir
    plot_type = args.file
    process_type = args.protype

    if plot_type == 'all':
        file_list = os.listdir(log_file_dir)
        for file_name in file_list:
            if re.match('^.+\.log', file_name):
                log_file_path = os.path.join(log_file_dir, file_name)
                if not os.path.exists(log_file_path):
                    print('File {:s} not exists!'.format(log_file_path))
                    continue
                view_logfile_training_curve(log_file_path, plot_title=file_name, process_type=process_type)
    else:
        log_file_name = plot_type
        log_file_path = os.path.join(log_file_dir, log_file_name)
        if not os.path.exists(log_file_path):
            print('File {:s} not exists!'.format(log_file_path))
            exit(-1)
        view_logfile_training_curve(log_file_path, process_type=process_type)


def view_logfile_training_curve(log_file_path, plot_title='Training curve', process_type='Epoch'):
    test_mse = OrderedDict()
    test_loss = OrderedDict()
    train_loss = OrderedDict()

    file_read = open(log_file_path)
    # Print some basic information
    print('-'*50)
    for i, file_line in enumerate(file_read):
        info_line_list = range(0, 14+1)
        if i in info_line_list:
            print(file_line.split('\n')[0])
        if i > max(info_line_list):
            break
    
    # Read and convert information
    data_counter = 0
    for file_line in file_read:
        # e.g. Epoch: [0][0/391]	Loss 0.7876	Time 10.973	Data 0.372	Prune 0.131	Prec@1 75.00	Prec@5 96.09
        # pattern_epoch = 'Epoch: \[[0-9]+]\[[0-9]+/[0-9]+]\tLoss [0-9]+\.[0-9]+\tTime [0-9]+\.[0-9]+\tData [0-9]+\.[0-9]+\tPrune [0-9]+\.[0-9]+\tPrec@1 [0-9]+\.[0-9]+\tPrec@5 [0-9]+\.[0-9]+'
        # e.g. ==train== Epoch[1]: [0/2333]	Loss 1.3657	Time 2.146	Data 1.525
        pattern_epoch = '==train== Epoch\[[0-9]+]: \[[0-9]+/[0-9]+]\tLoss [0-9]+\.[0-9]+\tTime [0-9]+\.[0-9]+\tData [0-9]+\.[0-9]+'
        pattern_epoch_match = re.search(pattern_epoch, file_line)
        if pattern_epoch_match:
            data_counter_valid = True
            int_num_list, float_num_list = extract_num(pattern_epoch_match.group())
            if process_type == 'Epoch':  # display the last data in each epoch
                data_counter = int_num_list[0]
            elif process_type == 'Iteration' and int_num_list[1] > 20:  # data less than 20 iterations are arbitrary
                if data_counter > int_num_list[0] * int_num_list[2] + int_num_list[1]:
                    print(data_counter, int_num_list, float_num_list, file_line)
                data_counter = int_num_list[0] * int_num_list[2] + int_num_list[1]
            else:
                data_counter_valid = False

            if data_counter_valid:
                train_loss[data_counter] = float_num_list[0]

        # e.g.  * Prec@1 74.800 Prec@5 98.380 Error@1 25.200
        # pattern_test_result = '\* Prec@1 [0-9]+\.[0-9]+ Prec@5 [0-9]+\.[0-9]+ Error@1 [0-9]+\.[0-9]+'
        # e.g. ==evaluate== Epoch[1]: [100/2333]	Loss 0.0082	Time 0.132	Data 0.015	MSE 0.008
        pattern_test_result = '==evaluate== Epoch\[[0-9]+]: \[[0-9]+/[0-9]+]\tLoss [0-9]+\.[0-9]+\tTime [0-9]+\.[0-9]+\tData [0-9]+\.[0-9]+\tMSE [0-9]+\.[0-9]+'
        pattern_test_result_match = re.search(pattern_test_result, file_line)
        if pattern_test_result_match:
            int_num_list, float_num_list = extract_num(pattern_test_result_match.group())

            # Give out values
            test_mse[data_counter] = float_num_list[0]
            test_loss[data_counter] = float_num_list[3]

    # Display
    loss_list = [
        [list(test_loss.keys()), list(test_loss.values()), 'test loss'],
        [list(train_loss.keys()), list(train_loss.values()), 'train loss'],
    ]
    metric_list = [
        [list(test_mse.keys()), list(test_mse.values()), 'test mse'],
    ]
    for _, loss_value, loss_name in loss_list:
        if len(loss_value) == 0:
            continue
        loss_val = loss_value[int(len(loss_value) * 0.25):]
        print('Minimum {:s} = {:.4f}'.format(loss_name, min(loss_val)))
    for _, accuracy_value, accuracy_name in metric_list:
        if len(accuracy_value) == 0:
            continue
        accuracy_val = accuracy_value[int(len(accuracy_value) * 0.25):]
        print('Maximum {:s} = {:.3f}%'.format(accuracy_name, max(accuracy_val)))
    display_plot(loss_list, metric_list, plot_title=plot_title, plot_xlabel=process_type)


def str2int(intNumStr):
    intNum = 0
    for i, NumStr in enumerate(reversed(intNumStr)):
        intNum += int(NumStr) * (10 ** i)
    return intNum


def str2float(s):
    intNumStr,floatNumStr=s.split('.')
    intNum = 0
    floatNum = 0
    for i, NumStr in enumerate(reversed(intNumStr)):
        intNum += int(NumStr) * (10 ** i)
    for i, NumStr in enumerate(floatNumStr):
        floatNum += int(NumStr) * (0.1 ** (i + 1))
    return intNum + floatNum


def extract_num(str_in):
    '''
    Extract numbers from a string
    :param str_in: Input string
    :return intNumList: A list that contains int number extracted from string
    :return floatNumList: A list that contains float number extracted from string
    '''
    intNumList = []
    floatNumList = []

    pattern_int = '[0-9]+'
    pattern_float = '[0-9]+\.[0-9]+'

    pattern_float_found = re.findall(pattern_float, str_in)
    if pattern_float_found:
        # Convert string to number
        for num_str in pattern_float_found:
            floatNumList.append(str2float(num_str))
        # Remove float number from string
        for num_str in pattern_float_found:
            str_in = str_in.replace(num_str, '', 1)

    pattern_int_found = re.findall(pattern_int, str_in)
    if pattern_int_found:
        # Convert string to number
        for num_str in pattern_int_found:
            intNumList.append(str2int(num_str))

    return intNumList, floatNumList


def display_plot(loss_list, accuracy_list, plot_title='Training curve', plot_xlabel='Epoch'):
    '''
    Display plot for accuracy and loss line. Colors for lines may refer to
    https://www.cnblogs.com/zwt20120701/p/10872023.html
    :param max_display_x: number of data for axis x
    :param loss_list: list in format [ [ [data_x], [data_y], name] ] which contains loss information
    :param accuracy_list: list in format [ [ [data], name] ] which contains accuracy information
    :param plot_title: title for curve
    :param plot_xlabel: label for x axis
    :return:
    '''
    color_list_loss = ['dodgerblue', 'cornflowerblue', 'lightskyblue', 'navy']
    color_list_accuracy = ['tomato', 'darkorange', 'darkseagreen', 'lightgreen']

    fig, ax = plt.subplots()
    for i, [data_x, data_y, name] in enumerate(accuracy_list):
        ax.plot(data_x, data_y, label=name, color=color_list_accuracy[i])

    ax1 = ax.twinx()
    for i, [data_x, data_y, name] in enumerate(loss_list):
        ax1.plot(data_x, data_y, label=name, color=color_list_loss[i])

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax1.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='center right')

    ax.set_xlabel(plot_xlabel)
    ax.set_ylabel('Accuracy (%)')
    ax1.set_ylabel('Loss')

    plt.grid()
    plt.title(plot_title)
    plt.show()


if __name__ == '__main__':
    main()
