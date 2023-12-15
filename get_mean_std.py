# 读取一个以'.out'为后缀名的文件
# 将其中最佳准确率提取出来，并计算所有最佳准确率的平均值和标准差。
from statistics import mean
import numpy as np


def compute_accuracy(file_name):
    acc = []

    with open(file_name, 'r') as f:
        is_best = False
        for line in f.readlines():
            if is_best:
                acc.append(float(line))
                is_best = False
            elif 'Best accuracy' in line:
                is_best = True

    if len(acc) > 0:
        print("最佳准确率列表：", acc)
        print("平均准确率：", mean(acc) * 100)
        print("标准差：", np.std(acc) * 100)
    else:
        print("未找到最佳准确率")


if __name__ == '__main__':
    file_name = input("请输入文件名：") + '.out'
    compute_accuracy(file_name)
