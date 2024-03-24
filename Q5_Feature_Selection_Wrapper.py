import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def splitData(data):
    dataSize = len(data)
    idx = random.sample(range(0, dataSize), dataSize)

    x_train = data[idx[: int(0.6 * dataSize)], 1:-1]
    y_train = data[idx[: int(0.6 * dataSize)], -1]
    x_sel = data[idx[int(0.6 * dataSize) : int(0.8 * dataSize)], 1:-1]
    y_sel = data[idx[int(0.6 * dataSize) : int(0.8 * dataSize)], -1]
    x_test = data[idx[int(0.8 * dataSize) :], 1:-1]
    y_test = data[idx[int(0.8 * dataSize) :], -1]
    return x_train, y_train, x_sel, y_sel, x_test, y_test


attr_map = {
    0: "Clump_thickness",
    1: "Uniformity_of_cell_size",
    2: "Uniformity_of_cell_shape",
    3: "Marginal_adhesion",
    4: "Single_epithelial_cell_size",
    5: "Bare_nuclei",
    6: "Bland_chromatin",
    7: "Normal_nucleoli",
    8: "Mitoses",
}

with open("./breast+cancer+wisconsin+original/breast-cancer-wisconsin.data", "r") as f:
    data = f.read()
data = data.split("\n")
data = data[:-1]
data = [x.split(",") for x in data]
data = np.array(data)
clean_data = []
wrong_data = []
for i in data:
    try:
        clean_data.append(i.astype(int))
    except:
        wrong_data.append(i)
clean_data = np.array(clean_data)
wrong_data = np.array(wrong_data)
print("Numbers of clean data = {}".format(len(clean_data)))
print("Numbers of wrong data = {}".format(len(wrong_data)))
acc_test_list = []
for t in range(10):

    x_train, y_train, x_sel, y_sel, x_test, y_test = splitData(clean_data)

    picked_attr = []
    for _ in range(3):
        acc_sel_best = 0
        best_attribute = -1
        for i in range(9):
            if i in picked_attr:
                continue
            else:
                tmp_attr = picked_attr.copy()
                tmp_attr.append(i)
                model = GaussianNB()
                model.fit(x_train[:, tmp_attr], y_train)
                y_pred = model.predict(x_sel[:, tmp_attr])
                acc_sel = accuracy_score(y_sel, y_pred)
                if acc_sel > acc_sel_best:
                    acc_sel_best = acc_sel
                    best_attribute = i
        picked_attr.append(best_attribute)
    model = GaussianNB()
    print("Trial :{}".format(t))
    print(
        "picked_attr :{}, {}, {}".format(
            attr_map[picked_attr[0]], attr_map[picked_attr[1]], attr_map[picked_attr[2]]
        )
    )
    model.fit(x_train[:, picked_attr], y_train)
    y_pred = model.predict(x_test[:, picked_attr])
    acc_test = accuracy_score(y_test, y_pred)
    print("accurate :{}".format(acc_test))
    acc_test_list.append(acc_test)

plt.figure(1)
plt.title(
    "Test accuracy in feature selection (wrapper method). Avg.={}".format(
        sum(acc_test_list) / 10
    )
)
plt.xlabel("Trails")
plt.ylabel("Accuracy")
plt.plot(acc_test_list)

acc_test_full_list = []
for _ in range(10):
    x_train, y_train, _, _, x_test, y_test = splitData(clean_data)
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc_test_full = accuracy_score(y_test, y_pred)
    acc_test_full_list.append(acc_test_full)

plt.figure(2)
plt.title(
    "Test accuracy without feature selection. Avg.={}".format(
        sum(acc_test_full_list) / 10
    )
)
plt.xlabel("Trails")
plt.ylabel("Accuracy")
plt.plot(acc_test_full_list)
