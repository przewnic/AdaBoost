"""
Project: AdaBoost Implementation
Authors: przewnic
Date: 01.2021
"""
# Testing work of own implementation of AdaBoost
from Database import Database
from Forest import Forest
from sklearn.model_selection import KFold
import numpy as np
from AdaBoost import AdaBoost
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import time


if __name__ == "__main__":
    # read .data and prepare dataset
    db = Database()
    db.load_from_file("Lists/data/processed.cleveland.data")
    X = np.array(db.get_X())
    y = np.array(db.get_y())
    # Find best weak learners
    f = Forest()
    forest = f.create_forest(db)

    NUMBER_OF_ITERATIONS = 40
    # Create splits for k-fold validation
    NUMBER_OF_SPLITS = 5
    kf = KFold(n_splits=NUMBER_OF_SPLITS)

    ERR_sum = 0
    ada = AdaBoost()
    err_table = [[], []]
    err_table_sklearn = [[], []]
    for i in range(1, NUMBER_OF_ITERATIONS+1):
        # Own Ada
        tic = time.perf_counter()
        for train, test in kf.split(X):
            ada.fit(forest, db.get_sub_db(train), i)  # Learing own Ada
            Err = ada.test(db.get_sub_db(test))  # Testing own Ada
            ERR_sum += Err
        toc = time.perf_counter()
        err_table[0].append(ERR_sum/NUMBER_OF_SPLITS)
        err_table[1].append(round(toc-tic, 4))
        print(f"Own Ada for {i} weak learners: ", ERR_sum/NUMBER_OF_SPLITS)
        ERR_sum = 0
        # Sklearn Ada
        tic = time.perf_counter()
        clf = AdaBoostClassifier(n_estimators=i+1)
        scores = cross_val_score(clf, X, y, cv=5)
        scores_mean = scores.mean()
        toc = time.perf_counter()
        err_table_sklearn[1].append(round(toc-tic, 4))
        print(f"Sklearn error for i:{i} weak learners:", 1-scores_mean)
        err_table_sklearn[0].append(1-scores_mean)

    # Create a plot for errors
    plt.plot(range(1, NUMBER_OF_ITERATIONS+1, 1), err_table[0], label='own Ada')
    plt.plot(range(1, NUMBER_OF_ITERATIONS+1, 1), err_table_sklearn[0], label='sklearn Ada')
    plt.title("Error based on number of weak learners")
    plt.xlabel("number of weak learners")
    plt.ylabel("error value")
    plt.legend()
    plt.show()

    # Create a plot for times
    plt.plot(range(1, NUMBER_OF_ITERATIONS+1, 1), err_table[1], label='own Ada')
    plt.plot(range(1, NUMBER_OF_ITERATIONS+1, 1), err_table_sklearn[1], label='sklearn Ada')
    plt.title("Time based on number of weak learners")
    plt.xlabel("number of weak learners")
    plt.ylabel("time of execution [s]")
    plt.legend()
    plt.show()
