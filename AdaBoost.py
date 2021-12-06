"""
Project: AdaBoost Implementation
Authors: przewnic
Date: 01.2021
"""
# Implementation of AdaBoost Alghorithm
from numpy import log as ln
import math
from random import random


class AdaBoost():
    """ Implementation of AdaBoost algorithm """
    def __init__(self):
        """
        classifier: a list of pairs of trees and their parameters
        """
        self.classifier = []

    def fit(self, forest, db, n_estimators=50):
        """
        :param: forest: a list of stamp trees, that can be used in algorithm
        :param: db:  a data base of processed data (instantion of Database)
        :param: n_estimators: number of estimators
        """
        self.classifier = []
        db_lenght = db.get_len()
        w = [1/db_lenght for _ in db.people]  # w: samples' weight

        for _ in range(n_estimators):
            weak = self.get_best_weak(db, forest)
            e = [w_i for i, w_i in enumerate(w) if weak.loss_function(db.people[i]) == 1]
            e = sum(e)
            alpha = self.get_alpha(e)

            self.classifier.append([weak, alpha])

            w = self.update_weights(w, alpha, weak, db)
            w = self.normalize_w(w)

            distribution = list(self.calculate_distribution(w))
            db = self.create_new_dataset(db, distribution)
            w = [1/db.get_len() for _ in w]  # reset weights

    def get_alpha(self, e):
        """ Calculate new alpha
        :param: e
        """
        try:
            alpha = float(0.5*ln((1-e)/e+0.0001))
        except ZeroDivisionError:
            alpha = 0
        return alpha

    def get_best_weak(self, db, forest):
        """ Find best weak classificator
        :param: db
        :param: forest
        returns: best tree-stump
        """
        err = []  # list of errors for all possible trees from forest
        for weak in forest:
            l, r = [], []
            for i in range(db.get_len()):
                if db.people[i].x[weak.attribute] <= weak.value:
                    l.append(db.people[i])
                else:
                    r.append(db.people[i])
            err.append(self.partition_test([l, r]))
        index_min = err.index(min(err))
        return forest[index_min]

    def partition_test(self, split):
        """ Calculate gini index for chosen split of data
        :param: split
        """
        gini = [1, 1]
        number_of_samples = [len(split[0]), len(split[1])]
        for count, _ in enumerate(split):
            probability = [0, 0]
            for person in _:
                if person.num == 0:
                    probability[0] += 1
                else:
                    probability[1] += 1
            if number_of_samples[count] != 0:
                gini[count] -= (probability[0]/number_of_samples[count])**2
                gini[count] -= (probability[1]/number_of_samples[count])**2
            else:
                gini[count] = 0

        return gini[0]*(number_of_samples[0]/sum(number_of_samples)) + \
            gini[1]*(number_of_samples[1]/sum(number_of_samples))

    def calculate_distribution(self, weights):
        """ Crate distribution based on current weights
        :params: weights
        """
        total = 0
        for w in weights:
            total += w
            yield total

    def draw_sample(self, distribution, point):
        """ Given distribution choose sample's index
        :param: distribution
        :param: point
        """
        index = -1
        for i in range(len(distribution) - 1):
            if (point > 0) & (point <= distribution[1]):
                index = distribution.index(distribution[1])
                return index
            elif (point > distribution[i]) & (point <= distribution[i + 1]):
                index = distribution.index(distribution[i + 1])
                return index

    def create_new_dataset(self, db, distribution):
        """ Create dataset of the same length based on distribution
        :param: db
        :param: distribution
        """
        indices = []
        for _ in range(len(db.get_y())):
            sample = self.draw_sample(distribution, random())
            indices.append(sample)
        return db.get_sub_db(indices)

    def update_w(self, previous_w, alpha, weak, sample):
        """
        :param: previous_w
        :param: alpha
        :param: weak
        :param: sample
        """
        return previous_w*(math.e**(alpha*weak.loss_function(sample)))

    def update_weights(self, w, alpha, weak, db):
        """
        :param: w: list of weights
        :param: alpha
        :param: weak: weak classificator- tree stump
        :db: data
        """
        return [self.update_w(w_i, alpha, weak, db.people[i]) for i, w_i in enumerate(w)]

    def normalize_w(self, w):
        """
        :param: w
        """
        w_sum = sum(w)
        for i in range(len(w)):
            w[i] = w[i]/w_sum
        return w

    def classify(self, sample):
        """ Makes Classification of given sample
        :param: sample
        """
        if not self.classifier:
            raise Exception("Ada was not fitted")
        class_0, class_1 = 0, 0
        for wk in self.classifier:
            if wk[0].classify(sample) == 0:
                class_0 += wk[1]
            else:
                class_1 += wk[1]
        if class_1 > class_0:
            return 1
        else:
            return 0

    def loss_function(self, sample):
        """ Counts value of loss function, 0 if classification  was correct
        :param: sample
        """
        if self.classify(sample) == sample.num:
            return 0
        return 1

    def test(self, test_db):
        """ Counts error for given test data set (Database)
        :param: test_db
        """
        err = 0
        for sample in test_db.people:
            err += self.loss_function(sample)
        return err/len(test_db.people)
