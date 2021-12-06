"""
Project: AdaBoost Implementation
Authors: przewnic
Date: 01.2021
"""
# Implemenation of class helping choose best decision trees
# for all atributes of data and values that are used for making decisions
import statistics
from Tree import Tree


class Forest():

    def partition(self, data, attribute, value):
        """ Splits data
        :param: data
        :param: attribute
        :param: values
        """
        left = [person for person in data if person.x[attribute] <= value]
        right = [person for person in data if person.x[attribute] > value]
        return left, right

    def split(self, data, attribute, values):
        """ Split data on given attribute
        :param: data
        :param: attribute
        :param: values
        """
        split = []
        values = list(values)
        l, r = None, data
        for value in values:
            l, r = self.partition(r, attribute, value)
            split.append(l)
        split.append(r)
        return split

    def partition_test(self, split, data, attribute):
        """ Counts gini index for given split
        :param: split
        :param: data
        :param: attribute
        :param: attributes
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
            gini[count] -= (probability[0]/number_of_samples[count])**2
            gini[count] -= (probability[1]/number_of_samples[count])**2

        return gini[0]*(number_of_samples[0]/sum(number_of_samples)) + \
            gini[1]*(number_of_samples[1]/sum(number_of_samples))

    def get_splitting_values(self, data, attribute):
        """ Creates list of values that can
        split given data on given attribute
        :param: data
        :param: attribute
        """
        values = list(set([person.x[attribute] for person in data]))
        values.sort()
        chcecked_values = []
        for i in range(len(values)-1):
            chcecked_values.append(statistics.mean([values[i], values[i+1]]))
        return chcecked_values

    def create_forest(self, data):
        """ Creates list of best weak learners
        based on given data, one for each attribute
        :param: data
        """
        forest = []
        for attribute in range(len(data.people[0].x)):
            splitting_values = self.get_splitting_values(data.people,
                                                         attribute)
            result = []
            for split_value in splitting_values:
                s = self.split(data.people, attribute, [split_value])
                split_gini = self.partition_test(s, data.people, attribute)
                result.append([split_value, split_gini])
            value = min(result, key=lambda i: i[1])
            forest.append(Tree(attribute, value[0]))
        return forest
