"""
Project: AdaBoost Implementation
Authors: przewnic
Date: 01.2021
"""
# Implementation of simple Decision Tree-Stamp


class Tree():
    def __init__(self, attribute, value):
        """
        :param: attribute: which attribute is considered-index
        :param: value: directs which node to choose next
        """
        self.attribute = attribute
        self.value = value

    def classify(self, sample):
        """ Classifies sample's class. """
        if sample.x[self.attribute] <= self.value:
            return 0.0
        return 1.0

    def loss_function(self, sample):
        """ Counts value of loss function, -1 if classification was correct."""
        if self.classify(sample) == sample.num:
            return -1
        return 1

    def __str__(self):
        return f"Atribute: {self.attribute} Value: {self.value}"
