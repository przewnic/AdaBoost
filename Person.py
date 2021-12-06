"""
Project: AdaBoost Implementation
Authors: przewnic
Date: 01.2021
"""


class Person():
    """ Represents person's atributes """
    def __init__(self, age, sex, cp,
                 trestbps, chol, fbs,
                 restecg, thalach, exang,
                 oldpeak, slope, ca, thal,
                 num):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.tresbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal
        self.num = num  # the predicted value

        self.x = self.get_atributes()  # atributes

    def get_atributes(self):
        return [self.age, self.sex, self.cp, self.tresbps, self.chol, self.fbs,
                self.restecg, self.thalach, self.exang, self.oldpeak,
                self.slope, self.ca, self.thal]

    def predicted_value(self):
        return self.num

    def __str__(self):
        return f"Atributes: {self.get_atributes()}, \
                 Predicted: {self.predicted_value()}"
