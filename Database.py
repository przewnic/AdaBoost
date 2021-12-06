"""
Project: AdaBoost Implementation
Authors: przewnic
Date: 01.2021
"""
# Implementation of class helping reading and manipulating
# and checking data
import csv
from Person import Person
import re


class MalformedData(Exception):
    def __init__(self, msg, row=None):
        super().__init__(msg)
        self.row = row


class Database():
    """ Database for reading and processing data from file """
    def __init__(self, people=[]):
        self.people = people
        self.class_values = []

    def load_from_file(self, path):
        """
        :para: path
        """
        with open(path, newline="") as file:
            """ fieldnames=['age', 'sex', 'cp', 'trestbps',
                            'chol', 'fbs', 'restecg', 'thalach',
                            'exang', 'oldpeak', 'slope', 'ca',
                            'thal', 'num']
            """
            reader = csv.reader(file)
            self.people = self.form_data(reader)
            self.class_values = self.get_class_values()

    def form_data(self, reader):
        """
        :param: reader: iterable
        """
        result = []
        regnumber = re.compile(r'\?')
        try:
            for row in reader:
                if regnumber.search(' '.join(row)) is None:
                    for _ in range(14):
                        row[_] = float(row[_])
                    if row[13] > 1.0:
                        row[13] = 1.0
                    result.append(
                            Person(
                                row[0], row[1], row[2],
                                row[3], row[4], row[5],
                                row[6], row[7], row[8],
                                row[9], row[10], row[11],
                                row[12], row[13]
                                )
                                )
            return result
        except IndexError:
            raise MalformedData(f"Missing column in file", row)

    def get_class_values(self):
        values = set()
        for person in self.people:
            values.add(person.num)
        return list(values)

    def get_len(self):
        return len(self.people)

    def get_X(self):
        X = []
        for person in self.people:
            X.append(person.get_atributes())
        return X

    def get_y(self):
        y = []
        for person in self.people:
            y.append(person.predicted_value())
        return y

    def sort(self, attribute):
        """
        :param: attribute: sorting based on index of attribute
        """
        self.people.sort(key=lambda person: person.x[attribute])

    def get_sub_db(self, indices):
        """
        Creates a subset of database for testing and learning
        :param: indices: a list of indices
        """
        sub_people = []
        for idx in indices:
            sub_people.append(self.people[idx])
        return Database(sub_people)

    def __str__(self):
        people = f""
        for person in self.people:
            people += f"{person}\n"
        return people


if __name__ == "__main__":
    try:
        db = Database()
        db.load_from_file("data/processed.cleveland.data")
        print(db)
        db.sort(0)
        print(db)
    except MalformedData as e:
        print(
            f"Error: {e}\n"
            f"Row: {e.row}"
        )
