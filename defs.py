import numpy as np

class Person:
    'A person is a collection of 7 emotions and id'

    def __init__(self, id):
        self.id = id
        self.expressions = []
        self.data = []