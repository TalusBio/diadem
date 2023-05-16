"""Implements a processor class, which is a simple class
that can be used to process a list of items in parallel.

It is build passing it a list of functions and executes the
functions in order using the outputs of the previous function
as input for the next one.
"""

from dataclasses import dataclass


@dataclass
class Processor:
    functions = list[callable]

    def process(self, items):
        results = []
        for func in self.functions:
            if not results:
                result = [func(item) for item in items]
            else:
                result = [func(item) for item in results]
            results = result
        return results
