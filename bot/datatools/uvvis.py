from .base import FunctionValuedExperiment

class UVVisExperiemnt(FunctionValuedExperiment):
    def __init__(self, directory, iterations):
        super().__init__(directory, iterations)