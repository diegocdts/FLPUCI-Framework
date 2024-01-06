from enum import Enum


class LearningApproach(Enum):
    CEN = 'Centralized'
    FED = 'FL-based'

    def __str__(self):
        return str(self.value)


class WindowStrategy(Enum):
    ACC = 'ACC'
    SLI = 'SLI'

    def __str__(self):
        return str(self.value)
