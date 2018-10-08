class CoarsenError(Exception):
    def __init__(self):
        super().__init__(self)

    def __str__(self):
        return 'Error: the number of connected components must be smaller than the threshold'
