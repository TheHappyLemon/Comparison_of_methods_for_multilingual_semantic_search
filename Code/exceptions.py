class EmbeddingsError(Exception):
    def __init__(self, message, indexes):            
        super().__init__(message)
        self.indexes = indexes