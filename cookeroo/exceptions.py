class InvalidDirectoryStructureException(ValueError):
    def __init__(self, *args, **kwargs):
        super(ValueError, self).__init__(*args, **kwargs)
