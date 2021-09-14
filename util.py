class Config(dict):
    def __init__(self, *args, **kwargs): 
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value
