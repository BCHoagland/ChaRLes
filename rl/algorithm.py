class Algorithm:
    def __init__(self):
        pass

    def __getattr__(self, k):
        return getattr(self.agent, k)
