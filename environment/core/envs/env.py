

class Env():
    def __init__(
        self, 
        buffer
    ):
        self.buffer=buffer

    
    async def setup(self):
        raise NotImplementedError