@app.service
class MyService(faust.Service):

    async def on_start(self) -> None:
        self.log.info('STARTED')

    async def on_stop(self) -> None:
        self.log.info('STOPPED')

    async def on_restart(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass