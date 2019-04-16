import faust

@app.service
class BinanceKlineIngress(faust.Service):

    # Called only the first time the service
    # is started.
    async def on_first_start(self) -> None:
        self.log.info('STARTED')

    # Called every time before the service
    # is started/restarted.
    async def on_start(self) -> None:
        self.log.info('STARTED')

    # Called every time after the service
    # is started/restarted.
    async def on_started(self) -> None:
        self.log.info('STARTED')

    # Called every time before the service 
    # is stopped/restarted.
    async def on_stop(self) -> None:
        self.log.info('STOPPED')

    # Called every time when the service 
    # is restarted.
    async def on_restart(self) -> None:
        pass

    # Called every time after the service 
    # is stopped/restarted
    async def on_shutdown(self) -> None:
        pass

    # Callback to be used to add service
    # dependencies.
    async def on_init_dependencies(self) -> None:
        pass