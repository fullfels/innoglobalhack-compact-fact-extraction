"""Single physical inference slot, owned by the task, never by its HTTP waiter."""
import asyncio
from src.pipeline.adapters import PipelineError

class InferenceWorker:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.task = None
        self.closing = False

    async def run(self, call):
        if self.closing:
            raise PipelineError('WORKER_SHUTTING_DOWN', 503, 'worker')
        if self.task is not None and not self.task.done():
            raise PipelineError('WORKER_BUSY', 429, 'worker')
        # No await between checking the slot and assigning its owner.
        self.task = asyncio.create_task(asyncio.to_thread(self.pipeline.run, call))
        self.task.add_done_callback(self._consume_result)
        return await asyncio.shield(self.task)

    @staticmethod
    def _consume_result(task):
        # Detached HTTP waiters must not leave unobserved task exceptions.
        if not task.cancelled():
            task.exception()

    async def close(self):
        self.closing = True
        if self.task is not None:
            try:
                await asyncio.shield(self.task)
            except Exception:
                pass
        # Native inference must finish before model resources may be released.
