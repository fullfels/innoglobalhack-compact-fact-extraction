"""Single physical inference slot, owned by the task, never by its HTTP waiter."""
import asyncio
import json
import logging
from uuid import uuid4
from src.pipeline.adapters import PipelineError

logger = logging.getLogger('mws.worker')

class InferenceWorker:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.task = None
        self.closing = False

    async def run(self, call, *, request_id=None):
        if self.closing:
            raise PipelineError('WORKER_SHUTTING_DOWN', 503, 'worker')
        if self.task is not None and not self.task.done():
            raise PipelineError('WORKER_BUSY', 429, 'worker')
        correlation_id = request_id or str(uuid4())
        detached = False
        logged = False

        def consume_result(task):
            nonlocal logged
            if task.cancelled():
                return
            exc = task.exception()  # Always observe detached exceptions.
            if exc is not None and detached and not logged:
                logged = True
                logger.error(json.dumps({
                    'event': 'detached_inference_failed',
                    'request_id': correlation_id,
                    'module': exc.module if isinstance(exc, PipelineError) else 'worker',
                    'error_code': exc.code if isinstance(exc, PipelineError) else 'INFERENCE_FAILED',
                    'error_type': (exc.error_type or type(exc).__name__) if isinstance(exc, PipelineError) else type(exc).__name__,
                }))

        # No await between checking the slot and assigning its owner.
        task = asyncio.create_task(asyncio.to_thread(self.pipeline.run, call))
        self.task = task
        task.add_done_callback(consume_result)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            detached = True
            # Completion callback may already have run while the waiter was active.
            if task.done():
                consume_result(task)
            raise

    async def close(self):
        self.closing = True
        if self.task is not None:
            try:
                await asyncio.shield(self.task)
            except Exception:
                pass
        # Native inference must finish before model resources may be released.
