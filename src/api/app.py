import asyncio
import importlib
import os
from contextlib import asynccontextmanager
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from src.api.config import Settings
from src.contracts.models import CallInput, ExtractResponse, Meta
from src.pipeline.adapters import MockExtractor, PendingRealAdapter, PipelineError, ModelDescriptor
from src.api.worker import InferenceWorker
from src.pipeline.service import Pipeline

logger = logging.getLogger('mws')
logging.basicConfig(level=logging.INFO, format='%(message)s')
ROOT = Path(__file__).resolve().parents[2]


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if pipeline is None and os.getenv('MWS_PIPELINE_FACTORY'):
        module,sep,name=os.environ['MWS_PIPELINE_FACTORY'].partition(':')
        if not sep:raise ValueError('MWS_PIPELINE_FACTORY must be module:function')
        pipeline=getattr(importlib.import_module(module),name)()
    pipeline = pipeline or Pipeline(MockExtractor() if settings.mode == 'mock' else PendingRealAdapter())
    descriptor = getattr(pipeline.extractor, 'descriptor', None)
    if descriptor is None:
        # Legacy test/custom extractors are permitted only as explicitly mock.
        if settings.mode != 'mock':
            raise ValueError('Real extractor requires ModelDescriptor')
        descriptor = ModelDescriptor('custom-mock', 'mock')
    if settings.mode != descriptor.mode:
        raise ValueError('Configured mode disagrees with extractor descriptor')
    worker = InferenceWorker(pipeline)

    @asynccontextmanager
    async def lifespan(app):
        adapter=pipeline.extractor
        if callable(getattr(adapter,'load',None)):
            try:await asyncio.to_thread(adapter.load)
            except PipelineError as exc:
                logger.error(json.dumps({'event':'model_load_failed','module':exc.module,'error_code':exc.code,'error_type':exc.error_type}))
        try:
            yield
        finally:
            await worker.close()
            if callable(getattr(adapter,'close',None)):
                try:await asyncio.to_thread(adapter.close)
                except PipelineError as exc:
                    logger.error(json.dumps({'event':'model_close_failed','module':exc.module,'error_code':exc.code,'error_type':exc.error_type}))

    app = FastAPI(title='MWS CompactFactExtraction', version='0.1.0', lifespan=lifespan)
    app.state.worker = worker

    def model_state():
        return getattr(pipeline.extractor, 'state', 'ready' if descriptor.mode == 'mock' else 'unloaded')

    @app.middleware('http')
    async def request_context(request: Request, call_next):
        request.state.request_id = str(uuid4())
        if request.method == 'POST':
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > settings.max_body_bytes:
                    return JSONResponse(status_code=413, content={'error': {'module': 'input', 'code': 'BODY_TOO_LARGE', 'request_id': request.state.request_id}})
            request._body = bytes(body)
        response = await call_next(request)
        response.headers['X-Request-ID'] = request.state.request_id
        return response

    def error(request, status, module, code, error_type=None):
        logger.warning(json.dumps({'request_id': request.state.request_id, 'module': module, 'error_code': code, 'error_type': error_type}))
        return JSONResponse(status_code=status, content={'error': {'module': module, 'code': code, 'request_id': request.state.request_id}})

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return error(request, 422, 'input', 'INVALID_REQUEST')

    @app.exception_handler(PipelineError)
    async def pipeline_error(request, exc):
        return error(request, exc.status, exc.module, exc.code, exc.error_type)

    @app.get('/health')
    def health():
        ready = model_state() == 'ready' and not worker.closing
        return JSONResponse(status_code=200 if ready else 503, content={'process_alive': True, 'model_state': model_state(), 'ready': ready, 'mode': descriptor.mode, 'is_mock': descriptor.mode == 'mock'})

    @app.get('/model/info')
    def info():
        return {'model_id': descriptor.model_id, 'mode': descriptor.mode, 'is_mock': descriptor.mode == 'mock', 'model_state': model_state(), 'artifact': None, **{k:getattr(pipeline.extractor,'public_info',{}).get(k) for k in ['revision','hash','parameters','dtype','context','manifest_id']}, 'engine': descriptor.engine, 'limits': {'body_bytes': settings.max_body_bytes, 'turns': 500, 'concurrency': 1, 'queue': 0}}

    @app.post('/extract', response_model=ExtractResponse)
    async def extract(call: CallInput, request: Request):
        if model_state() != 'ready':
            raise PipelineError('MODEL_NOT_READY', 503, 'extractor')
        started = perf_counter()
        try:
            prediction, warnings = await asyncio.wait_for(
                worker.run(call, request_id=request.state.request_id), timeout=settings.inference_timeout_sec)
        except TimeoutError:
            raise PipelineError('INFERENCE_TIMEOUT_SLOT_RETAINED',504,'worker') from None
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError('INFERENCE_FAILED', 503, 'pipeline', type(exc).__name__) from None
        elapsed = (perf_counter() - started) * 1000
        logger.info(json.dumps({'request_id': request.state.request_id, 'call_hash': hashlib.sha256(call.call_id.encode()).hexdigest()[:16], 'model_id': descriptor.model_id, 'facts': len(prediction.facts), 'elapsed_ms': elapsed}))
        return ExtractResponse(prediction=prediction, meta=Meta(request_id=request.state.request_id, model_id=descriptor.model_id, mode=descriptor.mode, elapsed_ms=elapsed, warnings=warnings, is_mock=descriptor.mode == 'mock'))

    app.mount('/static', StaticFiles(directory=ROOT / 'frontend'), name='static')

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(ROOT / 'frontend/index.html')
    return app

app = create_app()
