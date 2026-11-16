"""Engine-independent resource lifecycle. ML supplies the loader and inference bridge."""
from dataclasses import dataclass, field
import threading
from typing import Callable
from src.contracts.models import CallInput, CandidateBatch
from .adapters import InjectedAdapter, ModelResources, ModelDescriptor, PipelineError

@dataclass
class LoadedModel:
    inference: Callable[[CallInput, ModelResources], CandidateBatch]
    close: Callable[[],None] = field(default=lambda:None)

class ManagedAdapter(InjectedAdapter):
    def __init__(self, resources: ModelResources, loader: Callable[[ModelResources],LoadedModel], *, descriptor: ModelDescriptor,
                 revision=None, artifact_sha256=None, dtype=None, context=None, parameters=None, manifest_id=None):
        super().__init__(resources,self._infer,descriptor=descriptor)
        self.loader=loader
        self._lock=threading.Lock()
        self._closing=threading.Event()
        self._resource_thread=None
        self._session=None
        self._attempted=False
        self.public_info={'revision':revision,'hash':artifact_sha256,'dtype':dtype,'context':context,'parameters':parameters,'manifest_id':manifest_id}

    def load(self):
        with self._lock:
            if self._closing.is_set():raise PipelineError('MODEL_STOPPING',503,'loader')
            if self.state=='ready':return
            if self._attempted:raise PipelineError('MODEL_LOAD_ALREADY_ATTEMPTED',503,'loader')
            self._attempted=True;self.set_state('loading')
            session=None
            self._resource_thread=threading.get_ident()
            try:
                session=self.loader(self.resources)
                if not isinstance(session,LoadedModel) or not callable(session.inference) or not callable(session.close):
                    raise TypeError('loader must return LoadedModel')
                self._session=session;self.set_state('ready')
            except Exception as exc:
                self.set_state('failed')
                if isinstance(session,LoadedModel) and callable(session.close):
                    try:session.close()
                    except Exception as cleanup:
                        raise PipelineError('MODEL_LOAD_CLEANUP_FAILED',503,'loader',type(cleanup).__name__) from None
                raise PipelineError('MODEL_LOAD_FAILED',503,'loader',type(exc).__name__) from None
            finally:self._resource_thread=None

    @property
    def stopping(self):return self._closing.is_set()

    def _infer(self, call, resources):
        with self._lock:
            if self.stopping or self.state!='ready' or self._session is None:
                raise PipelineError('MODEL_NOT_READY',503,'extractor')
            self._resource_thread=threading.get_ident()
            try:return self._session.inference(call,resources)
            finally:self._resource_thread=None

    def close(self):
        if self._resource_thread==threading.get_ident():
            raise PipelineError('MODEL_CLOSE_FROM_RESOURCE_CALLBACK',503,'loader')
        self._closing.set()
        try:
            with self._lock:
                session,self._session=self._session,None
                if session is not None:
                    try:session.close()
                    except Exception as exc:
                        self.set_state('failed')
                        raise PipelineError('MODEL_CLOSE_FAILED',503,'loader',type(exc).__name__) from None
                self.set_state('unloaded')
        finally:self._closing.clear()
