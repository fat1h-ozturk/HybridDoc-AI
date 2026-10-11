"""Optional cached CodeFormulaV2 worker; no imports of model packages in base usage."""
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path


def _worker(connection, cache_dir):
    sys.stdout = sys.stderr
    # Offline settings affect this isolated process only, never the caller.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_HUB_CACHE'] = str(Path(cache_dir) / 'hub')
    model = None
    try:
        while True:
            png = connection.recv()
            if png is None:
                return
            try:
                import io
                import torch
                from PIL import Image
                from docling_core.types.doc import DoclingDocument, DocItemLabel
                from docling.datamodel.base_models import ItemAndImageEnrichmentElement
                from docling.datamodel.pipeline_options import CodeFormulaVlmOptions
                from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
                from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
                from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel
                if model is None:
                    torch.set_num_threads(2)
                    options = CodeFormulaVlmOptions.from_preset('codeformulav2')
                    options.model_spec = options.model_spec.model_copy(update={
                        'revision': 'ecedbe111d15c2dc60bfd4a823cbe80127b58af4'})
                    options.engine_options = TransformersVlmEngineOptions(torch_dtype='float32')
                    CodeFormulaVlmModel.elements_batch_size = 1
                    model = CodeFormulaVlmModel(
                        enabled=True, enable_remote_services=False, artifacts_path=None,
                        options=options, accelerator_options=AcceleratorOptions(
                            device=AcceleratorDevice.CPU, num_threads=2))
                    if model.engine.vlm_model.dtype != torch.float32:
                        raise RuntimeError('formula model must use CPU float32')
                doc = DoclingDocument(name='local-formula')
                item = doc.add_text(label=DocItemLabel.FORMULA, text='')
                with Image.open(io.BytesIO(png)) as image:
                    with torch.inference_mode():
                        result = list(model(doc, [ItemAndImageEnrichmentElement(
                            item=item, image=image.convert('RGB'))]))
                connection.send(('ok', result[0].text))
            except Exception as exc:
                connection.send(('error', f'{type(exc).__name__}: {exc}'))
                return
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class LocalFormulaRecognizer:
    """Batch one, CPU float32, cached weights only; bounded wall time including load.

    Use as a context manager. After a failure the worker is disabled for the
    remainder of this document; subsequent crops remain explicitly unresolved.
    """
    def __init__(self, cache_dir=None, timeout_seconds=120, budget_seconds=480):
        for value in (timeout_seconds, budget_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('timeouts and budget must be positive finite seconds')
        self.cache_dir = str(Path(cache_dir).resolve()) if cache_dir else None
        self.timeout_seconds = timeout_seconds
        self.budget_seconds = budget_seconds
        self._process = None
        self._connection = None
        self._started = None
        self._failure = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._process.close()
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def recognize(self, png_bytes):
        now = time.monotonic()
        if self._started is None:
            self._started = now
        remaining = self.budget_seconds - (now - self._started)
        if remaining <= 0:
            raise TimeoutError('document recognition budget exhausted')
        if self._failure:
            raise RuntimeError(f'formula worker unavailable: {self._failure}')
        try:
            if self._process is None:
                context = multiprocessing.get_context('spawn')
                self._connection, child = context.Pipe()
                self._process = context.Process(target=_worker, args=(child, self.cache_dir))
                self._process.start()
                child.close()
            self._connection.send(png_bytes)
            remaining = self.budget_seconds - (time.monotonic() - self._started)
            if not self._connection.poll(max(0, min(self.timeout_seconds, remaining))):
                raise TimeoutError('formula crop timed out (includes model loading)')
            status, value = self._connection.recv()
            if status != 'ok':
                raise RuntimeError(value)
            return value
        except Exception as exc:
            self._failure = str(exc)
            self.close()
            raise
