"""Synchronous NumPy inference for static TensorRT engines using CUDA buffers."""

from pathlib import Path

import numpy as np


class TensorRTEngine:
    def __init__(self, path):
        import tensorrt as trt
        from cuda.bindings import runtime as cuda

        self.cuda = cuda
        self.buffers = {}
        self.stream = None
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        self.inputs = []
        self.outputs = {}
        try:
            self.stream = self.check(cuda.cudaStreamCreate())[0]
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                shape = tuple(self.engine.get_tensor_shape(name))
                if any(size < 0 for size in shape):
                    raise ValueError("Only static TensorRT shapes are supported")
                host = np.empty(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
                pointer = self.check(cuda.cudaMalloc(host.nbytes))[0]
                self.buffers[name] = (host, pointer)
                self.context.set_tensor_address(name, int(pointer))
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append(name)
                else:
                    self.outputs[name] = host
        except Exception:
            self.close()
            raise

    def check(self, result):
        if result[0] != self.cuda.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA operation failed: {result[0]}")
        return result[1:]

    def __call__(self, inputs):
        if set(inputs) != set(self.inputs):
            raise ValueError(f"Expected TensorRT inputs {self.inputs}")
        cuda = self.cuda
        for name, value in inputs.items():
            host, pointer = self.buffers[name]
            value = np.ascontiguousarray(value, dtype=host.dtype)
            if value.shape != host.shape:
                raise ValueError(f"{name}: expected {host.shape}, got {value.shape}")
            self.check(cuda.cudaMemcpy(pointer, value.ctypes.data, value.nbytes,
                                       cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("TensorRT execution failed")
        self.check(cuda.cudaStreamSynchronize(self.stream))
        for name, host in self.outputs.items():
            self.check(cuda.cudaMemcpy(host.ctypes.data, self.buffers[name][1], host.nbytes,
                                       cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return self.outputs

    def close(self):
        for _, pointer in self.buffers.values():
            self.check(self.cuda.cudaFree(pointer))
        self.buffers.clear()
        if self.stream is not None:
            self.check(self.cuda.cudaStreamDestroy(self.stream))
            self.stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
