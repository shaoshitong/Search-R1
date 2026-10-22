"""Private-service experiment component; not installed in production.

E5 retains its original encode([query]) calls. Only CPU float32 vectors are
joined for a top3 FAISS search. Encoder and index stay on their owner event-loop
thread; this module creates no thread, service, actor, or alternate encoder.
"""
import asyncio
import math
import threading

import numpy as np


class SingletonFaissBatcher:
    def __init__(self, encoder, index, max_batch_size=16, max_wait_ms=2):
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError('max_batch_size must be a positive integer')
        if not math.isfinite(max_wait_ms) or max_wait_ms < 0:
            raise ValueError('max_wait_ms must be finite and nonnegative')
        self.encoder, self.index = encoder, index
        self.max_batch_size, self.max_wait_seconds = max_batch_size, max_wait_ms / 1000
        self.loop = asyncio.get_running_loop()
        self.owner_thread = threading.get_ident()
        self.queue = asyncio.Queue()
        self.closing = False
        self.worker = self.loop.create_task(self._run())

    def _owner(self):
        if threading.get_ident() != self.owner_thread or asyncio.get_running_loop() is not self.loop:
            raise RuntimeError('Encoder/FAISS batcher must stay on its owner event-loop thread')

    async def submit(self, query):
        """Return this request's ordered (scores, ids), or its batch exception."""
        self._owner()
        if self.closing:
            raise RuntimeError('Encoder/FAISS batcher is closed')
        result = self.loop.create_future()
        self.queue.put_nowait((query, result))
        return await result

    async def close(self):
        """Stop admitting requests, drain accepted requests, and await the worker."""
        self._owner()
        if not self.closing:
            self.closing = True
            self.queue.put_nowait(None)
        await asyncio.shield(self.worker)

    def _execute(self, batch):
        self._owner()
        batch = [(query, future) for query, future in batch if not future.cancelled()]
        if not batch:
            return
        try:
            vectors = np.concatenate([self.encoder.encode([query]) for query, _ in batch], axis=0)
            if vectors.dtype != np.float32 or vectors.ndim != 2 or len(vectors) != len(batch):
                raise ValueError('Original singleton E5 must return one CPU float32 vector per query')
            scores, ids = self.index.search(vectors, 3)
            if scores.shape != ids.shape or scores.shape != (len(batch), 3):
                raise ValueError('FAISS must return one ordered top3 row per query')
        except Exception as error:
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)
            return
        for row, (_, future) in enumerate(batch):
            if not future.done():
                future.set_result((scores[row].copy(), ids[row].copy()))

    async def _run(self):
        self._owner()
        while True:
            first = await self.queue.get()
            if first is None:
                return
            batch, stop = [first], False
            deadline = self.loop.time() + self.max_wait_seconds
            while len(batch) < self.max_batch_size:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - self.loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), remaining)
                    except asyncio.TimeoutError:
                        break
                if item is None:
                    stop = True
                    break
                batch.append(item)
            self._execute(batch)
            if stop:
                return
