from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable
from contextlib import asynccontextmanager

from agent_system.environments.scienceworld.envs.scienceworld_env import ScienceWorldEnvAdapter


AdapterFactory = Callable[[], ScienceWorldEnvAdapter]


class ScienceWorldEnvPool:
    def __init__(
        self,
        *,
        size: int = 1,
        adapter_factory: AdapterFactory | None = None,
        env_step_limit: int = 100,
    ) -> None:
        if int(size) <= 0:
            raise ValueError("ScienceWorld pool size must be positive")
        self.size = int(size)
        self._adapter_factory = adapter_factory or (
            lambda: ScienceWorldEnvAdapter(env_step_limit=env_step_limit)
        )
        self._queue: asyncio.Queue[ScienceWorldEnvAdapter] | None = None
        self._adapters: list[ScienceWorldEnvAdapter] = []
        self._initialize_lock = asyncio.Lock()
        self._closed = False
        self._atexit_registered = False

    async def _ensure_initialized(self) -> None:
        if self._closed:
            raise RuntimeError("ScienceWorld pool is closed")
        if self._queue is not None:
            return
        async with self._initialize_lock:
            if self._queue is not None:
                return
            results = await asyncio.gather(
                *[
                    asyncio.to_thread(self._adapter_factory)
                    for _ in range(self.size)
                ],
                return_exceptions=True,
            )
            created = [
                result
                for result in results
                if isinstance(result, ScienceWorldEnvAdapter)
            ]
            failures = [
                result
                for result in results
                if isinstance(result, BaseException)
            ]
            if failures:
                await asyncio.gather(
                    *[asyncio.to_thread(adapter.close) for adapter in created],
                    return_exceptions=True,
                )
                raise failures[0]
            queue: asyncio.Queue[ScienceWorldEnvAdapter] = asyncio.Queue(
                maxsize=self.size
            )
            for adapter in created:
                queue.put_nowait(adapter)
            self._adapters = created
            self._queue = queue
            if not self._atexit_registered:
                atexit.register(self._close_at_exit)
                self._atexit_registered = True

    def _close_at_exit(self) -> None:
        for adapter in self._adapters:
            try:
                adapter.close()
            except Exception:
                pass

    @asynccontextmanager
    async def lease(self):
        await self._ensure_initialized()
        if self._closed or self._queue is None:
            raise RuntimeError("ScienceWorld pool is closed")
        adapter = await self._queue.get()
        try:
            yield adapter
        finally:
            if self._closed:
                await asyncio.to_thread(adapter.close)
            else:
                self._queue.put_nowait(adapter)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        adapters = list(self._adapters)
        await asyncio.gather(
            *[asyncio.to_thread(adapter.close) for adapter in adapters],
            return_exceptions=False,
        )
        self._adapters.clear()
        self._queue = None


__all__ = ["ScienceWorldEnvPool"]
