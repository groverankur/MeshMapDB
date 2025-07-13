import inspect
import threading
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Tuple

Hook = Callable[..., None]
AsyncHook = Callable[..., Awaitable[None]]
EventHook = Callable[..., Any]  # can return None or an awaitable


class HookManager:
    """
    Manages synchronous and async hooks with priorities.
    """

    def __init__(self):
        self._hooks: Dict[str, List[Tuple[int, EventHook]]] = defaultdict(list)
        self._lock = threading.Lock()

    def register(self, event: str, hook: EventHook, priority: int = 0):
        """
        Register a hook under an event name. Lower priority runs first.
        """
        with self._lock:
            self._hooks[event].append((priority, hook))
            # sort once on insertion
            self._hooks[event].sort(key=lambda p_h: p_h[0])

    def unregister(self, event: str, hook: EventHook):
        """
        Remove a previously registered hook.
        """
        with self._lock:
            self._hooks[event] = [
                (p, h) for (p, h) in self._hooks[event] if h is not hook
            ]

    async def trigger(self, event: str, **kwargs: Any):
        """
        Trigger all hooks for an event. Synchronous hooks are called directly,
        async ones are awaited. Exceptions are caught and printed.
        """
        with self._lock:
            hooks = list(self._hooks.get(event, []))

        for _, hook in hooks:
            try:
                result = hook(**kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                # replace with proper logging
                print(f"Error in hook '{event}' ({hook}): {e}")
