#!/usr/bin/env python3
"""
harness/telemetry/events.py -- Structured Event Dispatcher & Observability Pipeline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    TURN_START = "turn_start"
    SCHEMA_LINKED = "schema_linked"
    VALUE_GROUNDED = "value_grounded"
    SQL_GENERATED = "sql_generated"
    QUERY_EXECUTED = "query_executed"
    REPAIR_ATTEMPTED = "repair_attempted"
    TURN_END = "turn_end"


@dataclass
class LatencyBreakdown:
    schema_linking_ms: float = 0.0
    value_grounding_ms: float = 0.0
    memory_recall_ms: float = 0.0
    generation_ms: float = 0.0
    execution_ms: float = 0.0
    repair_ms: float = 0.0
    total_turn_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "schema_linking_ms": round(self.schema_linking_ms, 2),
            "value_grounding_ms": round(self.value_grounding_ms, 2),
            "memory_recall_ms": round(self.memory_recall_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "execution_ms": round(self.execution_ms, 2),
            "repair_ms": round(self.repair_ms, 2),
            "total_turn_ms": round(self.total_turn_ms, 2),
        }


@dataclass
class AgentEvent:
    event_type: EventType
    session_id: str
    db_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


EventListener = Callable[[AgentEvent], None]


class TelemetryDispatcher:
    """
    Pub/Sub event dispatcher for monitoring, tracing, and benchmarking agent executions.
    """

    def __init__(self, verbose: bool = False):
        self.listeners: List[EventListener] = []
        self.event_log: List[AgentEvent] = []
        self.verbose = verbose

    def subscribe(self, listener: EventListener) -> None:
        self.listeners.append(listener)

    def dispatch(self, event_type: EventType, session_id: str, db_name: str, **kwargs) -> AgentEvent:
        ev = AgentEvent(
            event_type=event_type,
            session_id=session_id,
            db_name=db_name,
            data=kwargs,
        )
        self.event_log.append(ev)

        if self.verbose:
            print(f"[Telemetry] {event_type.value.upper()} | session={session_id} | db={db_name} | {kwargs}")

        for listener in self.listeners:
            try:
                listener(ev)
            except Exception:
                pass

        return ev

    def clear(self) -> None:
        self.event_log.clear()
