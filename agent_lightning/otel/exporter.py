"""
OTel compatibility — export Agent Lightning spans as OpenTelemetry traces.

Lets you plug into Jaeger, Datadog, Honeycomb, Grafana Tempo, etc.
with zero changes to your agent code.
"""
from __future__ import annotations
import json, logging
from typing import Any, Dict, List, Optional
from ..core.models import Span, StepType

log = logging.getLogger(__name__)


class OTelSpanAdapter:
    """
    Convert Agent Lightning Span → OpenTelemetry-compatible dict.
    Can be sent to any OTel collector endpoint.
    """

    # Map our StepType → OTel span kind
    KIND_MAP = {
        StepType.LLM_RESPONSE: "CLIENT",
        StepType.TOOL_CALL:    "CLIENT",
        StepType.TOOL_RESULT:  "SERVER",
        StepType.PROMPT:       "INTERNAL",
        StepType.REWARD:       "INTERNAL",
    }

    @classmethod
    def to_otel(cls, span: Span) -> Dict[str, Any]:
        """Convert to OTLP-compatible span dict."""
        kind = cls.KIND_MAP.get(span.step_type, "INTERNAL") if span.step_type else "INTERNAL"

        attributes = {
            **span.attributes,
            "al.step_type":    span.step_type.value if span.step_type else "unknown",
            "al.rollout_id":   span.rollout_id or "",
            "al.attempt_id":   span.attempt_id or "",
            "al.sequence_id":  span.sequence_id or 0,
        }
        if span.model:          attributes["llm.model"]            = span.model
        if span.input_tokens:   attributes["llm.input_tokens"]     = span.input_tokens
        if span.output_tokens:  attributes["llm.output_tokens"]    = span.output_tokens
        if span.latency_ms:     attributes["al.latency_ms"]        = span.latency_ms
        if span.tool_name:      attributes["tool.name"]            = span.tool_name
        if span.reward:         attributes["al.reward"]            = span.reward.value
        if span.content and isinstance(span.content, str):
            attributes["al.content_preview"] = span.content[:200]

        return {
            "traceId":  span.trace_id.replace("-", ""),
            "spanId":   span.span_id.replace("-", "")[:16],
            "parentSpanId": span.parent_id.replace("-","")[:16] if span.parent_id else None,
            "name":     span.name,
            "kind":     kind,
            "startTimeUnixNano": int(span.start_time.timestamp() * 1e9),
            "endTimeUnixNano":   int(span.end_time.timestamp() * 1e9) if span.end_time else None,
            "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in attributes.items()],
            "status":   {"code": 2 if span.status == "ERROR" else 1},
        }

    @classmethod
    def to_otel_batch(cls, spans: List[Span]) -> Dict[str, Any]:
        """Format spans as OTLP batch export payload."""
        return {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "agent-lightning"}},
                        {"key": "service.version", "value": {"stringValue": "0.2.0"}},
                    ]
                },
                "scopeSpans": [{
                    "scope": {"name": "agent_lightning.tracer", "version": "0.2.0"},
                    "spans": [cls.to_otel(s) for s in spans],
                }]
            }]
        }


class OTelExporter:
    """
    Export spans to any OTLP-compatible collector (Jaeger, Datadog, etc.)

    Example:
        exporter = OTelExporter(endpoint="http://jaeger:4318/v1/traces")
        await exporter.export(spans)
    """

    def __init__(self, endpoint: str = "http://localhost:4318/v1/traces",
                 headers: Dict[str, str] = None):
        self.endpoint = endpoint
        self.headers  = headers or {}

    async def export(self, spans: List[Span]) -> bool:
        """Send spans to OTel collector."""
        if not spans:
            return True
        try:
            import aiohttp
        except ImportError:
            log.warning("aiohttp not installed — OTel export skipped")
            return False

        payload = OTelSpanAdapter.to_otel_batch(spans)
        headers = {"Content-Type": "application/json", **self.headers}

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                    if resp.status not in (200, 204):
                        log.warning("OTel export failed: HTTP %d", resp.status)
                        return False
            log.debug("Exported %d spans to %s", len(spans), self.endpoint)
            return True
        except Exception as e:
            log.warning("OTel export error: %s", e)
            return False

    def export_to_file(self, spans: List[Span], path: str) -> int:
        """Write spans to JSONL file (for offline analysis)."""
        import json
        with open(path, "a") as f:
            for span in spans:
                f.write(json.dumps(OTelSpanAdapter.to_otel(span)) + "\n")
        return len(spans)
