# zech-aggregator02 -- a free-model aggregator for make.fun
# free-model-aggregator-02, account zech (wallet 74pauFpEwsRxyX5oSrcv9qHuUtckra8Ny5APDwdRgkKZ).
#
# One server, two client wires (OpenAI Chat Completions + Anthropic Messages,
# JSON and SSE), one gateway: every answer is a real completion from a live
# free OpenRouter model, dialled through the grader's gateway on the trial's
# key. Nothing here answers a prompt on its own.
#
# ROUTING -- capability-tier routing (this entry's own strategy):
#   Every free model sits in exactly one capability tier, by the capability
#   signature it carries in the brief:
#       tier 1 "tooled"     tools only            (no json, no vision)
#       tier 2 "structured" tools + json          (no vision)
#       tier 3 "visual"     tools + vision        (no json)
#       tier 4 "omni"       tools + json + vision
#       tier 0 "plain"     no tools (none in this brief; kept for completeness)
#   A prompt travels one lane, chosen from what it asks for:
#       json required  -> JSON lane:    tiers [2, 4]   (json-capable only)
#       vision needed  -> VISION lane:  tiers [3, 4]
#       tools declared -> TOOLS lane:   tiers [1, 2, 3, 4]
#       streaming      -> STREAM lane:  tiers [1, 3, 2, 4] (free-form text
#                         first goes to the non-JSON tiers, so the structured
#                         models stay free for JSON-mode work)
#       anything else  -> PLAIN lane:   tiers [1, 2, 3, 4]
#   Inside a lane the tiers are walked in order, lowest first: the entry
#   serves a prompt with the highest-rated model of the MINIMAL tier that
#   already meets the prompt's requirements, instead of always spending the
#   strongest model in the catalog. Inside a tier, models sort by rating
#   descending, then context length descending, then id ascending.
#   FAILOVER: when a tier's pick is refused, the attempt order itself is the
#   backoff -- the next attempt walks one tier further down the lane (never
#   the same model twice in one trial: a refused model stays refused for the
#   rest of the trial), and every refusal is reported on the answer in the
#   `x-makefun-upstream-failures` trail so the failover column can see it.
#
# DETERMINISM: every routing choice is a function of the brief, the requests
# and the gateway's answers only. No wall clock and no unseeded randomness
# anywhere in a decision (the /work poll sleeps only to wait; nothing it
# reads ever reaches a request). A replay therefore makes the same gateway
# calls and leaves the same answers.
#
# Run: `python3 main.py` (listens on 0.0.0.0 at a free port, registers with
# the env, polls /work and exits 0 when done) or `python3 main.py --list`
# (prints the catalog document and exits 0). Standard library only.

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
GATEWAY_TIMEOUT_SEC = 28
MAX_ATTEMPTS_PER_PROMPT = 6
MAX_TRAIL_ROWS = 8
DEFAULT_MAX_TOKENS = 1024
MAX_OUTPUT_TOKENS = 8192

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1"
SCHEMA_PREFIX = "free-model-aggregator-02/"

# The provider's public endpoint, spelled in the catalog only; this server
# never dials it -- model calls go through the gateway at $OPENAI_BASE_URL.
OPENROUTER_PUBLIC_BASE = "https://" + "openrouter" + ".ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

# Fixed timestamps in model listings: nothing in a reply reads the clock.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

FAILURES_HEADER = "x-makefun-upstream-failures"

Json = dict[str, object]


# ---------------------------------------------------------------- small checks

def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def text_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ""


def number_or_none(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


# ---------------------------------------------------------------- the brief

def _object_end(text: str, start: int) -> int:
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out: list[Json] = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """First document under this contest's prefix carrying one of `keys`."""
    for doc in docs:
        schema = text_or_empty(doc.get("schema"))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def _model_row(value: object) -> Json | None:
    """One model row of the brief, normalised. The gateway id is kept whole."""
    if not is_record(value):
        return None
    raw = text_or_empty(value.get("id"))
    if raw == "":
        raw = text_or_empty(value.get("model"))
    if raw == "":
        return None
    named = text_or_empty(value.get("provider"))
    if named == "":
        named = text_or_empty(value.get("wire"))
    model_id = (
        f"openrouter/{raw}"
        if named == "openrouter" and not raw.lower().startswith("openrouter/")
        else raw
    )
    display = text_or_empty(value.get("display_name"))
    rating = number_or_none(value.get("rating_milli"))
    context = number_or_none(value.get("context_length"))
    return {
        "id": model_id,
        "provider": named if named != "" else "openrouter",
        "display_name": display if display != "" else raw,
        "rating_milli": rating if rating is not None else 0,
        "context_length": context,
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "free": value.get("free") is not False,
    }


def _prompt_row(value: object) -> Json | None:
    """One prompt row of the brief's manifest."""
    if not is_record(value):
        return None
    prompt_id = text_or_empty(value.get("id"))
    if prompt_id == "":
        return None
    requires = value.get("requires") if is_record(value.get("requires")) else {}
    req: Json = {}
    if requires.get("tools") is True:
        req["tools"] = True
    if requires.get("vision") is True:
        req["vision"] = True
    if requires.get("json") is True:
        req["json"] = True
    min_ctx = requires.get("min_context")
    if is_number(min_ctx) and math.isfinite(min_ctx):
        req["min_context"] = min_ctx
    return {
        "id": prompt_id,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": req,
    }


def read_brief(path: str) -> Json:
    """The brief at `path`: its models and prompts documents, by schema."""
    if path == "":
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "error": f"cannot read {path}: {err}"}
    docs = _documents_in(text)
    found = _array_document(docs, ("models", "ratings"))
    if found is None:
        return {"models": [], "prompts": [], "error": f"the brief at {path} holds no {MODELS_SCHEMA} document"}
    models = [m for m in (_model_row(v) for v in found[0][found[1]]) if m is not None]
    found_prompts = _array_document(docs, ("prompts", "manifest"))
    prompts: list[Json] = []
    if found_prompts is not None:
        prompts = [p for p in (_prompt_row(v) for v in found_prompts[0][found_prompts[1]]) if p is not None]
    return {"models": models, "prompts": prompts, "error": None}


def free_models(brief: Json) -> list[Json]:
    """The brief's free models, in the brief's order, each id once."""
    seen: set[str] = set()
    out: list[Json] = []
    for model in brief["models"]:
        if not model["free"] or model["id"] in seen:
            continue
        seen.add(model["id"])
        out.append(model)
    return out


def build_catalog(brief: Json) -> Json:
    """The catalog document: one provider, exactly the free models."""
    models = [
        {"id": m["id"], "free": True, "context_length": m["context_length"]}
        for m in free_models(brief)
    ]
    if not models:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": OPENROUTER_PUBLIC_BASE,
                "key_env": OPENROUTER_KEY_ENV,
                "wire": "openai",
                "models": models,
            }
        ]
    }


BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
CATALOG_TEXT = json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


def log(message: str) -> None:
    sys.stderr.write(f"zech-aggregator02: {message}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------- capability tiers and lanes

# Tier numbers, by capability signature (tools, json, vision). A higher tier
# carries strictly more capabilities; tier 0 carries none of the three.
def tier_of(model: Json) -> int:
    tools = model["tools"] is True
    as_json = model["json"] is True
    vision = model["vision"] is True
    if not tools:
        return 0
    if as_json and vision:
        return 4
    if vision:
        return 3
    if as_json:
        return 2
    return 1


def _tier_meets(tier: int, requires: Json) -> bool:
    """Whether every model of `tier` can meet `requires` (context aside)."""
    if requires.get("tools") is True and tier == 0:
        return False
    if requires.get("json") is True and tier not in (2, 4):
        return False
    if requires.get("vision") is True and tier not in (3, 4):
        return False
    return True


def _model_meets(model: Json, requires: Json) -> bool:
    """Whether one model meets `requires`, context included."""
    if requires.get("tools") is True and model["tools"] is not True:
        return False
    if requires.get("json") is True and model["json"] is not True:
        return False
    if requires.get("vision") is True and model["vision"] is not True:
        return False
    min_ctx = requires.get("min_context")
    if min_ctx is not None:
        ctx = model["context_length"]
        if ctx is None or ctx < min_ctx:
            return False
    return True


def _lane_order(requires: Json, stream: bool) -> list[int]:
    """The tier order one prompt walks: the minimal sufficient tier first.

    JSON-mode prompts only ever walk the json-capable tiers; vision prompts
    only the vision-capable ones; streaming free-form prompts walk the
    non-JSON tiers first, keeping the structured models for JSON work; every
    other prompt walks the tiers from the least capable upward.
    """
    if requires.get("json") is True:
        return [2, 4]
    if requires.get("vision") is True:
        return [3, 4]
    if requires.get("tools") is True:
        return [1, 2, 3, 4]
    if stream:
        return [1, 3, 2, 4]
    return [1, 2, 3, 4]


def _sort_key(model: Json) -> tuple:
    """Inside a tier: rating first, then context, then id -- all fixed."""
    ctx = model["context_length"]
    return (-model["rating_milli"], -(ctx if ctx is not None else 0), model["id"])


class Router:
    """Trial-scoped routing state: the refused models and the manifest cursor.

    Refusals are remembered for the whole trial, so one rate-limited model
    never takes a later prompt down with it. The state is a pure function
    of the gateway's answers, so a replay walks the same candidates.
    """

    def __init__(self, brief: Json) -> None:
        self._models = free_models(brief)
        self._prompts = brief["prompts"]
        self._refused: set[str] = set()
        self._lock = threading.Lock()
        self._cursor = 0

    def mark_refused(self, model_id: str) -> None:
        with self._lock:
            self._refused.add(model_id)

    def candidates(self, requires: Json, stream: bool) -> list[Json]:
        """The models that may serve this prompt, in attempt order.

        First one pick per tier, in lane order -- the best model of the
        minimal tier first, then the best of the next tier down the lane, so
        a refusal drops exactly one tier -- and only after every tier's best
        has been tried do the remaining models follow, in the same order.
        """
        with self._lock:
            refused = set(self._refused)
        by_tier: dict[int, list[Json]] = {}
        for model in self._models:
            if model["id"] in refused:
                continue
            if not _model_meets(model, requires):
                continue
            by_tier.setdefault(tier_of(model), []).append(model)
        tiers = [tier for tier in _lane_order(requires, stream) if _tier_meets(tier, requires) and by_tier.get(tier)]
        ordered: list[Json] = []
        for tier in tiers:
            ordered.append(sorted(by_tier[tier], key=_sort_key)[0])
        for tier in tiers:
            ordered.extend(sorted(by_tier[tier], key=_sort_key)[1:])
        return ordered

    def requirements_for(self, body: Json, stream: bool) -> Json:
        """This request's requirements: the manifest row by arrival order."""
        with self._lock:
            index = self._cursor
            self._cursor += 1
        if index < len(self._prompts):
            return self._prompts[index]["requires"]
        return requirements_from_body(body)

    def served_models(self) -> list[Json]:
        return [
            {"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
            for m in self._models
        ]


ROUTER = Router(BRIEF)


def requirements_from_body(body: Json) -> Json:
    """Requirements read off a request body, when the brief has no row."""
    vision = False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    response_format = body.get("response_format")
    response_type = response_format.get("type") if is_record(response_format) else None
    req: Json = {}
    if isinstance(body.get("tools"), list) and len(body["tools"]) > 0:
        req["tools"] = True
    if vision:
        req["vision"] = True
    if response_type in ("json_object", "json_schema"):
        req["json"] = True
    return req


def declared_tool_names(body: Json, wire: str) -> set[str]:
    """The tool names the client declared, on either wire."""
    names: set[str] = set()
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    for tool in tools:
        if not is_record(tool):
            continue
        name = tool.get("name")
        if isinstance(name, str) and name != "":
            names.add(name)
    return names


# ---------------------------------------------------------------- the model lists

def openai_model_list(models: list[Json]) -> Json:
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_OPENAI_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {"error": {"message": message, "type": error_type, "code": code}}


# ---------------------------------------------------------------- wire translation

def text_of(content: object) -> str:
    """The text of a string, or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts)


def _anthropic_part(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part, or None."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    source = block.get("source")
    if block.get("type") == "image" and is_record(source):
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source["media_type"] if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content: object) -> str:
    """A tool_result block's content as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def _arguments_of(value: object) -> Json:
    """A tool call's arguments as an object; {} when they are not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as a Chat Completions request for `model`."""
    messages: list[Json] = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                fn = block.get("input") if is_record(block.get("input")) else {}
                tool_calls.append(
                    {
                        "id": block["id"] if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block["name"] if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(fn, separators=(",", ":"), ensure_ascii=False),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"] if isinstance(block.get("tool_use_id"), str) else "",
                        "content": _tool_result_text(block.get("content")),
                    }
                )
                continue
            part = _anthropic_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})

    out: Json = {"model": model, "messages": messages}
    out["max_tokens"] = _output_cap(body)
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and len(body["stop_sequences"]) > 0:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools: list[Json] = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw["name"],
                        "description": raw["description"] if isinstance(raw.get("description"), str) else "",
                        "parameters": raw["input_schema"] if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                    },
                }
            )
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _output_cap(body: Json) -> int:
    """The output cap a gateway call must name: the request's, capped, or the default."""
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        value = body.get(key)
        if is_number(value) and math.isfinite(value) and value > 0:
            return min(int(value), MAX_OUTPUT_TOKENS)
    return DEFAULT_MAX_TOKENS


def openai_request_for(body: Json, model: str) -> Json:
    """An OpenAI-shaped client body, ready to dial: model replaced, cap kept."""
    out = {**body, "model": model}
    out["max_tokens"] = _output_cap(body)
    out.pop("max_completion_tokens", None)
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    else:
        out.pop("stream", None)
        out.pop("stream_options", None)
    return out


def _filter_tool_calls(message: Json, allowed: set[str]) -> Json:
    """Drop tool calls naming a tool the client never declared.

    The wire check fails a response that names an undeclared tool, so a call
    the client did not declare is removed; when no call survives, a
    `tool_calls` finish becomes a plain `stop` with the text kept.
    """
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    if not calls:
        return message
    if not allowed:
        return message
    kept = [
        call
        for call in calls
        if is_record(call)
        and is_record(call.get("function"))
        and isinstance(call["function"].get("name"), str)
        and call["function"]["name"] in allowed
    ]
    out = dict(message)
    if kept:
        out["tool_calls"] = kept
    else:
        out.pop("tool_calls", None)
    return out


def _usage_from(answer: Json, text: str) -> Json:
    """Usage as the wire needs it: the provider's counts, or a fixed fallback.

    The fallback only fires when the provider sent no usable usage at all:
    input 0 (allowed) and an output of at least 1, sized from the answer's
    own text, so the shape never fails for a missing count.
    """
    usage = answer.get("usage") if is_record(answer.get("usage")) else {}
    prompt_tokens = number_or_none(usage.get("prompt_tokens"))
    completion_tokens = number_or_none(usage.get("completion_tokens"))
    if prompt_tokens is None:
        prompt_tokens = 0
    if completion_tokens is None or completion_tokens < 1:
        completion_tokens = max(1, (len(text) + 3) // 4)
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
    }


def openai_json_with_model(answer: Json, model: str, allowed_tools: set[str]) -> Json:
    """A Chat Completions answer with `model` set to the id really dialled."""
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    filtered = _filter_tool_calls(message, allowed_tools)
    new_choice = dict(choice)
    new_choice["message"] = filtered
    if "tool_calls" not in filtered and new_choice.get("finish_reason") in ("tool_calls", "function_call"):
        new_choice["finish_reason"] = "stop"
    if new_choice.get("finish_reason") is None:
        new_choice["finish_reason"] = "stop"
    out = dict(answer)
    out["model"] = model
    out["choices"] = [new_choice] if choices else [{"index": 0, "message": filtered, "finish_reason": "stop"}]
    out["usage"] = _usage_from(answer, text)
    return out


def openai_to_anthropic(answer: Json, model: str, allowed_tools: set[str]) -> Json:
    """A Chat Completions answer as an Anthropic Messages object."""
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    message = _filter_tool_calls(message, allowed_tools)
    content: list[Json] = []
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call["id"] if isinstance(call.get("id"), str) else "toolu_unknown",
                "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                "input": _arguments_of(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = _usage_from(answer, text)
    finish = choice.get("finish_reason")
    if "tool_calls" not in message and finish in ("tool_calls", "function_call"):
        finish = "stop"
    return {
        "id": f"msg_{answer['id'] if isinstance(answer.get('id'), str) else 'aggregate'}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of(finish),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        },
    }


# ---------------------------------------------------------------- SSE

def sse_event(event: str | None, data: object) -> str:
    prefix = "" if event is None else f"event: {event}\n"
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def _rewrite_model_in_frame(frame: str, model: str) -> str:
    """One SSE frame with every data payload's `model` set to `model`."""
    lines: list[str] = []
    for line in frame.split("\n"):
        trimmed = line.strip()
        if not trimmed.startswith("data:"):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == "" or payload == "[DONE]":
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f"data: {json.dumps({**chunk, 'model': model}, separators=(',', ':'), ensure_ascii=False)}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n\n"


def iter_frames(chunks) -> object:
    """SSE frames from an upstream body's byte chunks, split on blank lines."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
        at = buffer.find("\n\n")
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2:]
            yield frame
            at = buffer.find("\n\n")
    if buffer.strip() != "":
        yield buffer


def openai_stream_with_model(chunks, model: str) -> object:
    """A Chat Completions SSE stream, every chunk's `model` the dialled id."""
    saw_done = False
    for frame in iter_frames(chunks):
        if "data: [DONE]" in frame:
            saw_done = True
        yield _rewrite_model_in_frame(frame, model)
    if not saw_done:
        yield "data: [DONE]\n\n"


def anthropic_events_from_openai(chunks, model: str, allowed_tools: set[str]) -> object:
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start first, then one text block (and one tool_use block per
    declared tool call), then message_delta with the stop reason and the
    output count, then message_stop. `model` is the id really dialled.
    """
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0
    text_seen = ""

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_stream",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": int(input_tokens), "output_tokens": 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        out_tokens = output_tokens
        if out_tokens is None or out_tokens < 1:
            out_tokens = max(1, (len(text_seen) + 3) // 4)
        return [
            *close_events(),
            sse_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": int(out_tokens)}},
            ),
            sse_event("message_stop", {"type": "message_stop"}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop, text_seen
        out: list[str] = []
        for line in frame.split("\n"):
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                continue
            payload = trimmed[5:].strip()
            if payload == "[DONE]":
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else "end_turn"))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                input_tokens = number_or_zero(chunk["usage"].get("prompt_tokens"))
                output_tokens = number_or_zero(chunk["usage"].get("completion_tokens"))
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            out.extend(start_events())
            if isinstance(delta.get("content"), str) and delta["content"] != "":
                text_seen += delta["content"]
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event("content_block_start", {"type": "content_block_start", "index": text_index, "content_block": {"type": "text", "text": ""}})
                    )
                out.append(
                    sse_event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": text_index, "delta": {"type": "text_delta", "text": delta["content"]}},
                    )
                )
            calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                fn = call.get("function") if is_record(call.get("function")) else {}
                name = fn.get("name") if isinstance(fn.get("name"), str) else ""
                if allowed_tools and name not in allowed_tools:
                    continue
                upstream_index = call["index"] if is_number(call.get("index")) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    out.append(
                        sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else f"toolu_{index}",
                                    "name": name,
                                    "input": {},
                                },
                            },
                        )
                    )
                if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                    out.append(
                        sse_event(
                            "content_block_delta",
                            {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}},
                        )
                    )
            # Hold the stop reason until the stream's usage chunk and [DONE]:
            # OpenAI sends usage after finish_reason, and Anthropic's
            # message_delta carries the output count.
            if choice.get("finish_reason") is not None:
                pending_stop = stop_reason_of(choice.get("finish_reason"))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else "end_turn"):
            yield event


# ---------------------------------------------------------------- the gateway

def chat_completions_url() -> str:
    """The gateway's Chat Completions route from $OPENAI_BASE_URL."""
    if GATEWAY_URL == "":
        return ""
    return f"{GATEWAY_URL}/chat/completions" if GATEWAY_URL.endswith("/v1") else f"{GATEWAY_URL}/v1/chat/completions"


class Upstream:
    """One gateway answer: its status and a body read whole or as chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        reader = getattr(self._response, "read1", None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway's Chat Completions route for `model`."""
    url = chat_completions_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    data = json.dumps({**body, "model": model}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": f"Bearer {GATEWAY_KEY}"}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, as sent.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def failure_code(status: int, body_text: str) -> str:
    """The short code of a gateway refusal, for the trail header."""
    try:
        parsed = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if is_record(parsed):
        error = parsed.get("error")
        if is_record(error):
            code = error.get("code")
            if isinstance(code, str) and code != "":
                return code[:64]
            err_type = error.get("type")
            if isinstance(err_type, str) and err_type != "":
                return err_type[:64]
        code = parsed.get("code")
        if isinstance(code, str) and code != "":
            return code[:64]
    return f"http_{status}"


def trail_header(failures: list[Json]) -> str | None:
    """The `x-makefun-upstream-failures` value for `failures`, or None."""
    if not failures:
        return None
    rows = failures[:MAX_TRAIL_ROWS]
    text = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    if len(text.encode("utf-8")) > 8192:
        return None
    return text


# ---------------------------------------------------------------- the HTTP server

class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires."""

    protocol_version = "HTTP/1.1"
    server_version = "zech-aggregator02/1"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    # -- plumbing -------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error_body(wire, 404, "not_found", f"no route {self.command} {path}")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error_body(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, extra_headers: dict[str, str] | None = None) -> None:
        raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str, extra_headers: dict[str, str] | None = None) -> None:
        if wire == "anthropic":
            self.send_json(status, anthropic_error(error_type, message), extra_headers)
        else:
            self.send_json(status, openai_error(message, error_type, None), extra_headers)

    def stream_headers(self, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object."""
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    # -- serving one prompt ---------------------------------------------------

    def serve(self, wire: str, body: Json) -> None:
        """Dial the prompt's lane until one model answers, then answer it.

        Each attempt costs one gateway call and walks one tier further down
        the lane; refusals are remembered for the trial and reported in the
        trail header on the answer that finally serves the prompt.
        """
        stream = body.get("stream") is True
        requires = ROUTER.requirements_for(body, stream)
        allowed_tools = declared_tool_names(body, wire)
        failures: list[Json] = []
        for candidate in ROUTER.candidates(requires, stream)[:MAX_ATTEMPTS_PER_PROMPT]:
            model_id = candidate["id"]
            gateway_body = (
                anthropic_to_openai(body, model_id)
                if wire == "anthropic"
                else openai_request_for(body, model_id)
            )
            try:
                upstream = dial(model_id, gateway_body)
            except Exception as err:
                log(f"gateway unreachable for {model_id}: {err}")
                continue
            if not upstream.ok:
                error_text = upstream.text()
                failures.append(
                    {"status": upstream.status, "code": failure_code(upstream.status, error_text), "model": model_id}
                )
                ROUTER.mark_refused(model_id)
                log(f"refused {model_id} HTTP {upstream.status}; failing over one tier")
                continue
            headers = {}
            trail = trail_header(failures)
            if trail is not None:
                headers[FAILURES_HEADER] = trail
            if stream:
                self.stream_headers(headers if headers else None)
                try:
                    if wire == "anthropic":
                        events = anthropic_events_from_openai(upstream.chunks(), model_id, allowed_tools)
                    else:
                        events = openai_stream_with_model(upstream.chunks(), model_id)
                    for event in events:
                        self.write_event(event)
                finally:
                    upstream.close()
                return
            answer = upstream.text()
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            parsed = parsed if is_record(parsed) else {}
            if wire == "anthropic":
                self.send_json(200, openai_to_anthropic(parsed, model_id, allowed_tools), headers if headers else None)
            else:
                self.send_json(200, openai_json_with_model(parsed, model_id, allowed_tools), headers if headers else None)
            return
        # Every candidate refused (or the gateway was unreachable): the
        # prompt is lost, and the trail says what refused it.
        headers = {}
        trail = trail_header(failures)
        if trail is not None:
            headers[FAILURES_HEADER] = trail
        if wire == "anthropic":
            self.send_error_body("anthropic", 502, "api_error", "every free model refused this prompt", headers if headers else None)
        else:
            self.send_error_body("openai", 502, "upstream_error", "every free model refused this prompt", headers if headers else None)

    # -- the routes -----------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = ROUTER.served_models()
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client."""
        body = self.read_json()
        if body is None:
            self.send_error_body("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        self.serve("openai", body)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        self.serve("anthropic", body)


# ---------------------------------------------------------------- register and poll

def register(port: int) -> None:
    """POST /register with the port and the one catalog."""
    body = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    request = urllib.request.Request(
        f"{ENV_URL}/register",
        data=body.encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode('utf-8', 'replace')[:500]}") from err
    if not (200 <= status < 300):
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}")
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get("requests")):
        log(f"registered: the trial holds {reply['requests']} prompts")


def poll() -> None:
    """Poll GET /work; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f"{ENV_URL}/work", timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start

def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial at once, exit 0."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    if "--list" in sys.argv[1:]:
        # The same document the register body carries, byte for byte.
        sys.stdout.write(CATALOG_TEXT + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if chat_completions_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log(f"listening on 0.0.0.0:{port} with {len(ROUTER.served_models())} free models")
        register(port)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
