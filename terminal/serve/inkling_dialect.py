"""Inkling-2-Mini's wire dialect, translated to Strata's OutputParser / JsonOutputParser form.

Inkling-2-Mini renders its assistant output with explicit channel tokens:
    <|message_model|><|content_thinking|> ... <|end_message|>
    <|message_model|><|content_text|> ... <|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_text|>...<|end_message|>
    <|content_model_end_sampling|>

`InklingDialect.feed()` streams these tokens into:
    <think>...</think>
    plain text
    <tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>

Using JSON inside `<tool_call>...</tool_call>` (read by InklingOutputParser -> JsonOutputParser) preserves
arbitrary XML special characters (`<`, `>`, `&`, `</parameter>`), nested quotes, booleans/numbers/arrays,
and multiline code losslessly while reusing all downstream OpenAI / Anthropic / Responses / MCP streaming!
"""
from __future__ import annotations

import json
import re

MSG_MODEL = "<|message_model|>"
CONTENT_THINK = "<|content_thinking|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_TOOL_TEXT = "<|content_invoke_tool_text|>"
CONTENT_XML = "<|content_xml|>"
END_MSG = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
ENDOFTEXT = "<|endoftext|>"

_HEADER_TAGS = (CONTENT_THINK, CONTENT_TEXT, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, CONTENT_XML, END_MSG, END_SAMPLING, ENDOFTEXT, MSG_MODEL)
_IDLE_TAGS = (MSG_MODEL, CONTENT_THINK, CONTENT_TEXT, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, CONTENT_XML, END_MSG, END_SAMPLING, ENDOFTEXT)
_BODY_TAGS = (END_MSG, END_SAMPLING, ENDOFTEXT, MSG_MODEL)
_MAX_TAG = max(len(t) for t in _HEADER_TAGS)
# the form the chat template writes a call in, up to where its args object opens: {"name":"NAME","args":
_CANON_CALL = re.compile(r'\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"args"\s*:\s*(?=\{)')
_CANON_GIVE_UP = 512                 # a body this long that still is not that form is held whole (the older path)


def _safe_json_call(name: str, arguments) -> str:
    """Serialize a tool call as `<tool_call>\\n{...}\\n</tool_call>` with `</` escaped as `<\\/` so literal `</tool_call>` in args never breaks framing."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False).replace("</", "<\\/")
    return f"<tool_call>\n{payload}\n</tool_call>"


def _parse_tool_body(header_name: str, body: str, kind: str) -> str:
    raw = body.strip()
    fallback_name = header_name.strip()
    if kind == "tool_json":
        try:
            obj = json.loads(raw)
        except ValueError:
            # Recover common trailing garbage / unclosed brace if cut
            obj = None
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except ValueError:
                    obj = None
        if isinstance(obj, dict):
            name = str(obj.get("name") or fallback_name).strip()
            args = obj.get("args") if "args" in obj else obj.get("arguments")
            if args is None:
                args = {k: v for k, v in obj.items() if k != "name"} if "name" in obj else {}
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict):
                        args = parsed
                except ValueError:
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"value": args}
            if name:
                return _safe_json_call(name, args)
        if fallback_name:
            return _safe_json_call(fallback_name, {"raw": raw} if raw else {})
        return raw
    # tool_text / content_xml: check if it is JSON or XML `<function=...>` / raw text
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                name = str(obj.get("name") or fallback_name).strip()
                args = obj.get("args") if "args" in obj else obj.get("arguments", {})
                if name and isinstance(args, dict):
                    return _safe_json_call(name, args)
        except ValueError:
            pass
    if "<function=" in raw:
        return f"<tool_call>\n{raw}\n</tool_call>"
    if "<tool_call" in raw or "<invoke" in raw:
        m_call = re.search(r'<(?:tool_call|invoke)\b[^>]*\bname=["\']([^"\']+)["\'][^>]*>(.*?)</(?:tool_call|invoke)>', raw, re.S)
        if m_call:
            xname = m_call.group(1).strip()
            xbody = m_call.group(2)
            xargs = {
                pm.group(1).strip(): pm.group(2)
                for pm in re.finditer(r'<(?:param|parameter)\b[^>]*\bname=["\']([^"\']+)["\'][^>]*>(.*?)</(?:param|parameter)>', xbody, re.S)
            }
            if xname:
                return _safe_json_call(xname, xargs)
    if fallback_name:
        return _safe_json_call(fallback_name, {"input": raw} if raw else {})
    return raw


class InklingDialect:
    """Stateful streaming translator: one instance per assistant reply."""

    def __init__(self):
        self.state = "idle"          # idle | header | think | text | tool_json | tool_text
        self.buf = ""
        self.header_acc = ""
        self.tool_acc = ""
        self.calls = 0
        self._in_think = False
        # Streamed calls (the parser turns these on): a call in the canonical `{"name":NAME,"args":{...}}` form is reported while it is being
        # written - ("tool_start", NAME), then ("tool_args", piece) for the text of the args object - and closed by ("tool_end", frame).
        # Off (the default), or for any other form, the whole call is held and given as one `<tool_call>` frame, as before.
        self.stream = False
        self.allowed = None          # callable(name) -> bool: the names a call may be streamed for (the request's declared tools)
        self._reset_stream()

    def _reset_stream(self) -> None:
        self._s_mode = None          # None: not decided yet | "on": streaming | "off": held (not the canonical form)
        self._s_pos = 0              # the next character of tool_acc the args scanner reads
        self._s_depth = 0
        self._s_str = False
        self._s_esc = False
        self._s_done = False         # the args object is closed

    def _stream_tool(self, out: list[tuple[str, str]]) -> None:
        """Report the call being written in tool_acc as far as it has arrived (see `stream`)."""
        if not self.stream or self.state != "tool_json" or self._s_mode == "off" or self._s_done:
            return
        if self._s_mode is None:
            m = _CANON_CALL.match(self.tool_acc)
            if not m:
                if len(self.tool_acc) > _CANON_GIVE_UP:
                    self._s_mode = "off"
                return
            try:
                name = json.loads('"' + m.group(1) + '"')
            except ValueError:
                self._s_mode = "off"
                return
            if not name or (self.allowed is not None and not self.allowed(name)):
                self._s_mode = "off"
                return
            self._s_mode, self._s_pos = "on", m.end()
            out.append(("tool_start", name))
        acc, i, n = self.tool_acc, self._s_pos, len(self.tool_acc)
        start = i
        while i < n:
            ch = acc[i]
            i += 1
            if self._s_str:
                if self._s_esc:
                    self._s_esc = False
                elif ch == "\\":
                    self._s_esc = True
                elif ch == '"':
                    self._s_str = False
            elif ch == '"':
                self._s_str = True
            elif ch == "{":
                self._s_depth += 1
            elif ch == "}":
                self._s_depth -= 1
                if self._s_depth == 0:
                    self._s_done = True
                    break
        if i > start:
            out.append(("tool_args", acc[start:i]))
        self._s_pos = i

    def _tags_for(self):
        if self.state == "idle":
            return _IDLE_TAGS
        if self.state == "header":
            return _HEADER_TAGS
        return _BODY_TAGS

    def _split(self, tags):
        best, which = -1, None
        for t in tags:
            i = self.buf.find(t)
            if i >= 0 and (best < 0 or i < best):
                best, which = i, t
        if best >= 0:
            return best, which, -1
        start = max(0, len(self.buf) - _MAX_TAG + 1)
        j = self.buf.find("<|", start)
        while j >= 0:
            tail = self.buf[j:]
            if any(t.startswith(tail) for t in tags):
                return -1, None, j
            j = self.buf.find("<|", j + 1)
        if self.buf.endswith("<") and any(t.startswith("<") for t in tags):
            return -1, None, len(self.buf) - 1
        return -1, None, -1

    def feed_channels(self, text: str = "", final: bool = False) -> list[tuple[str, str]]:
        """Stream raw Inkling-2-Mini output into typed `("reasoning" | "content", text)` chunks.
        Tool invocations are emitted in the `"content"` channel as `<tool_call>\\n{...}\\n</tool_call>` JSON frames."""
        self.buf += text
        out: list[tuple[str, str]] = []
        while True:
            tags = self._tags_for()
            i, tag, hold = self._split(tags)
            if tag is None:
                end = len(self.buf) if (hold < 0 or final) else hold
                chunk, self.buf = self.buf[:end], self.buf[end:]
                self._emit_typed(chunk, out)
                break
            chunk, self.buf = self.buf[:i], self.buf[i + len(tag):]
            self._emit_typed(chunk, out)
            self._handle_tag_typed(tag, out)
        if final:
            self._finish_typed(out)
        return out

    def feed(self, text: str = "", final: bool = False) -> str:
        """Stream raw Inkling-2-Mini output into `<think>...</think>` + text + `<tool_call>...</tool_call>` string."""
        parts: list[str] = []
        for kind, chunk in self.feed_channels(text, final=final):
            if kind in ("tool_start", "tool_args"):
                continue
            if kind == "reasoning":
                if not self._in_think:
                    parts.append("<think>")
                    self._in_think = True
                parts.append(chunk)
            else:
                if self._in_think:
                    parts.append("</think>")
                    self._in_think = False
                parts.append(chunk)
        if final and self._in_think:
            parts.append("</think>")
            self._in_think = False
        return "".join(parts)

    def _emit_typed(self, chunk: str, out: list[tuple[str, str]]) -> None:
        if not chunk:
            return
        if self.state == "think":
            # The model writes a <|content_thinking|> span even at "Thinking effort level: 0" (seen on SQ-20: 4,102 tokens before one
            # 79-character call).  It was dropped here, so the client saw a long silence and never got the reasoning to send back.
            out.append(("reasoning", chunk))
        elif self.state in ("idle", "text"):
            out.append(("content", chunk))
        elif self.state == "header":
            self.header_acc += chunk
        elif self.state in ("tool_json", "tool_text"):
            self.tool_acc += chunk
            self._stream_tool(out)

    def _close_call(self, out: list[tuple[str, str]]) -> None:
        streamed = self._s_mode == "on"
        if streamed and not self._s_done:
            self._reset_stream()     # the output ended inside the args: no closing frame, so the call is not reported as complete
            return
        rendered = _parse_tool_body(self.header_acc, self.tool_acc, self.state)
        if rendered:
            if "<tool_call>" in rendered:
                self.calls += 1
            out.append(("tool_end" if streamed else "content", rendered))
        self._reset_stream()

    def _close_channel_typed(self, out: list[tuple[str, str]]) -> None:
        if self.state in ("tool_json", "tool_text"):
            self._close_call(out)
            self.tool_acc = ""
            self.header_acc = ""
        elif self.state == "header":
            self.header_acc = ""

    def _handle_tag_typed(self, tag: str, out: list[tuple[str, str]]) -> None:
        if tag in (END_SAMPLING, ENDOFTEXT):
            self._close_channel_typed(out)
            self.state = "idle"
            return
        if tag == MSG_MODEL:
            self._close_channel_typed(out)
            self.state = "header"
            self.header_acc = ""
            return
        if tag == END_MSG:
            self._close_channel_typed(out)
            self.state = "idle"
            return
        if tag == CONTENT_THINK:
            self.header_acc = ""
            self.state = "think"
            return
        if tag == CONTENT_TEXT:
            self.header_acc = ""
            self.state = "text"
            return
        if tag == CONTENT_TOOL_JSON:
            self.state = "tool_json"
            self.tool_acc = ""
            self._reset_stream()
            return
        if tag in (CONTENT_TOOL_TEXT, CONTENT_XML):
            self.state = "tool_text"
            self.tool_acc = ""
            return

    def _finish_typed(self, out: list[tuple[str, str]]) -> None:
        if self.state in ("tool_json", "tool_text") and (self.tool_acc.strip() or self.header_acc.strip()):
            self._close_call(out)
        self.header_acc = ""
        self.tool_acc = ""
        self.state = "idle"


def force_text(force):
    """Translate Qwen-style forced tool call opener into Inkling-2-Mini's native invocation prefix."""
    if not isinstance(force, str):
        return force
    m = re.fullmatch(r"<tool_call>\n(?:<function=([^>\n]+)>\n)?", force)
    if not m:
        return force
    name = m.group(1) or ""
    if name:
        return f'{MSG_MODEL}{name}{CONTENT_TOOL_JSON}{{"name":{json.dumps(name)},"args":'
    return MSG_MODEL


def translate(text: str) -> str:
    return InklingDialect().feed(text, final=True)


# ---- thinking: the model's own template takes "Thinking effort level" 0 .. 0.99 (none, minimal, low, medium, high, xhigh, max) ----
# The default stays "none" (what Strata always sent), but a request that names a level now gets it, and what the model thinks is
# passed on as reasoning instead of being dropped.
DEFAULT_EFFORT = "none"
EFFORT = {
    "none": "none", "off": "none", "no_think": "none", "disabled": "none", "false": "none", "0": "none", "0.0": "none",
    "minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max",
    "on": "high", "enabled": "high", "true": "high",
}


def normalize_effort(value=None) -> str:
    """A client's reasoning effort -> one of the template's words; nothing, or a word it does not know, is the default ("none")."""
    if value is None or value is False:
        return DEFAULT_EFFORT
    if value is True:
        return "high"
    if isinstance(value, (int, float)):
        v = float(value)
        return "none" if v <= 0.0 else "minimal" if v <= 0.1 else "low" if v <= 0.2 else "medium" if v <= 0.7 else "high" if v <= 0.9 else "max"
    return EFFORT.get(str(value).strip().lower(), DEFAULT_EFFORT)


def template_kwargs(kwargs: dict) -> dict:
    """Normalize kwargs for Inkling-2-Mini's chat_template.jinja: `reasoning_effort` is one of the template's words
    (`enable_thinking: false` and no effort at all both mean "none")."""
    out = dict(kwargs)
    mode = out.pop("thinking_mode", None)
    effort = out.get("reasoning_effort")
    if out.get("enable_thinking") is False:
        level = "none"
    elif effort not in (None, ""):
        level = normalize_effort(effort)
    elif mode not in (None, ""):
        level = normalize_effort(mode)
    else:
        level = DEFAULT_EFFORT
    out["enable_thinking"] = level != "none"
    out["reasoning_effort"] = level
    return out


def prepare_messages(messages: list[dict]) -> list[dict]:
    """Canonicalize assistant tool_call arguments from JSON strings to dicts so Inkling-2-Mini's chat_template.jinja
    never raises `tool call arguments must be a parsed object, not a JSON string`."""
    out = []
    for m in messages:
        if not isinstance(m, dict) or not m.get("tool_calls"):
            out.append(m)
            continue
        new_calls = []
        changed = False
        for tc in m["tool_calls"]:
            if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                new_calls.append(tc)
                continue
            fn = dict(tc["function"])
            args = fn.get("arguments")
            if isinstance(args, str):
                changed = True
                try:
                    parsed = json.loads(args) if args.strip() else {}
                    fn["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
                except ValueError:
                    fn["arguments"] = {"raw": args}
            elif args is None:
                changed = True
                fn["arguments"] = {}
            new_calls.append({**tc, "function": fn})
        out.append({**m, "tool_calls": new_calls} if changed else m)
    return out
