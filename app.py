"""
app.py  —  Chainlit UI for the Synera AI Agent
Run with:  chainlit run app.py
"""

import asyncio
import json
import os
import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from langchain_core.messages import AIMessage, ToolMessage

from agent import (
    cancel,
    graph,
    resume,
    save_conversation,
    save_message_image,
    stream_tokens,
)

# ---------------------------------------------------------------------------
# Chainlit data layer — persists threads/steps in PostgreSQL for history sidebar
# ---------------------------------------------------------------------------
_PG_ASYNC_URI = os.environ.get("POSTGRES_URI", "").replace(
    "postgresql://", "postgresql+asyncpg://"
)
if _PG_ASYNC_URI:
    cl_data._data_layer = SQLAlchemyDataLayer(conninfo=_PG_ASYNC_URI)

_ROOT       = Path(__file__).parent
_IMAGES_DIR = _ROOT / "images"
_IMAGES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_tool(name: str, args: dict) -> str:
    if name == "kiki_recommend":
        condition = args.get("bone_preset", "unknown")
        weight    = args.get("body_wt", "?")
        return (
            f"I have everything I need. I'll run the **KIKI ML model** to recommend "
            f"the optimal lattice for a **{condition}** patient weighing **{weight} kg**."
        )
    if name == "reconstruct_lattice":
        cell = args.get("cell_type", "?")
        vf   = args.get("vol_frac", 0.25)
        rx   = args.get("x_rotation", 0)
        ry   = args.get("y_rotation", 0)
        rz   = args.get("z_rotation", 0)
        rot  = f"{rx}° / {ry}° / {rz}°" if any([rx, ry, rz]) else "no rotation"
        return (
            f"I'll reconstruct a **{cell}** lattice in Synera — "
            f"volume fraction **{vf}**, rotation {rot}."
        )
    if name == "create_sphere":
        return f"I'll create a sphere with radius **{args.get('radius', '?')}** in Synera."
    if name == "create_pattern":
        labels = {0: "Curved Beam", 1: "Glass Sponge 1", 2: "Glass Sponge 2"}
        pt = labels.get(args.get("pattern_type", 1), "surface pattern")
        return f"I'll create a **{pt}** surface pattern in Synera."
    return f"I'm ready to call `{name}`."


def _format_args(name: str, args: dict) -> str:
    lines = [f"**Current parameters for `{name}`:**"]
    for k, v in args.items():
        lines.append(f"- `{k}`: {v}")
    return "\n".join(lines)


def _save_image(png_paths: list[str], thread_id: str) -> list[Path]:
    """Save every PNG export and return all saved paths (one per tool result)."""
    saved = []
    for raw_path in png_paths:
        raw = Path(raw_path)
        if raw.exists():
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dest = _IMAGES_DIR / f"{raw.stem}_{ts}.png"
            shutil.copy2(raw, dest)
            save_message_image(thread_id, str(dest))
            saved.append(dest)
    return saved


def _get_pending_tool(thread_id: str) -> list[dict] | None:
    """Check graph state for a HITL interrupt and return ALL pending tool calls."""
    config   = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot.next:
        return None

    tool_calls: list[dict] = []

    try:
        sub_snap = graph.get_state(config, subgraphs=True)
        for task in sub_snap.tasks:
            if hasattr(task, "state") and task.state:
                for msg in reversed(task.state.values.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls.append({"name": tc["name"], "args": tc["args"]})
                        break
        if tool_calls:
            return tool_calls
    except Exception:
        pass

    try:
        for msg in reversed(snapshot.values.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"name": tc["name"], "args": tc["args"]})
                break
    except Exception:
        pass

    return tool_calls if tool_calls else None


# ---------------------------------------------------------------------------
# Async streaming helper
# ---------------------------------------------------------------------------

async def _stream_response(
    thread_id: str, user_message: str, msg: cl.Message
) -> tuple[list[str], bool]:
    """
    Run stream_tokens() in a background thread, forwarding each token to the
    Chainlit message. Returns (png_paths, streamed_any).
    streamed_any is True if at least one visible text token was forwarded.
    """
    q: queue.Queue = queue.Queue()
    png_paths: list[str] = []
    streamed_any = False

    def _producer():
        for item in stream_tokens(thread_id, user_message):
            q.put(item)
        q.put(None)

    threading.Thread(target=_producer, daemon=True).start()

    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            break

        token_or_msg, node = item

        # Skip supervisor routing output — it's internal JSON, not user-facing
        if node == "supervisor":
            continue

        if token_or_msg == "__error__":
            await msg.stream_token(f"\n\nError: {node}")
            break
        elif isinstance(token_or_msg, str):
            # Skip any stray routing JSON that slips through
            if '{"next"' in token_or_msg:
                continue
            await msg.stream_token(token_or_msg)
            streamed_any = True
        elif isinstance(token_or_msg, ToolMessage):
            try:
                payload = (
                    token_or_msg.content
                    if isinstance(token_or_msg.content, dict)
                    else json.loads(token_or_msg.content)
                )
                if payload.get("success") and payload.get("exports"):
                    for v in payload["exports"].values():
                        if str(v).endswith(".png"):
                            png_paths.append(str(v))
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    return png_paths, streamed_any


# ---------------------------------------------------------------------------
# HITL — show confirmation buttons attached to a message
# ---------------------------------------------------------------------------

async def _show_hitl(tool_infos: list[dict]) -> None:
    """Show description(s) for one or more pending tool calls and action buttons."""
    cl.user_session.set("pending_interrupt", tool_infos[0])  # first tool for cancel/change
    descriptions = "\n\n".join(
        _describe_tool(t["name"], t["args"]) for t in tool_infos
    )
    await cl.Message(
        content=f"{descriptions}\n\nShall I proceed?",
        actions=[
            cl.Action(name="hitl_yes",    payload={"value": "yes"},    label="✓ Yes"),
            cl.Action(name="hitl_no",     payload={"value": "no"},     label="✗ No"),
            cl.Action(name="hitl_change", payload={"value": "change"}, label="✏ Change values"),
        ],
    ).send()


async def _resume_and_reply() -> None:
    """Call resume(), display the reply, and chain to the next interrupt if any."""
    thread_id = cl.user_session.get("thread_id")
    try:
        loop = asyncio.get_running_loop()
        reply, png_paths, next_interrupt = await loop.run_in_executor(None, resume, thread_id)
    except Exception as e:
        await cl.Message(content=f"Error resuming agent: {e}").send()
        return

    reply_msg = cl.Message(content=reply or "")
    image_paths = _save_image(png_paths, thread_id)
    if image_paths:
        reply_msg.elements = [
            cl.Image(path=str(p), name=f"result_{i}", display="inline")
            for i, p in enumerate(image_paths)
        ]
    await reply_msg.send()

    if next_interrupt:
        pending_tools = _get_pending_tool(thread_id)
        if pending_tools:
            await _show_hitl(pending_tools)


# ---------------------------------------------------------------------------
# Action callbacks — triggered when the user clicks a HITL button
# ---------------------------------------------------------------------------

@cl.action_callback("hitl_yes")
async def on_hitl_yes(action: cl.Action):
    cl.user_session.set("pending_interrupt", None)
    await _resume_and_reply()


@cl.action_callback("hitl_no")
async def on_hitl_no(action: cl.Action):
    thread_id = cl.user_session.get("thread_id")
    cl.user_session.set("pending_interrupt", None)
    cancel_text, _, _ = cancel(thread_id)
    await cl.Message(content=cancel_text).send()


@cl.action_callback("hitl_change")
async def on_hitl_change(action: cl.Action):
    thread_id  = cl.user_session.get("thread_id")
    tool_info  = cl.user_session.get("pending_interrupt") or {}
    cl.user_session.set("pending_interrupt", None)
    cancel(thread_id)
    await cl.Message(
        content=(
            f"{_format_args(tool_info.get('name', ''), tool_info.get('args', {}))}\n\n"
            "Please send a new message with the updated values, for example:\n"
            "*\"Reconstruct a GYR lattice with volume fraction 0.3 and 45° Z rotation.\"*"
        )
    ).send()


# ---------------------------------------------------------------------------
# Authentication — auto-login (no password screen) for local/single-user use
# ---------------------------------------------------------------------------

@cl.header_auth_callback
async def header_auth_callback(headers) -> Optional[cl.User]:
    return cl.User(identifier="local-user", metadata={"role": "admin"})


# ---------------------------------------------------------------------------
# Chainlit lifecycle hooks
# ---------------------------------------------------------------------------

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="BCC Lattice",
            message="Reconstruct a BCC lattice with 30% volume fraction",
        ),
        cl.Starter(
            label="Gyroid (rotated)",
            message="Reconstruct a Gyroid lattice rotated 45° around Z",
        ),
        cl.Starter(
            label="Glass Sponge Pattern",
            message="Create a Glass Sponge surface pattern with 0.3 spacing",
        ),
        cl.Starter(
            label="Patient Recommendation",
            message="Recommend a lattice for an elderly patient weighing 70 kg",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    # Use Chainlit's own thread_id so the data layer and LangGraph share the same key
    thread_id = cl.context.session.thread_id
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("first_message", True)
    cl.user_session.set("pending_interrupt", None)

    await cl.Message(
        content=(
            "## Synera AI\n"
            "Conversational 3D geometry assistant powered by LangGraph.\n\n"
            "Describe what you want to design — lattice structures, spheres, or surface patterns "
            "— and I'll run the Synera workflow automatically. "
            "Use the quick-start suggestions above, or type your own request."
        )
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Restore LangGraph thread when the user opens a past conversation from the sidebar."""
    thread_id = thread.get("id") or cl.context.session.thread_id
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("first_message", False)
    cl.user_session.set("pending_interrupt", None)


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")

    # Name the thread on the first user message
    if cl.user_session.get("first_message"):
        title = message.content[:60]
        save_conversation(thread_id, title)
        layer = cl_data.get_data_layer()
        if layer:
            await layer.update_thread(thread_id=thread_id, name=title)
        cl.user_session.set("first_message", False)

    # Snapshot message count before running the agent so the fallback can
    # limit its search to messages added during THIS turn only.
    config     = {"configurable": {"thread_id": thread_id}}
    pre_count  = len(graph.get_state(config).values.get("messages", []))

    # Stream the agent response
    reply_msg = cl.Message(content="")
    await reply_msg.send()

    png_paths, streamed_any = await _stream_response(thread_id, message.content, reply_msg)

    # Attach any images produced by Synera tools (non-HITL path)
    image_paths = _save_image(png_paths, thread_id)
    if image_paths:
        reply_msg.elements = [
            cl.Image(path=str(p), name=f"result_{i}", display="inline")
            for i, p in enumerate(image_paths)
        ]

    # Sub-graph tokens (kiki, synera, knowledge) don't reach the parent stream.
    # Only use the state fallback when nothing was streamed, and restrict the
    # scan to messages produced in this turn (post pre_count) to avoid
    # accidentally surfacing replies from earlier turns.
    if not streamed_any:
        state    = graph.get_state(config)
        new_msgs = state.values.get("messages", [])[pre_count:]
        for m in reversed(new_msgs):
            if isinstance(m, AIMessage) and not m.tool_calls:
                content = m.content
                if isinstance(content, list):
                    content = "\n".join(
                        b["text"] for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if content:
                    reply_msg.content = content
                    break

    await reply_msg.update()

    # Check for HITL interrupt and show confirmation buttons
    pending_tools = _get_pending_tool(thread_id)
    if pending_tools:
        await _show_hitl(pending_tools)
