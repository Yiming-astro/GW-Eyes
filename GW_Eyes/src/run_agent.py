from __future__ import annotations

import argparse
import asyncio

from agno.agent import RunEvent
from GW_Eyes.client.collector_agent import CollectorClient
from GW_Eyes.client.executor_agent import ExecutorClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run GW_Eyes agent in single-turn or multi-turn mode.")
    p.add_argument(
        "--agent",
        choices=["collector", "executor"],
        default="executor",
        help="Agent: collector for download, executor for else",
    )
    p.add_argument(
        "--mode",
        choices=["multi", "single"],
        default="multi",
        help="Run mode: 'multi' for interactive chat loop, 'single' for one-shot prompt.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text for single mode; in multi mode, optional first message to send.",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print tool-call events and streaming output (default: True). Use --no-debug to disable.",
    )
    p.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG knowledge retrieval (default: disabled).",
    )
    return p


async def _run_one(agent, prompt: str, debug: bool) -> None:
    if not debug:
        resp = await agent.arun(prompt)
        print(resp.content)
        metrics = getattr(resp, "metrics", None)
        if metrics is not None:
            print(f"\n[input_tokens] {getattr(metrics, 'input_tokens', None)}")
            print(f"[output_tokens] {getattr(metrics, 'output_tokens', None)}")
        return

    async for ev in agent.arun(prompt, stream=True, stream_events=True):
        if ev.event == RunEvent.tool_call_started:
            tool = getattr(ev, "tool", None)
            if tool is not None:
                print(f"\n[tool_call_started] {tool.tool_name} args={tool.tool_args}")
            else:
                print("\n[tool_call_started] (no tool metadata)")

        elif ev.event == RunEvent.tool_call_completed:
            tool = getattr(ev, "tool", None)
            tool_name = tool.tool_name if tool is not None else "(unknown_tool)"
            print(f"\n[tool_call_completed] {tool_name} result={ev.content}")

        elif ev.event == RunEvent.run_content:
            print(ev.content, end="", flush=True)

        elif ev.event == RunEvent.run_error:
            print(f"\n[run_error] {ev.content}")

    print("\n")


async def run_single(agent_type:str, prompt: str, debug: bool, enable_rag: bool) -> None:
    if agent_type == 'collector':
        client = CollectorClient()
    else:
        client = ExecutorClient(enable_rag=enable_rag)
    agent = await client.connect()
    try:
        await _run_one(agent, prompt, debug)
    finally:
        await client.close()
        await asyncio.sleep(0.05)


async def run_multi(agent_type:str, initial_prompt: str | None, debug: bool, enable_rag: bool) -> None:
    if agent_type == 'collector':
        client = CollectorClient()
    else:
        client = ExecutorClient(enable_rag=enable_rag)
    agent = await client.connect()

    print("GW agent REPL. Type 'exit' or 'quit' to stop.\n")

    try:
        if initial_prompt:
            await _run_one(agent, initial_prompt, debug)

        while True:
            try:
                user_text = await asyncio.to_thread(input, "> ")
            except EOFError:
                break

            user_text = user_text.strip()
            if not user_text:
                continue
            if user_text.lower() in ("exit", "quit"):
                break

            await _run_one(agent, user_text, debug)

    finally:
        await client.close()
        await asyncio.sleep(0.05)


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "single" and not args.prompt:
        raise SystemExit("Error: --prompt is required when --mode=single")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        if args.mode == "single":
            loop.run_until_complete(run_single(args.agent, args.prompt, args.debug, args.rag))
        else:
            loop.run_until_complete(run_multi(args.agent, args.prompt, args.debug, args.rag))

        loop.run_until_complete(loop.shutdown_asyncgens())
        if hasattr(loop, "shutdown_default_executor"):
            loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
