"""ReAct agent executor — Thought → Action → Observation loop.

Uses OmniFF runtime as tool backend. LLM decides which tools to call
and synthesizes final answer from observations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from omniff.agent.tools import Tool, get_builtin_tools
from omniff.observability import get_logger

_log = get_logger("agent")

_TOOL_CALL_RE = re.compile(
    r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\}|\S.*?)$",
    re.MULTILINE | re.DOTALL,
)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


@dataclass
class AgentStep:
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0

    @property
    def trace(self) -> str:
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}:")
            lines.append(f"  Thought: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}({step.action_input})")
                lines.append(f"  Observation: {step.observation}")
        lines.append(f"Answer: {self.answer}")
        return "\n".join(lines)


def _build_system_prompt(tools: list[Tool]) -> str:
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}. Parameters: {json.dumps(t.parameters)}"
        for t in tools
    )
    return f"""You are an AI agent with access to tools. Solve the user's task step by step.

Available tools:
{tool_descriptions}

For each step, use this format:
Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: {{"param1": "value1", "param2": "value2"}}

After receiving an Observation, continue reasoning.

When you have enough information to answer, respond with:
Thought: I now have enough information to answer.
Final Answer: <your complete answer>

Rules:
- Use tools when you need external information or capabilities
- Think step by step
- Be concise in observations
- Maximum 10 steps before you must give a Final Answer"""


class AgentExecutor:
    """ReAct agent that uses OmniFF runtime as tool backend."""

    def __init__(
        self,
        runtime: Any,
        tools: list[Tool] | None = None,
        max_steps: int = 10,
    ) -> None:
        self.runtime = runtime
        self.tools = {t.name: t for t in (tools or get_builtin_tools())}
        self.max_steps = max_steps

    def run(self, task: str, context: str = "") -> AgentResult:
        from omniff.models.llm import LLMModel

        system_prompt = _build_system_prompt(list(self.tools.values()))
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": f"Context: {context}\n\nTask: {task}"})
        else:
            messages.append({"role": "user", "content": task})

        steps: list[AgentStep] = []

        for step_num in range(self.max_steps):
            prompt = self._messages_to_prompt(messages)

            llm = self._get_llm()
            response = llm.infer({"prompt": prompt, "thinking": False, "max_new_tokens": 1024})
            text = response["text"].strip()

            _log.info("agent step %d: %s", step_num + 1, text[:200])

            final_match = _FINAL_ANSWER_RE.search(text)
            if final_match:
                answer = final_match.group(1).strip()
                thought = text[:final_match.start()].replace("Thought:", "").strip()
                steps.append(AgentStep(thought=thought or "Final answer ready."))
                return AgentResult(answer=answer, steps=steps, total_steps=step_num + 1)

            tool_match = _TOOL_CALL_RE.search(text)
            if tool_match:
                action_name = tool_match.group(1).strip()
                action_input_raw = tool_match.group(2).strip()

                try:
                    action_input = json.loads(action_input_raw)
                except json.JSONDecodeError:
                    action_input = {"text": action_input_raw}

                thought = text[:tool_match.start()].replace("Thought:", "").strip()

                observation = self._execute_tool(action_name, action_input)

                step = AgentStep(
                    thought=thought,
                    action=action_name,
                    action_input=action_input,
                    observation=observation,
                )
                steps.append(step)

                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                steps.append(AgentStep(thought=text))
                return AgentResult(answer=text, steps=steps, total_steps=step_num + 1)

        return AgentResult(
            answer="Maximum steps reached. Based on gathered information: " +
                   (steps[-1].observation or steps[-1].thought if steps else "No conclusion."),
            steps=steps,
            total_steps=self.max_steps,
        )

    def _execute_tool(self, name: str, params: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'. Available: {', '.join(self.tools.keys())}"

        try:
            return tool.fn(self.runtime, **params)
        except Exception as e:
            _log.warning("tool %s failed: %s", name, e)
            return f"Error executing {name}: {e}"

    def _get_llm(self) -> Any:
        from omniff.models.llm import LLMModel

        if not hasattr(self, "_llm") or self._llm is None:
            self._llm = self.runtime._ensure_model(
                "llm", LLMModel, model_id="Qwen/Qwen3-4B", device="auto"
            )
        return self._llm

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        self.tools.pop(name, None)
