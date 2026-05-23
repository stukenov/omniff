"""Tests for agent executor and tools."""

from unittest.mock import MagicMock, patch

from omniff.agent.executor import AgentExecutor, AgentResult, AgentStep, _build_system_prompt
from omniff.agent.tools import Tool, get_builtin_tools


class TestTools:
    def test_builtin_tools_count(self):
        tools = get_builtin_tools()
        assert len(tools) == 8

    def test_builtin_tool_names(self):
        tools = get_builtin_tools()
        names = {t.name for t in tools}
        assert "translate" in names
        assert "transcribe" in names
        assert "describe_image" in names
        assert "generate_image" in names
        assert "summarize" in names
        assert "read_document" in names
        assert "generate_code" in names
        assert "web_search" in names

    def test_tool_has_required_fields(self):
        tools = get_builtin_tools()
        for tool in tools:
            assert tool.name
            assert tool.description
            assert tool.parameters
            assert callable(tool.fn)

    def test_custom_tool(self):
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"input": "Test input"},
            fn=lambda runtime, input: f"Result: {input}",
        )
        assert tool.name == "test_tool"
        result = tool.fn(None, input="hello")
        assert result == "Result: hello"


class TestSystemPrompt:
    def test_includes_tool_names(self):
        tools = get_builtin_tools()
        prompt = _build_system_prompt(tools)
        assert "translate" in prompt
        assert "transcribe" in prompt
        assert "Final Answer" in prompt

    def test_includes_format_instructions(self):
        tools = get_builtin_tools()
        prompt = _build_system_prompt(tools)
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Action Input:" in prompt


class TestAgentExecutor:
    def test_init_default_tools(self):
        runtime = MagicMock()
        executor = AgentExecutor(runtime=runtime)
        assert len(executor.tools) == 8
        assert executor.max_steps == 10

    def test_init_custom_tools(self):
        runtime = MagicMock()
        custom = [Tool(name="t", description="d", parameters={}, fn=lambda r: "")]
        executor = AgentExecutor(runtime=runtime, tools=custom)
        assert len(executor.tools) == 1

    def test_add_remove_tool(self):
        runtime = MagicMock()
        executor = AgentExecutor(runtime=runtime, tools=[])
        tool = Tool(name="new", description="d", parameters={}, fn=lambda r: "")
        executor.add_tool(tool)
        assert "new" in executor.tools
        executor.remove_tool("new")
        assert "new" not in executor.tools

    def test_execute_unknown_tool(self):
        runtime = MagicMock()
        executor = AgentExecutor(runtime=runtime)
        result = executor._execute_tool("nonexistent", {})
        assert "Unknown tool" in result

    def test_execute_tool_error(self):
        runtime = MagicMock()
        bad_tool = Tool(
            name="bad", description="d", parameters={},
            fn=lambda runtime: (_ for _ in ()).throw(ValueError("boom")),
        )
        executor = AgentExecutor(runtime=runtime, tools=[bad_tool])
        result = executor._execute_tool("bad", {})
        assert "Error" in result

    def test_final_answer_parsed(self):
        runtime = MagicMock()
        executor = AgentExecutor(runtime=runtime)

        mock_llm = MagicMock()
        mock_llm.infer.return_value = {
            "text": "Thought: Simple question.\nFinal Answer: 42"
        }

        with patch.object(executor, "_get_llm", return_value=mock_llm):
            result = executor.run("What is 6*7?")

        assert result.answer == "42"
        assert result.total_steps == 1

    def test_tool_call_then_answer(self):
        runtime = MagicMock()
        executor = AgentExecutor(runtime=runtime)

        mock_llm = MagicMock()
        mock_llm.infer.side_effect = [
            {"text": 'Thought: Need to search.\nAction: web_search\nAction Input: {"query": "test"}'},
            {"text": "Thought: Got it.\nFinal Answer: Found the answer."},
        ]

        mock_tool = MagicMock(return_value="Search result: 42")
        executor.tools["web_search"].fn = lambda runtime, **kw: "Search result: 42"

        with patch.object(executor, "_get_llm", return_value=mock_llm):
            result = executor.run("Search for test")

        assert result.answer == "Found the answer."
        assert result.total_steps == 2
        assert len(result.steps) == 2
        assert result.steps[0].action == "web_search"
        assert result.steps[0].observation == "Search result: 42"

    def test_max_steps_limit(self):
        runtime = MagicMock()
        executor = AgentExecutor(runtime=runtime, max_steps=2)

        mock_llm = MagicMock()
        mock_llm.infer.return_value = {
            "text": 'Thought: Keep going.\nAction: web_search\nAction Input: {"query": "more"}'
        }
        executor.tools["web_search"].fn = lambda runtime, **kw: "result"

        with patch.object(executor, "_get_llm", return_value=mock_llm):
            result = executor.run("Loop forever")

        assert result.total_steps == 2
        assert "Maximum steps" in result.answer


class TestAgentResult:
    def test_trace_format(self):
        result = AgentResult(
            answer="42",
            steps=[
                AgentStep(thought="Think", action="search", action_input={"q": "x"}, observation="found"),
                AgentStep(thought="Done"),
            ],
            total_steps=2,
        )
        trace = result.trace
        assert "Step 1:" in trace
        assert "Step 2:" in trace
        assert "Answer: 42" in trace


class TestRouterAgent:
    def test_route_agent_keyword(self):
        from omniff.router.keyword_router import KeywordRouter
        router = KeywordRouter()
        assert router.route("agent investigate this problem").route_class == "AGENT"

    def test_route_agent_ru(self):
        from omniff.router.keyword_router import KeywordRouter
        router = KeywordRouter()
        assert router.route("исследуй эту тему").route_class == "AGENT"

    def test_route_agent_step_by_step(self):
        from omniff.router.keyword_router import KeywordRouter
        router = KeywordRouter()
        assert router.route("solve this step by step").route_class == "AGENT"
