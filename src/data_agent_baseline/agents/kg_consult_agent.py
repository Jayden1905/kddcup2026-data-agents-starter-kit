"""ConsultAgent: dedicated reasoning agent over the ApproachGraph.

Maintains its own conversation context, receives all approach recordings,
builds a running synthesis of what has been tried, and provides strategic
recommendations to the main agent. Consulted both when blocked (reactive)
and proactively every N turns to steer exploration.
"""

from __future__ import annotations

from data_agent_baseline.agents.kg_approach_graph import ApproachGraph, ApproachNode
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage

MEMORY_AGENT_SYSTEM = """\
You are a strategy advisor for a data analysis agent. Your role is to observe \
the history of attempted SQL approaches and provide strategic guidance.

You receive:
1. The original question being answered
2. The full approach history (every SQL tried, its structural fingerprint, result)
3. Any context about the database schema and available tables

Your job:
- Identify PATTERNS in the failures (why do they keep failing?)
- Identify BLIND SPOTS (what hasn't been tried that could work?)
- Recommend a CONCRETE next strategy (specific tables, joins, filters, or \
aggregation patterns to try)
- Flag if the data might not exist in SQL and suggest alternatives (documents, \
python extraction)

Be specific and actionable. Don't repeat what failed. Think about WHY \
approaches failed and what that implies about the data structure.

Respond with a JSON object:
{
  "synthesis": "1-2 sentence summary of the pattern you see in the failures",
  "blind_spots": ["list of strategies not yet attempted"],
  "recommendation": "specific next approach to try with reasoning",
  "confidence": "high/medium/low — how confident you are this will work",
  "pivot_needed": true/false — whether the agent should abandon the current direction entirely
}
"""


class ConsultAgent:
    """Dedicated reasoning agent that maintains context over approach history.

    Unlike the main agent which gets a pruned conversation window, the memory
    agent accumulates ALL approach history and builds a running understanding
    of what has been tried and why things are failing.
    """

    def __init__(self, model: ModelAdapter, question: str) -> None:
        self.model = model
        self.question = question
        self._context: list[ModelMessage] = [
            ModelMessage(role="system", content=MEMORY_AGENT_SYSTEM),
        ]
        self._turn_count = 0
        self._formula: str = ""

    def set_formula(self, formula: str) -> None:
        """Store the CoT formula so the consult agent can reference it."""
        self._formula = formula
        self._context.append(
            ModelMessage(
                role="user",
                content=f"[Formula established]\n{formula}",
            )
        )
        self._context.append(
            ModelMessage(
                role="assistant",
                content="Noted. I will check all approaches against this formula.",
            )
        )
        self._last_synthesis: str = ""

    def create_resolution_plan(self, schema_overview: str = "") -> str:
        """Identify unknowns in the formula and create a tool-call plan.

        Returns a structured plan of which tools to call in what order to
        resolve all unknowns before writing SQL. Called once after formula
        is established.
        """
        if not self._formula:
            return ""

        prompt = (
            f"[Resolution Planning]\n"
            f"Question: {self.question}\n"
            f"Formula:\n{self._formula}\n"
            f"\nSchema overview:\n{schema_overview}\n\n"
            f"Identify all UNKNOWNS in this formula that need resolution "
            f"before SQL can be written. For each unknown, specify which tool "
            f"to use:\n"
            f"- recall_schema: check if we've seen this before\n"
            f"- resolve: map informal name to DB value\n"
            f"- knowledge: look up domain definitions/thresholds\n"
            f"- find_value: locate where a value lives in the schema\n"
            f"- distribution: check numeric ranges\n"
            f"- run_sql: verify with SELECT DISTINCT\n\n"
            f"Output a JSON list of steps:\n"
            f'[{{"unknown": "what needs resolving", "tool": "tool_name", '
            f'"params": {{...}}}}, ...]\n'
            f"If no unknowns, respond: []"
        )
        self._context.append(ModelMessage(role="user", content=prompt))

        response = self._call(self._context)
        if response:
            self._context.append(ModelMessage(role="assistant", content=response))
            self._resolution_plan = response
        return response or ""

    def refine_formula(self, discovery: str) -> str:
        """Refine the formula based on new schema/data discovery.

        Triggers an LLM call that sees all prior context + the new finding,
        and returns an updated formula. Called after significant discoveries
        (schema, topology, first successful run_sql).
        """
        if not self._formula:
            return ""

        prompt = (
            f"[New Discovery]\n{discovery}\n\n"
            f"Current formula:\n{self._formula}\n\n"
            f"Does this discovery change the formula? If yes, output the "
            f"corrected formula (FORMULA, NUMERATOR, DENOMINATOR, JOIN, "
            f"CONSTRAINT). If no change needed, respond: NO CHANGE"
        )
        self._context.append(ModelMessage(role="user", content=prompt))

        response = self._call(self._context)
        if response:
            self._context.append(ModelMessage(role="assistant", content=response))
            stripped = response.strip()
            if "NO CHANGE" not in stripped.upper():
                self._formula = stripped
                return stripped
        return ""

    def get_formula(self) -> str:
        """Return the current formula."""
        return self._formula

    def observe(self, node: ApproachNode) -> None:
        """Feed a new approach result to the memory agent's context.

        Does not trigger an LLM call — just accumulates observations.
        """
        fp = node.fingerprint
        msg = (
            f"[Approach recorded — Turn {node.turn}]\n"
            f"SQL: {node.sql}\n"
            f"Tables: {fp.tables}\n"
            f"Joins: {fp.joins}\n"
            f"Select: {fp.select_cols}\n"
            f"Filters: {fp.filter_cols} = {fp.filter_values}\n"
            f"Aggregations: {fp.aggs}\n"
            f"Group By: {fp.group_by}\n"
            f"Order By: {fp.order_by}\n"
            f"Grain: {fp.grain}\n"
            f"Temporal: {fp.temporal}\n"
            f"Distinct: {fp.has_distinct} | Subquery: {fp.has_subquery}\n"
            f"Result: {node.result}\n"
            f"Why Failed: {node.reason}"
        )
        self._context.append(ModelMessage(role="user", content=msg))
        self._context.append(
            ModelMessage(role="assistant", content="Observed. Awaiting consultation.")
        )

    def consult(
        self,
        approaches: ApproachGraph,
        schema_context: str = "",
        trigger: str = "blocked",
    ) -> str:
        """Ask the memory agent for strategic guidance.

        Args:
            approaches: the full approach graph
            schema_context: optional schema/table info for reference
            trigger: why we're consulting ("blocked", "proactive", "stuck")

        Returns:
            The raw LLM response (JSON with synthesis + recommendation)
        """
        self._turn_count += 1

        # Build the consultation prompt
        prompt_parts = [
            f"[Consultation — trigger: {trigger}]",
            f"Question: {self.question}",
            "",
            f"Total approaches tried: {len(approaches.nodes)}",
        ]

        dead = approaches.dead_ends()
        if dead:
            prompt_parts.append(
                f"Dead ends (tried 2+ times): {len(dead)}"
            )
            for d in dead:
                fp = d.fingerprint
                prompt_parts.append(
                    f"  ⊘ tables={fp.tables} filters={fp.filter_cols} "
                    f"aggs={fp.aggs} grain={fp.grain}"
                )

        if schema_context:
            prompt_parts.append(f"\nAvailable schema:\n{schema_context}")

        prompt_parts.append(
            "\nBased on everything you've observed, what should the agent try next? "
            "Be specific — name tables, columns, join paths, or suggest "
            "abandoning SQL entirely if appropriate."
        )

        self._context.append(
            ModelMessage(role="user", content="\n".join(prompt_parts))
        )

        # Call the LLM
        response = self._call(self._context)

        if response:
            self._context.append(ModelMessage(role="assistant", content=response))
            self._last_synthesis = response

        return response

    def get_last_synthesis(self) -> str:
        """Return the most recent strategic recommendation."""
        return self._last_synthesis

    def render_guidance(self) -> str:
        """Render the latest guidance for injection into the main agent's prompt."""
        if not self._last_synthesis:
            return ""
        return f"[Strategy Advisor]\n{self._last_synthesis}"

    def _call(self, messages: list[ModelMessage]) -> str:
        """Call the LLM with the memory agent's full context."""
        try:
            result = self.model.complete(messages)
            return result if result else ""
        except RuntimeError:
            return ""
