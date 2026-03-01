"""LangGraph StateGraph — connects all 6 agents into an executable pipeline."""

import logging
from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from prompter.agents.analyzer import analyze
from prompter.agents.architect import architect
from prompter.agents.communication_designer import design_communication
from prompter.agents.critic import critique
from prompter.agents.packager import package
from prompter.agents.refiner import refine
from prompter.config import Settings
from prompter.state import PipelineState

logger = logging.getLogger(__name__)

# Canonical node names (match last_checkpoint values set by each agent)
NODE_ANALYZE = "analyze"
NODE_ARCHITECT = "architect"
NODE_DESIGN_COMMUNICATION = "design_communication"
NODE_CRITIQUE = "critique"
NODE_REFINE = "refine"
NODE_PACKAGE = "package"

# Ordered list for determining resume targets (linear portion only)
NODE_ORDER = [
    NODE_ANALYZE,
    NODE_ARCHITECT,
    NODE_DESIGN_COMMUNICATION,
    NODE_CRITIQUE,
    NODE_REFINE,
    NODE_PACKAGE,
]

# Maps node names to their agent callable
_AGENT_MAP = {
    NODE_ANALYZE: analyze,
    NODE_ARCHITECT: architect,
    NODE_DESIGN_COMMUNICATION: design_communication,
    NODE_CRITIQUE: critique,
    NODE_REFINE: refine,
    NODE_PACKAGE: package,
}


def check_clarification_needed(
    state: PipelineState,
) -> Literal["proceed", "needs_clarification"]:
    """Route after analyzer: exit early if the idea needs clarification."""
    if state.get("needs_clarification", False):
        return "needs_clarification"
    return "proceed"


def should_continue_refining(
    state: PipelineState,
) -> Literal["refine", "package"]:
    """Route after critic: refine if quality fails, package if all pass or iterations exhausted."""
    if state.get("all_passed", False):
        return "package"
    if state.get("current_iteration", 0) >= state.get("max_iterations", 3):
        return "package"
    return "refine"


def build_graph(
    settings: Settings,
    entry_node: str = NODE_ANALYZE,
) -> CompiledStateGraph:
    """Build and compile the LangGraph StateGraph with settings bound to each node.

    Uses functools.partial to inject the Settings instance into each agent
    function, since LangGraph nodes receive only the state dict.

    Args:
        settings: Settings instance with API credentials and configuration.
        entry_node: Node to start execution from. Defaults to "analyze".
                    Set to a later node for resume-from-checkpoint scenarios.

    Returns:
        Compiled StateGraph ready for invoke() or stream().

    Raises:
        ValueError: If entry_node is not a valid node name.
    """
    if entry_node not in _AGENT_MAP:
        raise ValueError(
            f"Unknown entry node: {entry_node!r}. "
            f"Valid nodes: {list(_AGENT_MAP.keys())}"
        )

    graph = StateGraph(PipelineState)

    # Determine which nodes are reachable from the entry point.
    # For critique/refine loop: if entering at critique, refine and package are also needed.
    entry_idx = NODE_ORDER.index(entry_node)
    reachable = set(NODE_ORDER[entry_idx:])

    # Register reachable nodes with settings partially applied
    for node_name in NODE_ORDER:
        if node_name in reachable:
            agent_fn = _AGENT_MAP[node_name]
            graph.add_node(node_name, partial(agent_fn, settings=settings))

    # Entry edge
    graph.add_edge(START, entry_node)

    # Conditional edge after analyze: clarification check
    if NODE_ANALYZE in reachable:
        graph.add_conditional_edges(
            NODE_ANALYZE,
            check_clarification_needed,
            {"proceed": NODE_ARCHITECT, "needs_clarification": END},
        )

    # Linear edges
    if NODE_ARCHITECT in reachable and NODE_DESIGN_COMMUNICATION in reachable:
        graph.add_edge(NODE_ARCHITECT, NODE_DESIGN_COMMUNICATION)

    if NODE_DESIGN_COMMUNICATION in reachable and NODE_CRITIQUE in reachable:
        graph.add_edge(NODE_DESIGN_COMMUNICATION, NODE_CRITIQUE)

    # Critic → refine/package conditional edge
    if NODE_CRITIQUE in reachable:
        targets = {}
        if NODE_REFINE in reachable:
            targets["refine"] = NODE_REFINE
        if NODE_PACKAGE in reachable:
            targets["package"] = NODE_PACKAGE
        graph.add_conditional_edges(NODE_CRITIQUE, should_continue_refining, targets)

    # Refine loops back to critique
    if NODE_REFINE in reachable and NODE_CRITIQUE in reachable:
        graph.add_edge(NODE_REFINE, NODE_CRITIQUE)

    # Terminal edge
    if NODE_PACKAGE in reachable:
        graph.add_edge(NODE_PACKAGE, END)

    logger.info(f"Graph built: entry={entry_node}, nodes={sorted(reachable)}")
    return graph.compile()


def get_next_node(last_checkpoint: str, state: PipelineState) -> str | None:
    """Determine which node to resume from based on the last completed checkpoint.

    Args:
        last_checkpoint: The last_checkpoint value from a loaded state.
        state: The loaded pipeline state (used for critique/refine routing).

    Returns:
        The node name to resume from, or None if the pipeline is complete.

    Raises:
        ValueError: If last_checkpoint is not recognized.
    """
    if last_checkpoint == NODE_PACKAGE:
        return None  # Pipeline already complete

    if last_checkpoint == NODE_CRITIQUE:
        # Re-enter the routing decision: should we refine or package?
        route = should_continue_refining(state)
        return NODE_REFINE if route == "refine" else NODE_PACKAGE

    if last_checkpoint == NODE_REFINE:
        return NODE_CRITIQUE  # After refine, always go back to critique

    if last_checkpoint == NODE_ANALYZE:
        # Check if clarification was needed
        if state.get("needs_clarification", False):
            return None  # Pipeline ended at clarification
        return NODE_ARCHITECT

    # For linear nodes (architect, design_communication), advance to next in order
    try:
        idx = NODE_ORDER.index(last_checkpoint)
        if idx + 1 < len(NODE_ORDER):
            return NODE_ORDER[idx + 1]
    except ValueError:
        raise ValueError(f"Unknown last_checkpoint: {last_checkpoint!r}")

    return None
