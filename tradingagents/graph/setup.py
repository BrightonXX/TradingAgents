# TradingAgents/graph/setup.py

from typing import Any, Dict, Tuple
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic

# Maps analyst type keys to their state report keys
_REPORT_KEY_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        portfolio_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.portfolio_manager_memory = portfolio_manager_memory
        self.conditional_logic = conditional_logic

    # ------------------------------------------------------------------
    # Helper: create the set of analyst/tool/delete nodes
    # ------------------------------------------------------------------
    def _create_analyst_nodes(self, selected_analysts):
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(self.quick_thinking_llm)
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(self.quick_thinking_llm)
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(self.quick_thinking_llm)
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(self.quick_thinking_llm)
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        return analyst_nodes, delete_nodes, tool_nodes

    # ------------------------------------------------------------------
    # Parallel mode: individual analyst sub-graphs + debate graph
    # ------------------------------------------------------------------
    def setup_parallel(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ) -> Tuple[Dict[str, Any], Dict[str, str], Any]:
        """Create individual analyst sub-graphs and a debate graph.

        Returns:
            (analyst_subgraphs, report_key_map, debate_graph)
        """
        if not selected_analysts:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        analyst_nodes, delete_nodes, tool_nodes = self._create_analyst_nodes(selected_analysts)

        # Build a mini sub-graph per analyst (analyst → tools → loop → END)
        analyst_subgraphs = {}
        for atype in selected_analysts:
            sg = StateGraph(AgentState)
            name = f"{atype.capitalize()} Analyst"
            tools_name = f"tools_{atype}"
            clear_name = f"Msg Clear {atype.capitalize()}"

            sg.add_node(name, analyst_nodes[atype])
            sg.add_node(tools_name, tool_nodes[atype])
            sg.add_node(clear_name, delete_nodes[atype])

            sg.add_edge(START, name)
            sg.add_conditional_edges(
                name,
                getattr(self.conditional_logic, f"should_continue_{atype}"),
                [tools_name, clear_name],
            )
            sg.add_edge(tools_name, name)
            sg.add_edge(clear_name, END)

            analyst_subgraphs[atype] = sg.compile()

        invest_debate_graph = self._build_invest_debate_graph()
        risk_debate_graph = self._build_risk_debate_graph()
        trader_pm_graph = self._build_trader_pm_graph()
        return analyst_subgraphs, _REPORT_KEY_MAP, invest_debate_graph, risk_debate_graph, trader_pm_graph

    # ------------------------------------------------------------------
    # Track A: Invest debate (Bull → Bear → Research Manager)
    # ------------------------------------------------------------------
    def _build_invest_debate_graph(self):
        bull_node = create_bull_researcher(self.quick_thinking_llm, self.bull_memory)
        bear_node = create_bear_researcher(self.quick_thinking_llm, self.bear_memory)
        rm_node = create_research_manager(self.deep_thinking_llm, self.invest_judge_memory)

        g = StateGraph(AgentState)
        g.add_node("Bull Researcher", bull_node)
        g.add_node("Bear Researcher", bear_node)
        g.add_node("Research Manager", rm_node)

        g.add_edge(START, "Bull Researcher")
        g.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
        )
        g.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
        )
        g.add_edge("Research Manager", END)
        return g.compile()

    # ------------------------------------------------------------------
    # Track B: Risk debate (Aggressive → Conservative → Neutral)
    # ------------------------------------------------------------------
    def _build_risk_debate_graph(self):
        aggr_node = create_aggressive_debator(self.quick_thinking_llm)
        cons_node = create_conservative_debator(self.quick_thinking_llm)
        neut_node = create_neutral_debator(self.quick_thinking_llm)

        g = StateGraph(AgentState)
        g.add_node("Aggressive Analyst", aggr_node)
        g.add_node("Conservative Analyst", cons_node)
        g.add_node("Neutral Analyst", neut_node)

        g.add_edge(START, "Aggressive Analyst")
        g.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Conservative Analyst": "Conservative Analyst", "Portfolio Manager": END},
        )
        g.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Neutral Analyst": "Neutral Analyst", "Portfolio Manager": END},
        )
        g.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Aggressive Analyst": "Aggressive Analyst", "Portfolio Manager": END},
        )
        return g.compile()

    # ------------------------------------------------------------------
    # Tail: Trader → Portfolio Manager
    # ------------------------------------------------------------------
    def _build_trader_pm_graph(self):
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)
        pm_node = create_portfolio_manager(self.deep_thinking_llm, self.portfolio_manager_memory)

        g = StateGraph(AgentState)
        g.add_node("Trader", trader_node)
        g.add_node("Portfolio Manager", pm_node)

        g.add_edge(START, "Trader")
        g.add_edge("Trader", "Portfolio Manager")
        g.add_edge("Portfolio Manager", END)
        return g.compile()

    # ------------------------------------------------------------------
    # Full sequential debate graph (fallback / non-parallel)
    # ------------------------------------------------------------------
    def _build_debate_graph(self):
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm, self.bull_memory)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm, self.bear_memory)
        research_manager_node = create_research_manager(self.deep_thinking_llm, self.invest_judge_memory)
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm, self.portfolio_manager_memory)

        workflow = StateGraph(AgentState)

        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        workflow.add_edge(START, "Bull Researcher")
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Conservative Analyst": "Conservative Analyst", "Portfolio Manager": "Portfolio Manager"},
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Neutral Analyst": "Neutral Analyst", "Portfolio Manager": "Portfolio Manager"},
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Aggressive Analyst": "Aggressive Analyst", "Portfolio Manager": "Portfolio Manager"},
        )
        workflow.add_edge("Portfolio Manager", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Sequential mode (original behaviour)
    # ------------------------------------------------------------------
    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph (sequential).

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        analyst_nodes, delete_nodes, tool_nodes = self._create_analyst_nodes(selected_analysts)

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(
            self.deep_thinking_llm, self.portfolio_manager_memory
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        # Compile and return
        return workflow.compile()
