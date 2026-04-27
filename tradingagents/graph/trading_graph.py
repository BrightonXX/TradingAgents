# TradingAgents/graph/trading_graph.py

import os
import time
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
        parallel: bool = False,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
            parallel: If True, run analysts concurrently via ThreadPoolExecutor
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []
        self.user_portfolio = self.config.get("user_portfolio", "")
        self.parallel = parallel
        self.selected_analysts = selected_analysts

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.portfolio_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        if self.parallel:
            (
                self.analyst_subgraphs,
                self.report_key_map,
                self.invest_debate_graph,
                self.risk_debate_graph,
                self.trader_pm_graph,
            ) = self.graph_setup.setup_parallel(selected_analysts)
            self.graph = None
        else:
            self.graph = self.graph_setup.setup_graph(selected_analysts)
            self.analyst_subgraphs = None
            self.invest_debate_graph = None
            self.risk_debate_graph = None
            self.trader_pm_graph = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, self.user_portfolio
        )
        args = self.propagator.get_graph_args()

        if self.parallel:
            final_state = self._propagate_parallel(init_agent_state, args)
        elif self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)

    # ------------------------------------------------------------------
    # Parallel analyst execution
    # ------------------------------------------------------------------
    def _propagate_parallel(self, init_state, args):
        """Run analysts in parallel, then run the debate graph.

        Each analyst runs in its own thread with its own sub-graph.
        Individual analyst failures are retried independently so a single
        failure doesn't discard the work of the other analysts.
        """
        import random

        max_retries = 3
        base_backoff = 8  # seconds

        # --- Phase 1: Run analyst sub-graphs in parallel (2 batches) ---
        print("Phase 1: Running analysts in parallel...", flush=True)
        results = {}
        timings = {}
        t0 = time.time()

        def _run_one(name, graph):
            """Run a single analyst sub-graph with retry logic."""
            print(f"  [{name}] starting...", flush=True)
            for attempt in range(1, max_retries + 1):
                try:
                    t = time.time()
                    result = graph.invoke(init_state)
                    elapsed = time.time() - t
                    return name, result, elapsed, None
                except Exception as exc:
                    err_msg = str(exc).lower()
                    is_rate_limit = "429" in err_msg or "rate" in err_msg or "overload" in err_msg
                    if is_rate_limit and attempt < max_retries:
                        wait = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 2)
                        print(f"  [{name}] rate-limited (attempt {attempt}/{max_retries}), "
                              f"retrying in {wait:.1f}s...")
                        time.sleep(wait)
                        continue
                    return name, None, 0, exc
            return name, None, 0, RuntimeError(f"{name}: exhausted retries")

        # Run in batches of 2 to avoid overwhelming yfinance / API proxy
        analyst_items = list(self.analyst_subgraphs.items())
        max_concurrent = 2
        for batch_start in range(0, len(analyst_items), max_concurrent):
            batch = analyst_items[batch_start:batch_start + max_concurrent]
            batch_names = [n for n, _ in batch]
            print(f"  Batch ({', '.join(batch_names)})...", flush=True)
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = {
                    pool.submit(_run_one, name, graph): name
                    for name, graph in batch
                }
                for future in as_completed(futures):
                    name, result, elapsed, error = future.result()
                    if error:
                        print(f"  [{name}] FAILED: {error}", flush=True)
                        results[name] = {}
                    else:
                        report_key = self.report_key_map.get(name, "")
                        report_val = result.get(report_key, "") if result else ""
                        print(f"  [{name}] done in {elapsed:.1f}s "
                              f"({len(report_val)} chars)", flush=True)
                        results[name] = result
                        timings[name] = elapsed

        analyst_elapsed = time.time() - t0
        print(f"Phase 1 complete: {analyst_elapsed:.1f}s "
              f"(wall-clock for {len(timings)} analysts)\n", flush=True)

        # Merge reports into a state dict for the debate tracks
        merged = dict(init_state)
        for name, result in results.items():
            key = self.report_key_map.get(name)
            if key and result:
                merged[key] = result.get(key, "")

        # Brief cooldown to avoid rate limits between phases
        print("Cooling down 10s...", flush=True)
        time.sleep(10)

        # --- Phase 2: Two debate tracks in parallel ---
        print("Phase 2: Running debate tracks in parallel...", flush=True)
        t2 = time.time()

        # Track A: invest debate uses merged state as-is
        invest_state = dict(merged)

        # Track B: risk debate needs a placeholder trader plan since Trader
        # hasn't run yet.  The debators use it as context but their core
        # arguments come from the analyst reports.
        risk_state = dict(merged)
        if not risk_state.get("trader_investment_plan"):
            risk_state["trader_investment_plan"] = (
                "No specific trader plan yet — assess risk/reward based on "
                "the analyst reports provided."
            )

        def _run_graph(tag, graph, state):
            for attempt in range(1, max_retries + 1):
                try:
                    t = time.time()
                    result = graph.invoke(state)
                    elapsed = time.time() - t
                    return tag, result, elapsed, None
                except Exception as exc:
                    err_msg = str(exc).lower()
                    is_rl = "429" in err_msg or "rate" in err_msg or "overload" in err_msg
                    if is_rl and attempt < max_retries:
                        wait = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 2)
                        print(f"  [{tag}] rate-limited (attempt {attempt}), "
                              f"retry {wait:.0f}s...", flush=True)
                        time.sleep(wait)
                        continue
                    return tag, None, 0, exc
            return tag, None, 0, RuntimeError(f"{tag}: exhausted retries")

        debate_results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = {
                pool.submit(_run_graph, "invest_debate", self.invest_debate_graph, invest_state): "invest",
                pool.submit(_run_graph, "risk_debate", self.risk_debate_graph, risk_state): "risk",
            }
            for f in as_completed(futs):
                tag, result, elapsed, error = f.result()
                if error:
                    print(f"  [{tag}] FAILED: {error}", flush=True)
                else:
                    print(f"  [{tag}] done in {elapsed:.1f}s", flush=True)
                    debate_results[tag] = (result, elapsed)

        debate_elapsed = time.time() - t2
        print(f"Phase 2 complete: {debate_elapsed:.1f}s\n", flush=True)

        # Merge debate results into a single state for Trader + PM
        pm_state = dict(merged)
        invest_r, _ = debate_results.get("invest_debate", ({}, 0))
        risk_r, _ = debate_results.get("risk_debate", ({}, 0))
        if invest_r:
            for k in ("investment_debate_state", "investment_plan"):
                if k in invest_r:
                    pm_state[k] = invest_r[k]
        if risk_r:
            if "risk_debate_state" in risk_r:
                pm_state["risk_debate_state"] = risk_r["risk_debate_state"]

        # --- Phase 3: Trader → Portfolio Manager ---
        print("Phase 3: Trader → Portfolio Manager...", flush=True)
        t3 = time.time()

        for attempt in range(1, max_retries + 1):
            try:
                trace = []
                _last_t = t3
                for chunk in self.trader_pm_graph.stream(pm_state, **args):
                    for node_name in chunk:
                        if node_name.startswith("__"):
                            continue
                        now = time.time()
                        dt = now - _last_t
                        print(f"  [{node_name}] {dt:.0f}s", flush=True)
                        _last_t = now
                    trace.append(chunk)
                final_state = trace[-1] if trace else pm_state
                break
            except Exception as exc:
                err_msg = str(exc).lower()
                is_rl = "429" in err_msg or "rate" in err_msg or "overload" in err_msg
                if is_rl and attempt < max_retries:
                    wait = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 2)
                    print(f"  Trader/PM rate-limited (attempt {attempt}), "
                          f"retry {wait:.0f}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise

        pm_elapsed = time.time() - t3
        total_elapsed = time.time() - t0
        print(f"\n{'='*50}", flush=True)
        print(f"  Analysts:  {analyst_elapsed:.0f}s", flush=True)
        print(f"  Debates:   {debate_elapsed:.0f}s  (parallel)", flush=True)
        print(f"  Trader+PM: {pm_elapsed:.0f}s", flush=True)
        print(f"  TOTAL:     {total_elapsed:.0f}s", flush=True)
        print(f"{'='*50}\n", flush=True)

        # Ensure analyst reports are in final state
        for name, result in results.items():
            key = self.report_key_map.get(name)
            if key and result and key not in final_state:
                final_state[key] = result.get(key, "")

        return final_state
