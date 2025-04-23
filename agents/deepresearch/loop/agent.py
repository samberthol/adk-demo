# agents/deepresearch/loop/agent.py
import os
from google.adk.agents import LoopAgent
from ..analyser.agent import analysis_agent
from ..evaluator.agent import satisfaction_evaluator

analysis_loop_agent = LoopAgent(
    name="AnalysisLoopAgent",
    description="Iteratively analyzes research findings and evaluates sufficiency until criteria are met.",
    agent_to_loop=analysis_agent,
    evaluator=satisfaction_evaluator,
    exit_value="sufficient",
    max_loops=3,
)
