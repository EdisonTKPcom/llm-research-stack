"""A minimal tool-using agent scaffold.

Add tools (e.g., market data via ccxt/public endpoints), then route:
- tool -> analysis -> RAG -> answer
"""
from typing import Dict, Any, List, Callable

class Tool:
    def __init__(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]], description: str):
        self.name = name
        self.func = func
        self.description = description

class Agent:
    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}

    def plan(self, question: str) -> str:
        # Naive planner. Replace with LLM call.
        if "price" in question.lower():
            return "market_price"
        return "rag"

    def run(self, question: str) -> Dict[str, Any]:
        action = self.plan(question)
        if action in self.tools:
            return self.tools[action].func({"query": question})
        return {"answer": "Defer to RAG or general QA.", "action": action}
