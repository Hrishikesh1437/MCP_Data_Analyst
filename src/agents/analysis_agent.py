from src.analysis.data_analyzer import DataAnalyzer

class AnalysisAgent:
    def __init__(self, model_tool_callable):
        self.model_tool = model_tool_callable
        self.analyzer = DataAnalyzer()

    def analyze_text(self, text: str) -> str:
        payload = {'prompt': text}
        res = self.model_tool(payload)
        return res.get('result', '') if isinstance(res, dict) else str(res)
