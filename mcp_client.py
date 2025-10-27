# # Minimal MCP client for local testing using the MCP SDK.
# import asyncio
# try:
#     from mcp.client import connect_stdin_stdout
# except Exception:
#     raise RuntimeError('mcp.client.connect_stdin_stdout not available; ensure MCP SDK installed.')

# async def call_analyze(prompt: str):
#     async with connect_stdin_stdout() as (client, _):
#         res = await client.call_tool('data_analysis_agent', 'analyze_data', {'prompt': prompt})
#         print('Response:', res)

# if __name__ == '__main__':
#     import sys
#     prompt = ' '.join(sys.argv[1:]) or 'Please summarize dataset trends.'
#     asyncio.run(call_analyze(prompt))

import requests
import base64

API = "http://localhost:8000"

# --- Test 1: Gemini text analysis ---
r = requests.post(f"{API}/tools/analyze_data", json={"prompt": "Summarize PCA in data science"})
print(r.json())

# --- Test 2: Upload CSV ---
with open("sample.csv", "rb") as f:
    data = f.read()
payload = {"file_bytes": base64.b64encode(data).decode(), "filename": "sample.csv"}
r = requests.post(f"{API}/tools/upload_csv", json=payload)
print(r.json())

# --- Test 3: Ask Gemini to analyze CSV ---
with open("sample.csv", "rb") as f:
    data = f.read()
payload = {"file_bytes": base64.b64encode(data).decode(), "question": "Find correlations and insights."}
r = requests.post(f"{API}/tools/analyze_csv_with_gemini", json=payload)
print(r.json())
