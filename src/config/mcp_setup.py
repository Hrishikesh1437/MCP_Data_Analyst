# # src/config/mcp_setup.py
# import io
# import os
# import pandas as pd
# from fastapi import FastAPI, UploadFile, Form
# from fastapi.responses import JSONResponse
# import uvicorn
# import google.generativeai as genai

# # ──────────────────────────────────────────────
# # ✅ Gemini Setup
# # ──────────────────────────────────────────────
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise RuntimeError("❌ GEMINI_API_KEY not found. Please set it before running MCP server.")
# genai.configure(api_key=api_key)

# # ──────────────────────────────────────────────
# # ✅ FastAPI App
# # ──────────────────────────────────────────────
# app = FastAPI(title="MCP Data Analysis Server (Gemini-powered)")

# @app.get("/")
# def root():
#     return {"status": "running", "message": "🚀 MCP Gemini Data Analysis API active"}

# # ──────────────────────────────────────────────
# # 📤 Upload + Summarize CSV
# # ──────────────────────────────────────────────
# @app.post("/upload_csv/")
# async def upload_csv(file: UploadFile):
#     try:
#         df = pd.read_csv(io.BytesIO(await file.read()))
#         df = df.replace({float('inf'): None, float('-inf'): None})  # avoid NaN crash
#         summary = df.describe(include="all").replace({pd.NA: None}).to_dict()
#         return JSONResponse(content={
#             "filename": file.filename,
#             "rows": len(df),
#             "columns": df.columns.tolist(),
#             "summary": summary,
#         })
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # ──────────────────────────────────────────────
# # 🧠 Gemini CSV Analyzer
# # ──────────────────────────────────────────────
# @app.post("/analyze_csv_with_gemini/")
# async def analyze_csv_with_gemini(file: UploadFile, question: str = Form(...)):
#     try:
#         df = pd.read_csv(io.BytesIO(await file.read()))
#         df = df.replace({float('inf'): None, float('-inf'): None})
#         stats = {
#             "rows": len(df),
#             "columns": df.columns.tolist(),
#             "missing": df.isnull().sum().to_dict(),
#             "numeric_summary": df.describe().replace({pd.NA: None}).to_dict(),
#         }

#         model = genai.GenerativeModel("gemini-pro-latest")
#         prompt = (
#             f"You are a data analysis expert.\n\n"
#             f"Dataset summary:\n{stats}\n\n"
#             f"Question: {question}\n"
#             f"Provide concise, structured insights."
#         )
#         gemini_response = model.generate_content(prompt)
#         return JSONResponse(content={
#             "status": "success",
#             "summary": stats,
#             "gemini_insight": gemini_response.text,
#         })
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # ──────────────────────────────────────────────
# # 🚀 Entry Point
# # ──────────────────────────────────────────────
# if __name__ == "__main__":
#     print("🚀 Starting MCP + HTTP Wrapper on http://localhost:8000 ...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# src/config/mcp_setup.py
import io
import os
import math
import pandas as pd
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import google.generativeai as genai

# ──────────────────────────────────────────────
# ✅ Gemini Setup
# ──────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("❌ GEMINI_API_KEY not found. Please set it before running MCP server.")
genai.configure(api_key=api_key)

# ──────────────────────────────────────────────
# ✅ FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(title="MCP Data Analysis Server (Gemini-powered)")

@app.get("/")
def root():
    return {"status": "running", "message": "🚀 MCP Gemini Data Analysis API active"}

# ──────────────────────────────────────────────
# 🧹 Helper: Make JSON-safe
# ──────────────────────────────────────────────
def make_json_safe(obj):
    """Recursively convert NaN/inf/None to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

# ──────────────────────────────────────────────
# 📤 Upload + Summarize CSV
# ──────────────────────────────────────────────
@app.post("/upload_csv/")
async def upload_csv(file: UploadFile):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        summary = df.describe(include="all").to_dict()
        safe_summary = make_json_safe(summary)

        return JSONResponse(content={
            "filename": file.filename,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "summary": safe_summary,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ──────────────────────────────────────────────
# 🧠 Gemini CSV Analyzer
# ──────────────────────────────────────────────
@app.post("/analyze_csv_with_gemini/")
async def analyze_csv_with_gemini(file: UploadFile, question: str = Form(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))

        stats = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "missing": df.isnull().sum().to_dict(),
            "numeric_summary": make_json_safe(df.describe().to_dict()),
        }

        model = genai.GenerativeModel("gemini-pro-latest")
        prompt = (
            f"You are a data analysis expert.\n\n"
            f"Dataset summary:\n{stats}\n\n"
            f"Question: {question}\n"
            f"Provide concise, structured insights."
        )
        gemini_response = model.generate_content(prompt)
        return JSONResponse(content={
            "status": "success",
            "summary": stats,
            "gemini_insight": gemini_response.text,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ──────────────────────────────────────────────
# 🚀 Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting MCP + HTTP Wrapper on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
