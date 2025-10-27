# # src/config/mcp_setup.py
# import asyncio
# import google.generativeai as genai
# from mcp.server.fastmcp import FastMCP

# import os

# # Check for Gemini API key
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise RuntimeError("❌ GEMINI_API_KEY not found. Please set it before running MCP server.")

# # Configure Gemini
# genai.configure(api_key=api_key)

# # Create MCP server instance
# server = FastMCP("data_analysis_mcp")

# @server.tool()
# async def analyze_data(prompt: str) -> str:
#     """
#     Send a data analysis instruction to Gemini and return the result.
#     Example:
#         analyze_data("Summarize key patterns in this dataset.")
#     """
#     model = genai.GenerativeModel("gemini-1.5-pro")
#     response = model.generate_content(prompt)
#     return response.text

# # Entry point
# if __name__ == "__main__":
#     print("🚀 Starting MCP server on http://localhost:8000 ...")
#     asyncio.run(server.run(host="0.0.0.0", port=8000))

# src/config/mcp_setup.py
import asyncio
import io
import os
import pandas as pd
import google.generativeai as genai
from mcp.server.fastmcp import FastMCP

# ──────────────────────────────────────────────────────────────
# ✅ Configure Gemini API
# ──────────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("❌ GEMINI_API_KEY not found. Please set it before running MCP server.")

genai.configure(api_key=api_key)

# ──────────────────────────────────────────────────────────────
# ✅ Initialize MCP Server
# ──────────────────────────────────────────────────────────────
server = FastMCP("data_analysis_mcp")

# ──────────────────────────────────────────────────────────────
# 🧩 1. Basic Text Analysis Tool (Gemini)
# ──────────────────────────────────────────────────────────────
@server.tool()
async def analyze_data(prompt: str) -> str:
    """
    Send a text-based analysis prompt to Gemini and return the model output.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text


# ──────────────────────────────────────────────────────────────
# 🧩 2. CSV Upload + Basic Data Stats
# ──────────────────────────────────────────────────────────────
@server.tool()
async def upload_csv(file_bytes: bytes, filename: str = "uploaded.csv") -> dict:
    """
    Upload and summarize a CSV file.
    Returns: {columns, missing_values, summary_statistics}
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))

        # Basic metadata
        columns = df.columns.tolist()
        missing = df.isnull().sum().to_dict()
        summary = df.describe(include="all").to_dict()

        result = {
            "filename": filename,
            "num_rows": len(df),
            "num_columns": len(columns),
            "columns": columns,
            "missing_values": missing,
            "summary_statistics": summary,
        }
        return result
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────
# 🧩 3. CSV + Gemini Insight Tool
# ──────────────────────────────────────────────────────────────
@server.tool()
async def analyze_csv_with_gemini(file_bytes: bytes, question: str) -> dict:
    """
    Combine structured Pandas stats + Gemini insights.
    Example: ask questions like 'Which columns have the strongest correlations?'
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))

        # Describe the dataset
        stats = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "missing": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict(),
        }

        # Prepare text prompt for Gemini
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = (
            f"You are a data analysis assistant.\n\n"
            f"Dataset Summary:\n{stats}\n\n"
            f"Question: {question}\n"
            f"Provide concise, structured insights."
        )

        gemini_response = model.generate_content(prompt)

        return {
            "status": "success",
            "summary": stats,
            "gemini_insight": gemini_response.text,
        }
    except Exception as e:
        return {"error": str(e)}
mcp = FastMCP("data_analysis_mcp")

# -------------------------------------------------------------
#Changes to the code to make it work with the new version of the code
@mcp.tool()
def analyze_data(file_path: str) -> dict:
    """Analyze a CSV file and return structured JSON stats."""
    import pandas as pd, json
    df = pd.read_csv(file_path)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "columns_info": df.dtypes.astype(str).to_dict(),
    }

# -------------------------------------------------------------


# ──────────────────────────────────────────────────────────────
# ✅ Server Entry Point
# ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("🚀 Starting Enhanced MCP Data Analysis Server (Gemini-powered)...")
#     #asyncio.run(server.run(host="0.0.0.0", port=8000))
#     asyncio.run(server.run()) 

# if __name__ == "__main__":
#     print("🚀 Starting Enhanced MCP Data Analysis Server (Gemini-powered)...")
#     mcp.run(
#         host="0.0.0.0",  # Allow external access too
#         port=8000,
#         use_web=True,    # 👈 crucial: ensures FastMCP starts HTTP server
#     )

# if __name__ == "__main__":
#     import uvicorn
#     print("🚀 Starting Enhanced MCP Data Analysis Server (Gemini-powered)...")

#     # Create FastAPI app from MCP
#     app = server.app  # FastMCP exposes the ASGI app

#     # Run with Uvicorn manually
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# ✅ Server Entry Point
# if __name__ == "__main__":
#     import uvicorn
#     from fastapi import FastAPI

#     print("🚀 Starting Enhanced MCP Data Analysis Server (Gemini-powered)...")

#     # Manually create a FastAPI app and mount MCP routes
#     app = FastAPI(title="Data Analysis MCP Server")

#     # Mount the MCP routes onto FastAPI
#     server.register_to_fastapi(app)

#     # Start the FastAPI app
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# ✅ Server Entry Point (compatible with latest FastMCP)
# ✅ Server Entry Point (Final compatible version)
# ✅ HTTP Wrapper around MCP (so you can use curl / Postman)
from fastapi import FastAPI, UploadFile, Form
import uvicorn
import io
import pandas as pd
import asyncio
from fastapi.responses import JSONResponse
import numpy as np
import json


app = FastAPI(title="MCP Data Analysis Server (Gemini-powered)")

# Mount your existing MCP tools for internal use
@app.get("/")
def root():
    return {"status": "running", "message": "🚀 MCP Gemini Data Analysis API active"}

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile):
    """Handle CSV upload and basic summary."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        
        # Generate summary
        summary = df.describe(include="all").replace([np.nan, np.inf, -np.inf], None).to_dict()
        
        # Prepare response
        result = {
            "filename": file.filename,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "summary": summary,
        }

        # Explicitly serialize to valid JSON
        return JSONResponse(content=json.loads(json.dumps(result, default=str)))

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@app.post("/analyze_csv_with_gemini/")
async def analyze_csv_with_gemini(file: UploadFile, question: str = Form(...)):
    """Upload CSV + ask Gemini a question about it."""
    import google.generativeai as genai

    file_bytes = await file.read()
    df = pd.read_csv(io.BytesIO(file_bytes))

    stats = {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "missing": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict(),
    }

    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"You are a data analysis assistant.\n\nDataset summary:\n{stats}\n\nQuestion: {question}\nGive concise, structured insights."
    response = model.generate_content(prompt)

    return {"summary": stats, "gemini_insight": response.text}


if __name__ == "__main__":
    print("🚀 Starting MCP + HTTP Wrapper on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
