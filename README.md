
markdown
Copy code
# 🚀 Enhanced MCP Data Analysis System

This project provides an **AI-powered data analysis system** built with **FastAPI**, **Streamlit**, and **Gemini integration**.  
It allows users to upload CSV datasets, analyze them using advanced AI insights, and visualize results in an interactive Streamlit dashboard.

---

## 🧩 Features

- 📊 Upload and analyze CSV datasets
- 🤖 Gemini-powered data interpretation
- ⚡ REST API built with FastAPI
- 🧠 Automatic data summary and type inference
- 🌐 Streamlit frontend for visualization
- 🧱 Modular, production-ready structure

---

## 🏗️ Project Structure

mcp/
├── src/
│ ├── config/
│ │ └── mcp_setup.py # Main server setup file
│ ├── routes/
│ │ └── csv_routes.py # File upload and analysis endpoints
│ ├── tools/
│ │ ├── csv_tool.py # Data loading, cleaning, summary functions
│ │ └── gemini_tool.py # AI analysis logic
│ └── utils/
│ └── helpers.py # Utility/helper methods
├── streamlit_app.py # Streamlit UI for visualization
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd mcp
2️⃣ Create and Activate a Virtual Environment
bash
Copy code
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
🚀 Running the FastAPI Server
To start the backend API:

bash
Copy code
python src/config/mcp_setup.py
You should see:

csharp
Copy code
🚀 Starting MCP + HTTP Wrapper on http://localhost:8000 ...
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
✅ Then test it using curl or Postman:

bash
Copy code
curl http://localhost:8000/
Expected response:

json
Copy code
{"status":"running","message":"🚀 MCP Gemini Data Analysis API active"}
🧮 Upload and Analyze a CSV
You can upload a dataset using:

bash
Copy code
curl -X POST -F "file=@path/to/your.csv" http://localhost:8000/upload_csv/
💡 Streamlit Frontend
To run the Streamlit dashboard:

bash
Copy code
streamlit run streamlit_app.py
Then open the link it gives (usually http://localhost:8501) in your browser.

🧰 API Endpoints Summary
Endpoint	Method	Description
/	GET	Health check
/upload_csv/	POST	Upload a CSV and get summary
/analyze_csv_with_gemini/	POST	AI-powered analysis on uploaded data

🧠 Example Use Case
Upload a dataset like titanic.csv

The API summarizes data (rows, columns, stats)

Streamlit displays interactive visuals

Gemini interprets trends, correlations, and insights

🧾 Requirements
Python 3.10+

FastAPI

Uvicorn

Pandas

Streamlit

Google Gemini SDK (or compatible API)

🤝 Contributing
Feel free to fork and improve!
Pull requests are welcome for:

New AI analysis modes

Enhanced data visualization

Integration with databases (PostgreSQL, Neo4j, etc.)

🪪 License
MIT License © 2025 — Developed by Hrishikesh / BlackCoffer Training








