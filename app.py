import streamlit as st
import subprocess, sys, json, asyncio

st.title('MCP Data Analysis Agent (Local test client)')
prompt = st.text_area('Prompt to analyze', value='Summarize the following dataset: ...')

if st.button('Run analysis (via MCP client)'):
    st.info('This runs a local MCP client against a server listening on stdin/stdout. For a full test, run the MCP server in a separate terminal: python mcp_server.py')
    st.write('For quick local demo, ensure server is running in another shell.')
