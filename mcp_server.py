# Entry point to run the MCP server over stdin/stdout
from src.config.mcp_setup import server

if __name__ == '__main__':
    print('Starting MCP server (stdin/stdout)...')
    server.run()
