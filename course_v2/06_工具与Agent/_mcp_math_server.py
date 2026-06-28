"""
〔06/07〕配套的最小 MCP server —— 仅供同目录 07_mcp_tools.py 通过 stdio 连接演示。
本身不是一节课、没有新概念；用 MCP 官方 SDK(FastMCP) 暴露两个计算工具。
它会被 07_mcp_tools.py 以子进程(stdio)方式自动拉起，无需手动运行。
依赖：pip install mcp
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """计算两个整数相加。"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """计算两个整数相乘。"""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")        # 通过标准输入输出(stdio)对外提供工具
