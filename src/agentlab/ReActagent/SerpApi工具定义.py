import os
import sys

from dotenv import load_dotenv
from serpapi import Client

# Windows 控制台默认 GBK，打印 emoji 会报 UnicodeEncodeError，这里强制 UTF-8 输出
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 加载 .env 文件中的环境变量
load_dotenv()


def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        # serpapi 1.2.0 的用法：Client 只负责鉴权，检索参数交给 search()
        # 返回值是 SerpResults（dict 子类），可以直接用 in / [] / get()
        client = Client(api_key=api_key)
        results = client.search({
            "engine": "google",
            "q": query,
            "gl": "cn",    # 国家代码
            "hl": "zh-cn", # 语言代码
        })

        # 智能解析:优先寻找最直接的答案
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            if snippets:
                return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"
