"""ReAct 智能体模块。

包含手写构建 ReAct Agent 的核心组件：
- ReActAgent：思考-行动-观察的主循环
- 输出解析器：从 LLM 回复中解析 Thought / Action
- ToolExecutor：工具注册与执行
- prompts：系统提示词模板
- SerpApi工具定义：基于 SerpApi 的搜索工具
"""
