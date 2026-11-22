"""Plan-and-Solve 智能体中的规划器（Planner）。

职责：调用大模型，把用户的复杂问题拆解成一系列可执行的步骤，
并将模型返回的列表字符串解析为 Python 列表。
"""

import ast  # ast.literal_eval 只解析字面量，比 eval 安全，不会执行任意代码

from agentlab.Plan_and_Solve_agent.prompts import PLANNER_PROMPT_TEMPLATE
from agentlab.基础流式调用 import HelloAgentsLLM


def _extract_plan_text(response_text: str) -> str:
    """从模型回复中提取计划文本。

    优先取 ```python 代码块里的内容；没有该标记时，
    退而取任意 ``` 代码块的内容；都没有则把整段回复当作计划文本。
    """
    text = response_text.strip()
    if "```" not in text:
        return text

    # 按 ``` 切分后，奇数下标就是代码块内容
    blocks = text.split("```")
    for block in blocks[1::2]:
        # 去掉代码块首行的语言标记（如 python）
        if block.lstrip().lower().startswith("python"):
            return block.split("\n", 1)[1].strip() if "\n" in block else ""
    # 没有 python 标记时，取第一个代码块
    return blocks[1].strip()


class Planner:
    """规划器：根据用户问题生成行动计划。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        """注入 LLM 客户端，生成计划时通过它请求模型。"""
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """根据用户问题生成一个行动计划。

        参数:
            question: 用户提出的原始问题。

        返回:
            计划步骤列表；模型输出为空或无法解析时返回空列表。
        """
        # 把问题填入提示词模板，要求模型按指定格式输出步骤列表
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        # 组装 OpenAI 消息格式，这里只需要一条 user 消息
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        # 调用 LLM 获取完整计划文本（客户端内部为流式输出，最终返回拼接好的字符串）
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        # 模型没有返回内容，直接视为无效计划
        if not response_text.strip():
            print("❌ 模型未返回计划内容")
            return []

        # 提取代码块内容并安全地解析为 Python 列表
        try:
            plan = ast.literal_eval(_extract_plan_text(response_text))
        except (ValueError, SyntaxError, TypeError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []

        # 只接受"字符串列表"形式的计划，其他类型视为无效
        if not isinstance(plan, list) or not all(isinstance(step, str) for step in plan):
            print(f"❌ 计划格式不是字符串列表: {plan!r}")
            return []
        return plan
