"""Plan-and-Solve 智能体中的执行器（Executor）。

职责：按 Planner 生成的计划逐步调用大模型，
每一步都把"原始问题 + 完整计划 + 已完成步骤与结果 + 当前步骤"拼进提示词，
最后一步的输出即为最终答案。
"""

from agentlab.Plan_and_Solve_agent.prompts import EXECUTOR_PROMPT_TEMPLATE
from agentlab.基础流式调用 import HelloAgentsLLM


class Executor:
    """执行器：按计划逐步执行并汇总结果。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        """注入 LLM 客户端，执行每一步时通过它请求模型。"""
        self.llm_client = llm_client

    @staticmethod
    def _format_plan(plan: list[str]) -> str:
        """把计划列表格式化成带序号的文本，方便拼进提示词。"""
        return "\n".join(f"{i}. {step}" for i, step in enumerate(plan, 1))

    def execute(self, question: str, plan: list[str]) -> str:
        """根据计划，逐步执行并解决问题。

        参数:
            question: 用户提出的原始问题。
            plan: Planner 生成的步骤列表，不能为空。

        返回:
            最后一步的执行结果，即最终答案。

        异常:
            ValueError: 计划为空时抛出，避免未定义变量和空转。
        """
        if not plan:
            raise ValueError("计划为空，无法执行")

        # 用列表累积历史记录，避免字符串反复拼接带来的额外开销
        history_parts: list[str] = []
        # 先初始化最终答案，保证即使循环异常退出也有确定的返回值
        final_answer = ""

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")

            # 组装提示词：历史为空时填入"无"，让模型知道这是第一步
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=self._format_plan(plan),
                history="\n\n".join(history_parts) if history_parts else "无",
                current_step=step,
            )

            messages = [{"role": "user", "content": prompt}]
            # think() 调用失败时返回 None，这里统一转成空字符串并去掉首尾空白
            response_text = (self.llm_client.think(messages=messages) or "").strip()

            # 模型没有返回内容时给出占位说明，避免历史里出现空白，也避免下一步空等
            if not response_text:
                response_text = "（本步骤模型未返回内容）"

            # 更新历史记录，为下一步做准备
            history_parts.append(f"步骤 {i}: {step}\n结果: {response_text}")
            final_answer = response_text

            print(f"✅ 步骤 {i} 已完成，结果: {response_text}")

        # 循环结束后，最后一步的响应就是最终答案
        return final_answer
