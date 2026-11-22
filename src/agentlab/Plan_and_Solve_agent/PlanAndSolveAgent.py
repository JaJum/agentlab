"""Plan-and-Solve 智能体主流程。

流程：先由 Planner 生成计划，再由 Executor 按计划逐步执行，
最后一步的结果作为最终答案返回。
"""

from agentlab.Plan_and_Solve_agent.Planner import Planner
from agentlab.Plan_and_Solve_agent.Executor import Executor
from agentlab.基础流式调用 import HelloAgentsLLM


class PlanAndSolveAgent:
    """Plan-and-Solve 智能体：规划 + 执行。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        """初始化智能体，同时创建规划器和执行器实例。"""
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str) -> str | None:
        """运行智能体的完整流程：先规划，后执行。

        参数:
            question: 用户提出的原始问题。

        返回:
            最终答案；计划生成失败时返回 None。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return None

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        return final_answer


if __name__ == "__main__":
    # 直接运行本模块即可体验完整流程（需在 .env 中配置模型与 API Key）
    agent = PlanAndSolveAgent(HelloAgentsLLM())
    agent.run(
        "一个水果店周一卖出的苹果数量是周二的两倍。周二卖出30个。"
        "周三比周二少卖10个。三天总共卖了多少个苹果？"
    )
