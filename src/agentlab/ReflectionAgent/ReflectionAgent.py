"""反思式智能体：执行 → 反思 → 优化的迭代闭环。"""

from typing import Dict, List, Optional, Protocol

from agentlab.ReflectionAgent.ExecutionPrompt import INITIAL_PROMPT_TEMPLATE
from agentlab.ReflectionAgent.Memory import EXECUTION, REFLECTION, Memory
from agentlab.ReflectionAgent.RefinementPrompt import REFINE_PROMPT_TEMPLATE
from agentlab.ReflectionAgent.ReflectionPrompt import REFLECT_PROMPT_TEMPLATE

# 反思器判定「已达到最优」时输出的标记。
# 必须与 ReflectionPrompt.py 中约定的标记保持一致，否则收敛判断会失效。
OPTIMAL_MARKER = "<<OPTIMAL>>"


class LLMClient(Protocol):
    """本模块依赖的最小 LLM 客户端协议。

    think 接收 OpenAI 风格的消息列表，返回完整文本；调用失败时可返回 None。
    """

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
    ) -> Optional[str]:
        ...


class ReflectionAgent:
    """通过多轮「反思 → 优化」逐步改进模型输出的智能体。"""

    def __init__(self, llm_client: LLMClient, max_iterations: int = 3):
        """
        参数:
        - llm_client: 实现了 think() 的 LLM 客户端。
        - max_iterations: 最高的「反思 → 优化」轮数。
        """
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str) -> str:
        """
        依次执行初始生成与多轮反思优化，返回最终代码。

        参数:
        - task (str): 交给智能体的任务描述。
        """
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record(EXECUTION, initial_code)

        # --- 2. 迭代循环:反思与优化 ---
        converged = False
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self._get_last_code()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record(REFLECTION, feedback)

            # b. 检查是否需要停止：依赖明确的标记，避免把普通反馈中的
            #    「无需改进」字样误判为收敛
            if OPTIMAL_MARKER in feedback:
                print("\n反思认为代码已无需改进，任务完成。")
                converged = True
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback,
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record(EXECUTION, refined_code)

        # 跑满最大轮数时，最后一版优化结果没有再经过反思，这里如实告知
        if not converged:
            print(f"\n[注意] 已达到最大轮数 {self.max_iterations}，最后一版代码未再复审。")

        final_code = self._get_last_code()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _get_last_code(self) -> str:
        """从记忆中取出最新一版代码；不存在时直接报错，避免把 None 拼进提示词。"""
        last_code = self.memory.get_last_execution()
        if last_code is None:
            raise RuntimeError("记忆中没有可用的执行结果，无法继续反思。")
        return last_code

    def _get_llm_response(self, prompt: str) -> str:
        """调用 LLM 并返回完整的流式响应文本。

        响应为空时抛出异常：LLM 客户端失败会返回 None，若静默转成空串，
        后续会拿空代码继续反思，最终产出空结果却不报错。
        """
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages)
        if not response_text or not response_text.strip():
            raise RuntimeError("LLM 返回了空响应，请检查模型服务或网络后重试。")
        return response_text
