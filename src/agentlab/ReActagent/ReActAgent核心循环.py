from agentlab.ReActagent.prompts import AGENT_SYSTEM_PROMPT
from agentlab.ReActagent.工具执行器 import ToolExecutor
from agentlab.ReActagent.LLM输出的解析器 import _parse_output, _parse_action
from agentlab.基础流式调用 import HelloAgentsLLM
import re 


# 定义 ReAct 智能体类，核心思想是"思考(Thought) -> 行动(Action) -> 观察(Observation)"反复循环
class ReActAgent:
    # 构造方法：注入 LLM 客户端和工具执行器，并设置最大循环步数防止死循环
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        # 保存 LLM 客户端，后续每一步都要用它请求模型
        self.llm_client = llm_client
        # 保存工具执行器，解析出 Action 后要从这里取出对应的工具函数来执行
        self.tool_executor = tool_executor
        # 保存最大步数，作为 while 循环的终止上界（默认 5 步）
        self.max_steps = max_steps
        # 初始化历史记录列表，用于累积每一轮的 Thought/Action/Observation 文本
        self.history = []

    # 智能体主入口：接收一个问题，跑完整的 ReAct 循环直到得出最终答案
    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        # 每次调用都清空历史，避免上一个问题的上下文污染这一次
        self.history = [] # 每次运行时重置历史记录
        # 当前步数计数器初始化为 0，循环内每轮自增
        current_step = 0

        # 只要还没达到最大步数，就继续"思考-行动"循环
        while current_step < self.max_steps:
            # 进入新的一轮，步数加 1
            current_step += 1
            # 打印当前步数，方便观察 Agent 的推理进度
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            # 取出所有已注册工具的描述文本，让模型知道有哪些工具可用
            tools_desc = self.tool_executor.getAvailableTools()
            # 把历史记录列表按换行拼成一个字符串，作为模型的历史上下文
            history_str = "\n".join(self.history)
            # 用三个占位符填充系统提示词模板，生成这一轮完整的提示词
            prompt = AGENT_SYSTEM_PROMPT.format(
                # 填入可用工具清单，模型会据此决定该调用哪个工具
                tools=tools_desc,
                # 填入用户最初提出的问题
                question=question,
                # 填入此前累积的思考与观察记录
                history=history_str
            )

            # 2. 调用LLM进行思考
            # 把拼好的提示词包装成 OpenAI 消息格式，这里只发一条 user 消息
            messages = [{"role": "user", "content": prompt}]
            # 调用 LLM 请求推理结果，内部流式打印，返回值为完整响应文本
            response_text = self.llm_client.think(messages=messages)

            # 检查模型是否返回了有效内容（请求失败或返回空时会走到这里）
            if not response_text:
                # 打印错误原因提示，便于排查网络、密钥或内容过滤问题
                print("错误:LLM未能返回有效响应。")
                # 跳出循环，避免在没有响应的情况下空转
                break

            # ... (后续的解析、执行、整合步骤)
        # (这段逻辑在 run 方法的 while 循环内)
            # 3. 解析LLM的输出
            thought, action = _parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = _parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具
            # (这段逻辑紧随工具调用之后，在 while 循环的末尾)
            print(f"👀 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None
