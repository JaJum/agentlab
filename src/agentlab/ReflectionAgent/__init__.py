"""Reflection（反思）智能体模块。

包含：
- ReflectionAgent：执行 → 反思 → 优化的迭代主流程
- Memory：短期记忆，保存执行与反思轨迹
- ExecutionPrompt / ReflectionPrompt / RefinementPrompt：各阶段提示词模板
"""
