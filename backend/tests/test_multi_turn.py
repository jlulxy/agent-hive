"""
多轮对话测试集 - 10个场景，每个至少4轮

测试 DirectAgent 的 conversation_history 管理逻辑：
1. tool calling 链是否完整保存
2. 追问时能否引用上一轮的工具结果
3. 智能裁剪是否按轮次正确裁剪
4. 消息角色序列是否合法
5. extract_session_summary 是否兼容新结构

运行方式：
  cd backend && python -m pytest tests/test_multi_turn.py -v
或
  cd backend && python tests/test_multi_turn.py
"""

import asyncio
import json
import sys
import os
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

# 项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.provider import LLMMessage, LLMConfig, LLMProvider


# ============================================================
# Mock LLM Provider - 可编程的 LLM 响应
# ============================================================

class MockLLMProvider(LLMProvider):
    """可编程的 Mock LLM Provider，支持预设 tool calling 和文本回复序列
    
    增强功能：
    - call_log: 记录每次调用时完整的 messages，用于事后断言上下文
    - all_call_log: 跨 reset 保留的完整调用日志（用于多轮测试）
    """
    
    def __init__(self):
        self._responses = []  # 预设的响应队列
        self._call_idx = 0
        self.call_log: List[List[LLMMessage]] = []  # 当前轮的调用日志
        self.all_call_log: List[List[LLMMessage]] = []  # 跨 reset 的完整调用日志
    
    def add_response(self, content: str = "", tool_calls: Optional[List[Dict]] = None):
        """添加一个预设响应（按调用顺序消费）"""
        self._responses.append({
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else "stop"
        })
    
    async def chat_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        snapshot = list(messages)
        self.call_log.append(snapshot)
        self.all_call_log.append(snapshot)
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        return {"content": "", "tool_calls": None, "finish_reason": "stop"}
    
    async def chat(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        snapshot = list(messages)
        self.call_log.append(snapshot)
        self.all_call_log.append(snapshot)
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            content = resp.get("content", "")
            if content:
                for i in range(0, len(content), 20):
                    yield content[i:i+20]
        else:
            yield "[Mock] No more responses"
    
    def reset(self):
        """重置响应队列和当前轮日志（保留 all_call_log）"""
        self._responses.clear()
        self._call_idx = 0
        self.call_log.clear()
    
    def get_last_call_messages(self) -> List[LLMMessage]:
        """获取最后一次 LLM 调用收到的 messages"""
        return self.all_call_log[-1] if self.all_call_log else []
    
    def get_last_call_context_text(self) -> str:
        """获取最后一次 LLM 调用的全部上下文文本（用于关键词搜索）"""
        msgs = self.get_last_call_messages()
        return " ".join(m.content or "" for m in msgs)


# ============================================================
# Mock SkillSet - 可编程的技能执行
# ============================================================

@dataclass
class MockSkillResult:
    success: bool = True
    result: str = ""
    summary: str = ""
    error: Optional[str] = None

class MockSkillSet:
    """Mock SkillSet，让我们控制技能返回的数据"""
    
    def __init__(self):
        self._results: Dict[str, MockSkillResult] = {}
    
    def set_result(self, skill_name: str, result: str, summary: str = ""):
        self._results[skill_name] = MockSkillResult(
            success=True, result=result, summary=summary or result[:80]
        )
    
    async def execute_skill(self, skill_name: str, **kwargs) -> MockSkillResult:
        return self._results.get(skill_name, MockSkillResult(success=False, error="Unknown skill"))
    
    def get_tool_definitions(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "web-search",
                    "description": "搜索网络",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "搜索关键词"}
                        },
                        "required": ["task"]
                    }
                }
            }
        ]
    
    def list_skills(self):
        return list(self._results.keys())
    
    def assign_skills(self, names):
        return len(names)


# ============================================================
# 辅助函数
# ============================================================

def build_tool_call(func_name: str, args: Dict, call_id: str = None) -> Dict:
    """构建一个 tool_call 字典"""
    return {
        "id": call_id or f"call_{func_name}_{id(args)}",
        "type": "function",
        "function": {
            "name": func_name,
            "arguments": json.dumps(args, ensure_ascii=False)
        }
    }

def count_roles(history: List[LLMMessage]) -> Dict[str, int]:
    """统计 conversation_history 中各角色的消息数"""
    counts = {}
    for m in history:
        counts[m.role] = counts.get(m.role, 0) + 1
    return counts

def count_rounds(history: List[LLMMessage]) -> int:
    """统计对话轮次（user 消息数量）"""
    return sum(1 for m in history if m.role == "user")

def assert_context_contains(provider: MockLLMProvider, keywords: List[str], 
                           call_index: int = -1, description: str = "") -> List[str]:
    """断言 LLM 收到的 context 中包含指定关键词
    
    这是「回答质量」测试的核心：如果传给 LLM 的上下文包含了正确的历史信息，
    那么一个合格的 LLM 就应该能给出正确的回答。
    
    Args:
        provider: MockLLMProvider 实例
        keywords: 必须出现的关键词列表
        call_index: 检查第几次调用（-1 为最后一次）
        description: 描述信息
    
    Returns:
        错误列表（空列表表示全部通过）
    """
    errors = []
    if not provider.all_call_log:
        errors.append(f"[{description}] LLM 从未被调用")
        return errors
    
    try:
        messages = provider.all_call_log[call_index]
    except IndexError:
        errors.append(f"[{description}] 调用索引 {call_index} 超出范围 (共 {len(provider.all_call_log)} 次调用)")
        return errors
    
    context_text = " ".join(m.content or "" for m in messages)
    
    for kw in keywords:
        if kw not in context_text:
            errors.append(f"[{description}] LLM 上下文中缺少关键词: '{kw}'")
    
    return errors


def assert_context_has_role(provider: MockLLMProvider, role: str, 
                            call_index: int = -1, description: str = "") -> List[str]:
    """断言 LLM 收到的 messages 中包含指定角色的消息"""
    errors = []
    if not provider.all_call_log:
        errors.append(f"[{description}] LLM 从未被调用")
        return errors
    
    try:
        messages = provider.all_call_log[call_index]
    except IndexError:
        errors.append(f"[{description}] 调用索引 {call_index} 超出范围")
        return errors
    
    if not any(m.role == role for m in messages):
        errors.append(f"[{description}] LLM messages 中缺少 role='{role}' 的消息")
    
    return errors


def assert_response_quality(response: str, expected_keywords: List[str], 
                           description: str = "") -> List[str]:
    """断言 LLM 的回复中包含预期的关键内容
    
    在 Mock 场景下，这验证的是我们预设的回复是否合理。
    在真实 LLM 场景下，这验证的是模型是否正确利用了上下文。
    
    Args:
        response: LLM 的回复文本
        expected_keywords: 回复中应包含的关键词
        description: 描述信息
    
    Returns:
        错误列表
    """
    errors = []
    for kw in expected_keywords:
        if kw not in response:
            errors.append(f"[{description}] 回复中缺少预期内容: '{kw}'")
    return errors


def validate_message_sequence(history: List[LLMMessage]) -> List[str]:
    """验证消息序列的合法性，返回错误列表"""
    errors = []
    if not history:
        return errors
    
    # 第一条应该是 user
    if history[0].role != "user":
        errors.append(f"第一条消息应为 user，实际为 {history[0].role}")
    
    for i, msg in enumerate(history):
        # tool 消息前必须有 assistant(tool_calls) 消息
        if msg.role == "tool":
            # 向前找最近的 assistant 消息
            found_tc = False
            for j in range(i-1, -1, -1):
                if history[j].role == "assistant" and history[j].tool_calls:
                    found_tc = True
                    break
                if history[j].role == "user":
                    break
            if not found_tc:
                errors.append(f"第 {i} 条 tool 消息前缺少 assistant(tool_calls) 消息")
    
    return errors


# ============================================================
# 直接模拟 DirectAgent 的核心逻辑（不依赖完整框架）
# ============================================================

class DirectAgentSimulator:
    """
    模拟 DirectAgent 的 conversation_history 管理逻辑，
    用于测试而不需要完整的框架依赖（如 skills、memory、events 等）。
    
    这里复现了 execute_task 中消息构建和保存的核心逻辑。
    """
    
    def __init__(self, provider: MockLLMProvider, skill_set: MockSkillSet):
        self.provider = provider
        self.skill_set = skill_set
        self.conversation_history: List[LLMMessage] = []
        self.llm_config = LLMConfig(model="mock-model")
    
    async def execute_task(self, task: str) -> str:
        """模拟 execute_task 的核心逻辑"""
        
        # 构建消息
        messages = [
            LLMMessage(role="system", content="你是一个AI助手，正处于多轮对话中。"),
        ]
        messages.extend(self.conversation_history)
        messages.append(LLMMessage(role="user", content=task))
        
        # 记录 history 长度，用于后面提取新消息
        history_len = len(self.conversation_history)
        
        # Tool calling 循环
        tool_definitions = self.skill_set.get_tool_definitions()
        max_tool_rounds = 5
        full_response = ""
        
        for tool_round in range(max_tool_rounds):
            if not tool_definitions:
                break
            
            response = await self.provider.chat_complete(
                messages, self.llm_config, tools=tool_definitions
            )
            
            content = response.get("content", "")
            tool_calls = response.get("tool_calls")
            
            if not tool_calls:
                break
            
            # 有工具调用
            messages.append(LLMMessage(
                role="assistant",
                content=content or "",
                tool_calls=tool_calls
            ))
            
            for tc in tool_calls:
                tool_call_id = tc["id"]
                func_name = tc["function"]["name"]
                
                result = await self.skill_set.execute_skill(skill_name=func_name)
                tool_result_str = result.result if result.success else (result.error or "执行失败")
                
                messages.append(LLMMessage(
                    role="tool",
                    content=str(tool_result_str) if tool_result_str else "无结果",
                    tool_call_id=tool_call_id,
                ))
        
        # 最终流式回复
        async for chunk in self.provider.chat(messages, self.llm_config):
            full_response += chunk
        
        # ===== 更新对话历史（完整保存 tool calling 链）=====
        history_start_idx = 1 + len(self.conversation_history)  # 1 for system prompt
        new_messages = messages[history_start_idx:]
        
        for msg in new_messages:
            if msg.role == "tool" and msg.content and len(msg.content) > 1500:
                msg = LLMMessage(
                    role=msg.role,
                    content=msg.content[:1500] + "\n...(结果已截取前1500字符)",
                    tool_call_id=msg.tool_call_id,
                )
            self.conversation_history.append(msg)
        
        if full_response.strip():
            self.conversation_history.append(LLMMessage(role="assistant", content=full_response))
        
        # 智能裁剪
        self._trim_conversation_history(max_rounds=6)
        
        return full_response
    
    def _trim_conversation_history(self, max_rounds: int = 6):
        """基于对话轮次的智能裁剪 + token 预算裁剪"""
        if not self.conversation_history:
            return
        
        round_starts = []
        for i, msg in enumerate(self.conversation_history):
            if msg.role == "user":
                round_starts.append(i)
        
        # 基础裁剪：按轮次
        if len(round_starts) > max_rounds:
            trim_from = round_starts[-max_rounds]
            self.conversation_history = self.conversation_history[trim_from:]
            round_starts = [i for i, m in enumerate(self.conversation_history) if m.role == "user"]
        
        # Token 预算裁剪
        MAX_HISTORY_CHARS = 24000
        total_chars = sum(len(m.content or "") for m in self.conversation_history)
        
        while total_chars > MAX_HISTORY_CHARS and len(round_starts) > 2:
            next_round_start = round_starts[1] if len(round_starts) > 1 else len(self.conversation_history)
            removed_chars = sum(len(m.content or "") for m in self.conversation_history[:next_round_start])
            self.conversation_history = self.conversation_history[next_round_start:]
            total_chars -= removed_chars
            round_starts = [i for i, m in enumerate(self.conversation_history) if m.role == "user"]
    
    def extract_session_summary(self) -> Dict[str, Any]:
        """提取会话摘要"""
        final_report = ""
        if self.conversation_history:
            assistant_msgs = [
                m.content for m in self.conversation_history
                if m.role == "assistant" and m.content and not m.tool_calls
            ]
            if assistant_msgs:
                final_report = assistant_msgs[-1][:2000]
        
        return {"final_report": final_report}


# ============================================================
# 测试用例
# ============================================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors: List[str] = []
        self.details: List[str] = []
    
    def add_error(self, msg: str):
        self.passed = False
        self.errors.append(msg)
    
    def add_detail(self, msg: str):
        self.details.append(msg)
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [f"{status} {self.name}"]
        for d in self.details:
            lines.append(f"  📝 {d}")
        for e in self.errors:
            lines.append(f"  ❗ {e}")
        return "\n".join(lines)


async def test_01_basic_follow_up_with_tool_results():
    """测试1：基础追问 - 搜索推荐后追问具体内容（复现原始 bug）
    
    核心验证：
    - 轮次2追问"悬案解码"时，LLM context 中是否包含第1轮的搜索原始数据
    - 轮次3追问"真探"时，LLM context 中是否包含"True Detective"、"豆瓣9.2"等第1轮数据
    - 轮次4综合追问时，LLM context 中是否同时包含多部剧的数据
    """
    result = TestResult("基础追问 - 搜索推荐后追问（含上下文质量断言）")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    # --- 轮次 1：搜索推荐 ---
    search_result_data = """搜索结果：
1. 悬案解码 (Unresolved) - 2024年Netflix悬疑剧，豆瓣8.9分，讲述FBI探员调查连环悬案
2. 真探 (True Detective) - HBO经典悬疑剧，第一季豆瓣9.2分，马修·麦康纳主演
3. 暗黑 (Dark) - 德国悬疑科幻剧，豆瓣9.0分，时间旅行+悬疑
4. 利器 (Sharp Objects) - HBO迷你剧，艾米·亚当斯主演，心理悬疑"""
    skill_set.set_result("web-search", search_result_data, "搜索到4部海外悬疑剧")
    
    tc_id = "call_search_001"
    provider.add_response(
        content="让我搜索一下热门海外悬疑剧。",
        tool_calls=[build_tool_call("web-search", {"task": "好看的海外悬疑剧推荐"}, tc_id)]
    )
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="根据搜索结果，为你推荐以下海外悬疑剧：\n1. **悬案解码** - Netflix 2024年新作，豆瓣8.9\n2. **真探** - HBO经典，豆瓣9.2\n3. **暗黑** - 德国科幻悬疑，豆瓣9.0\n4. **利器** - HBO迷你剧，心理悬疑")
    
    resp1 = await agent.execute_task("推荐好看的海外悬疑剧")
    
    # 验证回复质量
    for e in assert_response_quality(resp1, ["悬案解码", "真探", "暗黑", "利器"], "轮次1回复"):
        result.add_error(e)
    
    # 验证 history 数据完整性
    if not any(m.role == "tool" for m in agent.conversation_history):
        result.add_error("轮次1后 conversation_history 中缺少 tool 消息")
    else:
        result.add_detail("✓ 搜索结果数据已保存到 history")
    
    # --- 轮次 2：追问"悬案解码" ---
    provider.reset()
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="**悬案解码 (Unresolved)** 是2024年Netflix推出的悬疑剧，豆瓣评分8.9分，讲述FBI探员深入调查一系列连环悬案...")
    
    resp2 = await agent.execute_task("你推荐的悬案解码能不能展开讲讲")
    
    # ★ 核心断言：轮次2时 LLM 收到的 context 中必须包含第1轮搜索的原始数据
    for e in assert_context_contains(provider, 
        ["悬案解码", "FBI探员", "豆瓣8.9"], description="轮次2上下文应含搜索数据"):
        result.add_error(e)
    for e in assert_context_has_role(provider, "tool", description="轮次2上下文应含tool消息"):
        result.add_error(e)
    # 回复应引用搜索中的具体数据
    for e in assert_response_quality(resp2, ["悬案解码", "Netflix", "2024"], "轮次2回复"):
        result.add_error(e)
    
    result.add_detail("✓ 轮次2: LLM context 包含第1轮搜索数据，回复引用了正确内容")
    
    # --- 轮次 3：追问"真探" ---
    provider.reset()
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="**真探 (True Detective)** 第一季是HBO经典悬疑剧，豆瓣9.2分，马修·麦康纳饰演的探员深入调查...")
    
    resp3 = await agent.execute_task("真探呢？")
    
    # ★ 核心断言：轮次3时 LLM context 仍包含第1轮的搜索数据（真探相关）
    for e in assert_context_contains(provider,
        ["True Detective", "豆瓣9.2", "马修·麦康纳"], description="轮次3上下文应含真探数据"):
        result.add_error(e)
    for e in assert_response_quality(resp3, ["真探", "HBO", "9.2"], "轮次3回复"):
        result.add_error(e)
    
    result.add_detail("✓ 轮次3: LLM context 包含真探的原始搜索数据")
    
    # --- 轮次 4：综合追问 ---
    provider.reset()
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="如果你是悬疑剧入门，我推荐从**真探第一季**开始，它豆瓣9.2分是最高的...")
    
    resp4 = await agent.execute_task("这几部哪部最适合入门？")
    
    # ★ 核心断言：轮次4时 LLM context 应同时包含多部剧的数据（才能做综合推荐）
    for e in assert_context_contains(provider,
        ["悬案解码", "真探", "暗黑", "利器"], description="轮次4上下文应含所有剧目"):
        result.add_error(e)
    for e in assert_response_quality(resp4, ["真探", "入门"], "轮次4回复"):
        result.add_error(e)
    
    result.add_detail("✓ 轮次4: LLM context 包含所有历史数据，可做综合判断")
    
    # 结构验证
    total_rounds = count_rounds(agent.conversation_history)
    if total_rounds != 4:
        result.add_error(f"应有4轮对话，实际 {total_rounds} 轮")
    
    seq_errors = validate_message_sequence(agent.conversation_history)
    for e in seq_errors:
        result.add_error(f"消息序列错误: {e}")
    
    return result


async def test_02_multi_tool_calls_in_one_round():
    """测试2：单轮多工具调用 - LLM 在一轮中调用多个工具
    
    轮次：
    1. 用户：对比北京和上海今天的天气
       LLM：调用搜索(北京天气) + 搜索(上海天气) → 综合回复
    2. 用户：哪个城市更适合户外活动？
    3. 用户：明天呢？
    4. 用户：总结一下
    """
    result = TestResult("单轮多工具调用")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    # --- 轮次 1：两个 tool calls ---
    skill_set.set_result("web-search", "北京今天：晴，25°C，适合户外")
    
    tc1 = build_tool_call("web-search", {"task": "北京今天天气"}, "call_bj")
    tc2 = build_tool_call("web-search", {"task": "上海今天天气"}, "call_sh")
    
    provider.add_response(content="我来查一下两个城市的天气。", tool_calls=[tc1, tc2])
    # 注意：MockSkillSet 对同一 skill 只有一个结果，这里简化
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="北京：晴25°C，上海：阴22°C。北京更适合户外。")
    
    await agent.execute_task("对比北京和上海今天的天气")
    
    # 验证多个 tool 消息都被保存
    tool_msgs = [m for m in agent.conversation_history if m.role == "tool"]
    result.add_detail(f"轮次1后 tool 消息数: {len(tool_msgs)}")
    if len(tool_msgs) < 2:
        result.add_error(f"应有2条 tool 消息（双工具调用），实际 {len(tool_msgs)} 条")
    
    # 验证 assistant(tool_calls) 消息保存了 tool_calls 字段
    tc_msgs = [m for m in agent.conversation_history if m.role == "assistant" and m.tool_calls]
    if not tc_msgs:
        result.add_error("assistant(tool_calls) 消息未保存 tool_calls 字段")
    else:
        result.add_detail(f"✓ assistant(tool_calls) 消息已保存，含 {len(tc_msgs[0].tool_calls)} 个工具调用")
    
    # --- 轮次 2-4 ---
    for q in ["哪个城市更适合户外活动？", "明天呢？", "总结一下两天的天气对比"]:
        provider.reset()
        provider.add_response(content="", tool_calls=None)
        provider.add_response(content=f"关于{q[:10]}的回复...")
        await agent.execute_task(q)
    
    total_rounds = count_rounds(agent.conversation_history)
    if total_rounds != 4:
        result.add_error(f"应有4轮，实际 {total_rounds} 轮")
    
    seq_errors = validate_message_sequence(agent.conversation_history)
    for e in seq_errors:
        result.add_error(f"消息序列错误: {e}")
    
    return result


async def test_03_no_tool_pure_conversation():
    """测试3：纯文本对话（无工具调用）- 验证多轮纯文本上下文传递
    
    核心验证：
    - 每轮 LLM 调用时是否能看到之前所有轮的问答
    - 轮次5（"前景如何"）时 context 是否包含前4轮讨论的量子计算概念
    """
    result = TestResult("纯文本对话（含上下文累积断言）")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    skill_set.get_tool_definitions = lambda: []
    agent = DirectAgentSimulator(provider, skill_set)
    
    qa_pairs = [
        ("什么是量子计算？", "量子计算是利用量子力学原理（叠加态、纠缠）进行计算的技术。"),
        ("它和经典计算有什么区别？", "经典计算使用0和1的比特，量子计算使用量子比特（qubit），可同时表示0和1。"),
        ("量子比特是什么？", "量子比特（qubit）是量子计算的基本单元，具有叠加态特性。"),
        ("用一个比喻来解释", "想象一个硬币：经典比特是正面或反面，量子比特是硬币在空中旋转时同时是两面。"),
        ("这个领域的前景如何？", "量子计算前景广阔，IBM和Google等都在推进，但仍面临退相干等技术挑战。"),
    ]
    
    for i, (q, a) in enumerate(qa_pairs):
        provider.reset()
        provider.add_response(content=a)
        await agent.execute_task(q)
    
    # ★ 核心断言：最后一轮 LLM 收到的 context 应包含前面讨论的关键概念
    for e in assert_context_contains(provider,
        ["量子力学", "叠加态", "qubit", "硬币"],
        description="轮次5上下文应含前4轮关键概念"):
        result.add_error(e)
    result.add_detail("✓ 最后一轮 LLM 上下文包含前4轮关键概念")
    
    total_rounds = count_rounds(agent.conversation_history)
    roles = count_roles(agent.conversation_history)
    
    if total_rounds != 5:
        result.add_error(f"应有5轮，实际 {total_rounds} 轮")
    if "tool" in roles:
        result.add_error("纯文本对话不应有 tool 消息")
    if roles.get("user", 0) != roles.get("assistant", 0):
        result.add_error(f"user({roles.get('user')}) 和 assistant({roles.get('assistant')}) 消息数不匹配")
    
    return result


async def test_04_pronoun_reference_across_rounds():
    """测试4：跨轮代词引用 - 验证 LLM 上下文中包含正确的历史数据支撑代词解析
    
    核心验证：
    - 轮次2追问"它的并发模型"时，LLM context 是否包含 Go/goroutine 的搜索数据
    - 轮次3追问"goroutine"时，LLM context 是否仍包含完整搜索数据+前轮回复
    - 轮次4追问对比时，LLM context 是否同时包含 Go 和 Python 的数据
    """
    result = TestResult("跨轮代词引用（含上下文质量断言）")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    search_data = "Go适合高并发后端，goroutine轻量级协程，由Go runtime管理；Python适合AI/ML和快速开发，asyncio提供异步IO，GIL限制多线程"
    skill_set.set_result("web-search", search_data)
    
    # 轮次1：搜索
    tc_id = "call_search_compare"
    provider.add_response(
        content="搜索中...",
        tool_calls=[build_tool_call("web-search", {"task": "Python vs Go 后端开发"}, tc_id)]
    )
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="Go更适合高并发场景，拥有goroutine。Python更适合AI/ML。")
    await agent.execute_task("Python 和 Go 哪个更适合写后端？")
    
    # 轮次2：代词"它"指代追问
    provider.reset()
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="Go 的并发模型基于 CSP（通信顺序进程），goroutine 是其核心...")
    resp2 = await agent.execute_task("它的并发模型是怎样的？")
    
    # ★ 核心断言：轮次2 LLM context 应包含搜索数据（才能解析"它"指 Go）
    for e in assert_context_contains(provider,
        ["goroutine", "Go适合高并发"], description="轮次2上下文应含Go搜索数据"):
        result.add_error(e)
    for e in assert_context_has_role(provider, "tool", description="轮次2应看到历史tool"):
        result.add_error(e)
    result.add_detail("✓ 轮次2: 上下文包含 goroutine/Go 数据，支持代词解析")
    
    # 轮次3：引用前轮回复中的具体术语
    provider.reset()
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="goroutine 是 Go 语言的轻量级线程，由 Go runtime 而非 OS 调度...")
    resp3 = await agent.execute_task("你刚才提到的 goroutine 是什么？")
    
    # ★ 核心断言：轮次3上下文应同时包含搜索数据和前轮回复中的"CSP"
    for e in assert_context_contains(provider,
        ["goroutine", "CSP"], description="轮次3上下文应含搜索数据+前轮回复"):
        result.add_error(e)
    result.add_detail("✓ 轮次3: 上下文包含 goroutine+CSP，可解析'你刚才提到的'")
    
    # 轮次4：对比追问
    provider.reset()
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="Go 的 goroutine 由 Go runtime 调度，可轻松创建上百万个；Python 的 asyncio 是单线程事件循环，受 GIL 限制...")
    resp4 = await agent.execute_task("和 Python 的协程比呢？")
    
    # ★ 核心断言：轮次4应同时包含 Go 和 Python 的数据
    for e in assert_context_contains(provider,
        ["goroutine", "asyncio", "GIL"], description="轮次4上下文应含Go+Python数据"):
        result.add_error(e)
    for e in assert_response_quality(resp4, ["goroutine", "asyncio"], "轮次4回复应对比两者"):
        result.add_error(e)
    result.add_detail("✓ 轮次4: 上下文同时含 Go+Python 数据，支持对比回答")
    
    total_rounds = count_rounds(agent.conversation_history)
    if total_rounds != 4:
        result.add_error(f"应有4轮，实际 {total_rounds} 轮")
    
    return result


async def test_05_tool_result_truncation():
    """测试5：工具结果截断 - 验证超长工具结果被正确截断
    
    轮次：
    1. 用户：搜索最新的AI论文 → 返回超长结果（>1500字符）
    2. 用户：第一篇论文讲了什么？
    3. 用户：它的方法论是什么？
    4. 用户：总结一下
    """
    result = TestResult("工具结果截断")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    # 构造超长搜索结果（>1500字符）
    long_result = "AI论文搜索结果：\n" + "\n".join([
        f"论文{i}: {'A' * 100} 摘要：{'B' * 100}" for i in range(20)
    ])
    assert len(long_result) > 1500, f"测试数据太短: {len(long_result)}"
    skill_set.set_result("web-search", long_result)
    
    # 轮次1
    tc_id = "call_search_papers"
    provider.add_response(
        content="搜索中...",
        tool_calls=[build_tool_call("web-search", {"task": "最新AI论文"}, tc_id)]
    )
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="搜索到以下AI论文：\n1. 论文0...\n2. 论文1...")
    await agent.execute_task("搜索最新的AI论文")
    
    # 验证截断
    tool_msgs = [m for m in agent.conversation_history if m.role == "tool"]
    if tool_msgs:
        tool_content_len = len(tool_msgs[0].content)
        result.add_detail(f"工具结果长度: {tool_content_len} (原始: {len(long_result)})")
        if tool_content_len > 1600:  # 1500 + 截断提示
            result.add_error(f"工具结果未被截断: {tool_content_len} > 1600")
        if "结果已截取" in tool_msgs[0].content:
            result.add_detail("✓ 截断提示已添加")
        else:
            result.add_error("截断提示缺失")
    else:
        result.add_error("tool 消息缺失")
    
    # 轮次2-4
    for q in ["第一篇论文讲了什么？", "它的方法论是什么？", "总结一下"]:
        provider.reset()
        provider.add_response(content="", tool_calls=None)
        provider.add_response(content=f"关于 {q[:10]} ...")
        await agent.execute_task(q)
    
    return result


async def test_06_trim_keeps_recent_rounds():
    """测试6：裁剪策略 - 超过 max_rounds 时正确保留最近轮次
    
    执行 8 轮对话（含 tool calling），验证裁剪后保留最近 6 轮
    """
    result = TestResult("裁剪策略 - 保留最近 N 轮")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    skill_set.get_tool_definitions = lambda: []  # 无工具，简化
    agent = DirectAgentSimulator(provider, skill_set)
    
    # 执行 8 轮纯文本对话
    for i in range(8):
        provider.reset()
        provider.add_response(content=f"这是第 {i+1} 轮的回复。")
        await agent.execute_task(f"第 {i+1} 个问题")
    
    total_rounds = count_rounds(agent.conversation_history)
    result.add_detail(f"8轮后实际保留轮次: {total_rounds}")
    
    if total_rounds != 6:
        result.add_error(f"应保留最近6轮，实际 {total_rounds} 轮")
    
    # 验证保留的是最近6轮（第3-8轮）
    first_user_msg = next(m for m in agent.conversation_history if m.role == "user")
    if "第 3 个问题" not in first_user_msg.content:
        result.add_error(f"最早的 user 消息应是第3轮，实际: {first_user_msg.content}")
    else:
        result.add_detail("✓ 裁剪正确保留第3-8轮")
    
    return result


async def test_07_trim_with_tool_calls():
    """测试7：带 tool calling 的裁剪 - 验证裁剪时保持 tool calling 链完整
    
    轮次1-3: 带搜索的对话
    轮次4-7: 纯文本对话
    验证裁剪后 tool calling 链的完整性
    """
    result = TestResult("带 tool calling 的裁剪完整性")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    search_data = "搜索结果：测试数据"
    skill_set.set_result("web-search", search_data)
    
    # 轮次1-3：带搜索
    for i in range(3):
        provider.reset()
        tc_id = f"call_{i}"
        provider.add_response(
            content="搜索...",
            tool_calls=[build_tool_call("web-search", {"task": f"query_{i}"}, tc_id)]
        )
        provider.add_response(content="", tool_calls=None)
        provider.add_response(content=f"搜索轮 {i+1} 的回复")
        await agent.execute_task(f"搜索问题 {i+1}")
    
    # 轮次4-7：纯文本
    skill_set.get_tool_definitions = lambda: []
    for i in range(4):
        provider.reset()
        provider.add_response(content=f"纯文本轮 {i+4} 的回复")
        await agent.execute_task(f"文本问题 {i+4}")
    
    total_rounds = count_rounds(agent.conversation_history)
    result.add_detail(f"7轮后保留轮次: {total_rounds}")
    
    # 验证保留6轮（轮次2-7）
    if total_rounds != 6:
        result.add_error(f"应保留6轮，实际 {total_rounds} 轮")
    
    # 验证消息序列合法性
    seq_errors = validate_message_sequence(agent.conversation_history)
    for e in seq_errors:
        result.add_error(f"消息序列错误: {e}")
    
    if not seq_errors:
        result.add_detail("✓ 裁剪后消息序列合法")
    
    return result


async def test_08_extract_summary_with_tool_chain():
    """测试8：extract_session_summary 兼容性 - 验证带 tool calling 时摘要提取正确
    
    轮次：
    1. 搜索 + 回复
    2. 追问
    3. 再追问
    4. 总结
    验证 extract_session_summary 返回最后一条纯文本 assistant 回复
    """
    result = TestResult("extract_session_summary 兼容性")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    skill_set.set_result("web-search", "搜索数据...")
    
    # 轮次1：搜索
    tc_id = "call_s1"
    provider.add_response(
        content="搜索中...",
        tool_calls=[build_tool_call("web-search", {"task": "test"}, tc_id)]
    )
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="搜索结果的总结回复")
    await agent.execute_task("搜索一下")
    
    # 轮次2-4：纯文本
    skill_set.get_tool_definitions = lambda: []
    for i, (q, a) in enumerate([
        ("追问1", "追问1的回复"),
        ("追问2", "追问2的回复"),
        ("总结一下", "这是最终的总结回复，包含所有关键信息。"),
    ]):
        provider.reset()
        provider.add_response(content=a)
        await agent.execute_task(q)
    
    summary = agent.extract_session_summary()
    result.add_detail(f"摘要: {summary['final_report'][:80]}...")
    
    # 应返回最后一条纯文本 assistant 回复
    if "最终的总结回复" not in summary["final_report"]:
        result.add_error("摘要应为最后一条纯文本回复")
    else:
        result.add_detail("✓ 摘要正确提取最后一条纯文本回复")
    
    # 不应返回中间的 assistant(tool_calls) 消息
    if "搜索中" in summary["final_report"]:
        result.add_error("摘要不应包含中间 tool calling 消息")
    
    return result


async def test_09_interleaved_tool_and_text():
    """测试9：交替使用工具和纯文本 - 验证跨工具/纯文本轮次的上下文完整性
    
    核心验证：
    - 轮次3追问"吉祥物"时，LLM context 包含轮次2的搜索结果（北京奥运2008）
    - 轮次5追问"相隔多少年"时，LLM context 同时包含两次搜索结果（2008+2024）
    """
    result = TestResult("交替使用工具和纯文本（含上下文质量断言）")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    search_data_1 = "搜索结果1：北京奥运2008年举办，主场馆鸟巢，吉祥物福娃"
    search_data_2 = "搜索结果2：巴黎奥运2024年举办，主场馆法兰西体育场"
    skill_set.set_result("web-search", search_data_1)
    
    # 轮次1：纯文本
    original_tool_defs = skill_set.get_tool_definitions
    skill_set.get_tool_definitions = lambda: []
    provider.add_response(content="你好！有什么可以帮你的？")
    await agent.execute_task("你好")
    skill_set.get_tool_definitions = original_tool_defs
    
    # 轮次2：搜索
    provider.reset()
    tc_id1 = "call_bj_olympic"
    provider.add_response(
        content="查询中...",
        tool_calls=[build_tool_call("web-search", {"task": "北京奥运会"}, tc_id1)]
    )
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="北京奥运会于2008年举办。")
    await agent.execute_task("北京奥运会是哪年举办的？")
    
    # 轮次3：纯文本追问（基于搜索结果）
    provider.reset()
    skill_set.get_tool_definitions = lambda: []
    provider.add_response(content="北京奥运会的吉祥物是福娃，由五个形象组成...")
    resp3 = await agent.execute_task("吉祥物是什么？")
    skill_set.get_tool_definitions = original_tool_defs
    
    # ★ 核心断言：轮次3 LLM context 应包含轮次2的搜索数据
    for e in assert_context_contains(provider,
        ["北京奥运2008", "福娃"], description="轮次3上下文应含北京奥运搜索数据"):
        result.add_error(e)
    result.add_detail("✓ 轮次3: 纯文本追问时上下文含轮次2搜索数据")
    
    # 轮次4：搜索
    provider.reset()
    skill_set.set_result("web-search", search_data_2)
    tc_id2 = "call_paris_olympic"
    provider.add_response(
        content="查询中...",
        tool_calls=[build_tool_call("web-search", {"task": "巴黎奥运会"}, tc_id2)]
    )
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="巴黎奥运会于2024年举办。")
    await agent.execute_task("最近一届奥运会呢？")
    
    # 轮次5：纯文本对比追问
    provider.reset()
    skill_set.get_tool_definitions = lambda: []
    provider.add_response(content="两届奥运会相隔16年，北京2008到巴黎2024。")
    resp5 = await agent.execute_task("两届相隔多少年？")
    
    # ★ 核心断言：轮次5 LLM context 应同时包含两次搜索数据
    for e in assert_context_contains(provider,
        ["北京奥运2008", "巴黎奥运2024"], description="轮次5上下文应含两次搜索数据"):
        result.add_error(e)
    for e in assert_response_quality(resp5, ["16年", "2008", "2024"], "轮次5回复应含两个年份"):
        result.add_error(e)
    result.add_detail("✓ 轮次5: 上下文同时含两次搜索数据，可做跨轮对比")
    
    total_rounds = count_rounds(agent.conversation_history)
    if total_rounds != 5:
        result.add_error(f"应有5轮，实际 {total_rounds} 轮")
    
    seq_errors = validate_message_sequence(agent.conversation_history)
    for e in seq_errors:
        result.add_error(f"消息序列错误: {e}")
    
    return result


async def test_10_deep_reference_chain():
    """测试10：深度引用链 - 第4轮引用第1轮的具体数据
    
    核心验证：
    - 轮次2追问"第3首歌"时，LLM context 包含歌曲列表（才能定位第3首）
    - 轮次3切换话题后，轮次4回到音乐话题时，LLM context 仍包含完整歌曲列表
    - 验证 LLM 传入的 messages 中搜索结果的数据粒度足以回答具体追问
    """
    result = TestResult("深度引用链 - 跨多轮回溯（含上下文质量断言）")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    song_data = """热门歌曲搜索结果：
1. "Die With A Smile" - Lady Gaga & Bruno Mars
2. "APT." - ROSÉ & Bruno Mars  
3. "Birds of a Feather" - Billie Eilish
4. "Espresso" - Sabrina Carpenter
5. "Beautiful Things" - Benson Boone"""
    skill_set.set_result("web-search", song_data)
    
    # 轮次1：搜索歌曲
    tc_id = "call_songs"
    provider.add_response(
        content="搜索中...",
        tool_calls=[build_tool_call("web-search", {"task": "2024年最火的5首歌"}, tc_id)]
    )
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="2024年最火的5首歌：1. Die With A Smile 2. APT. 3. Birds of a Feather 4. Espresso 5. Beautiful Things")
    await agent.execute_task("2024年最火的5首歌是什么？")
    
    # 轮次2：追问第3首
    provider.reset()
    provider.add_response(content="", tool_calls=None)
    provider.add_response(content="Birds of a Feather 是 Billie Eilish 的歌，来自专辑 HIT ME HARD AND SOFT...")
    resp2 = await agent.execute_task("第3首歌讲了什么？")
    
    # ★ 核心断言：轮次2 LLM 需要看到完整歌曲列表才能定位"第3首"
    for e in assert_context_contains(provider,
        ["Birds of a Feather", "Billie Eilish", "Die With A Smile", "Espresso"],
        description="轮次2上下文应含完整歌曲列表"):
        result.add_error(e)
    for e in assert_response_quality(resp2, ["Birds of a Feather", "Billie Eilish"], "轮次2回复"):
        result.add_error(e)
    result.add_detail("✓ 轮次2: 上下文含完整歌曲列表，可定位'第3首'")
    
    # 轮次3：切换话题
    provider.reset()
    skill_set.get_tool_definitions = lambda: []
    provider.add_response(content="今天天气不错，适合出门。")
    await agent.execute_task("今天天气怎么样？")
    
    # 轮次4：回到音乐话题，引用第1轮
    provider.reset()
    provider.add_response(content="你之前问的时候，第一首是 Die With A Smile，Lady Gaga 和 Bruno Mars 合作的。")
    resp4 = await agent.execute_task("你之前推荐的第一首歌是什么？")
    
    # ★ 核心断言：经过话题切换后，轮次4的 LLM context 仍包含第1轮的歌曲数据
    for e in assert_context_contains(provider,
        ["Die With A Smile", "Lady Gaga", "Bruno Mars"],
        description="轮次4上下文应仍含第1轮搜索数据"):
        result.add_error(e)
    for e in assert_context_has_role(provider, "tool",
        description="轮次4应仍能看到历史tool消息"):
        result.add_error(e)
    for e in assert_response_quality(resp4, ["Die With A Smile", "Lady Gaga"], "轮次4回复"):
        result.add_error(e)
    result.add_detail("✓ 轮次4: 话题切换后仍保留第1轮搜索数据")
    
    # 验证搜索数据确实在 conversation_history 的 tool 消息中
    tool_msgs = [m for m in agent.conversation_history if m.role == "tool"]
    has_song_data = any("Die With A Smile" in m.content for m in tool_msgs)
    if not has_song_data:
        result.add_error("轮次1的搜索原始数据在后续轮次中丢失了")
    
    total_rounds = count_rounds(agent.conversation_history)
    if total_rounds != 4:
        result.add_error(f"应有4轮，实际 {total_rounds} 轮")
    
    return result


async def test_11_token_budget_trim():
    """测试11：Token 预算裁剪 - 当历史超长时自动缩减轮次
    
    构造每轮产生大量字符（>5000字符的 tool 结果），
    验证 token 预算机制在保留轮次的同时控制总长度。
    """
    result = TestResult("Token 预算裁剪")
    
    provider = MockLLMProvider()
    skill_set = MockSkillSet()
    agent = DirectAgentSimulator(provider, skill_set)
    
    # 每轮搜索产生 6000 字符的结果（会被截断到 1500 + 后续还有 assistant 回复约 200 字符）
    # 6轮后：每轮约 1700 字符 * 6 = 10200 字符 → 在预算内
    # 但如果工具结果只截到 1500，实际每轮还有 user(~30) + assistant_tc(~20) + tool(1500) + assistant(200) = ~1750
    
    # 构造超大搜索结果
    huge_result = "搜索结果：" + "数据" * 3000  # 约 6006 字符
    skill_set.set_result("web-search", huge_result)
    
    for i in range(5):
        provider.reset()
        tc_id = f"call_big_{i}"
        provider.add_response(
            content="搜索...",
            tool_calls=[build_tool_call("web-search", {"task": f"big query {i}"}, tc_id)]
        )
        provider.add_response(content="", tool_calls=None)
        provider.add_response(content=f"这是第 {i+1} 轮的长回复。" + "内容" * 200)  # 约 400 字符
        await agent.execute_task(f"搜索大量数据 {i+1}")
    
    total_chars = sum(len(m.content or "") for m in agent.conversation_history)
    total_rounds = count_rounds(agent.conversation_history)
    
    result.add_detail(f"5轮长结果后: {total_rounds} 轮, {total_chars} 字符, {len(agent.conversation_history)} 消息")
    
    # 总字符应不超过 24000
    if total_chars > 24000:
        result.add_error(f"总字符数 {total_chars} 超过预算 24000")
    else:
        result.add_detail(f"✓ 总字符数 {total_chars} 在预算 24000 以内")
    
    # 至少保留 2 轮
    if total_rounds < 2:
        result.add_error(f"至少应保留2轮，实际 {total_rounds}")
    
    return result


# ============================================================
# 测试运行器
# ============================================================

ALL_TESTS = [
    test_01_basic_follow_up_with_tool_results,
    test_02_multi_tool_calls_in_one_round,
    test_03_no_tool_pure_conversation,
    test_04_pronoun_reference_across_rounds,
    test_05_tool_result_truncation,
    test_06_trim_keeps_recent_rounds,
    test_07_trim_with_tool_calls,
    test_08_extract_summary_with_tool_chain,
    test_09_interleaved_tool_and_text,
    test_10_deep_reference_chain,
    test_11_token_budget_trim,
]


async def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("多轮对话测试集 - DirectAgent conversation_history 管理")
    print("=" * 70)
    print()
    
    results: List[TestResult] = []
    
    for test_func in ALL_TESTS:
        try:
            r = await test_func()
            results.append(r)
        except Exception as e:
            r = TestResult(test_func.__doc__.split("\n")[0] if test_func.__doc__ else test_func.__name__)
            r.add_error(f"测试异常: {type(e).__name__}: {e}")
            import traceback
            r.add_detail(traceback.format_exc())
            results.append(r)
    
    # 输出结果
    print()
    for r in results:
        print(r)
        print()
    
    # 统计
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    print("=" * 70)
    print(f"测试结果: {passed}/{len(results)} 通过, {failed} 失败")
    print("=" * 70)
    
    # 分析
    if failed > 0:
        print("\n❌ 失败测试分析：")
        for r in results:
            if not r.passed:
                print(f"\n  [{r.name}]")
                for e in r.errors:
                    print(f"    - {e}")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(run_all_tests())
    # 退出码：有失败则返回 1
    sys.exit(0 if all(r.passed for r in results) else 1)
