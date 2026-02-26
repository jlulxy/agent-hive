"""
Direct Agent - 普通模式 Agent

不经过角色涌现/Subagent 编排，直接用单个 Agent 响应用户请求。
同样具备完整的 skills 和 memory 能力。

设计原则：
- 简单直接：一个 LLM 调用，流式输出
- 全能力保留：skills（tool calling）、memory 检索/摄入
- 复用已有基础设施：LLMProvider、SkillExecutor、MemoryService、AG-UI 事件
"""

import asyncio
import uuid
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

from core.models import AgentStatus, TaskSession
from llm.provider import LLMProviderFactory, LLMMessage, LLMConfig
from skills import list_skills, get_global_registry
from skills.executor import SkillExecutor, AgentSkillSet
from agui.events import (
    EventFactory,
    BaseEvent,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    AgentThinkingEvent,
)


# 普通模式系统提示
DIRECT_AGENT_SYSTEM_PROMPT = """你是一个强大的 AI 助手，具备以下能力：

## 当前时间
{current_time}

## 核心能力
1. **深度分析**：能够深入分析复杂问题，提供全面、专业的见解
2. **技能调用**：可以调用各种技能工具来辅助完成任务（如网络搜索、数据分析、代码执行等）
3. **记忆系统**：能记住用户的偏好和历史交互

## 工作原则
- 直接、清晰地回答用户问题
- 必要时主动调用工具获取信息，尤其是需要实时数据时（如股价、新闻、最新资讯等），务必调用 web-search 工具
- 使用 Markdown 格式组织输出
- 提供有深度和实用价值的回答
- **重要**：当你决定调用工具时，必须在调用前用简短的文字说明你的思考过程和行动计划（例如："让我先搜索一下最新的相关信息..."）。这段文字会作为"模型思考过程"展示给用户，帮助用户理解你的推理链路。

## 多轮对话
你正处于一个连续的多轮对话中。对话历史包含了之前所有轮次的完整信息，包括：
- 用户的每一轮提问
- 你的回复内容
- 你调用过的工具及其返回的原始数据

**重要规则：**
1. **主动引用历史**：回答追问时，应主动引用你之前回复中的关键信息（如具体数据、列表项、结论等），用"正如我之前提到的..."或"基于前面讨论的..."等方式建立连贯性，让用户感受到你完整记得对话内容。
2. **精确指代解析**：当用户使用代词（"它"、"那个"、"后者"）、序号引用（"第3个"、"第一本"）或回指表达（"你刚说的"、"上面的"）时，必须回溯对话历史精确定位指代对象，不可猜测或泛泛回答。
3. **递进式展开**：当用户在前几轮讨论的基础上深入追问时，应在前文基础上递进展开，避免重复已讲过的基础概念，体现对话的层层深入。
4. **纠错后认知更新**：如果用户纠正了你的某个回答，你应明确承认并修正，后续回复中必须使用修正后的正确信息，不可重复错误。
5. **工具结果复用**：利用之前工具调用获取的原始数据来丰富追问的回答，优先使用历史中已有的工具结果，必要时再发起新的工具调用补充信息。

## 引用与来源
如果你使用了搜索工具获取信息，**必须**在回复末尾列出参考来源链接。格式如下：

```
## 参考来源
- [标题](URL)
- [标题](URL)
```

确保每个引用的事实都能追溯到具体来源，不要遗漏搜索结果中的 URL。

{skills_prompt}

{memory_prompt}
"""


class DirectAgent:
    """
    普通模式 Agent - 直接对话，不涌现角色
    
    支持：
    - 流式 LLM 输出（AG-UI 事件格式）
    - Skills（通过 tool calling）
    - Memory（用户偏好记忆）
    """
    
    def __init__(
        self,
        provider_type: str = "openai",
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.provider_type = provider_type
        self.model = model
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        
        # LLM
        self.provider = LLMProviderFactory.get_provider(provider_type)
        self.llm_config = LLMProviderFactory.get_default_config(provider_type)
        if model:
            self.llm_config.model = model
        
        # Skills - 分配所有可用技能
        self.agent_id = f"direct-{self.session_id[:8]}"
        self.skill_set = AgentSkillSet(
            agent_id=self.agent_id,
            agent_name="Assistant",
        )
        self._init_all_skills()
        
        # 会话管理
        self.sessions: Dict[str, TaskSession] = {}
        self.active_subagents: Dict[str, Any] = {}  # 兼容 MasterAgent 接口
        
        # 对话历史（用于多轮对话）
        self.conversation_history: List[LLMMessage] = []
        
        print(f"[DirectAgent] Created for session: {self.session_id[:8]}...")
    
    def _init_all_skills(self):
        """分配所有可用技能"""
        all_skill_names = list_skills()
        assigned = self.skill_set.assign_skills(all_skill_names)
        if assigned > 0:
            print(f"[DirectAgent] Assigned {assigned} skills: {all_skill_names}")
    
    async def execute_task(self, task: str) -> AsyncGenerator[BaseEvent, None]:
        """
        执行任务 - 普通模式

        流程：
        1. 检索用户记忆
        2. 构建系统提示（含技能信息）
        3. LLM 流式生成 / tool calling 循环
        4. 输出结果
        
        Yields:
            AG-UI 协议事件流
        """
        session = TaskSession(task=task)
        self.sessions[session.id] = session
        
        thread_id = session.id
        run_id = str(uuid.uuid4())
        
        # 发送开始事件
        yield EventFactory.run_started(thread_id, run_id)
        
        try:
            # ===== 记忆检索 =====
            user_memory_text = ""
            if self.user_id:
                try:
                    from memory.service import get_memory_service
                    memory_service = get_memory_service()
                    if memory_service.is_enabled:
                        memories = await memory_service.retrieve(
                            user_id=self.user_id,
                            queries=[task],
                        )
                        user_memory_text = memory_service.format_for_prompt(memories)
                        if user_memory_text:
                            print(f"[DirectAgent] Retrieved user memory for {self.user_id[:8]}...")
                        
                        # 摄入用户输入
                        asyncio.create_task(memory_service.memorize(
                            user_id=self.user_id,
                            content=f"用户任务请求: {task}",
                            modality="conversation",
                        ))
                except Exception as e:
                    print(f"[DirectAgent] Memory retrieval failed (non-blocking): {e}")
            
            # ===== 构建系统提示 =====
            skills_prompt = ""
            tool_definitions = self.skill_set.get_tool_definitions()
            if tool_definitions:
                skills_prompt = "## 可用工具\n你可以调用以下工具来辅助完成任务。"
            
            memory_prompt = ""
            if user_memory_text:
                memory_prompt = f"## 👤 用户偏好与记忆\n{user_memory_text}"
            
            system_prompt = DIRECT_AGENT_SYSTEM_PROMPT.format(
                skills_prompt=skills_prompt,
                memory_prompt=memory_prompt,
                current_time=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S（%A）"),
            )
            
            # ===== 构建消息 =====
            messages = [
                LLMMessage(role="system", content=system_prompt),
            ]
            
            # 添加对话历史
            messages.extend(self.conversation_history)
            
            # 添加当前任务
            messages.append(LLMMessage(role="user", content=task))
            
            # ===== 执行 LLM（带 tool calling 循环）=====
            session.status = AgentStatus.RUNNING
            
            message_id = f"direct-{run_id}"
            
            full_response = ""
            max_tool_rounds = 5
            
            # 多轮工具调用循环
            # 策略：每轮用 chat_complete (非流式) 检测 LLM 是否需要工具
            # - 有 tool_calls → 执行工具 → 继续下一轮检测
            # - 无 tool_calls → 跳出循环进入流式最终回答
            # 最多 max_tool_rounds 轮，防止无限循环
            
            for tool_round in range(max_tool_rounds):
                if not tool_definitions:
                    break
                
                print(f"[DirectAgent] Tool round {tool_round + 1}/{max_tool_rounds}, calling LLM (non-streaming for tool detection)...")
                try:
                    # 防止后续轮次在非流式 tool 检测阶段长时间卡住
                    response = await asyncio.wait_for(
                        self.provider.chat_complete(messages, self.llm_config, tools=tool_definitions),
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    print(f"[DirectAgent] Tool detection timeout in round {tool_round + 1}, fallback to final streaming response")
                    yield AgentThinkingEvent(
                        agent_id=self.agent_id,
                        agent_name="Assistant",
                        thinking="工具检索达到时限，先基于已有信息继续生成完整结论。",
                    )
                    break
                
                content = response.get("content", "")
                tool_calls = response.get("tool_calls")
                
                if not tool_calls:
                    # LLM 不再需要工具 → 跳出循环走流式最终回答
                    print(f"[DirectAgent] No tool calls in round {tool_round + 1}, proceeding to final response")
                    break
                
                # 有工具调用：发出 thinking 事件
                if content:
                    yield AgentThinkingEvent(
                        agent_id=self.agent_id,
                        agent_name="Assistant",
                        thinking=content,
                    )
                
                print(f"[DirectAgent] Round {tool_round + 1}: {len(tool_calls)} tool call(s): {[tc.get('function', {}).get('name') for tc in tool_calls]}")
                
                messages.append(LLMMessage(
                    role="assistant",
                    content=content or "",
                    tool_calls=tool_calls
                ))
                
                for tc in tool_calls:
                    tool_call_id = tc["id"]
                    func_name = tc["function"]["name"]
                    func_args_str = tc["function"]["arguments"]
                    
                    yield ToolCallStartEvent(
                        tool_call_id=tool_call_id,
                        tool_call_name=func_name,
                        parent_message_id=message_id,
                    )
                    
                    yield ToolCallArgsEvent(
                        tool_call_id=tool_call_id,
                        delta=func_args_str if isinstance(func_args_str, str) else json.dumps(func_args_str, ensure_ascii=False),
                    )
                    
                    try:
                        func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                        task_desc = func_args.get("task", task)
                        
                        print(f"[DirectAgent] Executing skill: {func_name}, task: {task_desc[:80]}")
                        
                        result = await self.skill_set.execute_skill(
                            skill_name=func_name,
                            task=task_desc,
                            mode=self._get_skill_mode(func_name),
                            script_name=self._get_skill_script(func_name),
                            script_args=self._build_script_args(func_name, func_args),
                        )
                        
                        tool_result_str = result.result if result.success else (result.error or "执行失败")
                        
                        print(f"[DirectAgent] Skill {func_name} result: success={result.success}, summary={result.summary}")
                        
                        yield ToolCallEndEvent(tool_call_id=tool_call_id)
                        
                        yield ToolCallResultEvent(
                            tool_call_id=tool_call_id,
                            result=json.dumps({
                                "agent_id": self.agent_id,
                                "agent_name": "Assistant",
                                "skill_name": func_name,
                                "success": result.success,
                                "summary": result.summary or "",
                                "result_preview": str(tool_result_str)[:500] if tool_result_str else "",
                            }, ensure_ascii=False),
                        )
                        
                        messages.append(LLMMessage(
                            role="tool",
                            content=str(tool_result_str) if tool_result_str else "无结果",
                            tool_call_id=tool_call_id,
                        ))
                        
                    except Exception as e:
                        error_msg = f"技能执行错误: {str(e)}"
                        print(f"[DirectAgent] Skill {func_name} error: {e}")
                        
                        yield ToolCallEndEvent(tool_call_id=tool_call_id)
                        
                        yield ToolCallResultEvent(
                            tool_call_id=tool_call_id,
                            result=json.dumps({
                                "agent_id": self.agent_id,
                                "agent_name": "Assistant",
                                "skill_name": func_name,
                                "success": False,
                                "summary": error_msg,
                                "result_preview": "",
                            }, ensure_ascii=False),
                        )
                        messages.append(LLMMessage(
                            role="tool",
                            content=json.dumps({"success": False, "error": error_msg}, ensure_ascii=False),
                            tool_call_id=tool_call_id,
                        ))
                
                # 工具执行完毕，继续下一轮检测（LLM 可能还需要更多工具调用）
            else:
                # for-else: 达到 max_tool_rounds 上限
                print(f"[DirectAgent] Reached max tool rounds ({max_tool_rounds}), proceeding to final response")
            
            # TEXT_MESSAGE_START：在工具调用循环后发出
            yield TextMessageStartEvent(message_id=message_id, role="assistant")
            
            # 最终文本回复：流式输出（不带 tools 参数，LLM 纯文本生成最终回答）
            print(f"[DirectAgent] Final streaming response...")
            async for chunk in self.provider.chat(messages, self.llm_config):
                full_response += chunk
                yield TextMessageContentEvent(
                    message_id=message_id,
                    delta=chunk
                )
            
            yield TextMessageEndEvent(message_id=message_id)
            
            # ===== 更新对话历史（完整保存 tool calling 链）=====
            # 从 messages 中提取本轮产生的所有消息（跳过 system 和之前的 history）
            history_start_idx = 1 + len(self.conversation_history)  # 1 for system prompt
            new_messages = messages[history_start_idx:]  # user + assistant(tool_calls) + tool results...
            
            for msg in new_messages:
                if msg.role == "tool" and msg.content and len(msg.content) > 1500:
                    # 裁剪过长的工具结果，保留关键信息
                    msg = LLMMessage(
                        role=msg.role,
                        content=msg.content[:1500] + "\n...(结果已截取前1500字符)",
                        tool_call_id=msg.tool_call_id,
                    )
                self.conversation_history.append(msg)
            
            # 追加最终的流式回复（如果 tool calling 循环产生了结果，最后的流式回复也要保存）
            if full_response.strip():
                self.conversation_history.append(LLMMessage(role="assistant", content=full_response))
            
            # 智能裁剪：基于对话轮次，保留最近 N 轮完整对话
            self._trim_conversation_history(max_rounds=6)
            
            session.status = AgentStatus.COMPLETED
            session.final_report = full_response
            
            # 记忆摄入
            if self.user_id and full_response:
                try:
                    from memory.service import get_memory_service
                    memory_service = get_memory_service()
                    if memory_service.is_enabled:
                        # direct 模式没有介入消息，且开始时已摄入 task，结束时不重复摄入
                        pass
                except Exception:
                    pass
            
            yield EventFactory.run_finished(thread_id, run_id)
            
        except Exception as e:
            session.status = AgentStatus.FAILED
            yield TextMessageContentEvent(
                message_id=f"direct-{run_id}",
                delta=f"\n\n❌ 错误: {str(e)}"
            )
            yield TextMessageEndEvent(message_id=f"direct-{run_id}")
            yield EventFactory.run_error(str(e))
    
    def _get_skill_mode(self, skill_name: str) -> str:
        """根据技能名判断执行模式"""
        registry = get_global_registry()
        skill = registry.get(skill_name)
        if skill and skill.get_scripts():
            return "script"
        return "prompt"
    
    def _get_skill_script(self, skill_name: str) -> Optional[str]:
        """获取技能脚本名"""
        registry = get_global_registry()
        skill = registry.get(skill_name)
        if skill:
            scripts = skill.get_scripts()
            if scripts:
                return scripts[0].name
        return None
    
    def _build_script_args(self, skill_name: str, func_args: Dict) -> Optional[List[str]]:
        """
        构建脚本参数
        
        根据不同 skill 的脚本参数格式构建命令行参数。
        例如 web-search 的 search.py 需要 --query <keyword> 格式。
        """
        task_desc = func_args.get("task", "")
        
        # web-search skill: search.py 需要 --query 参数
        if skill_name == "web-search":
            args = ["--query", task_desc]
            # 可选参数
            options = func_args.get("options", {})
            if isinstance(options, dict):
                if options.get("type"):
                    args.extend(["--type", str(options["type"])])
                if options.get("max_results"):
                    args.extend(["--max-results", str(options["max_results"])])
                if options.get("region"):
                    args.extend(["--region", str(options["region"])])
                if options.get("time_range"):
                    args.extend(["--time-range", str(options["time_range"])])
            # 默认返回更多结果
            if "--max-results" not in args:
                args.extend(["--max-results", "8"])
            return args
        
        # sougou-search skill: search.py 需要 --query 参数
        if skill_name == "sougou-search":
            args = ["--query", task_desc]
            options = func_args.get("options", {})
            if isinstance(options, dict):
                if options.get("max_results"):
                    args.extend(["--max-results", str(options["max_results"])])
            if "--max-results" not in args:
                args.extend(["--max-results", "10"])
            return args
        
        # 其他 skill 使用通用格式
        if task_desc:
            return [task_desc]
        return None
    
    def _trim_conversation_history(self, max_rounds: int = 6):
        """基于对话轮次的智能裁剪，同时考虑 token 预算
        
        一个"轮次"从 user 消息开始，包含后续所有 assistant/tool 消息，直到下一个 user 消息。
        
        裁剪策略：
        1. 基础裁剪：保留最近 max_rounds 轮
        2. Token 预算裁剪：估算总字符数，若超过阈值则进一步缩减轮次
        
        Args:
            max_rounds: 保留的最大轮次数
        """
        if not self.conversation_history:
            return
        
        # 找到每一轮的起始位置（user 消息的索引）
        round_starts = []
        for i, msg in enumerate(self.conversation_history):
            if msg.role == "user":
                round_starts.append(i)
        
        # 基础裁剪：按轮次
        if len(round_starts) > max_rounds:
            trim_from = round_starts[-max_rounds]
            old_len = len(self.conversation_history)
            self.conversation_history = self.conversation_history[trim_from:]
            print(f"[DirectAgent] Trimmed by rounds: {old_len} -> {len(self.conversation_history)} messages "
                  f"(kept {max_rounds} rounds)")
            
            # 重新计算 round_starts
            round_starts = [i for i, m in enumerate(self.conversation_history) if m.role == "user"]
        
        # Token 预算裁剪：估算总字符数（粗略 1 中文字 ≈ 2 token, 1 英文词 ≈ 1.3 token）
        # 对话历史的 token 预算设为约 12K token（约 24K 中文字符）
        MAX_HISTORY_CHARS = 24000
        total_chars = sum(len(m.content or "") for m in self.conversation_history)
        
        while total_chars > MAX_HISTORY_CHARS and len(round_starts) > 2:
            # 移除最早的一轮
            next_round_start = round_starts[1] if len(round_starts) > 1 else len(self.conversation_history)
            removed_chars = sum(len(m.content or "") for m in self.conversation_history[:next_round_start])
            self.conversation_history = self.conversation_history[next_round_start:]
            total_chars -= removed_chars
            round_starts = [i for i, m in enumerate(self.conversation_history) if m.role == "user"]
            print(f"[DirectAgent] Trimmed by token budget: removed oldest round, "
                  f"remaining chars ≈ {total_chars}")
    
    def extract_session_summary(self) -> Dict[str, Any]:
        """提取会话摘要（追问支持，兼容 MasterAgent 接口）"""
        # 从最近的 session 中提取 final_report
        final_report = ""
        for session in self.sessions.values():
            if session.final_report:
                final_report = session.final_report
        
        # 如果 session 中没有，从对话历史中提取最后一条 assistant 纯文本回复
        if not final_report and self.conversation_history:
            # 只取 content 非空且没有 tool_calls 的 assistant 消息（即最终回复，而非中间 tool calling 消息）
            assistant_msgs = [
                m.content for m in self.conversation_history
                if m.role == "assistant" and m.content and not m.tool_calls
            ]
            if assistant_msgs:
                final_report = assistant_msgs[-1][:2000]
        
        return {
            "final_report": final_report,
            "plan": None,
            "intervention_summary": None,
            "roles": None,
        }
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话状态（兼容 MasterAgent 接口）"""
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        return {
            "id": session.id,
            "task": session.task,
            "status": session.status.value,
            "plan": None,
            "subagents": {},
            "final_report": session.final_report,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }
    
    def get_instance_info(self) -> Dict[str, Any]:
        """获取实例信息"""
        return {
            "session_id": self.session_id,
            "provider_type": self.provider_type,
            "model": self.model,
            "mode": "direct",
            "skills_count": len(self.skill_set.list_skills()),
            "conversation_turns": sum(1 for m in self.conversation_history if m.role == "user"),
        }
    
    def cleanup(self):
        """清理资源"""
        self.sessions.clear()
        self.conversation_history.clear()
        print(f"[DirectAgent] Session {self.session_id[:8]}... cleaned up")
