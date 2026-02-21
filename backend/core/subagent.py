"""
Subagent 运行时

每个 Subagent 是一个独立的执行单元，拥有：
1. 涌现的角色定义
2. 独立的思考和执行能力
3. 自适应中继触发机制
4. 动态技能执行能力
"""

import asyncio
import json
import uuid
import logging
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
from datetime import datetime

from core.models import (
    SubagentConfig,
    SubagentState,
    AgentStatus,
    RelayMessage,
    RelayType,
    ToolCall,
    InterventionType,
)
from llm.provider import LLMProviderFactory, LLMMessage, LLMConfig


logger = logging.getLogger(__name__)


class SubagentRuntime:
    """Subagent 运行时"""
    
    def __init__(
        self,
        config: SubagentConfig,
        provider_type: str = "openai",
        model: Optional[str] = None,
        on_thinking: Optional[Callable[[str, str], None]] = None,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
        on_relay_request: Optional[Callable[[RelayMessage], None]] = None,
        on_tool_call: Optional[Callable[[str, ToolCall], None]] = None,
        skill_executor = None,  # 技能执行器
        user_memory: str = "",  # 用户记忆偏好文本
    ):
        """
        Args:
            config: Subagent 配置
            provider_type: LLM 提供者类型
            model: 模型名称
            on_thinking: 思考过程回调 (agent_id, thinking_content)
            on_progress: 进度更新回调 (agent_id, progress, step)
            on_relay_request: 中继请求回调
            on_tool_call: 工具调用回调
            skill_executor: 技能执行器实例
            user_memory: 用户记忆偏好文本，注入到 system prompt
        """
        self.config = config
        self.state = SubagentState(
            id=config.id,
            config=config,
        )
        
        self.provider = LLMProviderFactory.get_provider(provider_type)
        self.llm_config = LLMProviderFactory.get_default_config(provider_type)
        if model:
            self.llm_config.model = model
        
        # 回调函数
        self.on_thinking = on_thinking
        self.on_progress = on_progress
        self.on_relay_request = on_relay_request
        self.on_tool_call = on_tool_call
        
        # 技能执行器
        self.skill_executor = skill_executor
        
        # 用户记忆偏好
        self.user_memory = user_memory
        self._init_skill_set()
        
        # 对话历史
        self.messages: List[LLMMessage] = []
        
        # 控制标志
        self._paused = False
        self._cancelled = False
        
        # 人工干预相关
        self._pending_acknowledgements: List[str] = []  # 待确认的干预消息ID
        self._intervention_history: List[Dict[str, Any]] = []  # 干预历史
        
        # 中继消息队列
        self.relay_inbox: asyncio.Queue = asyncio.Queue()
    
    def _init_skill_set(self):
        """初始化技能集 (v2 架构 - SKILL.md 格式)
        
        只暴露 master 分配给本 subagent 的技能，确保子 agent 
        无法访问未授权的能力。
        """
        self._tool_definitions: List[Dict[str, Any]] = []  # 仅已分配技能的 tool 定义
        self._skill_name_map: Dict[str, str] = {}  # tool function name -> skill name 映射
        
        try:
            from skills import AgentSkillSet, SkillExecutor, init_skills
            
            # 初始化技能库（如果尚未初始化）
            init_skills()
            
            # 创建技能集
            if self.skill_executor is None:
                self.skill_set = AgentSkillSet(
                    agent_id=self.config.id,
                    agent_name=self.config.role.name
                )
            else:
                self.skill_set = AgentSkillSet(
                    agent_id=self.config.id,
                    agent_name=self.config.role.name,
                    executor=self.skill_executor
                )
            
            # 分配角色配置的技能
            for skill_assignment in self.config.role.assigned_skills:
                # v2 技能使用连字符命名，兼容下划线命名
                skill_name = skill_assignment.skill_name.replace('_', '-')
                self.skill_set.assign_skill(
                    skill_name,
                    skill_assignment.config
                )
            
            # 从已分配技能生成 tool 定义（仅限已分配技能）
            self._tool_definitions = self.skill_set.get_tool_definitions()
            for td in self._tool_definitions:
                func_name = td.get("function", {}).get("name", "")
                if func_name:
                    self._skill_name_map[func_name] = func_name  # skill name == function name
            
            assigned = self.skill_set.list_skills()
            logger.info(
                "Subagent %s skills initialized: assigned=%s, tools=%d",
                self.agent_name, assigned, len(self._tool_definitions),
            )
        except ImportError as e:
            logger.warning(f"技能系统初始化失败: {e}")
            self.skill_set = None
    
    @property
    def agent_id(self) -> str:
        return self.config.id
    
    @property
    def agent_name(self) -> str:
        return self.config.role.name
    
    async def run(self) -> SubagentState:
        """执行 Subagent 任务"""
        try:
            self._update_status(AgentStatus.RUNNING)
            self._init_messages()
            
            iteration = 0
            max_iterations = self.config.max_iterations
            
            while iteration < max_iterations and not self._cancelled:
                if self._paused:
                    self._update_status(AgentStatus.PAUSED)
                    await asyncio.sleep(0.5)
                    continue
                
                iteration += 1
                self.state.iterations = iteration
                
                # 检查中继消息
                await self._process_relay_inbox()
                
                # 更新进度
                progress = min(95, (iteration / max_iterations) * 100)
                self._update_progress(progress, f"迭代 {iteration}/{max_iterations}")
                
                # 执行一次 LLM 调用
                response = await self._execute_iteration()
                
                # 检查是否完成（升级版：先检查中继站状态）
                task_complete = self._is_task_complete(response)
                
                if task_complete:
                    # 再次确认中继站状态
                    has_pending, pending_summary = self._check_pending_relay_messages()
                    
                    if has_pending and not self._can_complete_with_pending_messages(response, pending_summary):
                        # 有待处理消息，注入提示让 Agent 处理
                        self.messages.append(LLMMessage(
                            role="user",
                            content=self._build_pending_message_prompt(pending_summary)
                        ))
                        continue
                    
                    # 确认完成
                    self.state.final_result = self._extract_final_result(response)
                    break
                
                # 检查是否需要触发中继
                if self.config.relay_enabled:
                    await self._check_relay_trigger(response)
                
                # 添加继续迭代的引导消息
                if iteration < max_iterations - 1:
                    has_pending, pending_summary = self._check_pending_relay_messages()
                    self.messages.append(LLMMessage(
                        role="user",
                        content=self._build_continuation_prompt(
                            iteration, 
                            response,
                            pending_summary if has_pending else None
                        )
                    ))
            
            self._update_status(AgentStatus.COMPLETED)
            self._update_progress(100, "完成")
            
        except asyncio.CancelledError:
            self._update_status(AgentStatus.CANCELLED)
        except Exception as e:
            self.state.error = str(e)
            logger.exception(
                "Subagent failed: id=%s name=%s iterations=%s error=%s",
                self.agent_id,
                self.agent_name,
                self.state.iterations,
                str(e),
            )
            self._update_status(AgentStatus.FAILED)
        
        return self.state
    
    async def run_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """流式执行"""
        try:
            self._update_status(AgentStatus.RUNNING)
            yield {"type": "status", "status": AgentStatus.RUNNING.value}
            
            self._init_messages()
            
            iteration = 0
            max_iterations = self.config.max_iterations
            
            while iteration < max_iterations and not self._cancelled:
                if self._paused:
                    self._update_status(AgentStatus.PAUSED)
                    yield {"type": "status", "status": AgentStatus.PAUSED.value}
                    await asyncio.sleep(0.5)
                    continue
                
                iteration += 1
                self.state.iterations = iteration
                
                # 处理中继消息
                processed_relay = await self._process_relay_inbox()
                if processed_relay:
                    yield {
                        "type": "relay_processed",
                        "count": len(processed_relay),
                        "has_intervention": any(
                            m.type == RelayType.HUMAN_INTERVENTION 
                            for m in processed_relay
                        )
                    }
                
                # 更新进度
                progress = min(95, (iteration / max_iterations) * 100)
                self._update_progress(progress, f"迭代 {iteration}/{max_iterations}")
                yield {
                    "type": "progress",
                    "progress": progress,
                    "step": f"迭代 {iteration}/{max_iterations}",
                    "iterations": iteration
                }
                
                # 流式执行（支持 tool calling）—— 实时 yield 所有事件
                full_response = ""
                accumulated_thinking = ""
                async for event in self._stream_iteration_with_tools():
                    event_type = event["type"]
                    
                    if event_type == "thinking":
                        # 实时推送 thinking chunks 给前端
                        chunk = event["delta"]
                        accumulated_thinking += chunk
                        self.state.thinking = accumulated_thinking
                        yield event
                        if self.on_thinking:
                            self.on_thinking(self.agent_id, chunk)
                    elif event_type == "final_content":
                        # 最终完整内容（用于完成判断）
                        full_response = event["content"]
                    else:
                        # tool_call_start, tool_call_result 等事件直接转发
                        yield event
                
                # 检查是否完成（升级版：先检查中继站状态）
                task_complete = self._is_task_complete(full_response)
                
                if task_complete:
                    # 再次确认中继站状态
                    has_pending, pending_summary = self._check_pending_relay_messages()
                    
                    if has_pending and not self._can_complete_with_pending_messages(full_response, pending_summary):
                        # 有待处理消息，通知 Agent 需要先处理
                        yield {
                            "type": "completion_blocked",
                            "reason": "pending_relay_messages",
                            "pending_summary": pending_summary
                        }
                        
                        # 注入提示让 Agent 知道需要先处理消息
                        self.messages.append(LLMMessage(
                            role="user",
                            content=self._build_pending_message_prompt(pending_summary)
                        ))
                        continue
                    
                    # 确认完成 - 先更新状态并 yield，再提取结果
                    # 这样前端能立即看到 completed 状态，不用等 _extract_final_result 完成
                    self._update_status(AgentStatus.COMPLETED)
                    yield {"type": "status", "status": AgentStatus.COMPLETED.value}
                    
                    self.state.final_result = self._extract_final_result(full_response)
                    yield {"type": "result", "result": self.state.final_result}
                    break
                
                # 检查中继触发
                if self.config.relay_enabled:
                    relay_msg = await self._check_relay_trigger(full_response)
                    if relay_msg:
                        yield {"type": "relay", "message": relay_msg.model_dump()}
                
                # 添加继续迭代的引导消息（如果还没完成）
                if iteration < max_iterations - 1:
                    # 检查是否有待处理消息需要提醒
                    has_pending, pending_summary = self._check_pending_relay_messages()
                    continuation_prompt = self._build_continuation_prompt(
                        iteration, 
                        full_response,
                        pending_summary if has_pending else None
                    )
                    self.messages.append(LLMMessage(
                        role="user",
                        content=continuation_prompt
                    ))
            
            # 兜底：如果循环正常退出（达到 max_iterations）但没有在循环内 break
            # 此时状态可能还未更新为 COMPLETED
            if self.state.status != AgentStatus.COMPLETED:
                self._update_status(AgentStatus.COMPLETED)
                yield {"type": "status", "status": AgentStatus.COMPLETED.value}
            
        except asyncio.CancelledError:
            self._update_status(AgentStatus.CANCELLED)
            yield {"type": "status", "status": AgentStatus.CANCELLED.value}
        except Exception as e:
            self.state.error = str(e)
            logger.exception(
                "Subagent stream failed: id=%s name=%s iterations=%s error=%s",
                self.agent_id,
                self.agent_name,
                self.state.iterations,
                str(e),
            )
            self._update_status(AgentStatus.FAILED)
            yield {"type": "error", "error": str(e)}
    
    def pause(self):
        """暂停执行"""
        self._paused = True
    
    def resume(self):
        """恢复执行"""
        self._paused = False
    
    def cancel(self):
        """取消执行"""
        self._cancelled = True
    
    async def receive_relay_message(self, message: RelayMessage):
        """接收中继消息"""
        # 标记消息被此 Agent 查看
        message.mark_viewed(self.agent_id)
        
        await self.relay_inbox.put(message)
        self.state.relay_messages_received.append(message.model_dump())
    
    async def receive_intervention(self, message: RelayMessage, intervention=None):
        """接收人工干预消息 - 特殊处理通道
        
        这个方法用于处理需要特殊响应的人工干预，而不只是简单地注入到对话中。
        
        Args:
            message: 中继消息
            intervention: 原始干预对象（可选）
        """
        # 标记消息被此 Agent 查看
        message.mark_viewed(self.agent_id)
        
        # 记录到收到的消息中
        self.state.relay_messages_received.append(message.model_dump())
        
        # 标记需要确认
        if message.metadata.get("requires_acknowledgement"):
            self._pending_acknowledgements.append(message.id)
        
        # 记录干预历史
        self._intervention_history.append({
            "message_id": message.id,
            "intervention_type": message.metadata.get("intervention_type", "unknown"),
            "priority": message.metadata.get("priority", 5),
            "timestamp": datetime.now().isoformat(),
            "content_preview": message.content[:100]
        })
        
        # 根据干预类型决定处理方式
        intervention_type = message.metadata.get("intervention_type", "")
        
        if intervention_type == InterventionType.INJECT.value:
            # 注入信息 - 放入收件箱让下次迭代处理
            await self.relay_inbox.put(message)
        elif intervention_type == InterventionType.ADJUST.value:
            # 调整指令 - 也放入收件箱，但标记优先级
            message.importance = max(message.importance, 0.9)
            await self.relay_inbox.put(message)
        else:
            # 其他类型（暂停/恢复/取消等已在 MasterAgent 层处理）
            # 仍然放入收件箱让 Agent 知道发生了什么
            await self.relay_inbox.put(message)
    
    def inject_information(self, information: str):
        """人工注入信息 - 增强版
        
        直接注入信息到 Agent 的对话历史中，
        使用强调性提示确保 Agent 关注这个信息
        """
        injection_prompt = f"""⚠️ **[重要：人工注入信息]** ⚠️

以下是人类操作员直接注入给你的重要信息，请务必认真阅读并整合到你的工作中：

---
{information}
---

**你需要做的**：
1. 仔细阅读上述注入的信息
2. 评估这些信息与你当前任务的相关性  
3. 如果相关，将其整合到你的分析或工作中
4. 在你的下一轮输出中体现对这些信息的考虑
5. 如果信息要求你调整方向或关注点，请相应调整

请继续你的工作，并考虑上述注入的信息。"""
        
        self.messages.append(LLMMessage(
            role="user",
            content=injection_prompt
        ))
        
        # 记录注入次数
        self._injected_info_count = getattr(self, '_injected_info_count', 0) + 1
        print(f"[Subagent {self.agent_id}] Information injected (total: {self._injected_info_count})")
    
    def _init_messages(self):
        """初始化消息"""
        self.messages = [
            LLMMessage(
                role="system",
                content=self._build_system_prompt()
            ),
            LLMMessage(
                role="user",
                content=self._build_task_prompt()
            )
        ]
    
    def _build_system_prompt(self) -> str:
        """构建系统提示 - 增强版，包含完整角色信息和技能"""
        role = self.config.role
        
        # 基础身份
        prompt_parts = [
            role.system_prompt,
            "",
            f"## 🕐 当前时间",
            f"{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S（%A）')}",
            "",
            "## 🎭 你的身份",
            f"- **角色名称**：{role.name}",
            f"- **专业描述**：{role.description}",
            f"- **专业水平**：{role.expertise_level}",
        ]
        
        # 工作目标
        if role.work_objective:
            prompt_parts.extend([
                "",
                "## 🎯 你的工作目标",
                role.work_objective,
            ])
        
        # 预期交付物
        if role.deliverables:
            prompt_parts.extend([
                "",
                "## 📦 预期交付物",
            ])
            for deliverable in role.deliverables:
                prompt_parts.append(f"- {deliverable}")
        
        # 工作方法论
        if role.methodology:
            methodology = role.methodology
            prompt_parts.extend([
                "",
                "## 📋 工作方法论",
                f"**总体方法**：{methodology.approach}",
            ])
            
            if methodology.steps:
                prompt_parts.append("")
                prompt_parts.append("**工作步骤**：")
                for i, step in enumerate(methodology.steps, 1):
                    prompt_parts.append(f"{i}. {step}")
            
            if methodology.tools_and_frameworks:
                prompt_parts.append("")
                prompt_parts.append("**使用的工具和框架**：")
                for tool in methodology.tools_and_frameworks:
                    prompt_parts.append(f"- {tool}")
            
            if methodology.success_criteria:
                prompt_parts.append("")
                prompt_parts.append("**成功标准**：")
                for criteria in methodology.success_criteria:
                    prompt_parts.append(f"- {criteria}")
        
        # 核心能力
        if role.capabilities:
            prompt_parts.extend([
                "",
                "## 💪 你的核心能力",
            ])
            for cap in role.capabilities:
                prompt_parts.append(f"- {cap}")
        
        # 关注领域
        if role.focus_areas:
            prompt_parts.extend([
                "",
                "## 🔍 关注领域",
            ])
            for area in role.focus_areas:
                prompt_parts.append(f"- {area}")
        
        # 技能说明
        if role.assigned_skills:
            prompt_parts.extend([
                "",
                "## 🛠️ 你拥有的技能",
            ])
            for skill in role.assigned_skills:
                prompt_parts.append(f"- **{skill.skill_display_name}** ({skill.skill_name})")
                if skill.reason:
                    prompt_parts.append(f"  用途：{skill.reason}")
            
            prompt_parts.append("")
            prompt_parts.append("⚠️ **技能限制**：你只能使用以上已分配的技能，不得调用或假设未分配的技能能力。")
            
            # 添加技能使用说明
            if self.skill_set:
                skill_injection = self.skill_set.get_system_prompt_injection()
                if skill_injection:
                    prompt_parts.append("")
                    prompt_parts.append(skill_injection)
        
        # 工作方式
        prompt_parts.extend([
            "",
            "## 📝 工作规范",
            "1. 深入分析你被分配的任务，发挥你的专业能力",
            "2. 按照你的工作方法论系统性地开展工作",
            "3. 当你有重要发现时，明确标注 **[关键发现]**",
            "4. 当你完成任务时，用 **[任务完成]** 标记，并给出完整的分析结果",
            "5. **引用与来源**：如果你使用了搜索工具获取信息，必须在回复末尾的 **参考来源** 章节中列出所引用的链接。格式如下：",
            "   ```",
            "   ## 参考来源",
            "   - [标题](URL)",
            "   - [标题](URL)",
            "   ```",
            "   确保每个引用的事实都能追溯到具体来源，不要遗漏搜索结果中的 URL。",
        ])
        
        # 中继触发条件
        if role.relay_triggers:
            prompt_parts.extend([
                "",
                "## 🔄 中继协作机制",
                "### 触发中继的条件",
                "当出现以下情况时，你应该与其他 Agent 交换信息：",
            ])
            for trigger in role.relay_triggers:
                prompt_parts.append(f"- {trigger}")
            
            prompt_parts.extend([
                "",
                "### 中继消息格式",
                "**1. 请求对齐（向其他Agent请求协助）**：",
                "```",
                "[请求中继: 简短说明请求原因]",
                "具体描述你需要什么信息或确认，例如：",
                "- 需要哪个角色确认什么内容",
                "- 你目前的分析发现是什么",
                "- 具体的问题或疑问",
                "```",
                "",
                "**2. 响应对齐（回复其他Agent的请求）**：",
                "```",
                "[响应对齐: 针对XXX的请求]",
                "你的具体回复内容：",
                "- 对问题的直接回答",
                "- 你的相关发现或分析",
                "- 补充信息或建议",
                "```",
                "",
                "**3. 分享发现（主动分享重要信息）**：",
                "```",
                "[关键发现]",
                "详细描述你的发现内容，确保信息完整有意义。",
                "```",
                "",
                "⚠️ **重要**：所有中继消息必须包含完整、具体的内容，不要只写称呼或空泛的确认。",
            ])
        
        # 注入用户记忆偏好
        if self.user_memory:
            prompt_parts.extend([
                "",
                "## 👤 用户偏好与记忆",
                "以下是关于当前用户的偏好和历史记忆信息，请在执行任务时充分考虑这些信息：",
                self.user_memory,
            ])
        
        return "\n".join(prompt_parts)
    
    def _build_task_prompt(self) -> str:
        """构建任务提示"""
        role = self.config.role
        
        prompt_parts = [
            "## 🎯 你的任务",
            self.config.task_segment,
            "",
        ]
        
        # 如果有交付物要求，提醒
        if role.deliverables:
            prompt_parts.extend([
                "## 📦 请确保你的输出包含",
            ])
            for deliverable in role.deliverables:
                prompt_parts.append(f"- {deliverable}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## 📝 工作流程说明",
            "1. 你需要进行深入、多轮的分析，不要急于给出最终结论",
            "2. 每轮分析后，我会询问你是否需要继续深入或有新的发现",
            "3. 当你认为分析已经完整且深入时，使用 **[任务完成]** 标记，并给出完整的分析结果",
            "4. 如果发现重要信息需要与其他 Agent 共享，请使用 **[关键发现]** 标记",
            "",
            "请开始你的第一轮分析，先从整体框架入手，逐步深入。",
        ])
        
        return "\n".join(prompt_parts)
    
    async def _execute_iteration(self) -> str:
        """执行一次迭代（支持 tool calling）
        
        如果 subagent 有已分配技能的 tool 定义，会将 tools 传递给 LLM，
        并在 LLM 返回 tool_calls 时自动执行对应技能、将结果反馈给 LLM。
        """
        tools = self._tool_definitions if self._tool_definitions else None
        max_tool_rounds = 3  # 单次迭代最多执行 3 轮工具调用
        
        for tool_round in range(max_tool_rounds + 1):
            response = await self.provider.chat_complete(
                self.messages, self.llm_config, tools=tools
            )
            content = response.get("content", "")
            tool_calls = response.get("tool_calls")
            
            if not tool_calls or not self.skill_set:
                # 没有工具调用或无技能集，直接返回文本响应
                self.messages.append(LLMMessage(role="assistant", content=content))
                self.state.thinking = content
                self.state.partial_result = content
                if self.on_thinking:
                    self.on_thinking(self.agent_id, content)
                return content
            
            # 有工具调用 => 执行技能并继续对话
            logger.info(
                "Subagent %s tool calls: %s",
                self.agent_name,
                [tc.get("function", {}).get("name") for tc in tool_calls],
            )
            
            # 记录 assistant 消息（带 tool_calls）
            self.messages.append(LLMMessage(
                role="assistant",
                content=content or "",
                tool_calls=tool_calls,
            ))
            
            # 执行每个工具调用
            for tc in tool_calls:
                tc_id = tc.get("id", str(uuid.uuid4()))
                func_name = tc.get("function", {}).get("name", "")
                func_args_str = tc.get("function", {}).get("arguments", "{}")
                
                try:
                    func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                except json.JSONDecodeError:
                    func_args = {"task": func_args_str}
                
                skill_name = self._skill_name_map.get(func_name, func_name)
                task_desc = func_args.get("task", func_args.get("query", str(func_args)))
                
                logger.info("Subagent %s executing skill: %s task=%s", self.agent_name, skill_name, task_desc[:100])
                
                # 回调通知
                if self.on_tool_call:
                    self.on_tool_call(self.agent_id, ToolCall(
                        id=tc_id,
                        name=func_name,
                        arguments=func_args if isinstance(func_args, dict) else {"task": str(func_args)},
                    ))
                
                # 通过 AgentSkillSet 执行技能（确保权限检查）
                try:
                    # 判断技能是否有脚本可执行
                    skill_name_str = skill_name or func_name
                    skill_obj = self.skill_set.executor.registry.get(skill_name_str)
                    has_scripts = skill_obj and len(skill_obj.get_scripts()) > 0
                    
                    if has_scripts and skill_obj is not None:
                        # 有脚本 => 用 hybrid 或 script 模式执行
                        scripts = skill_obj.get_scripts()
                        script_name = scripts[0].path.split("/")[-1] if scripts else None
                        
                        # 构建脚本参数
                        script_args = self._build_script_args(skill_name_str, func_args)
                        
                        result = await self.skill_set.execute_skill(
                            skill_name=skill_name_str,
                            task=task_desc,
                            mode="script",
                            script_name=script_name,
                            script_args=script_args,
                        )
                    else:
                        # 无脚本 => prompt 注入模式
                        result = await self.skill_set.execute_skill(
                            skill_name=skill_name_str,
                            task=task_desc,
                            mode="prompt",
                        )
                    
                    tool_result_content = json.dumps({
                        "success": result.success,
                        "result": result.result,
                        "summary": result.summary,
                        "error": result.error,
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error("Skill execution error: %s %s", skill_name, e)
                    tool_result_content = json.dumps({
                        "success": False,
                        "error": str(e),
                    }, ensure_ascii=False)
                
                # 添加 tool 结果消息
                self.messages.append(LLMMessage(
                    role="tool",
                    content=tool_result_content,
                    tool_call_id=tc_id,
                    name=func_name,
                ))
            
            # 工具结果已加入消息，继续下一轮让 LLM 消化结果
        
        # 达到最大工具轮次，做一次不带 tools 的调用获取最终回复
        response = await self.provider.chat_complete(self.messages, self.llm_config)
        content = response.get("content", "")
        self.messages.append(LLMMessage(role="assistant", content=content))
        self.state.thinking = content
        self.state.partial_result = content
        if self.on_thinking:
            self.on_thinking(self.agent_id, content)
        return content
    
    def _build_script_args(self, skill_name: str, func_args: Dict[str, Any]) -> List[str]:
        """根据技能名和函数参数构建脚本命令行参数"""
        args = []
        
        if skill_name == "web-search":
            query = func_args.get("task", func_args.get("query", ""))
            args.extend(["--query", query])
            if "max_results" in func_args:
                args.extend(["--max-results", str(func_args["max_results"])])
            if "type" in func_args:
                args.extend(["--type", func_args["type"]])
            if "region" in func_args:
                args.extend(["--region", func_args["region"]])
            if "time_range" in func_args:
                args.extend(["--time-range", func_args["time_range"]])
            # 默认 JSON 输出
            args.extend(["--format", "json"])
        elif skill_name == "sougou-search":
            query = func_args.get("task", func_args.get("query", ""))
            args.extend(["--query", query])
            if "max_results" in func_args:
                args.extend(["--max-results", str(func_args["max_results"])])
            args.extend(["--format", "json"])
        else:
            # 通用：将 task 作为参数
            task = func_args.get("task", "")
            if task:
                args.extend(["--task", task])
        
        return args
    
    async def _stream_iteration_with_tools(self) -> AsyncGenerator[Dict[str, Any], None]:
        """流式迭代 + tool calling 支持
        
        改为 AsyncGenerator，实时 yield 事件：
        - thinking: LLM 的中间思考/分析（工具决策时的 content）
        - tool_call_start / tool_call_result: 工具调用过程
        - final_content: 最终无工具调用时的完整回复（用于完成判断）
        
        最终回复走流式 chat()，实时推送给前端。
        """
        tools = self._tool_definitions if self._tool_definitions else None
        max_tool_rounds = 3
        
        # 工具调用循环：用非流式检测 tool_calls
        for tool_round in range(max_tool_rounds):
            if not tools or not self.skill_set:
                break
            
            response = await self.provider.chat_complete(
                self.messages, self.llm_config, tools=tools
            )
            content = response.get("content", "")
            tool_calls = response.get("tool_calls")
            
            if not tool_calls:
                # 无工具调用 → 跳出循环走流式最终输出
                break
            
            # 有工具调用：先推送 LLM 的决策思考（content 是调用工具前的分析）
            if content:
                yield {"type": "thinking", "delta": content}
            
            logger.info(
                "Subagent %s stream tool calls: %s",
                self.agent_name,
                [tc.get("function", {}).get("name") for tc in tool_calls],
            )
            
            self.messages.append(LLMMessage(
                role="assistant",
                content=content or "",
                tool_calls=tool_calls,
            ))
            
            for tc in tool_calls:
                tc_id = tc.get("id", str(uuid.uuid4()))
                func_name = tc.get("function", {}).get("name", "")
                func_args_str = tc.get("function", {}).get("arguments", "{}")
                
                try:
                    func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                except json.JSONDecodeError:
                    func_args = {"task": func_args_str}
                
                skill_name = self._skill_name_map.get(func_name, func_name)
                task_desc = func_args.get("task", func_args.get("query", str(func_args)))
                
                logger.info("Stream subagent %s executing skill: %s task=%s", self.agent_name, skill_name, task_desc[:100])
                
                # 实时推送 tool_call_start
                yield {
                    "type": "tool_call_start",
                    "tool_call_id": tc_id,
                    "tool_name": func_name,
                    "skill_name": skill_name or func_name,
                    "arguments": func_args if isinstance(func_args, dict) else {"task": str(func_args)},
                }
                
                if self.on_tool_call:
                    self.on_tool_call(self.agent_id, ToolCall(
                        id=tc_id,
                        name=func_name,
                        arguments=func_args if isinstance(func_args, dict) else {"task": str(func_args)},
                    ))
                
                try:
                    skill_name_str = skill_name or func_name
                    skill_obj = self.skill_set.executor.registry.get(skill_name_str)
                    has_scripts = skill_obj and len(skill_obj.get_scripts()) > 0
                    
                    if has_scripts and skill_obj is not None:
                        scripts = skill_obj.get_scripts()
                        script_name = scripts[0].path.split("/")[-1] if scripts else None
                        script_args = self._build_script_args(skill_name_str, func_args)
                        
                        result = await self.skill_set.execute_skill(
                            skill_name=skill_name_str,
                            task=task_desc,
                            mode="script",
                            script_name=script_name,
                            script_args=script_args,
                        )
                    else:
                        result = await self.skill_set.execute_skill(
                            skill_name=skill_name_str,
                            task=task_desc,
                            mode="prompt",
                        )
                    
                    tool_result_content = json.dumps({
                        "success": result.success,
                        "result": result.result,
                        "summary": result.summary,
                        "error": result.error,
                    }, ensure_ascii=False)
                    
                    # 实时推送 tool_call_result
                    yield {
                        "type": "tool_call_result",
                        "tool_call_id": tc_id,
                        "tool_name": func_name,
                        "skill_name": skill_name or func_name,
                        "success": result.success,
                        "summary": result.summary or "",
                        "result_preview": (result.result or "")[:500],
                    }
                except Exception as e:
                    logger.error("Stream skill execution error: %s %s", skill_name, e)
                    tool_result_content = json.dumps({
                        "success": False,
                        "error": str(e),
                    }, ensure_ascii=False)
                    
                    yield {
                        "type": "tool_call_result",
                        "tool_call_id": tc_id,
                        "tool_name": func_name,
                        "skill_name": skill_name or func_name,
                        "success": False,
                        "summary": f"Error: {str(e)}",
                        "result_preview": "",
                    }
                
                self.messages.append(LLMMessage(
                    role="tool",
                    content=tool_result_content,
                    tool_call_id=tc_id,
                    name=func_name,
                ))
        
        # 最终回复：走流式 chat()，实时推送 thinking chunks
        full_response = ""
        async for chunk in self.provider.chat(self.messages, self.llm_config):
            full_response += chunk
            yield {"type": "thinking", "delta": chunk}
        
        self.messages.append(LLMMessage(role="assistant", content=full_response))
        
        # 标记最终内容（供 run_stream 判断完成和提取结果）
        yield {"type": "final_content", "content": full_response}
    
    def _is_task_complete(self, response: str) -> bool:
        """检查任务是否完成
        
        判断标准更严格：
        1. 需要明确的完成标记
        2. 必须先检查中继站是否有待处理的消息
        3. 结合消息内容和指令来决定是否真正完成
        """
        # 首先检查中继站是否有待处理的消息
        has_pending, pending_summary = self._check_pending_relay_messages()
        
        if has_pending:
            # 有待处理的中继消息，不能直接完成
            # 需要根据消息类型和内容决定是否可以完成
            can_complete = self._can_complete_with_pending_messages(response, pending_summary)
            if not can_complete:
                return False
        
        # 严格模式：必须是明确的完成标记
        strict_markers = ["[任务完成]", "[TASK_COMPLETE]", "**任务完成**", "## 任务完成"]
        
        # 检查严格标记
        if any(marker in response for marker in strict_markers):
            return True
        
        # 宽松检查：只有在迭代次数达到一定阈值后才生效
        # 这确保 Agent 至少进行了足够的思考
        if self.state.iterations >= 3:
            # 检查是否有完整的分析结论
            conclusion_patterns = [
                "综上所述",
                "总结如下", 
                "最终结论",
                "分析报告",
                "完整分析结果"
            ]
            # 需要同时满足：有结论性词汇 + 内容足够长（表示完整分析）
            has_conclusion = any(p in response for p in conclusion_patterns)
            is_substantial = len(response) > 800  # 确保是实质性的内容
            if has_conclusion and is_substantial:
                return True
        
        return False
    
    def _check_pending_relay_messages(self) -> tuple[bool, Dict[str, Any]]:
        """检查中继站是否有待处理的消息
        
        Returns:
            (has_pending, summary): 是否有待处理消息，以及消息摘要
        """
        summary = {
            "total_count": 0,
            "intervention_count": 0,
            "high_priority_count": 0,
            "unacknowledged_count": len(self._pending_acknowledgements),
            "message_types": [],
            "interventions": [],
            "requires_response": False,
        }
        
        # 检查队列中的消息（不取出，只窥视）
        pending_messages = []
        temp_queue = asyncio.Queue()
        
        while not self.relay_inbox.empty():
            try:
                message = self.relay_inbox.get_nowait()
                pending_messages.append(message)
                temp_queue.put_nowait(message)
            except asyncio.QueueEmpty:
                break
        
        # 将消息放回原队列
        while not temp_queue.empty():
            try:
                message = temp_queue.get_nowait()
                self.relay_inbox.put_nowait(message)
            except asyncio.QueueEmpty:
                break
        
        summary["total_count"] = len(pending_messages)
        
        for msg in pending_messages:
            msg_type = msg.type.value if hasattr(msg.type, 'value') else str(msg.type)
            summary["message_types"].append(msg_type)
            
            # 统计人工干预
            if msg.type == RelayType.HUMAN_INTERVENTION:
                summary["intervention_count"] += 1
                intervention_type = msg.metadata.get("intervention_type", "unknown")
                priority = msg.metadata.get("priority", 5)
                summary["interventions"].append({
                    "type": intervention_type,
                    "priority": priority,
                    "content_preview": msg.content[:100]
                })
                
                # 高优先级干预需要响应
                if priority >= 7:
                    summary["high_priority_count"] += 1
                    summary["requires_response"] = True
            
            # 高重要性消息需要处理
            if msg.importance >= 0.8:
                summary["high_priority_count"] += 1
        
        # 判断是否有待处理消息
        has_pending = (
            summary["total_count"] > 0 or 
            summary["unacknowledged_count"] > 0
        )
        
        return has_pending, summary
    
    def _can_complete_with_pending_messages(
        self, 
        response: str, 
        pending_summary: Dict[str, Any]
    ) -> bool:
        """判断在有待处理消息的情况下是否可以完成任务
        
        核心逻辑：
        1. 有高优先级人工干预 -> 不能完成，必须先处理
        2. 有未确认的干预消息 -> 不能完成
        3. 有普通中继消息 -> 检查响应是否已经考虑了这些消息
        
        Args:
            response: 当前的 LLM 响应
            pending_summary: 待处理消息摘要
        
        Returns:
            是否可以完成
        """
        # 规则1: 有高优先级干预，必须先处理
        if pending_summary["high_priority_count"] > 0:
            return False
        
        # 规则2: 有未确认的干预消息，不能完成
        if pending_summary["unacknowledged_count"] > 0:
            return False
        
        # 规则3: 有人工干预需要响应
        if pending_summary["requires_response"]:
            return False
        
        # 规则4: 检查是否有需要处理的干预类型
        blocking_intervention_types = [
            InterventionType.INJECT.value,
            InterventionType.ADJUST.value,
        ]
        for intervention in pending_summary.get("interventions", []):
            if intervention["type"] in blocking_intervention_types:
                return False
        
        # 规则5: 如果响应中明确表示已处理中继消息，则可以完成
        acknowledgement_patterns = [
            "已收到中继消息",
            "已整合中继信息",
            "已考虑人工干预",
            "已根据干预调整",
            "收到干预通知",
            "已确认收到",
        ]
        if any(pattern in response for pattern in acknowledgement_patterns):
            return True
        
        # 规则6: 只有普通低优先级消息，且响应足够完整，可以完成
        # （Agent 会在下一轮自然处理这些消息）
        if (
            pending_summary["intervention_count"] == 0 and
            pending_summary["total_count"] <= 2 and
            len(response) > 500
        ):
            return True
        
        # 默认：有待处理消息时不能完成
        return False
    
    def _extract_final_result(self, response: str) -> str:
        """提取最终结果"""
        # 尝试提取 [任务完成] 之后的内容
        markers = ["[任务完成]", "[TASK_COMPLETE]"]
        for marker in markers:
            if marker in response:
                idx = response.index(marker)
                return response[idx:].strip()
        
        # 如果没有标记，返回整个响应
        return response
    
    def _is_meaningless_content(self, content: str) -> bool:
        """检查内容是否无意义
        
        用于过滤掉无效的中继消息内容
        """
        if not content:
            return True
        
        # 去除空白和标点后检查
        cleaned = content.strip()
        
        # 太短
        if len(cleaned) < 5:
            return True
        
        # 只包含符号/标点
        import re
        if re.match(r'^[\s\*\#\-\=\_\.\,\。\，\、\；\：\"\"\'\'\（\）\【\】\《\》\！\？]+$', cleaned):
            return True
        
        # 只包含 markdown 格式符号
        if re.match(r'^[\*\#\-\>\s]+$', cleaned):
            return True
        
        # 常见无意义模式
        meaningless_patterns = [
            r'^\*+$',           # 只有星号
            r'^#+$',            # 只有井号
            r'^-+$',            # 只有横线
            r'^\s*$',           # 只有空白
            r'^\.+$',           # 只有点
            r'^\(.*\)$',        # 只有括号内容且很短
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, cleaned):
                return True
        
        return False
    
    def _is_semantically_incomplete(self, content: str, is_response_type: bool = False) -> bool:
        """检查内容是否语义不完整
        
        检测那些引用了后续内容但实际没有包含的情况
        例如："以下问题"、"如下内容" 但没有实际列出
        
        Args:
            content: 要检查的内容
            is_response_type: 是否是响应类型消息（响应对齐、回复、确认等）
                             响应类型允许以称呼开头，只要后续有实质内容
        """
        if not content:
            return True
        
        import re
        
        # 如果内容足够长（超过80字），通常是完整的
        if len(content) > 80:
            return False
        
        # 语义不完整的模式：提到了"以下/如下"但内容太短
        incomplete_indicators = [
            r'以下[问题|内容|分析|要点|建议]',
            r'如下[问题|内容|分析|要点|建议]',
            r'下列[问题|内容|分析|要点|建议]',
            r'以下是',
            r'如下：',
            r'包括：$',
            r'分别是：$',
        ]
        
        # 如果内容很短（少于50字）且包含这些指示词，可能是不完整的
        if len(content) < 50:
            for pattern in incomplete_indicators:
                if re.search(pattern, content):
                    return True
        
        # 检查是否只是一个称呼/问候（但要区分响应类型）
        # 对于响应类型，允许称呼开头，只要后面有内容
        content_stripped = content.strip()
        
        # 如果是响应类型，检查称呼后是否有实质内容
        if is_response_type:
            # 检查是否是"致XXX\n\n内容"的格式
            lines = content_stripped.split('\n')
            first_line = lines[0].strip() if lines else ""
            
            # 如果第一行是称呼，检查后续是否有内容
            greeting_first_line = re.match(r'^(致|向|@)[^\s\n]{2,15}[：:]?\s*$', first_line)
            if greeting_first_line:
                # 检查后续内容
                remaining_content = '\n'.join(lines[1:]).strip()
                # 后续有实质内容（超过10字符且不只是标点）
                if len(remaining_content) > 10 and not re.match(r'^[\s\*\#\-\=\_\.\,\。\，]+$', remaining_content):
                    return False  # 有实质内容，不是不完整的
                else:
                    return True  # 没有实质后续内容
            else:
                # 第一行不是称呼，按正常逻辑处理
                return False
        
        # 非响应类型的称呼检查（严格模式）
        greeting_patterns = [
            r'^致[^\s]{2,10}$',                    # "致XXX" 只有称呼
            r'^向[^\s]{2,10}$',                    # "向XXX"
            r'^请[^\s]{2,10}[确认|注意|查看]?$',   # "请XXX确认"
            r'^@[^\s]+$',                          # "@某人"
        ]
        
        for pattern in greeting_patterns:
            if re.match(pattern, content_stripped):
                return True
        
        return False
    
    async def _check_relay_trigger(self, response: str) -> Optional[RelayMessage]:
        """检查是否需要触发中继
        
        智能检测：除了显式标记外，也检测内容中的关键发现模式
        支持：发现、请求、响应、建议、确认等多种类型
        """
        import re
        
        relay_type = None
        reason = ""
        content = ""
        target_agent_ids = []  # 支持指定目标
        
        # === 辅助函数：提取标记后的完整内容 ===
        def extract_full_content(response: str, tag_pattern: str, tag_end: str = "]") -> tuple[str, str]:
            """
            提取标记内容 + 标记后的相关内容
            
            例如：[响应对齐: 致影评整合专家]\n\n以下是我的分析...
            返回：("致影评整合专家", "以下是我的分析...")
            """
            import re
            
            # 找到标记位置
            tag_match = re.search(tag_pattern, response)
            if not tag_match:
                return "", ""
            
            tag_content = tag_match.group(1).strip() if tag_match.lastindex else ""
            tag_end_pos = tag_match.end()
            
            # 提取标记后的内容（直到下一个标记或段落结束）
            remaining = response[tag_end_pos:].strip()
            
            # 查找后续内容的结束位置
            # 遇到新的标记、分隔线、或超过500字符时停止
            end_patterns = [
                r'\n\[',           # 新的标记
                r'\n---',          # 分隔线
                r'\n\*\*\[',       # 加粗的标记
                r'\n##',           # 标题
            ]
            
            end_pos = len(remaining)
            for end_pat in end_patterns:
                match = re.search(end_pat, remaining)
                if match and match.start() < end_pos:
                    end_pos = match.start()
            
            # 限制长度
            end_pos = min(end_pos, 800)
            following_content = remaining[:end_pos].strip()
            
            return tag_content, following_content
        
        # === 响应类消息检测（优先级最高）===
        
        # 检查响应对齐
        if "[响应对齐:" in response:
            tag_content, following = extract_full_content(response, r'\[响应对齐:\s*([^\]]+)\]')
            if tag_content:
                # 响应对齐需要有实质内容
                # 如果标记内容是称呼形式（致XXX），必须有后续内容
                is_greeting_format = re.match(r'^(致|向|针对)[^\s]{2,15}', tag_content.strip())
                
                if following and len(following) > 10:
                    # 有后续内容，合并
                    content = f"{tag_content}\n\n{following}"
                    relay_type = RelayType.ALIGNMENT_RESPONSE
                    reason = "响应对齐请求"
                elif not is_greeting_format and len(tag_content) > 20:
                    # 不是称呼格式，且标记内容本身够长，直接使用
                    content = tag_content
                    relay_type = RelayType.ALIGNMENT_RESPONSE
                    reason = "响应对齐请求"
                else:
                    # 是称呼格式但没有实质后续内容，记录日志但不发送
                    print(f"[SubAgent {self.agent_name}] 响应对齐内容不完整，跳过: tag='{tag_content[:30]}', following='{following[:30] if following else 'None'}'")
        
        # 检查回复
        elif "[回复:" in response:
            tag_content, following = extract_full_content(response, r'\[回复:\s*([^\]]+)\]')
            if tag_content:
                # 同样的逻辑
                is_greeting_format = re.match(r'^(致|向|针对)[^\s]{2,15}', tag_content.strip())
                
                if following and len(following) > 10:
                    content = f"{tag_content}\n\n{following}"
                    relay_type = RelayType.ALIGNMENT_RESPONSE
                    reason = "回复求助"
                elif not is_greeting_format and len(tag_content) > 20:
                    content = tag_content
                    relay_type = RelayType.ALIGNMENT_RESPONSE
                    reason = "回复求助"
                else:
                    print(f"[SubAgent {self.agent_name}] 回复内容不完整，跳过: tag='{tag_content[:30]}', following='{following[:30] if following else 'None'}'")
        
        # 检查确认
        elif "[确认:" in response:
            tag_content, following = extract_full_content(response, r'\[确认:\s*([^\]]+)\]')
            if tag_content:
                if following and len(following) > 10:
                    content = f"{tag_content}\n\n{following}"
                else:
                    content = tag_content
                relay_type = RelayType.CONFIRMATION
                reason = "确认/认可"
        
        # === 请求类消息检测 ===
        
        # 检查显式中继请求（请求对齐）
        elif "[请求中继:" in response:
            tag_content, following = extract_full_content(response, r'\[请求中继:\s*([^\]]+)\]')
            if tag_content:
                # 请求对齐需要完整的上下文
                if following and len(following) > 10:
                    content = f"请求对齐: {tag_content}\n\n{following}"
                else:
                    content = f"请求对齐: {tag_content}"
                relay_type = RelayType.ALIGNMENT_REQUEST
                reason = tag_content
        
        # 检查疑问/求助
        elif "[求助:" in response or "[疑问:" in response:
            help_match = re.search(r'\[(求助|疑问):\s*([^\]]+)\]', response)
            if help_match:
                tag_type = help_match.group(1)
                tag_content = help_match.group(2)
                # 提取后续内容
                tag_end_pos = help_match.end()
                remaining = response[tag_end_pos:tag_end_pos + 500].strip()
                # 简单截取到下一个标记
                next_tag = re.search(r'\n\[|\n---|\n##', remaining)
                following = remaining[:next_tag.start()].strip() if next_tag else remaining[:300].strip()
                
                if following and len(following) > 10:
                    content = f"{tag_type}: {tag_content}\n\n{following}"
                else:
                    content = f"{tag_type}: {tag_content}"
                relay_type = RelayType.QUESTION
                reason = tag_content
        
        # 检查建议
        elif "[建议:" in response:
            tag_content, following = extract_full_content(response, r'\[建议:\s*([^\]]+)\]')
            if tag_content:
                if following and len(following) > 10:
                    content = f"建议: {tag_content}\n\n{following}"
                else:
                    content = f"建议: {tag_content}"
                relay_type = RelayType.SUGGESTION
                reason = tag_content
        
        # === 发现类消息检测 ===
        
        # 检查显式关键发现标记
        elif "[关键发现]" in response or "**[关键发现]**" in response:
            discovery_match = re.search(r'\[关键发现\]\s*(.+?)(?:\n\n|\n-|$)', response, re.DOTALL)
            if discovery_match:
                content = discovery_match.group(1).strip()
                # 验证内容有效性
                if len(content) >= 10 and not self._is_meaningless_content(content):
                    relay_type = RelayType.DISCOVERY
                    reason = "发现关键信息"
                else:
                    content = ""  # 无效内容，不发送
        
        # 检查洞察
        elif "[洞察]" in response or "[核心洞察]" in response:
            insight_match = re.search(r'\[(核心)?洞察\]\s*(.+?)(?:\n\n|\n-|$)', response, re.DOTALL)
            if insight_match:
                content = insight_match.group(2).strip()
                # 验证内容有效性
                if len(content) >= 10 and not self._is_meaningless_content(content):
                    relay_type = RelayType.INSIGHT
                    reason = "核心洞察"
                else:
                    content = ""  # 无效内容，不发送
        
        # 方式6: 智能检测重要发现（基于内容模式）
        # 只在迭代足够多时启用，避免过早触发
        elif self.state.iterations >= 2:
            important_patterns = [
                (r'值得注意的是[：:]\s*(.{20,200})', "值得注意的发现", RelayType.DISCOVERY),
                (r'重要发现[：:]\s*(.{20,200})', "重要发现", RelayType.DISCOVERY),
                (r'关键点[：:]\s*(.{20,200})', "关键点", RelayType.DISCOVERY),
                (r'核心洞察[：:]\s*(.{20,200})', "核心洞察", RelayType.INSIGHT),
                (r'重大影响[：:]\s*(.{20,200})', "重大影响", RelayType.DISCOVERY),
                (r'需要其他.*?(?:配合|协作|确认)', "跨域协作需求", RelayType.ALIGNMENT_REQUEST),
                (r'建议.*?(?:考虑|采用|使用)', "协作建议", RelayType.SUGGESTION),
            ]
            
            for pattern, pattern_reason, pattern_type in important_patterns:
                match = re.search(pattern, response)
                if match:
                    content = match.group(1) if match.lastindex else match.group(0)
                    relay_type = pattern_type
                    reason = pattern_reason
                    break
        
        # 如果检测到需要中继，先验证内容有效性
        if relay_type and content:
            # 清理内容
            content = content.strip()
            
            # 判断是否是响应类型消息（响应对齐、回复、确认等）
            is_response_type = relay_type in [
                RelayType.ALIGNMENT_RESPONSE,
                RelayType.CONFIRMATION,
            ]
            
            # 验证内容有效性（最小长度 + 非无意义内容 + 语义完整性）
            if len(content) < 5:
                print(f"[SubAgent {self.agent_name}] Skipped too short relay content: '{content[:50]}...'")
                return None
            
            if self._is_meaningless_content(content):
                print(f"[SubAgent {self.agent_name}] Skipped meaningless relay content: '{content[:50]}...'")
                return None
            
            # 对响应类型使用宽松的语义完整性检查
            if self._is_semantically_incomplete(content, is_response_type=is_response_type):
                print(f"[SubAgent {self.agent_name}] Skipped semantically incomplete relay content: '{content[:50]}...'")
                return None
            
            relay_msg = RelayMessage(
                type=relay_type,
                source_agent_id=self.agent_id,
                source_agent_name=self.agent_name,
                target_agent_ids=target_agent_ids,  # 支持指定目标
                content=content[:1000],  # 增加到 1000 字符，保留更多信息
                importance=0.8,
                metadata={"reason": reason, "iteration": self.state.iterations}
            )
            
            self.state.relay_messages_sent.append(relay_msg.model_dump())
            
            if self.on_relay_request:
                self.on_relay_request(relay_msg)
            
            return relay_msg
        
        return None
    
    async def _process_relay_inbox(self):
        """处理中继收件箱 - 升级版，智能处理人工干预"""
        processed_messages = []
        intervention_messages = []
        regular_messages = []
        
        # 先收集所有消息并分类
        while not self.relay_inbox.empty():
            try:
                message: RelayMessage = self.relay_inbox.get_nowait()
                if message.type == RelayType.HUMAN_INTERVENTION:
                    intervention_messages.append(message)
                else:
                    regular_messages.append(message)
            except asyncio.QueueEmpty:
                break
        
        # 优先处理人工干预消息（按重要性排序）
        intervention_messages.sort(key=lambda m: m.importance, reverse=True)
        
        for message in intervention_messages:
            # 构建增强的干预提示
            intervention_content = self._build_intervention_prompt(message)
            
            self.messages.append(LLMMessage(
                role="user",
                content=intervention_content
            ))
            
            processed_messages.append(message)
        
        # 处理普通中继消息（根据类型给出不同的响应提示）
        for message in regular_messages:
            msg_type = message.type.value if hasattr(message.type, 'value') else str(message.type)
            
            # 根据消息类型构建不同的响应提示
            if message.type == RelayType.ALIGNMENT_REQUEST:
                # 对齐请求 - 需要响应
                prompt = f"""[来自 {message.source_agent_name} 的对齐请求 🔄]
内容: {message.content}

**这是一个需要响应的请求！** 请：
1. 考虑你的分析是否与此请求相关
2. 如果相关，请使用以下格式进行响应：

[响应对齐: 针对XXX的回复]
这里写你的实际响应内容，包括：
- 你的相关发现或分析结论
- 对请求问题的直接回答
- 你认为重要的补充信息

**注意**：响应内容要完整具体，不要只写称呼或空泛的确认。
"""
            elif message.type == RelayType.QUESTION:
                # 问题/求助 - 需要回答
                prompt = f"""[来自 {message.source_agent_name} 的求助 ❓]
内容: {message.content}

**这是一个求助请求！** 如果你有相关知识或见解：
请使用以下格式进行回复：

[回复: 针对XXX问题的解答]
这里写你的具体回答内容，包括：
- 对问题的直接回答
- 相关的分析或依据
- 如有必要，附上你的建议

**注意**：回复内容要具体有帮助，不要只写"已收到"或空泛确认。
"""
            elif message.type == RelayType.SUGGESTION:
                # 建议 - 可选采纳
                prompt = f"""[来自 {message.source_agent_name} 的建议 💡]
内容: {message.content}

这是一个建议，你可以：
1. 如果认为有价值，整合到你的分析中
2. 使用 [确认: 原因] 表示采纳
3. 忽略如果与你的任务无关
"""
            elif message.type == RelayType.ALIGNMENT_RESPONSE:
                # 对齐响应 - 仅供参考
                prompt = f"""[来自 {message.source_agent_name} 的对齐响应 ✅]
内容: {message.content}

这是对之前对齐请求的响应，请参考整合。
"""
            elif message.type == RelayType.CONFIRMATION:
                # 确认 - 仅供参考
                prompt = f"""[来自 {message.source_agent_name} 的确认 ✔️]
内容: {message.content}

其他 Agent 确认了你的发现/建议。
"""
            elif message.type == RelayType.INSIGHT:
                # 洞察 - 高价值信息
                prompt = f"""[来自 {message.source_agent_name} 的核心洞察 🎯]
内容: {message.content}

这是一个重要的洞察，请仔细考虑是否能整合到你的分析中。
"""
            else:
                # 默认处理（discovery 等）
                prompt = f"""[来自 {message.source_agent_name} 的中继消息]
类型: {msg_type}
内容: {message.content}

请考虑这个信息，如果它与你的分析相关，请进行整合和调整。
"""
            
            self.messages.append(LLMMessage(
                role="user",
                content=prompt
            ))
            processed_messages.append(message)
        
        return processed_messages
    
    def _build_intervention_prompt(self, message: RelayMessage) -> str:
        """构建人工干预的智能提示
        
        根据干预类型和内容，构建引导 Agent 正确响应的提示
        """
        intervention_type = message.metadata.get("intervention_type", "unknown")
        priority = message.metadata.get("priority", 5)
        payload = message.metadata.get("payload", {})
        
        # 基础框架
        prompt_parts = [
            f"⚠️ **[重要：人工干预通知 - 优先级 {priority}/10]**",
            f"",
            f"来自: {message.source_agent_name}",
            f"",
        ]
        
        # 根据干预类型添加具体指导
        if intervention_type == InterventionType.INJECT.value:
            info = payload.get("information", message.content)
            prompt_parts.extend([
                "**类型**: 信息注入",
                "",
                "**注入内容**:",
                info,
                "",
                "**请执行以下操作**:",
                "1. 仔细阅读上述注入的信息",
                "2. 评估这些信息与你当前任务的相关性",
                "3. 如果相关，将其整合到你的分析中",
                "4. 如果需要调整方向，说明调整原因",
                "5. 在下一轮输出中体现对这些信息的考虑",
            ])
        
        elif intervention_type == InterventionType.ADJUST.value:
            adjustments = payload.get("adjustments", {})
            prompt_parts.extend([
                "**类型**: 行为调整指令",
                "",
                "**调整要求**:",
            ])
            for key, value in adjustments.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.extend([
                "",
                "**请执行以下操作**:",
                "1. 理解上述调整要求",
                "2. 评估如何在保持任务目标的前提下融入这些调整",
                "3. 在后续工作中体现这些调整",
                "4. 如果某些调整与当前任务冲突，请说明原因",
            ])
        
        elif intervention_type == InterventionType.PAUSE.value:
            prompt_parts.extend([
                "**类型**: 暂停通知",
                "",
                message.content,
                "",
                "**注意**: 你可能即将被暂停，请在当前响应中总结进度。",
            ])
        
        elif intervention_type == InterventionType.RESUME.value:
            prompt_parts.extend([
                "**类型**: 恢复通知",
                "",
                message.content,
                "",
                "**请执行以下操作**:",
                "1. 回顾之前的工作进度",
                "2. 继续未完成的任务",
                "3. 如有新的信息需要考虑，请整合进来",
            ])
        
        elif intervention_type == InterventionType.CANCEL.value:
            prompt_parts.extend([
                "**类型**: 取消通知",
                "",
                message.content,
                "",
                "**注意**: 另一个 Agent 的任务已被取消。如果这影响到你的工作，请相应调整。",
            ])
        
        else:
            # 通用处理
            prompt_parts.extend([
                message.content,
                "",
                "请根据上述人工干预信息，适当调整你的工作。",
            ])
        
        # 添加确认要求
        if message.metadata.get("requires_acknowledgement"):
            prompt_parts.extend([
                "",
                "---",
                "📝 请在你的下一轮响应开头确认收到此干预通知。",
            ])
        
        return "\n".join(prompt_parts)
    
    def _build_continuation_prompt(
        self, 
        iteration: int, 
        last_response: str,
        pending_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建继续迭代的引导提示
        
        根据当前迭代阶段、上一轮输出和待处理消息，引导 Agent 继续深入分析
        
        Args:
            iteration: 当前迭代次数
            last_response: 上一轮响应
            pending_summary: 待处理消息摘要（如果有）
        """
        prompt_parts = []
        
        # 如果有待处理消息，优先提醒
        if pending_summary and pending_summary.get("total_count", 0) > 0:
            prompt_parts.append("⚠️ **注意：中继站有待处理的消息**")
            prompt_parts.append("")
            
            if pending_summary.get("intervention_count", 0) > 0:
                prompt_parts.append(f"- 人工干预消息: {pending_summary['intervention_count']} 条")
                for intervention in pending_summary.get("interventions", []):
                    prompt_parts.append(f"  - 类型: {intervention['type']}, 优先级: {intervention['priority']}")
            
            if pending_summary.get("total_count", 0) > pending_summary.get("intervention_count", 0):
                other_count = pending_summary["total_count"] - pending_summary.get("intervention_count", 0)
                prompt_parts.append(f"- 其他中继消息: {other_count} 条")
            
            prompt_parts.append("")
            prompt_parts.append("请先处理这些消息后再继续你的分析。")
            prompt_parts.append("")
            prompt_parts.append("---")
            prompt_parts.append("")
        
        # 根据迭代阶段添加引导
        if iteration == 1:
            # 第一轮后，引导深入细节
            prompt_parts.extend([
                "你的初步分析很好。现在请：",
                "1. 针对你提到的关键点，进行更深入的分析",
                "2. 考虑是否有遗漏的角度或维度",
                "3. 如果有重要发现，请用 **[关键发现]** 标记",
                "",
                "继续深入分析："
            ])
        elif iteration == 2:
            # 第二轮后，引导发现关联
            prompt_parts.extend([
                "分析正在深入。请：",
                "1. 思考你的发现之间有什么关联或模式",
                "2. 是否有需要与其他专家角色协作确认的问题？如有，请用 **[请求中继: 原因]** 标记",
                "3. 继续挖掘潜在的洞察",
                "",
                "继续分析："
            ])
        elif iteration == 3:
            # 第三轮后，引导整合
            prompt_parts.extend([
                "分析已经比较深入。请：",
                "1. 尝试整合你的各项发现",
                "2. 形成初步的结论框架",
                "3. 如果你认为分析已经完整，可以用 **[任务完成]** 标记并给出完整结论",
                "",
                "继续："
            ])
        else:
            # 后续轮次，给予更大自由度
            prompt_parts.extend([
                "请继续你的分析，如果你认为已经足够深入和完整，请用 **[任务完成]** 标记并给出最终分析结果。",
                "",
                "继续："
            ])
        
        return "\n".join(prompt_parts)
    
    def _build_pending_message_prompt(self, pending_summary: Dict[str, Any]) -> str:
        """构建待处理消息提示
        
        当 Agent 尝试完成任务但有未处理的中继消息时，
        引导其先处理这些消息
        
        Args:
            pending_summary: 待处理消息摘要
        """
        prompt_parts = [
            "⚠️ **任务完成被阻止**",
            "",
            "在标记任务完成之前，你需要先处理中继站中的待处理消息：",
            "",
        ]
        
        # 详细列出待处理内容
        if pending_summary.get("intervention_count", 0) > 0:
            prompt_parts.append(f"📢 **人工干预消息** ({pending_summary['intervention_count']} 条):")
            for i, intervention in enumerate(pending_summary.get("interventions", []), 1):
                prompt_parts.append(f"  {i}. 类型: {intervention['type']}")
                prompt_parts.append(f"     优先级: {intervention['priority']}/10")
                prompt_parts.append(f"     内容预览: {intervention['content_preview'][:80]}...")
            prompt_parts.append("")
        
        if pending_summary.get("unacknowledged_count", 0) > 0:
            prompt_parts.append(f"❗ **未确认的干预消息**: {pending_summary['unacknowledged_count']} 条")
            prompt_parts.append("")
        
        other_count = pending_summary.get("total_count", 0) - pending_summary.get("intervention_count", 0)
        if other_count > 0:
            prompt_parts.append(f"💬 **其他中继消息**: {other_count} 条")
            prompt_parts.append("")
        
        # 添加处理指导
        prompt_parts.extend([
            "---",
            "",
            "**请按以下步骤处理**：",
            "1. 仔细阅读上述待处理消息的内容",
            "2. 根据消息内容调整你的分析或结论",
            "3. 如果收到人工干预，请明确确认：「已收到干预通知，内容是...」",
            "4. 如果干预要求你调整方向，请说明你的调整",
            "5. 处理完所有消息后，再考虑是否可以完成任务",
            "",
            "请处理这些消息并给出你的响应："
        ])
        
        return "\n".join(prompt_parts)
    
    def _update_status(self, status: AgentStatus):
        """更新状态"""
        self.state.status = status
        self.state.updated_at = datetime.now()
    
    def _update_progress(self, progress: float, step: str):
        """更新进度"""
        self.state.progress = progress
        self.state.current_step = step
        self.state.updated_at = datetime.now()
        
        if self.on_progress:
            self.on_progress(self.agent_id, progress, step)
