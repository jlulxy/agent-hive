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
            yield TextMessageStartEvent(message_id=message_id, role="assistant")
            
            full_response = ""
            max_tool_rounds = 5
            
            # 工具调用循环：用非流式 chat_complete 处理 tool calling
            for tool_round in range(max_tool_rounds):
                if not tool_definitions:
                    break
                
                print(f"[DirectAgent] Tool round {tool_round + 1}, calling LLM (non-streaming for tool detection)...")
                response = await self.provider.chat_complete(
                    messages, self.llm_config, tools=tool_definitions
                )
                
                content = response.get("content", "")
                tool_calls = response.get("tool_calls")
                
                if not tool_calls:
                    # 没有工具调用 → 不使用这个非流式结果，跳出循环走流式输出
                    break
                
                # 有工具调用：发出 thinking 事件（LLM 在调用工具前的分析）
                if content:
                    yield AgentThinkingEvent(
                        agent_id=self.agent_id,
                        agent_name="Assistant",
                        thinking=content,
                    )
                
                # 处理工具调用
                messages.append(LLMMessage(
                    role="assistant",
                    content=content or "",
                    tool_calls=tool_calls
                ))
                
                for tc in tool_calls:
                    tool_call_id = tc["id"]
                    func_name = tc["function"]["name"]
                    func_args_str = tc["function"]["arguments"]
                    
                    # TOOL_CALL_START 事件
                    yield ToolCallStartEvent(
                        tool_call_id=tool_call_id,
                        tool_call_name=func_name,
                        parent_message_id=message_id,
                    )
                    
                    # TOOL_CALL_ARGS 事件
                    yield ToolCallArgsEvent(
                        tool_call_id=tool_call_id,
                        delta=func_args_str if isinstance(func_args_str, str) else json.dumps(func_args_str, ensure_ascii=False),
                    )
                    
                    # 执行技能
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
                            content=error_msg,
                            tool_call_id=tool_call_id,
                        ))
            
            # 最终文本回复：始终用流式输出
            print(f"[DirectAgent] Final streaming response...")
            async for chunk in self.provider.chat(messages, self.llm_config):
                full_response += chunk
                yield TextMessageContentEvent(
                    message_id=message_id,
                    delta=chunk
                )
            
            yield TextMessageEndEvent(message_id=message_id)
            
            # 更新对话历史
            self.conversation_history.append(LLMMessage(role="user", content=task))
            self.conversation_history.append(LLMMessage(role="assistant", content=full_response))
            
            # 保持对话历史在合理长度
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-16:]
            
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
            "conversation_turns": len(self.conversation_history) // 2,
        }
    
    def cleanup(self):
        """清理资源"""
        self.sessions.clear()
        self.conversation_history.clear()
        print(f"[DirectAgent] Session {self.session_id[:8]}... cleaned up")
