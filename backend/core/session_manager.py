"""
Session Manager - 会话管理器

核心职责：
1. 管理所有活跃会话，实现数据隔离
2. 为每个会话创建独立的 MasterAgent 实例
3. 提供会话生命周期管理
4. 支持会话级别的资源清理
5. 【新增】数据持久化 - 通过 Repository 存储到数据库
6. 【新增】订阅者管理 - 支持多客户端订阅同一会话的 SSE 事件流

设计原则：
- 每个 session_id 对应一个完全独立的 MasterAgent
- 不同会话之间的数据完全隔离（Agent、中继站、消息历史等）
- 支持会话超时自动清理
- 依赖注入 - Repository 可替换（SQLite/MySQL/PostgreSQL）
- 发布-订阅模式 - 支持多客户端同时订阅同一会话
"""

import asyncio
import uuid
import json
from typing import Dict, Optional, List, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """会话信息（内存缓存）"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)
    provider: str = "openai"
    model: Optional[str] = None
    task: Optional[str] = None
    status: str = "active"  # active, completed, expired, error
    plan: Optional[Dict] = None
    final_report: Optional[str] = None
    error: Optional[str] = None
    user_id: Optional[str] = None
    mode: str = "emergent"  # emergent(涌现模式) / direct(普通模式)
    # 追问支持：上一轮关键信息
    intervention_summary: Optional[str] = None  # 人工干预摘要
    task_history: Optional[List[Dict]] = None   # 历史任务 [{task, summary, roles, timestamp}]
    previous_roles: Optional[List[Dict]] = None  # 上一轮角色配置（用于角色复用）
    
    def touch(self):
        """更新最后活跃时间"""
        self.last_active_at = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """检查会话是否过期"""
        return datetime.now() - self.last_active_at > timedelta(minutes=timeout_minutes)
    
    def has_history(self) -> bool:
        """是否有历史任务结果（用于判断追问）
        
        对涌现模式要求 final_report 存在；
        对普通模式只要求状态已完成（DirectAgent 保留对话历史，即使没有 final_report 也能追问）。
        """
        if self.status not in ("completed", "cancelled", "failed", "error"):
            return False
        # direct 模式：只要已完成就可以追问（DirectAgent 内部维护 conversation_history）
        if self.mode == "direct":
            return True
        # emergent 模式：需要 final_report
        return self.final_report is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "provider": self.provider,
            "model": self.model,
            "task": self.task,
            "status": self.status,
            "plan": self.plan,
            "final_report": self.final_report,
            "error": self.error,
            "user_id": self.user_id,
            "mode": self.mode,
            "intervention_summary": self.intervention_summary,
            "task_history": self.task_history,
            "has_history": self.has_history(),
        }
    
    def build_followup_context(self, max_chars: int = 2500) -> str:
        """构建追问上下文摘要，注入新一轮 MasterAgent。
        
        3 层裁剪：
        1. final_report 截取前 1500 字符
        2. intervention_summary 保留（已在保存时裁剪）
        3. 总上下文不超过 max_chars
        
        Returns:
            格式化的追问上下文字符串
        """
        parts = []
        
        # 历史任务链（最近 3 轮）
        if self.task_history:
            parts.append("## 历史任务记录")
            for i, entry in enumerate(self.task_history[-3:], 1):
                parts.append(f"### 第 {i} 轮")
                parts.append(f"- 任务: {entry.get('task', '未知')}")
                summary = entry.get('summary', '')
                if summary:
                    parts.append(f"- 结论摘要: {summary[:500]}")
            parts.append("")
        
        # 上一轮最终报告（核心结论）
        if self.final_report:
            report_truncated = self.final_report[:1500]
            if len(self.final_report) > 1500:
                report_truncated += "\n...(报告已截取前 1500 字符)"
            parts.append("## 上一轮任务的最终报告")
            parts.append(report_truncated)
            parts.append("")
        
        # 人工干预记录
        if self.intervention_summary:
            parts.append("## 用户干预记录")
            parts.append(self.intervention_summary)
            parts.append("")
        
        context = "\n".join(parts)
        
        # 总长度裁剪
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...(上下文已截取)"
        
        return context


class SessionManager:
    """
    会话管理器 - 单例模式
    
    管理所有活跃会话，确保每个会话有独立的 MasterAgent 实例
    
    新特性：
    - 依赖注入 Repository 实现数据持久化
    - 支持 SQLite、MySQL、PostgreSQL
    - 内存缓存 + 数据库持久化双层架构
    - 【新增】订阅者管理：支持多客户端订阅同一会话的 SSE 事件流
    """
    
    _instance: Optional['SessionManager'] = None
    _lock: Lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 内存缓存 - 活跃会话
        self._sessions: Dict[str, SessionInfo] = {}
        
        # MasterAgent 实例存储 (延迟导入，避免循环依赖)
        self._agents: Dict[str, Any] = {}  # session_id -> MasterAgent
        
        # 配置
        self._session_timeout_minutes: int = 60
        self._max_sessions: int = 100
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # 数据仓库（依赖注入）
        self._repository = None
        
        # 【新增】订阅者管理
        # session_id -> List[asyncio.Queue] - 每个订阅者一个 queue
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        # 订阅者锁，保证线程安全
        self._subscriber_lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
        
        self._initialized = True
        logger.info("[SessionManager] Initialized (without repository)")
    
    def set_repository(self, repository):
        """注入数据仓库
        
        Args:
            repository: 实现了 BaseSessionRepository 等接口的仓库实例
        """
        self._repository = repository
        logger.info(f"[SessionManager] Repository injected: {type(repository).__name__}")
    
    def get_repository(self):
        """获取数据仓库（延迟初始化）"""
        if self._repository is None:
            from storage import get_repository
            self._repository = get_repository()
            logger.info(f"[SessionManager] Repository auto-initialized: {type(self._repository).__name__}")
        return self._repository
    
    async def create_session(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        task: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: str = "emergent"
    ) -> str:
        """
        创建新会话
        
        Args:
            provider: LLM 提供者
            model: 模型名称
            task: 任务描述
            user_id: 用户 ID（可选）
            mode: 会话模式 emergent/direct
            
        Returns:
            新创建的 session_id
        """
        # 检查是否超过最大会话数
        if len(self._sessions) >= self._max_sessions:
            await self._cleanup_expired_sessions()
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(f"Maximum sessions ({self._max_sessions}) reached")
        
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session_info = SessionInfo(
            session_id=session_id,
            created_at=now,
            last_active_at=now,
            provider=provider,
            model=model,
            task=task,
            user_id=user_id,
            mode=mode
        )
        
        # 保存到内存
        self._sessions[session_id] = session_info
        
        # 持久化到数据库
        try:
            repo = self.get_repository()
            from storage.base import SessionRecord
            record = SessionRecord(
                session_id=session_id,
                task=task or "",
                status="active",
                provider=provider,
                model=model,
                mode=mode,
                user_id=user_id,
                created_at=now,
                updated_at=now,
                last_active_at=now,
            )
            await repo.create_session(record)
            logger.info(f"[SessionManager] Session persisted to database: {session_id[:8]}...")
        except Exception as e:
            logger.error(f"[SessionManager] Failed to persist session: {e}")
        
        logger.info(f"[SessionManager] Created session: {session_id[:8]}...")
        return session_id
    
    # 同步版本（兼容旧代码）
    def create_session_sync(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        task: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: str = "emergent"
    ) -> str:
        """同步创建会话（内部使用 asyncio.run 或事件循环）
        
        注意：当事件循环正在运行时，直接在内存中创建会话，
        并异步持久化到数据库。不会重复创建会话。
        """
        import asyncio
        
        # 先生成唯一的 session_id
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session_info = SessionInfo(
            session_id=session_id,
            created_at=now,
            last_active_at=now,
            provider=provider,
            model=model,
            task=task,
            user_id=user_id,
            mode=mode
        )
        
        # 保存到内存
        self._sessions[session_id] = session_info
        
        # 异步持久化到数据库
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中，创建异步任务
                asyncio.create_task(self._persist_session(session_info))
            else:
                loop.run_until_complete(self._persist_session(session_info))
        except RuntimeError:
            # 没有事件循环，创建新的
            asyncio.run(self._persist_session(session_info))
        
        logger.info(f"[SessionManager] Created session (sync): {session_id[:8]}...")
        return session_id
    
    async def _persist_session(self, session_info: SessionInfo):
        """异步持久化会话"""
        try:
            repo = self.get_repository()
            from storage.base import SessionRecord
            record = SessionRecord(
                session_id=session_info.session_id,
                task=session_info.task or "",
                status=session_info.status,
                provider=session_info.provider,
                model=session_info.model,
                mode=session_info.mode,
                user_id=session_info.user_id,
                plan_json=json.dumps(session_info.plan) if session_info.plan else None,
                created_at=session_info.created_at,
                updated_at=session_info.last_active_at,
                last_active_at=session_info.last_active_at,
            )
            await repo.create_session(record)
        except Exception as e:
            logger.error(f"[SessionManager] Async persist failed: {e}")
    
    def get_or_create_agent(
        self,
        session_id: str,
        provider: str = "openai",
        model: Optional[str] = None
    ) -> Any:
        """
        获取或创建会话对应的 MasterAgent
        
        这是核心方法：确保每个 session_id 都有独立的 MasterAgent 实例
        
        Args:
            session_id: 会话 ID
            provider: LLM 提供者
            model: 模型名称
            
        Returns:
            该会话专属的 MasterAgent 实例
        """
        # 延迟导入，避免循环依赖
        from core.master_agent import MasterAgent
        
        # 如果会话不存在，先创建会话
        if session_id not in self._sessions:
            now = datetime.now()
            self._sessions[session_id] = SessionInfo(
                session_id=session_id,
                created_at=now,
                last_active_at=now,
                provider=provider,
                model=model
            )
            # 异步持久化
            asyncio.create_task(self._persist_session(self._sessions[session_id]))
            logger.info(f"[SessionManager] Auto-created session: {session_id[:8]}...")
        
        # 更新最后活跃时间
        self._sessions[session_id].touch()
        asyncio.create_task(self._update_session_activity(session_id))
        
        # 如果 Agent 不存在，创建新的
        if session_id not in self._agents:
            # 从 SessionInfo 中获取 user_id 传递给 MasterAgent
            session_user_id = self._sessions[session_id].user_id if session_id in self._sessions else None
            self._agents[session_id] = MasterAgent(
                provider_type=provider,
                model=model,
                session_id=session_id,  # 传入 session_id 用于内部隔离
                user_id=session_user_id,
            )
            logger.info(f"[SessionManager] Created MasterAgent for session: {session_id[:8]}...")
        
        return self._agents[session_id]
    
    def get_or_create_direct_agent(
        self,
        session_id: str,
        provider: str = "openai",
        model: Optional[str] = None
    ) -> Any:
        """
        获取或创建会话对应的 DirectAgent（普通模式）
        
        Args:
            session_id: 会话 ID
            provider: LLM 提供者
            model: 模型名称
            
        Returns:
            该会话专属的 DirectAgent 实例
        """
        from core.direct_agent import DirectAgent
        
        # 如果会话不存在，先创建
        if session_id not in self._sessions:
            now = datetime.now()
            self._sessions[session_id] = SessionInfo(
                session_id=session_id,
                created_at=now,
                last_active_at=now,
                provider=provider,
                model=model,
                mode="direct"
            )
            asyncio.create_task(self._persist_session(self._sessions[session_id]))
            logger.info(f"[SessionManager] Auto-created session (direct): {session_id[:8]}...")
        
        self._sessions[session_id].touch()
        asyncio.create_task(self._update_session_activity(session_id))
        
        # 如果 Agent 不存在或不是 DirectAgent，创建新的
        existing = self._agents.get(session_id)
        if existing is None or not isinstance(existing, DirectAgent):
            session_user_id = self._sessions[session_id].user_id if session_id in self._sessions else None
            self._agents[session_id] = DirectAgent(
                provider_type=provider,
                model=model,
                session_id=session_id,
                user_id=session_user_id,
            )
            logger.info(f"[SessionManager] Created DirectAgent for session: {session_id[:8]}...")
        
        return self._agents[session_id]
    
    async def _update_session_activity(self, session_id: str):
        """更新会话活跃时间到数据库"""
        try:
            repo = self.get_repository()
            await repo.touch_session(session_id)
        except Exception as e:
            logger.debug(f"[SessionManager] Failed to update activity: {e}")
    
    def get_agent(self, session_id: str) -> Optional[Any]:
        """
        获取指定会话的 MasterAgent
        
        Args:
            session_id: 会话 ID
            
        Returns:
            MasterAgent 实例，如果不存在返回 None
        """
        if session_id in self._sessions:
            self._sessions[session_id].touch()
        
        return self._agents.get(session_id)
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息（从内存缓存）"""
        return self._sessions.get(session_id)
    
    async def get_session_info_from_db(self, session_id: str) -> Optional[Dict[str, Any]]:
        """从数据库获取会话信息"""
        try:
            repo = self.get_repository()
            record = await repo.get_session(session_id)
            if record:
                return record.to_dict()
            return None
        except Exception as e:
            logger.error(f"[SessionManager] Failed to get session from DB: {e}")
            return None
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有活跃会话（内存缓存）
        
        数据隔离：严格按 user_id 过滤，只返回当前用户自己的会话。
        user_id 为 None 时返回空列表，防止未认证用户看到所有数据。
        """
        if not user_id:
            return []
        sessions = [info for info in self._sessions.values() if info.user_id == user_id]
        return [
            {
                "session_id": info.session_id,
                "created_at": info.created_at.isoformat() if info.created_at else None,
                "last_active_at": info.last_active_at.isoformat() if info.last_active_at else None,
                "provider": info.provider,
                "model": info.model,
                "task": info.task[:100] if info.task else None,
                "status": info.status,
                "mode": info.mode,
                "has_agent": info.session_id in self._agents,
                "user_id": info.user_id,
            }
            for info in sessions
        ]
    
    async def list_sessions_from_db(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """从数据库列出会话（支持分页）"""
        try:
            repo = self.get_repository()
            records = await repo.list_sessions(
                status=status,
                user_id=user_id,
                limit=limit,
                offset=offset,
                order_by="created_at",
                order_desc=True
            )
            
            result = []
            for record in records:
                data = record.to_dict()
                # 添加是否有活跃 Agent 的标记
                data["has_agent"] = record.session_id in self._agents
                data["is_active_in_memory"] = record.session_id in self._sessions
                result.append(data)
            
            return result
        except Exception as e:
            logger.error(f"[SessionManager] Failed to list sessions from DB: {e}")
            return []
    
    async def count_sessions_from_db(self, status: Optional[str] = None, user_id: Optional[str] = None) -> int:
        """统计数据库中的会话数量（按用户隔离）"""
        try:
            repo = self.get_repository()
            return await repo.count_sessions(status, user_id=user_id)
        except Exception as e:
            logger.error(f"[SessionManager] Failed to count sessions: {e}")
            return 0
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """更新会话信息
        
        同时更新内存缓存和数据库
        """
        # 更新内存缓存
        if session_id in self._sessions:
            session = self._sessions[session_id]
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.touch()
        
        # 更新数据库
        try:
            repo = self.get_repository()
            
            # 转换某些字段
            db_updates = {}
            for key, value in updates.items():
                if key == "plan" and value is not None:
                    db_updates["plan_json"] = json.dumps(value)
                else:
                    db_updates[key] = value
            
            db_updates["updated_at"] = datetime.now()
            db_updates["last_active_at"] = datetime.now()
            
            await repo.update_session(session_id, db_updates)
            return True
        except Exception as e:
            logger.error(f"[SessionManager] Failed to update session: {e}")
            return False
    
    async def close_session(self, session_id: str) -> bool:
        """
        关闭并清理会话
        
        Args:
            session_id: 会话 ID
            
        Returns:
            是否成功关闭
        """
        # 清理 MasterAgent
        if session_id in self._agents:
            agent = self._agents[session_id]
            # 清理 Agent 内部资源
            if hasattr(agent, 'cleanup'):
                try:
                    agent.cleanup()
                except Exception as e:
                    logger.error(f"[SessionManager] Error cleaning up agent: {e}")
            del self._agents[session_id]
        
        # 更新状态
        if session_id in self._sessions:
            self._sessions[session_id].status = "completed"
            del self._sessions[session_id]
        
        # 更新数据库状态（不删除，保留历史）
        try:
            repo = self.get_repository()
            await repo.update_session(session_id, {
                "status": "completed",
                "updated_at": datetime.now()
            })
        except Exception as e:
            logger.error(f"[SessionManager] Failed to update session status in DB: {e}")
        
        logger.info(f"[SessionManager] Closed session: {session_id[:8]}...")
        return True
    
    # 同步版本
    def close_session_sync(self, session_id: str) -> bool:
        """同步关闭会话"""
        if session_id not in self._sessions and session_id not in self._agents:
            return False
        
        # 清理 MasterAgent
        if session_id in self._agents:
            agent = self._agents[session_id]
            if hasattr(agent, 'cleanup'):
                try:
                    agent.cleanup()
                except Exception as e:
                    logger.error(f"[SessionManager] Error cleaning up agent: {e}")
            del self._agents[session_id]
        
        # 更新内存状态
        if session_id in self._sessions:
            self._sessions[session_id].status = "completed"
            del self._sessions[session_id]
        
        # 异步更新数据库
        asyncio.create_task(self._update_session_status_in_db(session_id, "completed"))
        
        logger.info(f"[SessionManager] Closed session: {session_id[:8]}...")
        return True
    
    def prepare_followup(
        self,
        session_id: str,
        provider: str = "openai",
        model: Optional[str] = None
    ) -> Any:
        """准备追问：cleanup 旧 Agent、保留 SessionInfo、创建新 MasterAgent。
        
        关键：不删除 SessionInfo（保留历史数据），只 cleanup 并替换 Agent 实例。
        
        Args:
            session_id: 会话 ID
            provider: LLM 提供者
            model: 模型名称
            
        Returns:
            新创建的 MasterAgent 实例
        """
        from core.master_agent import MasterAgent
        
        # 1. Cleanup 旧 Agent（如果存在）
        if session_id in self._agents:
            agent = self._agents[session_id]
            if hasattr(agent, 'cleanup'):
                try:
                    agent.cleanup()
                except Exception as e:
                    logger.error(f"[SessionManager] Error cleaning up agent for followup: {e}")
            del self._agents[session_id]
        
        # 2. 更新 SessionInfo 状态为 active（重新激活）
        if session_id in self._sessions:
            self._sessions[session_id].status = "active"
            self._sessions[session_id].touch()
        
        # 3. 创建新 MasterAgent
        session_user_id = self._sessions[session_id].user_id if session_id in self._sessions else None
        new_agent = MasterAgent(
            provider_type=provider,
            model=model,
            session_id=session_id,
            user_id=session_user_id,
        )
        self._agents[session_id] = new_agent
        
        logger.info(f"[SessionManager] Prepared followup for session: {session_id[:8]}... (new MasterAgent created)")
        return new_agent
    
    def save_task_completion(
        self,
        session_id: str,
        final_report: str,
        plan: Optional[Dict] = None,
        intervention_summary: Optional[str] = None,
        roles: Optional[List[Dict]] = None,
    ):
        """任务完成时保存关键信息到 SessionInfo（在 cleanup 前调用）。
        
        Args:
            session_id: 会话 ID
            final_report: 最终报告
            plan: 任务计划
            intervention_summary: 人工干预摘要
            roles: 角色配置列表
        """
        if session_id not in self._sessions:
            logger.warning(f"[SessionManager] Session not found for save_task_completion: {session_id[:8]}")
            return
        
        session = self._sessions[session_id]
        session.final_report = final_report
        if plan:
            session.plan = plan
        session.intervention_summary = intervention_summary
        session.previous_roles = roles
        
        # 追加到 task_history
        if session.task_history is None:
            session.task_history = []
        
        # 生成当前轮次的摘要条目
        summary = final_report[:500] if final_report else ""
        session.task_history.append({
            "task": session.task or "未知任务",
            "summary": summary,
            "roles": [r.get("name", "") for r in (roles or [])],
            "timestamp": datetime.now().isoformat(),
        })
        
        # 只保留最近 3 轮
        if len(session.task_history) > 3:
            session.task_history = session.task_history[-3:]
        
        logger.info(f"[SessionManager] Saved task completion for session: {session_id[:8]}... "
                     f"(report={len(final_report)}chars, roles={len(roles or [])})")

    async def _update_session_status_in_db(self, session_id: str, status: str):
        """异步更新数据库中的会话状态"""
        try:
            repo = self.get_repository()
            await repo.update_session(session_id, {
                "status": status,
                "updated_at": datetime.now()
            })
        except Exception as e:
            logger.error(f"[SessionManager] Failed to update status in DB: {e}")
    
    async def delete_session(self, session_id: str) -> bool:
        """彻底删除会话（包括数据库记录）"""
        # 先关闭
        await self.close_session(session_id)
        
        # 从数据库删除
        try:
            repo = self.get_repository()
            return await repo.delete_session(session_id)
        except Exception as e:
            logger.error(f"[SessionManager] Failed to delete session: {e}")
            return False
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        # 清理内存中的过期会话
        expired_sessions = [
            session_id
            for session_id, info in self._sessions.items()
            if info.is_expired(self._session_timeout_minutes)
        ]
        
        for session_id in expired_sessions:
            logger.info(f"[SessionManager] Cleaning up expired session: {session_id[:8]}...")
            await self.close_session(session_id)
        
        # 清理数据库中的过期会话
        try:
            repo = self.get_repository()
            count = await repo.cleanup_expired_sessions(self._session_timeout_minutes)
            if count > 0:
                logger.info(f"[SessionManager] Marked {count} expired sessions in DB")
        except Exception as e:
            logger.error(f"[SessionManager] Failed to cleanup DB sessions: {e}")
        
        if expired_sessions:
            logger.info(f"[SessionManager] Cleaned up {len(expired_sessions)} expired sessions")
    
    def _cleanup_expired_sessions_sync(self):
        """同步清理过期会话"""
        expired_sessions = [
            session_id
            for session_id, info in self._sessions.items()
            if info.is_expired(self._session_timeout_minutes)
        ]
        
        for session_id in expired_sessions:
            logger.info(f"[SessionManager] Cleaning up expired session: {session_id[:8]}...")
            self.close_session_sync(session_id)
        
        if expired_sessions:
            logger.info(f"[SessionManager] Cleaned up {len(expired_sessions)} expired sessions")
    
    async def start_cleanup_task(self, interval_minutes: int = 10):
        """启动定期清理任务"""
        if self._cleanup_task is not None:
            return
        
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval_minutes * 60)
                await self._cleanup_expired_sessions()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"[SessionManager] Started cleanup task (interval: {interval_minutes} minutes)")
    
    def stop_cleanup_task(self):
        """停止清理任务"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None
    
    @property
    def active_session_count(self) -> int:
        """获取活跃会话数量（内存中）"""
        return len(self._sessions)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        return {
            "active_sessions": len(self._sessions),
            "active_agents": len(self._agents),
            "max_sessions": self._max_sessions,
            "timeout_minutes": self._session_timeout_minutes,
            "has_repository": self._repository is not None,
        }
    
    async def get_full_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取完整统计信息（包括数据库，按用户隔离）"""
        stats = self.get_stats()
        
        try:
            total_count = await self.count_sessions_from_db(user_id=user_id)
            active_count = await self.count_sessions_from_db(status="active", user_id=user_id)
            completed_count = await self.count_sessions_from_db(status="completed", user_id=user_id)
            
            stats.update({
                "db_total_sessions": total_count,
                "db_active_sessions": active_count,
                "db_completed_sessions": completed_count,
            })
        except Exception as e:
            logger.error(f"[SessionManager] Failed to get DB stats: {e}")
        
        return stats
    
    # ========== Agent 数据持久化方法 ==========
    
    async def save_agent(
        self,
        session_id: str,
        agent_id: str,
        agent_data: Dict[str, Any]
    ) -> bool:
        """保存 Agent 数据到数据库"""
        try:
            repo = self.get_repository()
            from storage.base import AgentRecord
            
            # 检查是否已存在
            existing = await repo.get_agent(agent_id, session_id)
            
            if existing:
                # 更新
                await repo.update_agent(agent_id, session_id, agent_data)
            else:
                # 创建
                record = AgentRecord(
                    agent_id=agent_id,
                    session_id=session_id,
                    name=agent_data.get("name", ""),
                    role_name=agent_data.get("role_name", ""),
                    role_description=agent_data.get("role_description", ""),
                    capabilities=json.dumps(agent_data.get("capabilities", [])),
                    task_segment=agent_data.get("task_segment", ""),
                    status=agent_data.get("status", "pending"),
                    progress=agent_data.get("progress", 0),
                    current_step=agent_data.get("current_step", ""),
                    iterations=agent_data.get("iterations", 0),
                    thinking=agent_data.get("thinking", ""),
                    work_objective=agent_data.get("work_objective"),
                    deliverables=json.dumps(agent_data.get("deliverables", [])),
                    methodology=agent_data.get("methodology"),
                    assigned_skills=json.dumps(agent_data.get("assigned_skills", [])),
                    expertise_level=agent_data.get("expertise_level"),
                    focus_areas=json.dumps(agent_data.get("focus_areas", [])),
                )
                await repo.create_agent(record)
            
            return True
        except Exception as e:
            logger.error(f"[SessionManager] Failed to save agent: {e}")
            return False
    
    async def get_session_agents(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话的所有 Agent"""
        try:
            repo = self.get_repository()
            records = await repo.list_agents_by_session(session_id)
            return [r.to_dict() for r in records]
        except Exception as e:
            logger.error(f"[SessionManager] Failed to get agents: {e}")
            return []
    
    # ========== 中继站数据持久化方法 ==========
    
    async def save_relay_station(
        self,
        session_id: str,
        station_data: Dict[str, Any]
    ) -> bool:
        """保存中继站数据"""
        try:
            repo = self.get_repository()
            from storage.base import RelayStationRecord
            
            station_id = station_data.get("station_id") or station_data.get("id")
            existing = await repo.get_station(station_id, session_id)
            
            if existing:
                await repo.update_station(station_id, session_id, station_data)
            else:
                record = RelayStationRecord(
                    station_id=station_id,
                    session_id=session_id,
                    name=station_data.get("name", ""),
                    phase=station_data.get("phase", 0),
                    participating_agents=json.dumps(station_data.get("participating_agents", [])),
                    is_active=station_data.get("is_active", True),
                )
                await repo.create_station(record)
            
            return True
        except Exception as e:
            logger.error(f"[SessionManager] Failed to save relay station: {e}")
            return False
    
    async def save_relay_message(
        self,
        session_id: str,
        message_data: Dict[str, Any]
    ) -> bool:
        """保存中继消息"""
        try:
            repo = self.get_repository()
            from storage.base import RelayMessageRecord
            
            record = RelayMessageRecord(
                message_id=message_data.get("message_id") or message_data.get("id"),
                station_id=message_data.get("station_id"),
                session_id=session_id,
                relay_type=message_data.get("relay_type", ""),
                source_agent_id=message_data.get("source_agent_id", ""),
                source_agent_name=message_data.get("source_agent_name", ""),
                target_agent_ids=json.dumps(message_data.get("target_agent_ids", [])),
                content=message_data.get("content", ""),
                importance=message_data.get("importance", 5),
                viewed_by=json.dumps(message_data.get("viewed_by", [])),
                acknowledged_by=json.dumps(message_data.get("acknowledged_by", [])),
                viewed_timestamps=json.dumps(message_data.get("viewed_timestamps", {})),
            )
            await repo.create_relay_message(record)
            return True
        except Exception as e:
            logger.error(f"[SessionManager] Failed to save relay message: {e}")
            return False
    
    async def get_session_relay_history(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取会话的中继消息历史"""
        try:
            repo = self.get_repository()
            records = await repo.get_relay_messages_by_session(session_id, limit)
            return [r.to_dict() for r in records]
        except Exception as e:
            logger.error(f"[SessionManager] Failed to get relay history: {e}")
            return []
    
    # ========== 干预记录持久化 ==========
    
    async def save_intervention(
        self,
        session_id: str,
        intervention_data: Dict[str, Any]
    ) -> bool:
        """保存干预记录"""
        try:
            repo = self.get_repository()
            from storage.base import InterventionRecord
            
            record = InterventionRecord(
                intervention_id=intervention_data.get("id", str(uuid.uuid4())),
                session_id=session_id,
                intervention_type=intervention_data.get("type", ""),
                scope=intervention_data.get("scope", "single"),
                target_agent_id=intervention_data.get("target_agent_id"),
                target_agent_ids=json.dumps(intervention_data.get("target_agent_ids", [])),
                payload_json=json.dumps(intervention_data.get("payload")) if intervention_data.get("payload") else None,
                reason=intervention_data.get("reason", ""),
                priority=intervention_data.get("priority", 5),
            )
            await repo.create_intervention(record)
            return True
        except Exception as e:
            logger.error(f"[SessionManager] Failed to save intervention: {e}")
            return False
    
    async def get_session_interventions(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取会话的干预历史"""
        try:
            repo = self.get_repository()
            records = await repo.get_interventions_by_session(session_id, limit)
            return [r.to_dict() for r in records]
        except Exception as e:
            logger.error(f"[SessionManager] Failed to get interventions: {e}")
            return []
    
    # ========== 消息持久化 ==========
    
    async def save_message(
        self,
        session_id: str,
        message_data: Dict[str, Any]
    ) -> bool:
        """保存消息（Master Agent 的任务分析、思考过程等）"""
        try:
            repo = self.get_repository()
            from storage.base import MessageRecord
            
            record = MessageRecord(
                message_id=message_data.get("message_id") or message_data.get("id") or str(uuid.uuid4()),
                session_id=session_id,
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
                metadata_json=json.dumps(message_data.get("metadata")) if message_data.get("metadata") else None,
            )
            await repo.create_message(record)
            return True
        except Exception as e:
            logger.error(f"[SessionManager] Failed to save message: {e}")
            return False
    
    async def update_message(
        self,
        session_id: str,
        message_id: str,
        content: str
    ) -> bool:
        """更新消息内容（用于流式消息追加）"""
        try:
            repo = self.get_repository()
            # 获取现有消息
            messages = await repo.get_messages_by_session(session_id, limit=1000)
            for msg in messages:
                if msg.message_id == message_id:
                    # 追加内容而不是替换
                    new_content = msg.content + content
                    # 使用底层 SQL 更新
                    from storage.sqlalchemy_repository import MessageModel
                    with repo.get_db_session() as session:
                        model = session.query(MessageModel).filter_by(
                            session_id=session_id,
                            message_id=message_id
                        ).first()
                        if model:
                            model.content = new_content
                            return True
            return False
        except Exception as e:
            logger.error(f"[SessionManager] Failed to update message: {e}")
            return False
    
    async def get_session_messages(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取会话的消息历史"""
        try:
            repo = self.get_repository()
            records = await repo.get_messages_by_session(session_id, limit)
            return [r.to_dict() for r in records]
        except Exception as e:
            logger.error(f"[SessionManager] Failed to get messages: {e}")
            return []
    
    # ========== 订阅者管理 (新增 - 支持多客户端订阅) ==========
    
    def _ensure_subscriber_lock(self):
        """确保订阅者锁已初始化"""
        if self._subscriber_lock is None:
            try:
                self._subscriber_lock = asyncio.Lock()
            except RuntimeError:
                # 如果没有事件循环，先不初始化
                pass
    
    async def subscribe(self, session_id: str, maxsize: int = 100) -> asyncio.Queue:
        """订阅会话事件流
        
        创建一个 Queue 用于接收该会话的所有事件。
        支持多个客户端同时订阅同一会话。
        
        Args:
            session_id: 会话 ID
            maxsize: Queue 最大容量，防止内存溢出
            
        Returns:
            asyncio.Queue - 用于接收事件的队列
        """
        self._ensure_subscriber_lock()
        
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        
        if self._subscriber_lock:
            async with self._subscriber_lock:
                if session_id not in self._subscribers:
                    self._subscribers[session_id] = []
                self._subscribers[session_id].append(queue)
        else:
            if session_id not in self._subscribers:
                self._subscribers[session_id] = []
            self._subscribers[session_id].append(queue)
        
        subscriber_count = len(self._subscribers.get(session_id, []))
        logger.info(f"[SessionManager] Client subscribed to session {session_id[:8]}... (total: {subscriber_count})")
        
        return queue
    
    async def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """取消订阅会话事件流
        
        Args:
            session_id: 会话 ID
            queue: 要移除的 Queue
        """
        self._ensure_subscriber_lock()
        
        if self._subscriber_lock:
            async with self._subscriber_lock:
                if session_id in self._subscribers:
                    try:
                        self._subscribers[session_id].remove(queue)
                        if not self._subscribers[session_id]:
                            del self._subscribers[session_id]
                    except ValueError:
                        pass  # Queue 不在列表中
        else:
            if session_id in self._subscribers:
                try:
                    self._subscribers[session_id].remove(queue)
                    if not self._subscribers[session_id]:
                        del self._subscribers[session_id]
                except ValueError:
                    pass
        
        subscriber_count = len(self._subscribers.get(session_id, []))
        logger.info(f"[SessionManager] Client unsubscribed from session {session_id[:8]}... (remaining: {subscriber_count})")
    
    async def broadcast_event(self, session_id: str, event: Any) -> int:
        """广播事件到所有订阅该会话的客户端
        
        使用非阻塞方式发送，如果某个 Queue 已满则跳过（避免一个慢客户端阻塞其他客户端）
        
        Args:
            session_id: 会话 ID
            event: 要广播的事件（BaseEvent 实例）
            
        Returns:
            成功发送的客户端数量
        """
        if session_id not in self._subscribers:
            return 0
        
        sent_count = 0
        failed_queues = []
        
        # 复制列表避免迭代时修改
        queues = self._subscribers.get(session_id, [])[:]
        
        for queue in queues:
            try:
                # 非阻塞发送
                queue.put_nowait(event)
                sent_count += 1
            except asyncio.QueueFull:
                # Queue 已满，记录但不移除（客户端可能只是暂时慢）
                logger.warning(f"[SessionManager] Queue full for session {session_id[:8]}..., event dropped")
            except Exception as e:
                # 其他错误，标记为需要移除
                logger.error(f"[SessionManager] Failed to broadcast to queue: {e}")
                failed_queues.append(queue)
        
        # 清理失败的 queues
        if failed_queues:
            self._ensure_subscriber_lock()
            if self._subscriber_lock:
                async with self._subscriber_lock:
                    for queue in failed_queues:
                        try:
                            self._subscribers[session_id].remove(queue)
                        except (ValueError, KeyError):
                            pass
        
        if sent_count > 0:
            event_type = getattr(event, 'type', type(event).__name__)
            logger.debug(f"[SessionManager] Broadcasted {event_type} to {sent_count} subscribers for session {session_id[:8]}...")
        
        return sent_count
    
    async def broadcast_state_changed(
        self,
        session_id: str,
        change_type: str,
        summary: Dict[str, Any] = None
    ) -> int:
        """广播会话状态变更事件
        
        这是一个便捷方法，用于在重要状态变更时通知所有订阅者刷新。
        
        Args:
            session_id: 会话 ID
            change_type: 变更类型 ("agent_added", "agent_status_changed", "completed", etc.)
            summary: 变更摘要信息
            
        Returns:
            成功发送的客户端数量
        """
        from agui.events import SessionStateChangedEvent
        
        event = SessionStateChangedEvent(
            session_id=session_id,
            change_type=change_type,
            summary=summary or {}
        )
        
        return await self.broadcast_event(session_id, event)
    
    def get_subscriber_count(self, session_id: str) -> int:
        """获取会话的订阅者数量"""
        return len(self._subscribers.get(session_id, []))
    
    def get_all_subscriber_stats(self) -> Dict[str, int]:
        """获取所有会话的订阅者统计"""
        return {
            session_id: len(queues)
            for session_id, queues in self._subscribers.items()
        }


# 全局会话管理器实例
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局会话管理器实例"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
