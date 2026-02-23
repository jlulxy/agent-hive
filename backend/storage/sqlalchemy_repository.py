"""
SQLAlchemy 仓库实现

支持 SQLite、MySQL、PostgreSQL
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc

from storage.base import (
    BaseSessionRepository,
    BaseAgentRepository,
    BaseMessageRepository,
    BaseRelayRepository,
    BaseInterventionRepository,
    BaseUserRepository,
    SessionRecord,
    AgentRecord,
    MessageRecord,
    RelayStationRecord,
    RelayMessageRecord,
    InterventionRecord,
    UserRecord,
)
from storage.sqlalchemy_models import (
    SessionModel,
    AgentModel,
    MessageModel,
    RelayStationModel,
    RelayMessageModel,
    InterventionModel,
    UserModel,
    create_database_engine,
    create_tables,
    get_session_factory,
)
from storage.config import StorageConfig


class SQLAlchemyRepository(
    BaseSessionRepository,
    BaseAgentRepository,
    BaseMessageRepository,
    BaseRelayRepository,
    BaseInterventionRepository,
    BaseUserRepository
):
    """SQLAlchemy 统一仓库实现
    
    实现所有数据访问接口，支持 SQLite、MySQL、PostgreSQL
    """
    
    def __init__(self, config: StorageConfig):
        """
        Args:
            config: 存储配置
        """
        self.config = config
        self._engine = None
        self._session_factory = None
        self._initialized = False
    
    def initialize(self):
        """初始化数据库连接"""
        if self._initialized:
            return
        
        # 确保 SQLite 数据目录存在
        if self.config.storage_type.value == "sqlite":
            import os
            db_dir = os.path.dirname(self.config.sqlite_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
        
        # 创建引擎
        self._engine = create_database_engine(
            self.config.get_connection_url(),
            echo=self.config.echo_sql,
            pool_size=self.config.pool_size,
        )
        
        # 创建表
        if self.config.auto_create_tables:
            create_tables(self._engine)
        
        # 创建会话工厂
        self._session_factory = get_session_factory(self._engine)
        
        self._initialized = True
        print(f"[SQLAlchemyRepository] Initialized: {self.config}")
    
    @contextmanager
    def get_db_session(self):
        """获取数据库会话（上下文管理器）"""
        if not self._initialized:
            self.initialize()
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    # ========== Session Repository ==========
    
    async def create_session(self, record: SessionRecord) -> SessionRecord:
        """创建会话"""
        with self.get_db_session() as session:
            model = SessionModel(
                session_id=record.session_id,
                task=record.task,
                status=record.status,
                provider=record.provider,
                model=record.model,
                mode=record.mode,
                user_id=record.user_id,
                plan_json=record.plan_json,
                final_report=record.final_report,
                error=record.error,
                created_at=record.created_at,
                updated_at=record.updated_at,
                last_active_at=record.last_active_at,
                metadata_json=record.metadata_json,
            )
            session.add(model)
            session.flush()
            
            return self._model_to_session_record(model)
    
    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """获取会话"""
        with self.get_db_session() as session:
            model = session.query(SessionModel).filter_by(session_id=session_id).first()
            if model:
                return self._model_to_session_record(model)
            return None
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> Optional[SessionRecord]:
        """更新会话"""
        with self.get_db_session() as session:
            model = session.query(SessionModel).filter_by(session_id=session_id).first()
            if not model:
                return None
            
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            model.updated_at = datetime.now()
            session.flush()
            
            return self._model_to_session_record(model)
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话（级联删除相关数据）"""
        with self.get_db_session() as session:
            model = session.query(SessionModel).filter_by(session_id=session_id).first()
            if model:
                session.delete(model)
                return True
            return False
    
    async def list_sessions(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[SessionRecord]:
        """列出会话"""
        with self.get_db_session() as session:
            query = session.query(SessionModel)
            
            if status:
                query = query.filter_by(status=status)
            
            if user_id:
                # 严格按用户隔离，只返回当前用户自己的会话
                query = query.filter(SessionModel.user_id == user_id)
            
            # 排序
            order_column = getattr(SessionModel, order_by, SessionModel.created_at)
            if order_desc:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(asc(order_column))
            
            # 分页
            models = query.offset(offset).limit(limit).all()
            
            return [self._model_to_session_record(m) for m in models]
    
    async def count_sessions(self, status: Optional[str] = None, user_id: Optional[str] = None) -> int:
        """统计会话数量（按用户隔离）"""
        with self.get_db_session() as session:
            query = session.query(SessionModel)
            if status:
                query = query.filter_by(status=status)
            if user_id:
                query = query.filter(SessionModel.user_id == user_id)
            return query.count()
    
    async def touch_session(self, session_id: str) -> bool:
        """更新会话最后活跃时间"""
        with self.get_db_session() as session:
            model = session.query(SessionModel).filter_by(session_id=session_id).first()
            if model:
                model.last_active_at = datetime.now()
                return True
            return False
    
    async def cleanup_expired_sessions(self, timeout_minutes: int = 60) -> int:
        """清理过期会话"""
        with self.get_db_session() as session:
            cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
            
            # 找出过期会话
            expired = session.query(SessionModel).filter(
                and_(
                    SessionModel.status == "active",
                    SessionModel.last_active_at < cutoff_time
                )
            ).all()
            
            count = len(expired)
            for model in expired:
                model.status = "expired"
            
            return count
    
    def _model_to_session_record(self, model: SessionModel) -> SessionRecord:
        """模型转记录"""
        return SessionRecord(
            session_id=model.session_id,
            task=model.task or "",
            status=model.status or "active",
            provider=model.provider or "openai",
            model=model.model,
            mode=getattr(model, 'mode', None) or "emergent",
            user_id=model.user_id,
            plan_json=model.plan_json,
            final_report=model.final_report,
            error=model.error,
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_active_at=model.last_active_at,
            metadata_json=model.metadata_json,
        )
    
    # ========== Agent Repository ==========
    
    async def create_agent(self, record: AgentRecord) -> AgentRecord:
        """创建 Agent 记录"""
        with self.get_db_session() as session:
            model = AgentModel(
                agent_id=record.agent_id,
                session_id=record.session_id,
                name=record.name,
                role_name=record.role_name,
                role_description=record.role_description,
                capabilities=record.capabilities,
                task_segment=record.task_segment,
                status=record.status,
                progress=record.progress,
                current_step=record.current_step,
                iterations=record.iterations,
                thinking=record.thinking,
                partial_result=record.partial_result,
                final_result=record.final_result,
                work_objective=record.work_objective,
                deliverables=record.deliverables,
                methodology=record.methodology,
                assigned_skills=record.assigned_skills,
                expertise_level=record.expertise_level,
                focus_areas=record.focus_areas,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            session.add(model)
            session.flush()
            
            return self._model_to_agent_record(model)
    
    async def get_agent(self, agent_id: str, session_id: str) -> Optional[AgentRecord]:
        """获取 Agent"""
        with self.get_db_session() as session:
            model = session.query(AgentModel).filter_by(
                agent_id=agent_id,
                session_id=session_id
            ).first()
            if model:
                return self._model_to_agent_record(model)
            return None
    
    async def update_agent(self, agent_id: str, session_id: str, updates: Dict[str, Any]) -> Optional[AgentRecord]:
        """更新 Agent"""
        with self.get_db_session() as session:
            model = session.query(AgentModel).filter_by(
                agent_id=agent_id,
                session_id=session_id
            ).first()
            if not model:
                return None
            
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            model.updated_at = datetime.now()
            session.flush()
            
            return self._model_to_agent_record(model)
    
    async def list_agents_by_session(self, session_id: str) -> List[AgentRecord]:
        """获取会话的所有 Agent"""
        with self.get_db_session() as session:
            models = session.query(AgentModel).filter_by(
                session_id=session_id
            ).order_by(AgentModel.created_at).all()
            
            return [self._model_to_agent_record(m) for m in models]
    
    async def delete_agents_by_session(self, session_id: str) -> int:
        """删除会话的所有 Agent"""
        with self.get_db_session() as session:
            count = session.query(AgentModel).filter_by(session_id=session_id).delete()
            return count
    
    def _model_to_agent_record(self, model: AgentModel) -> AgentRecord:
        return AgentRecord(
            agent_id=model.agent_id,
            session_id=model.session_id,
            name=model.name,
            role_name=model.role_name,
            role_description=model.role_description or "",
            capabilities=model.capabilities or "[]",
            task_segment=model.task_segment or "",
            status=model.status or "pending",
            progress=model.progress or 0,
            current_step=model.current_step or "",
            iterations=model.iterations or 0,
            thinking=model.thinking or "",
            partial_result=model.partial_result,
            final_result=model.final_result,
            work_objective=model.work_objective,
            deliverables=model.deliverables or "[]",
            methodology=model.methodology,
            assigned_skills=model.assigned_skills or "[]",
            expertise_level=model.expertise_level,
            focus_areas=model.focus_areas or "[]",
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
    
    # ========== Message Repository ==========
    
    async def create_message(self, record: MessageRecord) -> MessageRecord:
        """创建消息"""
        with self.get_db_session() as session:
            model = MessageModel(
                message_id=record.message_id,
                session_id=record.session_id,
                role=record.role,
                content=record.content,
                timestamp=record.timestamp,
                metadata_json=record.metadata_json,
            )
            session.add(model)
            session.flush()
            
            return self._model_to_message_record(model)
    
    async def get_messages_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[MessageRecord]:
        """获取会话的消息"""
        with self.get_db_session() as session:
            models = session.query(MessageModel).filter_by(
                session_id=session_id
            ).order_by(MessageModel.timestamp).offset(offset).limit(limit).all()
            
            return [self._model_to_message_record(m) for m in models]
    
    async def delete_messages_by_session(self, session_id: str) -> int:
        """删除会话的所有消息"""
        with self.get_db_session() as session:
            count = session.query(MessageModel).filter_by(session_id=session_id).delete()
            return count
    
    def _model_to_message_record(self, model: MessageModel) -> MessageRecord:
        return MessageRecord(
            message_id=model.message_id,
            session_id=model.session_id,
            role=model.role,
            content=model.content,
            timestamp=model.timestamp,
            metadata_json=model.metadata_json,
        )
    
    # ========== Relay Repository ==========
    
    async def create_station(self, record: RelayStationRecord) -> RelayStationRecord:
        """创建中继站"""
        with self.get_db_session() as session:
            model = RelayStationModel(
                station_id=record.station_id,
                session_id=record.session_id,
                name=record.name,
                phase=record.phase,
                participating_agents=record.participating_agents,
                is_active=record.is_active,
                summary=record.summary,
                created_at=record.created_at,
                closed_at=record.closed_at,
            )
            session.add(model)
            session.flush()
            
            return self._model_to_station_record(model)
    
    async def get_station(self, station_id: str, session_id: str) -> Optional[RelayStationRecord]:
        """获取中继站"""
        with self.get_db_session() as session:
            model = session.query(RelayStationModel).filter_by(
                station_id=station_id,
                session_id=session_id
            ).first()
            if model:
                return self._model_to_station_record(model)
            return None
    
    async def update_station(self, station_id: str, session_id: str, updates: Dict[str, Any]) -> Optional[RelayStationRecord]:
        """更新中继站"""
        with self.get_db_session() as session:
            model = session.query(RelayStationModel).filter_by(
                station_id=station_id,
                session_id=session_id
            ).first()
            if not model:
                return None
            
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            session.flush()
            
            return self._model_to_station_record(model)
    
    async def list_stations_by_session(self, session_id: str) -> List[RelayStationRecord]:
        """获取会话的所有中继站"""
        with self.get_db_session() as session:
            models = session.query(RelayStationModel).filter_by(
                session_id=session_id
            ).order_by(RelayStationModel.created_at).all()
            
            return [self._model_to_station_record(m) for m in models]
    
    async def create_relay_message(self, record: RelayMessageRecord) -> RelayMessageRecord:
        """创建中继消息"""
        with self.get_db_session() as session:
            model = RelayMessageModel(
                message_id=record.message_id,
                station_id=record.station_id,
                session_id=record.session_id,
                relay_type=record.relay_type,
                source_agent_id=record.source_agent_id,
                source_agent_name=record.source_agent_name,
                target_agent_ids=record.target_agent_ids,
                content=record.content,
                importance=record.importance,
                viewed_by=record.viewed_by,
                acknowledged_by=record.acknowledged_by,
                viewed_timestamps=record.viewed_timestamps,
                timestamp=record.timestamp,
                metadata_json=record.metadata_json,
            )
            session.add(model)
            session.flush()
            
            return self._model_to_relay_message_record(model)
    
    async def get_relay_messages_by_station(
        self,
        station_id: str,
        session_id: str,
        limit: int = 100
    ) -> List[RelayMessageRecord]:
        """获取中继站的消息"""
        with self.get_db_session() as session:
            models = session.query(RelayMessageModel).filter_by(
                station_id=station_id,
                session_id=session_id
            ).order_by(RelayMessageModel.timestamp).limit(limit).all()
            
            return [self._model_to_relay_message_record(m) for m in models]
    
    async def get_relay_messages_by_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[RelayMessageRecord]:
        """获取会话的所有中继消息"""
        with self.get_db_session() as session:
            models = session.query(RelayMessageModel).filter_by(
                session_id=session_id
            ).order_by(desc(RelayMessageModel.timestamp)).limit(limit).all()
            
            return [self._model_to_relay_message_record(m) for m in models]
    
    def _model_to_station_record(self, model: RelayStationModel) -> RelayStationRecord:
        return RelayStationRecord(
            station_id=model.station_id,
            session_id=model.session_id,
            name=model.name,
            phase=model.phase or 0,
            participating_agents=model.participating_agents or "[]",
            is_active=model.is_active if model.is_active is not None else True,
            summary=model.summary,
            created_at=model.created_at,
            closed_at=model.closed_at,
        )
    
    def _model_to_relay_message_record(self, model: RelayMessageModel) -> RelayMessageRecord:
        return RelayMessageRecord(
            message_id=model.message_id,
            station_id=model.station_id,
            session_id=model.session_id,
            relay_type=model.relay_type,
            source_agent_id=model.source_agent_id,
            source_agent_name=model.source_agent_name,
            target_agent_ids=model.target_agent_ids or "[]",
            content=model.content or "",
            importance=model.importance or 5,
            viewed_by=model.viewed_by or "[]",
            acknowledged_by=model.acknowledged_by or "[]",
            viewed_timestamps=model.viewed_timestamps or "{}",
            timestamp=model.timestamp,
            metadata_json=model.metadata_json,
        )
    
    # ========== Intervention Repository ==========
    
    async def create_intervention(self, record: InterventionRecord) -> InterventionRecord:
        """创建干预记录"""
        with self.get_db_session() as session:
            model = InterventionModel(
                intervention_id=record.intervention_id,
                session_id=record.session_id,
                intervention_type=record.intervention_type,
                scope=record.scope,
                target_agent_id=record.target_agent_id,
                target_agent_ids=record.target_agent_ids,
                payload_json=record.payload_json,
                reason=record.reason,
                priority=record.priority,
                timestamp=record.timestamp,
            )
            session.add(model)
            session.flush()
            
            return self._model_to_intervention_record(model)
    
    async def get_interventions_by_session(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[InterventionRecord]:
        """获取会话的干预记录"""
        with self.get_db_session() as session:
            models = session.query(InterventionModel).filter_by(
                session_id=session_id
            ).order_by(desc(InterventionModel.timestamp)).limit(limit).all()
            
            return [self._model_to_intervention_record(m) for m in models]
    
    def _model_to_intervention_record(self, model: InterventionModel) -> InterventionRecord:
        return InterventionRecord(
            intervention_id=model.intervention_id,
            session_id=model.session_id,
            intervention_type=model.intervention_type,
            scope=model.scope or "single",
            target_agent_id=model.target_agent_id,
            target_agent_ids=model.target_agent_ids or "[]",
            payload_json=model.payload_json,
            reason=model.reason or "",
            priority=model.priority or 5,
            timestamp=model.timestamp,
        )
    
    # ========== User Repository ==========
    
    async def create_user(self, record: UserRecord) -> UserRecord:
        """创建用户"""
        with self.get_db_session() as session:
            model = UserModel(
                user_id=record.user_id,
                username=record.username,
                password_hash=record.password_hash,
                display_name=record.display_name,
                created_at=record.created_at,
                updated_at=record.updated_at,
                metadata_json=record.metadata_json,
            )
            session.add(model)
            session.flush()
            return self._model_to_user_record(model)
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        """通过 user_id 获取用户"""
        with self.get_db_session() as session:
            model = session.query(UserModel).filter_by(user_id=user_id).first()
            if model:
                return self._model_to_user_record(model)
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        """通过 username 获取用户"""
        with self.get_db_session() as session:
            model = session.query(UserModel).filter_by(username=username).first()
            if model:
                return self._model_to_user_record(model)
            return None
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[UserRecord]:
        """更新用户信息"""
        with self.get_db_session() as session:
            model = session.query(UserModel).filter_by(user_id=user_id).first()
            if not model:
                return None
            
            for key, value in updates.items():
                if hasattr(model, key) and key not in ("user_id", "username", "password_hash"):
                    setattr(model, key, value)
            
            model.updated_at = datetime.now()
            session.flush()
            return self._model_to_user_record(model)
    
    def _model_to_user_record(self, model: UserModel) -> UserRecord:
        return UserRecord(
            user_id=model.user_id,
            username=model.username,
            password_hash=model.password_hash,
            display_name=model.display_name or "",
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata_json=model.metadata_json,
        )
