"""
FastAPI 路由定义

实现 AG-UI 协议的 HTTP 接口

重要改动：
- 移除全局单例 master_agent
- 使用 SessionManager 实现会话级别的数据隔离
- 每个请求必须携带 session_id，确保操作只影响对应会话
- 【新增】数据持久化接口 - 支持从数据库查询会话历史
"""

from fastapi import APIRouter, HTTPException, Request, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import json
from datetime import datetime

from core import MasterAgent, DirectAgent, InterventionType, InterventionScope, HumanIntervention
from core.session_manager import get_session_manager, SessionManager
from auth.dependencies import get_optional_user


router = APIRouter()


def get_agent_for_session(
    session_id: str,
    provider: str = "openai",
    model: Optional[str] = None
) -> MasterAgent:
    """
    获取指定会话的 MasterAgent 实例
    
    这是核心入口：确保每个 session_id 都有独立的 Agent 实例
    
    Args:
        session_id: 会话 ID
        provider: LLM 提供者
        model: 模型名称
        
    Returns:
        该会话专属的 MasterAgent 实例
    """
    session_manager = get_session_manager()
    return session_manager.get_or_create_agent(session_id, provider, model)


# ========== 请求/响应模型 ==========

class TaskRequest(BaseModel):
    """任务请求"""
    task: str
    context: Optional[str] = None
    provider: str = "openai"  # openai/claude
    model: Optional[str] = None
    session_id: Optional[str] = None  # 可选：指定会话 ID，如果不指定则自动创建
    mode: str = "emergent"  # emergent(涌现模式) / direct(普通模式)


class InterventionRequest(BaseModel):
    """人工干预请求 - 升级版"""
    session_id: str                                      # 必须：目标会话 ID
    agent_id: Optional[str] = None                       # 单个目标 Agent
    agent_ids: Optional[List[str]] = None                # 多个目标 Agent
    intervention_type: str                               # pause/resume/cancel/inject/adjust/broadcast
    payload: Optional[Dict[str, Any]] = None             # 干预负载
    reason: str = ""                                     # 干预原因
    priority: int = 5                                    # 优先级 1-10
    scope: str = "single"                                # 作用范围: single/selected/all/broadcast
    broadcast_to_relay: bool = True                      # 是否广播到中继站


class StatusResponse(BaseModel):
    """状态响应"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# ========== SSE 事件流接口 ==========

@router.post("/task/stream")
async def execute_task_stream(request: TaskRequest, user_id: Optional[str] = Depends(get_optional_user)):
    """
    执行任务 - SSE 事件流
    
    这是主接口，支持完整的 AG-UI 协议事件流
    
    重要：每个请求会创建或使用独立的会话实例
    
    【重要改进】改为同步持久化，确保数据写入后再发送 SSE：
    - Agent 状态 (AGENT_SPAWNED, AGENT_STATUS_CHANGED, AGENT_PROGRESS)
    - 中继消息 (RELAY_MESSAGE_SENT)
    - 任务计划 (PLAN_GENERATED)
    
    同时广播事件给所有订阅该会话的客户端（支持多浏览器同步）
    """
    from agui.events import (
        AgentSpawnedEvent,
        AgentStatusChangedEvent,
        AgentProgressEvent,
        RelayStationOpenedEvent,
        RelayMessageSentEvent,
        PlanGeneratedEvent,
        RunFinishedEvent,
        RunErrorEvent,
        TextMessageStartEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
    )
    
    session_manager = get_session_manager()
    
    # 追问相关上下文（默认 None，仅追问时构建）
    previous_context: Optional[str] = None
    previous_roles: Optional[list] = None
    is_followup = False
    
    # 创建或获取会话
    if request.session_id:
        session_id = request.session_id
        
        # 追问检测：session 已存在且已完成，且有历史数据
        session_info = session_manager.get_session_info(session_id)
        if session_info and session_info.has_history():
            is_followup = True
            print(f"[SSE] Followup detected for session {session_id[:8]}... "
                  f"(status={session_info.status}, has_report={bool(session_info.final_report)})")
            
            # 1. 构建追问上下文（裁剪后的上一轮摘要）
            previous_context = session_info.build_followup_context()
            
            # 2. 提取上一轮角色配置供角色引擎复用
            previous_roles = session_info.previous_roles
            
            # 3. 更新 SessionInfo 的 task 为新任务
            session_info.task = request.task
            session_info.status = "active"
            
            # 4. Cleanup 旧 Agent、创建新 MasterAgent（串行，避免竞态）
            if request.mode == "direct":
                # Direct 模式：复用已有 DirectAgent（保留 conversation_history 实现多轮对话）
                agent = session_manager.get_or_create_direct_agent(
                    session_id=session_id,
                    provider=request.provider,
                    model=request.model
                )
            else:
                agent = session_manager.prepare_followup(
                    session_id=session_id,
                    provider=request.provider,
                    model=request.model
                )
    else:
        session_id = session_manager.create_session_sync(
            provider=request.provider,
            model=request.model,
            task=request.task,
            user_id=user_id,
            mode=request.mode
        )
    
    # 非追问场景：正常获取或创建 Agent
    if not is_followup:
        if request.mode == "direct":
            agent = session_manager.get_or_create_direct_agent(
                session_id=session_id,
                provider=request.provider,
                model=request.model
            )
        else:
            agent = get_agent_for_session(
                session_id=session_id,
                provider=request.provider,
                model=request.model
            )
    
    async def persist_and_broadcast(event):
        """持久化事件数据到数据库，并广播给所有订阅者
        
        关键事件（状态变更、Agent创建、计划生成等）同步等待持久化完成；
        高频非关键事件（thinking、progress、text content）异步持久化不阻塞 SSE 流。
        """
        change_type = None
        summary = {}
        
        # 判断是否为关键事件（需要同步等待持久化完成的事件）
        is_critical = isinstance(event, (
            AgentSpawnedEvent,
            AgentStatusChangedEvent,
            PlanGeneratedEvent,
            RunFinishedEvent,
            RunErrorEvent,
            RelayStationOpenedEvent,
            RelayMessageSentEvent,
        ))
        
        async def _do_persist():
            """实际的持久化逻辑"""
            nonlocal change_type, summary
            try:
                if isinstance(event, AgentSpawnedEvent):
                    await session_manager.save_agent(
                        session_id=session_id,
                        agent_id=event.agent_id,
                        agent_data={
                            "name": event.agent_name,
                            "role_name": event.role_name,
                            "role_description": event.role_description,
                            "capabilities": event.capabilities,
                            "task_segment": event.task_segment,
                            "status": "running",
                            "progress": 0,
                            "current_step": "",
                            "iterations": 0,
                            "thinking": "",
                            "work_objective": event.work_objective,
                            "deliverables": event.deliverables,
                            "methodology": str(event.methodology) if event.methodology else None,
                            "assigned_skills": event.assigned_skills,
                            "expertise_level": event.expertise_level,
                            "focus_areas": event.focus_areas,
                        }
                    )
                    change_type = "agent_added"
                    summary = {
                        "agent_id": event.agent_id,
                        "agent_name": event.agent_name,
                        "role_name": event.role_name,
                    }
                    
                elif isinstance(event, AgentStatusChangedEvent):
                    await session_manager.save_agent(
                        session_id=session_id,
                        agent_id=event.agent_id,
                        agent_data={"status": event.new_status}
                    )
                    change_type = "agent_status_changed"
                    summary = {
                        "agent_id": event.agent_id,
                        "previous_status": event.previous_status,
                        "new_status": event.new_status,
                    }
                    
                elif isinstance(event, AgentProgressEvent):
                    await session_manager.save_agent(
                        session_id=session_id,
                        agent_id=event.agent_id,
                        agent_data={
                            "progress": event.progress,
                            "current_step": event.current_step,
                            "iterations": event.iterations,
                        }
                    )
                    
                elif isinstance(event, RelayStationOpenedEvent):
                    await session_manager.save_relay_station(
                        session_id=session_id,
                        station_data={
                            "station_id": event.station_id,
                            "name": event.station_name,
                            "phase": event.phase,
                            "participating_agents": [a["id"] for a in event.participating_agents],
                            "is_active": True,
                        }
                    )
                    
                elif isinstance(event, RelayMessageSentEvent):
                    await session_manager.save_relay_message(
                        session_id=session_id,
                        message_data={
                            "message_id": event.message_id,
                            "station_id": event.station_id,
                            "relay_type": event.relay_type,
                            "source_agent_id": event.source_agent_id,
                            "source_agent_name": event.source_agent_name,
                            "target_agent_ids": event.target_agent_ids,
                            "content": event.content,
                            "importance": event.importance,
                            "timestamp": event.timestamp,
                            "viewed_by": event.viewed_by,
                            "acknowledged_by": event.acknowledged_by,
                            "viewed_timestamps": event.viewed_timestamps,
                            "metadata": event.metadata,
                        }
                    )
                    
                elif isinstance(event, PlanGeneratedEvent):
                    plan_data = {
                        "id": event.plan_id,
                        "original_task": event.original_task,
                        "analysis": event.analysis,
                        "phases": [
                            {
                                "phase_number": p.get("phase_number"),
                                "name": p.get("name"),
                                "description": p.get("description"),
                                "participating_roles": p.get("participating_roles", []),
                            }
                            for p in event.phases
                        ],
                        "estimated_duration": event.estimated_duration,
                        "total_agents": event.total_agents,
                    }
                    await session_manager.update_session(
                        session_id=session_id,
                        updates={"plan_json": json.dumps(plan_data)}
                    )
                    change_type = "plan_generated"
                    summary = {"plan_id": event.plan_id, "total_agents": event.total_agents}
                    
                elif isinstance(event, RunFinishedEvent):
                    change_type = "completed"
                    summary = {"status": "completed"}
                    
                    # 从 Agent 中提取关键信息保存到 SessionInfo（追问支持）
                    try:
                        current_agent = session_manager.get_agent(session_id)
                        if current_agent and hasattr(current_agent, 'extract_session_summary'):
                            agent_summary = current_agent.extract_session_summary()
                            session_manager.save_task_completion(
                                session_id=session_id,
                                final_report=agent_summary.get("final_report", ""),
                                plan=agent_summary.get("plan"),
                                intervention_summary=agent_summary.get("intervention_summary"),
                                roles=agent_summary.get("roles"),
                            )
                    except Exception as e:
                        print(f"[SSE] Failed to extract session summary for followup: {e}")
                    
                    await session_manager.update_session(
                        session_id=session_id,
                        updates={"status": "completed"}
                    )
                    
                elif isinstance(event, RunErrorEvent):
                    change_type = "error"
                    summary = {"error": event.message}
                    await session_manager.update_session(
                        session_id=session_id,
                        updates={"status": "error", "error": event.message}
                    )
                
                elif isinstance(event, TextMessageStartEvent):
                    await session_manager.save_message(
                        session_id=session_id,
                        message_data={
                            "message_id": event.message_id,
                            "role": event.role,
                            "content": "",
                        }
                    )
                    
                elif isinstance(event, TextMessageContentEvent):
                    await session_manager.update_message(
                        session_id=session_id,
                        message_id=event.message_id,
                        content=event.delta
                    )
                    
            except Exception as e:
                print(f"[SSE] Failed to persist event {type(event).__name__}: {e}")
        
        if is_critical:
            # 关键事件：同步等待持久化完成后再继续
            await _do_persist()
        else:
            # 非关键事件：fire-and-forget 异步持久化，不阻塞 SSE 流
            asyncio.create_task(_do_persist())
        
        # 广播事件给所有订阅者（始终同步，确保前端收到）
        await session_manager.broadcast_event(session_id, event)
        
        # 广播状态变更通知（用于刷新会话列表）
        if change_type:
            await session_manager.broadcast_state_changed(session_id, change_type, summary)
    
    async def event_generator():
        """事件生成器"""
        try:
            # 首先发送 session_id 给前端
            session_event = {
                "type": "SESSION_CREATED",
                "session_id": session_id,
                "timestamp": ""
            }
            yield f"event: SESSION_CREATED\ndata: {json.dumps(session_event)}\n\n"
            
            # MasterAgent 支持追问上下文，DirectAgent 仅接受 task
            if request.mode == "direct":
                task_stream = agent.execute_task(request.task)
            else:
                task_stream = agent.execute_task(
                    request.task,
                    previous_context=previous_context,
                    previous_roles=previous_roles,
                )
            async for event in task_stream:
                # 【重要改动】同步等待持久化和广播完成
                await persist_and_broadcast(event)
                
                # 转换为 SSE 格式
                yield event.to_sse()
                await asyncio.sleep(0.01)  # 小延迟避免过快
        except Exception as e:
            # 发送错误事件
            error_event = {
                "type": "RUN_ERROR",
                "message": str(e),
                "session_id": session_id,
                "timestamp": ""
            }
            yield f"event: RUN_ERROR\ndata: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/task/{session_id}/stream")
async def get_task_stream(session_id: str):
    """
    获取任务事件流 - 用于重连
    """
    session_manager = get_session_manager()
    agent = session_manager.get_agent(session_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    state = agent.get_session_state(session_id)
    
    if not state:
        raise HTTPException(status_code=404, detail="Task session not found in agent")
    
    async def event_generator():
        """事件生成器 - 发送当前状态快照"""
        # 发送状态快照
        snapshot_event = {
            "type": "STATE_SNAPSHOT",
            "snapshot": state,
            "session_id": session_id,
            "timestamp": ""
        }
        yield f"event: STATE_SNAPSHOT\ndata: {json.dumps(snapshot_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ========== 状态查询接口 ==========

@router.get("/task/{session_id}/state")
async def get_task_state(session_id: str) -> StatusResponse:
    """获取任务状态"""
    session_manager = get_session_manager()
    agent = session_manager.get_agent(session_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # 尝试获取任务会话状态
    state = None
    for task_session_id in agent.sessions.keys():
        state = agent.get_session_state(task_session_id)
        if state:
            break
    
    if not state:
        # 返回 Agent 实例信息
        return StatusResponse(
            success=True,
            message="Session found, no active task",
            data=agent.get_instance_info()
        )
    
    return StatusResponse(
        success=True,
        message="State retrieved",
        data=state
    )


@router.get("/sessions")
async def list_sessions(
    status: Optional[str] = Query(None, description="Filter by status: active, completed, expired"),
    source: str = Query("memory", description="Data source: memory or db"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = Depends(get_optional_user)
) -> StatusResponse:
    """列出所有会话
    
    Args:
        status: 按状态筛选
        source: 数据源 - memory(内存缓存) 或 db(数据库)
        limit: 分页大小
        offset: 偏移量
    """
    session_manager = get_session_manager()
    
    if source == "db":
        # 从数据库查询
        sessions = await session_manager.list_sessions_from_db(
            status=status,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        total = await session_manager.count_sessions_from_db(status)
        stats = await session_manager.get_full_stats()
    else:
        # 从内存查询
        sessions = session_manager.list_sessions(user_id=user_id)
        if status:
            sessions = [s for s in sessions if s.get("status") == status]
        total = len(sessions)
        sessions = sessions[offset:offset + limit]
        stats = session_manager.get_stats()
    
    return StatusResponse(
        success=True,
        message=f"Found {len(sessions)} sessions (total: {total})",
        data={
            "sessions": sessions,
            "total": total,
            "limit": limit,
            "offset": offset,
            "stats": stats
        }
    )


@router.delete("/session/{session_id}")
async def close_session(session_id: str) -> StatusResponse:
    """关闭并清理会话"""
    session_manager = get_session_manager()
    
    if session_manager.close_session_sync(session_id):
        return StatusResponse(
            success=True,
            message=f"Session {session_id} closed and cleaned up"
        )
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@router.get("/session/{session_id}")
async def get_session_detail(session_id: str) -> StatusResponse:
    """获取单个会话的详细信息（从数据库）"""
    session_manager = get_session_manager()
    
    # 尝试从数据库获取
    session_data = await session_manager.get_session_info_from_db(session_id)
    
    if not session_data:
        # 回退到内存
        session_info = session_manager.get_session_info(session_id)
        if session_info:
            session_data = session_info.to_dict()
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # 添加是否有活跃 Agent 的标记
    session_data["has_agent"] = session_manager.get_agent(session_id) is not None
    
    return StatusResponse(
        success=True,
        message="Session found",
        data=session_data
    )


@router.get("/session/{session_id}/agents")
async def get_session_agents(session_id: str) -> StatusResponse:
    """获取会话的所有 Agent"""
    session_manager = get_session_manager()
    agents = await session_manager.get_session_agents(session_id)
    
    return StatusResponse(
        success=True,
        message=f"Found {len(agents)} agents",
        data={"agents": agents}
    )


@router.get("/session/{session_id}/relay-history")
async def get_session_relay_history(
    session_id: str,
    limit: int = Query(100, ge=1, le=500)
) -> StatusResponse:
    """获取会话的中继消息历史（从数据库）"""
    session_manager = get_session_manager()
    messages = await session_manager.get_session_relay_history(session_id, limit)
    
    return StatusResponse(
        success=True,
        message=f"Found {len(messages)} relay messages",
        data={"messages": messages}
    )


@router.get("/session/{session_id}/interventions")
async def get_session_interventions(
    session_id: str,
    limit: int = Query(50, ge=1, le=200)
) -> StatusResponse:
    """获取会话的干预历史（从数据库）"""
    session_manager = get_session_manager()
    interventions = await session_manager.get_session_interventions(session_id, limit)
    
    return StatusResponse(
        success=True,
        message=f"Found {len(interventions)} interventions",
        data={"interventions": interventions}
    )


# ========== 人工干预接口 (升级版 - 通过中继站广播) ==========

@router.post("/intervention")
async def apply_intervention(request: InterventionRequest) -> StatusResponse:
    """
    应用人工干预 - 升级版
    
    重要：干预操作只会影响指定 session_id 对应的会话
    不会影响其他会话的 Agent
    
    支持通过中继站广播干预消息，让所有相关 Agent 都能感知
    返回生成的中继事件供前端更新状态
    """
    session_manager = get_session_manager()
    agent = session_manager.get_agent(request.session_id)
    
    if not agent:
        raise HTTPException(
            status_code=404, 
            detail=f"Session {request.session_id} not found. Cannot apply intervention."
        )
    
    try:
        # 解析作用范围
        scope_map = {
            "single": InterventionScope.SINGLE,
            "selected": InterventionScope.SELECTED,
            "all": InterventionScope.ALL,
            "broadcast": InterventionScope.BROADCAST,
        }
        scope = scope_map.get(request.scope, InterventionScope.SINGLE)
        
        # 解析干预类型
        type_map = {
            "pause": InterventionType.PAUSE,
            "resume": InterventionType.RESUME,
            "cancel": InterventionType.CANCEL,
            "inject": InterventionType.INJECT,
            "adjust": InterventionType.ADJUST,
            "restart": InterventionType.RESTART,
        }
        
        if request.intervention_type not in type_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown intervention type: {request.intervention_type}. Valid types: {list(type_map.keys())}"
            )
        
        intervention_type = type_map[request.intervention_type]
        
        # 处理不同类型的干预
        if request.intervention_type == "pause":
            if request.agent_id:
                success = await agent.pause_agent(
                    request.agent_id, 
                    reason=request.reason, 
                    broadcast=request.broadcast_to_relay
                )
            else:
                # 暂停所有
                success = True
                for aid in list(agent.active_subagents.keys()):
                    result = await agent.pause_agent(aid, reason=request.reason, broadcast=False)
                    success = success and result
                # 发送一个总的广播
                if request.broadcast_to_relay:
                    await agent.broadcast_to_all_agents(
                        f"所有 Agent 已被暂停。原因: {request.reason or '用户请求'}",
                        reason="批量暂停",
                        priority=request.priority
                    )
            message = "Agent(s) paused and broadcast to relay station" if success else "No agents to pause"
        
        elif request.intervention_type == "resume":
            if request.agent_id:
                success = await agent.resume_agent(
                    request.agent_id,
                    reason=request.reason,
                    broadcast=request.broadcast_to_relay
                )
            else:
                success = True
                for aid in list(agent.active_subagents.keys()):
                    result = await agent.resume_agent(aid, reason=request.reason, broadcast=False)
                    success = success and result
                if request.broadcast_to_relay:
                    await agent.broadcast_to_all_agents(
                        f"所有 Agent 已恢复工作。原因: {request.reason or '用户请求'}",
                        reason="批量恢复",
                        priority=request.priority
                    )
            message = "Agent(s) resumed and broadcast to relay station" if success else "No agents to resume"
        
        elif request.intervention_type == "cancel":
            if request.agent_id:
                success = await agent.cancel_agent(
                    request.agent_id,
                    reason=request.reason,
                    broadcast=request.broadcast_to_relay
                )
            else:
                success = True
                for aid in list(agent.active_subagents.keys()):
                    result = await agent.cancel_agent(aid, reason=request.reason, broadcast=False)
                    success = success and result
                if request.broadcast_to_relay:
                    await agent.broadcast_to_all_agents(
                        f"所有任务已取消。原因: {request.reason or '用户请求'}",
                        reason="批量取消",
                        priority=8
                    )
            message = "Agent(s) cancelled and broadcast to relay station" if success else "No agents to cancel"
        
        elif request.intervention_type == "inject":
            if not request.payload or "information" not in request.payload:
                raise HTTPException(
                    status_code=400,
                    detail="payload.information required for inject"
                )
            
            information = request.payload["information"]
            
            # 用户的 inject 消息即时摄入记忆系统（用户可能在对话中透露身份、偏好等信息）
            if hasattr(agent, 'user_id') and agent.user_id:
                try:
                    from memory.service import get_memory_service
                    memory_service = get_memory_service()
                    if memory_service.is_enabled:
                        import asyncio
                        asyncio.create_task(memory_service.memorize(
                            user_id=agent.user_id,
                            content=f"用户说: {information}",
                            modality="conversation",
                        ))
                except Exception:
                    pass  # 记忆摄入失败不影响主流程
            
            if request.agent_id:
                # 注入到单个 Agent
                success = await agent.inject_to_agent(
                    request.agent_id,
                    information,
                    broadcast=request.broadcast_to_relay,
                    priority=request.priority
                )
                message = "Information injected to agent and broadcast to relay station" if success else "Failed to inject"
            elif request.agent_ids:
                # 注入到选定的多个 Agent
                success = True
                for aid in request.agent_ids:
                    result = await agent.inject_to_agent(aid, information, broadcast=False, priority=request.priority)
                    success = success and result
                if request.broadcast_to_relay:
                    intervention = HumanIntervention(
                        type=InterventionType.INJECT,
                        target_agent_ids=request.agent_ids,
                        scope=InterventionScope.SELECTED,
                        payload={"information": information},
                        reason=request.reason or f"向 {len(request.agent_ids)} 个 Agent 注入信息",
                        priority=request.priority,
                    )
                    await agent.relay_coordinator.broadcast_intervention(intervention)
                message = f"Information injected to {len(request.agent_ids)} agents" if success else "Failed to inject"
            else:
                # 广播给所有 Agent
                success = await agent.broadcast_to_all_agents(
                    information,
                    reason=request.reason,
                    priority=request.priority,
                    force_action=(scope == InterventionScope.ALL)
                )
                message = "Information broadcast to all agents via relay station" if success else "Failed to broadcast"
        
        elif request.intervention_type == "adjust":
            if not request.payload or "adjustments" not in request.payload:
                raise HTTPException(
                    status_code=400,
                    detail="payload.adjustments required for adjust"
                )
            if not request.agent_id:
                raise HTTPException(
                    status_code=400,
                    detail="agent_id required for adjust"
                )
            
            success = await agent.adjust_agent(
                request.agent_id,
                request.payload["adjustments"],
                reason=request.reason,
                broadcast=request.broadcast_to_relay
            )
            message = "Agent adjusted and broadcast to relay station" if success else "Failed to adjust agent"
        
        elif request.intervention_type == "broadcast":
            # 纯广播模式 - 只广播消息，不强制执行
            if not request.payload or "message" not in request.payload:
                raise HTTPException(
                    status_code=400,
                    detail="payload.message required for broadcast"
                )
            
            success = await agent.broadcast_to_all_agents(
                request.payload["message"],
                reason=request.reason,
                priority=request.priority,
                force_action=False
            )
            message = "Message broadcast to relay station" if success else "Failed to broadcast"
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown intervention type: {request.intervention_type}"
            )
        
        # 获取生成的中继事件
        relay_events = agent.get_pending_relay_events()
        relay_messages = [
            {
                "station_id": e.station_id,
                "message_id": e.message_id,
                "source_agent_id": e.source_agent_id,
                "source_agent_name": e.source_agent_name,
                "target_agent_ids": e.target_agent_ids,
                "relay_type": e.relay_type,
                "content": e.content,
                "importance": e.importance,
                "metadata": e.metadata,
                "viewed_by": e.viewed_by,
                "acknowledged_by": e.acknowledged_by,
                "viewed_timestamps": e.viewed_timestamps,
                "timestamp": e.timestamp,
            }
            for e in relay_events
        ]
        
        return StatusResponse(
            success=success,
            message=message,
            data={
                "intervention_type": request.intervention_type,
                "scope": request.scope,
                "broadcast_to_relay": request.broadcast_to_relay,
                "target_agent_id": request.agent_id,
                "target_agent_ids": request.agent_ids,
                "relay_messages": relay_messages,  # 新增：返回生成的中继消息
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return StatusResponse(
            success=False,
            message=str(e)
        )


@router.post("/intervention/broadcast")
async def broadcast_to_relay(
    session_id: str,
    message: str,
    reason: str = "",
    priority: int = 5,
    force_action: bool = False
) -> StatusResponse:
    """
    向中继站广播消息 - 简化接口
    
    这是一个快捷接口，用于快速向指定会话的所有 Agent 广播消息
    """
    session_manager = get_session_manager()
    agent = session_manager.get_agent(session_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    try:
        success = await agent.broadcast_to_all_agents(
            message,
            reason=reason,
            priority=priority,
            force_action=force_action
        )
        
        return StatusResponse(
            success=success,
            message="Broadcast sent to relay station" if success else "No active agents to broadcast to",
            data={"session_id": session_id}
        )
    except Exception as e:
        return StatusResponse(
            success=False,
            message=str(e)
        )


@router.get("/relay/{session_id}/history")
async def get_relay_history(session_id: str, limit: int = 20) -> StatusResponse:
    """获取指定会话的中继站消息历史"""
    session_manager = get_session_manager()
    agent = session_manager.get_agent(session_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    messages = [
        {
            "id": msg.id,
            "type": msg.type.value,
            "source_agent_name": msg.source_agent_name,
            "content": msg.content,  # 返回完整内容，由前端决定如何截断显示
            "importance": msg.importance,
            "timestamp": msg.timestamp.isoformat(),
            "is_intervention": msg.type.value == "human_intervention",
            "viewed_by": msg.viewed_by,
            "acknowledged_by": msg.acknowledged_by,
            "viewed_timestamps": msg.viewed_timestamps,
        }
        for msg in agent.relay_coordinator.message_history[-limit:]
    ]
    
    return StatusResponse(
        success=True,
        message=f"Found {len(messages)} messages",
        data={"messages": messages}
    )


@router.get("/relay/{session_id}/message/{message_id}")
async def get_relay_message_detail(session_id: str, message_id: str) -> StatusResponse:
    """获取指定会话中的单条中继消息详情"""
    session_manager = get_session_manager()
    agent = session_manager.get_agent(session_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # 查找消息
    for msg in agent.relay_coordinator.message_history:
        if msg.id == message_id:
            return StatusResponse(
                success=True,
                message="Message found",
                data={
                    "id": msg.id,
                    "type": msg.type.value,
                    "source_agent_id": msg.source_agent_id,
                    "source_agent_name": msg.source_agent_name,
                    "target_agent_ids": msg.target_agent_ids,
                    "content": msg.content,
                    "importance": msg.importance,
                    "metadata": msg.metadata,
                    "timestamp": msg.timestamp.isoformat(),
                    "viewed_by": msg.viewed_by,
                    "acknowledged_by": msg.acknowledged_by,
                    "viewed_timestamps": msg.viewed_timestamps,
                    "session_id": session_id,
                }
            )
    
    raise HTTPException(status_code=404, detail="Message not found")


@router.get("/relay/{session_id}/interventions")
async def get_intervention_history(session_id: str, limit: int = 10) -> StatusResponse:
    """获取指定会话的人工干预历史"""
    session_manager = get_session_manager()
    agent = session_manager.get_agent(session_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    interventions = [
        {
            "id": i.id,
            "type": i.type.value,
            "scope": i.scope.value,
            "target_agent_id": i.target_agent_id,
            "reason": i.reason,
            "priority": i.priority,
            "timestamp": i.timestamp.isoformat(),
            "session_id": session_id,
        }
        for i in agent.relay_coordinator.get_intervention_history(limit)
    ]
    
    return StatusResponse(
        success=True,
        message=f"Found {len(interventions)} interventions",
        data={"interventions": interventions}
    )


# ========== 健康检查 ==========

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    session_manager = get_session_manager()
    stats = await session_manager.get_full_stats()
    return {
        "status": "healthy",
        "service": "Agent Swarm",
        "version": "1.1.0",
        "session_manager_stats": stats
    }


@router.get("/stats")
async def get_detailed_stats() -> StatusResponse:
    """获取详细统计信息"""
    session_manager = get_session_manager()
    stats = await session_manager.get_full_stats()
    
    return StatusResponse(
        success=True,
        message="Stats retrieved",
        data=stats
    )


@router.get("/session/{session_id}/live-state")
async def get_session_live_state(session_id: str) -> StatusResponse:
    """
    获取会话的完整状态
    
    优先级：
    1. 如果会话在内存中活跃 → 返回实时状态
    2. 如果会话不在内存中 → 从数据库加载历史数据
    
    返回数据包括：
    - 所有 Agent 的状态
    - 所有中继站及消息
    - 任务执行状态和计划
    
    这个 API 用于切换到任意会话时，恢复完整的前端状态
    """
    session_manager = get_session_manager()
    agent = session_manager.get_agent(session_id)
    
    if agent:
        # ===== 会话在内存中活跃，返回实时状态 =====
        
        # 收集所有 Agent 的状态
        agents_data = []
        for aid, subagent in agent.active_subagents.items():
            # SubagentRuntime 的数据存储在 config 和 state 中
            # - config.role: 角色信息 (name, description, capabilities 等)
            # - config.task_segment: 任务片段
            # - state: 运行状态 (status, progress, iterations, thinking 等)
            role = subagent.config.role
            state = subagent.state
            
            agent_info = {
                "agent_id": aid,
                "name": role.name,
                "role_name": role.name,
                "role_description": role.description,
                "capabilities": role.capabilities,
                "task_segment": subagent.config.task_segment,
                "status": state.status.value if hasattr(state.status, 'value') else str(state.status),
                "progress": state.progress,
                "current_step": state.current_step,
                "iterations": state.iterations,
                "thinking": state.thinking[-1000:] if state.thinking else "",  # 最近 1000 字符
                "work_objective": role.work_objective,
                "deliverables": role.deliverables,
                "methodology": str(role.methodology) if role.methodology else None,
            }
            agents_data.append(agent_info)
        
        # 收集中继站状态
        relay_stations_data = []
        if hasattr(agent, 'relay_coordinator'):
            # 获取所有中继站 (属性名是 stations 不是 relay_stations)
            for station_id, station in agent.relay_coordinator.stations.items():
                station_data = {
                    "station_id": station_id,
                    "name": station.name,
                    "phase": station.phase,
                    "participating_agents": list(station.participating_agents),
                    "is_active": station.is_active,
                    "messages": []
                }
                
                # 获取该中继站的消息
                for msg in agent.relay_coordinator.message_history:
                    if hasattr(msg, 'station_id') and msg.station_id == station_id:
                        msg_data = {
                            "message_id": msg.id,
                            "station_id": msg.station_id if hasattr(msg, 'station_id') else station_id,
                            "source_agent_id": msg.source_agent_id,
                            "source_agent_name": msg.source_agent_name,
                            "target_agent_ids": msg.target_agent_ids,
                            "relay_type": msg.type.value if hasattr(msg.type, 'value') else str(msg.type),
                            "content": msg.content,
                            "importance": msg.importance,
                            "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp),
                            "viewed_by": msg.viewed_by,
                            "acknowledged_by": msg.acknowledged_by,
                        }
                        station_data["messages"].append(msg_data)
                
                relay_stations_data.append(station_data)
        
        # 获取会话基本信息
        session_info = session_manager.get_session_info(session_id)
        
        # 从数据库获取已持久化的消息
        messages_data = await session_manager.get_session_messages(session_id, limit=100)
        
        return StatusResponse(
            success=True,
            message="Live state retrieved from memory",
            data={
                "is_live": True,
                "session_id": session_id,
                "task": session_info.task if session_info else None,
                "status": session_info.status if session_info else "active",
                "plan": agent.current_plan if hasattr(agent, 'current_plan') and agent.current_plan else None,
                "agents": agents_data,
                "relay_stations": relay_stations_data,
                "messages": messages_data,  # 新增：返回消息历史
                "total_messages": len(agent.relay_coordinator.message_history) if hasattr(agent, 'relay_coordinator') else 0,
            }
        )
    
    # ===== 会话不在内存中，从数据库加载历史数据 =====
    
    # 1. 获取 Session 基本信息
    session_data = await session_manager.get_session_info_from_db(session_id)
    if not session_data:
        return StatusResponse(
            success=False,
            message=f"Session {session_id} not found",
            data={
                "is_live": False,
                "session_id": session_id,
                "agents": [],
                "relay_stations": [],
                "total_messages": 0,
            }
        )
    
    # 2. 从数据库加载 Agents
    agents_from_db = await session_manager.get_session_agents(session_id)
    agents_data = [
        {
            "agent_id": a.get("agent_id"),
            "name": a.get("name"),
            "role_name": a.get("role_name"),
            "role_description": a.get("role_description"),
            "capabilities": a.get("capabilities", []),
            "task_segment": a.get("task_segment"),
            "status": a.get("status", "completed"),  # 历史数据默认已完成
            "progress": a.get("progress", 100),
            "current_step": a.get("current_step", ""),
            "iterations": a.get("iterations", 0),
            "thinking": a.get("thinking", ""),
            "work_objective": a.get("work_objective"),
            "deliverables": a.get("deliverables", []),
            "methodology": a.get("methodology"),
        }
        for a in agents_from_db
    ]
    
    # 3. 从数据库加载中继消息
    relay_messages = await session_manager.get_session_relay_history(session_id, limit=500)
    
    # 按 station_id 分组消息
    stations_map = {}
    for msg in relay_messages:
        station_id = msg.get("station_id", "default-station")
        if station_id not in stations_map:
            stations_map[station_id] = {
                "station_id": station_id,
                "name": f"中继站 {station_id[:8]}",
                "phase": 0,
                "participating_agents": [],
                "is_active": False,  # 历史数据，已关闭
                "messages": []
            }
        
        stations_map[station_id]["messages"].append({
            "message_id": msg.get("message_id"),
            "station_id": station_id,
            "source_agent_id": msg.get("source_agent_id"),
            "source_agent_name": msg.get("source_agent_name"),
            "target_agent_ids": msg.get("target_agent_ids", []),
            "relay_type": msg.get("relay_type"),
            "content": msg.get("content"),
            "importance": msg.get("importance", 5),
            "timestamp": msg.get("timestamp"),
            "viewed_by": msg.get("viewed_by", []),
            "acknowledged_by": msg.get("acknowledged_by", []),
        })
    
    relay_stations_data = list(stations_map.values())
    
    # 4. 解析计划（如果有）
    plan = None
    if session_data.get("plan"):
        plan = session_data["plan"]
    
    # 5. 从数据库加载消息历史
    messages_data = await session_manager.get_session_messages(session_id, limit=100)
    
    return StatusResponse(
        success=True,
        message="Historical state retrieved from database",
        data={
            "is_live": False,  # 标记为历史数据
            "session_id": session_id,
            "task": session_data.get("task"),
            "status": session_data.get("status", "completed"),
            "plan": plan,
            "agents": agents_data,
            "relay_stations": relay_stations_data,
            "messages": messages_data,  # 新增：返回消息历史
            "total_messages": len(relay_messages),
        }
    )


# ========== 多客户端订阅接口 (新增) ==========

@router.get("/session/{session_id}/subscribe")
async def subscribe_to_session(session_id: str):
    """
    订阅会话事件流 - 支持多客户端
    
    这个端点允许多个浏览器/标签页同时订阅同一个会话的实时事件流。
    
    流程：
    1. 首先发送当前状态快照（STATE_SNAPSHOT）
    2. 然后持续推送后续的实时事件
    3. 客户端断开时自动取消订阅
    
    使用场景：
    - 在新浏览器中打开已有会话
    - 多设备同步查看任务进度
    - 页面刷新后重新连接
    """
    session_manager = get_session_manager()
    
    # 检查会话是否存在（内存或数据库）
    agent = session_manager.get_agent(session_id)
    session_info = session_manager.get_session_info(session_id)
    
    if not agent and not session_info:
        # 尝试从数据库获取
        session_data = await session_manager.get_session_info_from_db(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # 订阅事件流
    event_queue = await session_manager.subscribe(session_id)
    
    async def event_generator():
        """生成 SSE 事件流"""
        try:
            # 1. 首先发送状态快照
            snapshot = await _build_session_snapshot(session_manager, session_id)
            snapshot_event = {
                "type": "STATE_SNAPSHOT",
                "session_id": session_id,
                "snapshot": snapshot,
                "timestamp": datetime.now().isoformat()
            }
            yield f"event: STATE_SNAPSHOT\ndata: {json.dumps(snapshot_event)}\n\n"
            
            # 2. 持续监听并推送后续事件
            while True:
                try:
                    # 等待新事件，超时 30 秒发送心跳
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    
                    # 发送事件
                    if hasattr(event, 'to_sse'):
                        yield event.to_sse()
                    else:
                        # 普通字典事件
                        event_type = event.get('type', 'UNKNOWN')
                        yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
                        
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    heartbeat = {
                        "type": "HEARTBEAT",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"event: HEARTBEAT\ndata: {json.dumps(heartbeat)}\n\n"
                    
        except asyncio.CancelledError:
            # 客户端断开连接
            pass
        finally:
            # 取消订阅
            await session_manager.unsubscribe(session_id, event_queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def _build_session_snapshot(session_manager: SessionManager, session_id: str) -> Dict[str, Any]:
    """构建会话状态快照
    
    优先从内存获取实时数据，否则从数据库加载历史数据
    """
    agent = session_manager.get_agent(session_id)
    
    if agent:
        # 从内存获取实时状态
        agents_data = []
        for aid, subagent in agent.active_subagents.items():
            role = subagent.config.role
            state = subagent.state
            agents_data.append({
                "agent_id": aid,
                "name": role.name,
                "role_name": role.name,
                "role_description": role.description,
                "capabilities": role.capabilities,
                "task_segment": subagent.config.task_segment,
                "status": state.status.value if hasattr(state.status, 'value') else str(state.status),
                "progress": state.progress,
                "current_step": state.current_step,
                "iterations": state.iterations,
                "thinking": state.thinking[-500:] if state.thinking else "",
            })
        
        relay_stations_data = []
        if hasattr(agent, 'relay_coordinator'):
            for station_id, station in agent.relay_coordinator.stations.items():
                messages = []
                for msg in agent.relay_coordinator.message_history:
                    if hasattr(msg, 'station_id') and msg.station_id == station_id:
                        messages.append({
                            "message_id": msg.id,
                            "source_agent_name": msg.source_agent_name,
                            "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                            "relay_type": msg.type.value if hasattr(msg.type, 'value') else str(msg.type),
                            "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp),
                        })
                
                relay_stations_data.append({
                    "station_id": station_id,
                    "name": station.name,
                    "phase": station.phase,
                    "is_active": station.is_active,
                    "messages": messages[-20:],  # 最近 20 条
                })
        
        session_info = session_manager.get_session_info(session_id)
        
        # 从数据库获取已持久化的消息
        messages_data = await session_manager.get_session_messages(session_id, limit=100)
        
        return {
            "is_live": True,
            "task": session_info.task if session_info else None,
            "status": session_info.status if session_info else "active",
            "plan": agent.current_plan if hasattr(agent, 'current_plan') else None,
            "agents": agents_data,
            "relay_stations": relay_stations_data,
            "messages": messages_data,  # 新增：消息历史
            "subscriber_count": session_manager.get_subscriber_count(session_id),
        }
    
    # 从数据库加载历史数据
    session_data = await session_manager.get_session_info_from_db(session_id)
    agents_from_db = await session_manager.get_session_agents(session_id)
    relay_messages = await session_manager.get_session_relay_history(session_id, limit=100)
    
    # 【新增】从数据库加载消息历史
    messages_data = await session_manager.get_session_messages(session_id, limit=100)
    
    # 按 station_id 分组消息
    stations_map = {}
    for msg in relay_messages:
        station_id = msg.get("station_id", "default")
        if station_id not in stations_map:
            stations_map[station_id] = {
                "station_id": station_id,
                "name": f"中继站 {station_id[:8]}",
                "is_active": False,
                "messages": []
            }
        stations_map[station_id]["messages"].append({
            "message_id": msg.get("message_id"),
            "source_agent_name": msg.get("source_agent_name"),
            "content": msg.get("content", "")[:200],
            "relay_type": msg.get("relay_type"),
            "timestamp": msg.get("timestamp"),
        })
    
    return {
        "is_live": False,
        "task": session_data.get("task") if session_data else None,
        "status": session_data.get("status", "completed") if session_data else "unknown",
        "plan": session_data.get("plan") if session_data else None,
        "agents": [
            {
                "agent_id": a.get("agent_id"),
                "name": a.get("name"),
                "role_name": a.get("role_name"),
                "status": a.get("status", "completed"),
                "progress": a.get("progress", 100),
            }
            for a in agents_from_db
        ],
        "relay_stations": list(stations_map.values()),
        "messages": messages_data,  # 【新增】历史会话也返回消息
        "subscriber_count": session_manager.get_subscriber_count(session_id),
    }


@router.get("/session/{session_id}/subscribers")
async def get_session_subscribers(session_id: str) -> StatusResponse:
    """获取会话的当前订阅者数量"""
    session_manager = get_session_manager()
    count = session_manager.get_subscriber_count(session_id)
    
    return StatusResponse(
        success=True,
        message=f"Session has {count} active subscribers",
        data={
            "session_id": session_id,
            "subscriber_count": count
        }
    )


@router.get("/subscribers/stats")
async def get_all_subscriber_stats() -> StatusResponse:
    """获取所有会话的订阅者统计"""
    session_manager = get_session_manager()
    stats = session_manager.get_all_subscriber_stats()
    
    return StatusResponse(
        success=True,
        message=f"Found {len(stats)} sessions with subscribers",
        data={
            "sessions": stats,
            "total_subscribers": sum(stats.values())
        }
    )
