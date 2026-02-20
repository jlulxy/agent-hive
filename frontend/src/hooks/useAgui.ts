/**
 * AG-UI 协议 Hook
 * 
 * 处理 SSE 事件流，管理应用状态
 * 
 * 重构：支持多会话（Session）数据隔离
 * 
 * 核心改动：
 * 1. 每次执行任务时，后端会返回新的 session_id
 * 2. 所有操作都绑定到具体的 session_id
 * 3. 干预操作必须指定 session_id，只影响对应会话
 * 4. 新增 subscribeToSession：支持订阅已有会话的实时事件流（跨浏览器共享）
 */

import { useCallback, useRef, useState } from 'react';
import { useStore } from '../store';
import { getAuthHeader, getStoredToken } from '../auth/api';
import {
  EventType,
  AguiEvent,
  AgentStatus,
  Agent,
  RelayMessage,
  StateSnapshotEvent,
  SessionStateChangedEvent,
} from '../types/agui';

export function useAgui() {
  const eventSourceRef = useRef<EventSource | null>(null);
  const subscriptionEventSourceRef = useRef<EventSource | null>(null);  // 新增：订阅连接
  const [isSubscribing, setIsSubscribing] = useState(false);  // 新增：订阅状态
  
  const {
    createSession,
    setActiveSession,
    sessionId: currentSessionId,
    setSessionId,
    setTask,
    setStatus,
    setMode: setStoreMode,
    setPlan,
    addAgent,
    updateAgent,
    updateAgentThinking,
    addAgentToolCall,
    updateAgentToolCall,
    addRelayStation,
    addRelayMessage,
    closeRelayStation,
    addMessage,
    appendMessageContent,
    addStreamToolCall,
    updateStreamToolCall,
    appendStreamThinking,
    setFinalReport,
    setError,
  } = useStore();
  
  // 记录当前任务的 mode，用于在 SESSION_CREATED 后设置
  const currentModeRef = useRef<'emergent' | 'direct'>('emergent');

  /**
   * 解析 SSE 事件
   */
  const parseEvent = useCallback((eventType: string, data: string): AguiEvent | null => {
    try {
      const parsed = JSON.parse(data);
      return { ...parsed, type: eventType } as AguiEvent;
    } catch (e) {
      console.error('Failed to parse event:', e);
      return null;
    }
  }, []);

  /**
   * 处理事件
   * 
   * 重要改动：添加 targetSessionId 参数，用于会话隔离
   * - 如果事件有 session_id，只更新对应的会话
   * - 如果当前活跃会话与事件的 session_id 不匹配，跳过更新
   * - 对于 SESSION_CREATED 事件，会自动切换到新会话
   */
  const handleEvent = useCallback((event: AguiEvent, targetSessionId?: string) => {
    // 获取当前活跃的 session_id
    const { activeSessionId } = useStore.getState();
    
    // 从事件中提取 session_id（某些事件可能有这个字段）
    const eventSessionId = (event as any).session_id || targetSessionId;
    
    // 【核心逻辑】会话隔离校验
    // 如果事件明确属于某个会话，且该会话不是当前活跃会话，则跳过
    // 除了 SESSION_CREATED 事件（这个事件会切换会话）
    if (eventSessionId && 
        activeSessionId && 
        eventSessionId !== activeSessionId && 
        event.type !== EventType.SESSION_CREATED) {
      console.log(`[useAgui] Skipping event for session ${eventSessionId}, active session is ${activeSessionId}`);
      return;
    }
    
    switch (event.type) {
      // 新增：处理会话创建事件
      case EventType.SESSION_CREATED:
        const newSessionId = (event as any).session_id;
        console.log(`[useAgui] Session created: ${newSessionId}, mode: ${currentModeRef.current}`);
        createSession(newSessionId);
        setSessionId(newSessionId);
        // 立即设置 mode，避免 createSession 默认 'emergent' 覆盖
        setStoreMode(currentModeRef.current);
        break;

      case EventType.RUN_STARTED:
        // 如果有 thread_id 且与当前 session 不同，可能需要同步
        if (event.thread_id && event.thread_id !== currentSessionId) {
          console.log(`[useAgui] RUN_STARTED with different thread_id: ${event.thread_id}`);
        }
        setStatus(AgentStatus.RUNNING);
        break;

      case EventType.RUN_FINISHED:
        setStatus(AgentStatus.COMPLETED);
        break;

      case EventType.RUN_ERROR:
        setStatus(AgentStatus.FAILED);
        setError(event.message);
        break;

      case EventType.TEXT_MESSAGE_START:
        addMessage({
          id: event.message_id,
          role: event.role as 'assistant',
          content: '',
          timestamp: event.timestamp,
        });
        break;

      case EventType.TEXT_MESSAGE_CONTENT:
        appendMessageContent(event.message_id, event.delta);
        break;

      case EventType.TEXT_MESSAGE_END:
        // 消息结束，可以触发一些后处理
        break;

      case EventType.AGENT_SPAWNED:
        addAgent({
          id: event.agent_id,
          name: event.agent_name,
          roleName: event.role_name,
          roleDescription: event.role_description,
          capabilities: event.capabilities,
          taskSegment: event.task_segment,
          status: AgentStatus.PENDING,
          progress: 0,
          currentStep: '',
          iterations: 0,
          thinking: '',
          // 新增字段
          workObjective: event.work_objective,
          deliverables: event.deliverables,
          methodology: event.methodology,
          assignedSkills: event.assigned_skills,
          expertiseLevel: event.expertise_level,
          focusAreas: event.focus_areas,
        });
        break;

      case EventType.AGENT_STATUS_CHANGED:
        updateAgent(event.agent_id, {
          status: event.new_status as AgentStatus,
        });
        break;

      case EventType.AGENT_PROGRESS:
        updateAgent(event.agent_id, {
          progress: event.progress,
          currentStep: event.current_step,
          iterations: event.iterations,
        });
        break;

      // AGENT_THINKING 在下方统一处理（区分涌现/普通模式）

      case EventType.RELAY_STATION_OPENED:
        addRelayStation({
          id: event.station_id,
          name: event.station_name,
          phase: event.phase,
          participatingAgents: event.participating_agents,
          messages: [],
          isActive: true,
        });
        break;

      case EventType.RELAY_MESSAGE_SENT:
        addRelayMessage(event.station_id, {
          id: event.message_id,
          stationId: event.station_id,
          sourceAgentId: event.source_agent_id,
          sourceAgentName: event.source_agent_name,
          targetAgentIds: event.target_agent_ids,
          relayType: event.relay_type,
          content: event.content,
          importance: event.importance,
          timestamp: event.timestamp,
          metadata: (event as any).metadata || {},
          viewedBy: (event as any).viewed_by || [],
          acknowledgedBy: (event as any).acknowledged_by || [],
          viewedTimestamps: (event as any).viewed_timestamps || {},
        });
        break;

      case EventType.RELAY_STATION_CLOSED:
        closeRelayStation(event.station_id, event.summary);
        break;

      case EventType.PLAN_GENERATED:
        setPlan({
          id: event.plan_id,
          originalTask: event.original_task,
          analysis: event.analysis,
          phases: event.phases.map(p => ({
            phaseNumber: p.phase_number,
            name: p.name,
            description: p.description,
            participatingRoles: p.participating_roles,
          })),
          estimatedDuration: event.estimated_duration,
          totalAgents: event.total_agents,
        });
        break;

      case EventType.ROLE_EMERGED:
        // 角色涌现事件，可以在 UI 上显示
        console.log('Role emerged:', event.role_name);
        break;

      case EventType.TOOL_CALL_START: {
        const tcStart = event as any;
        const agentIdForTool = tcStart.parent_message_id || '';
        
        // 检查是否是已知 agent 发起的 tool call（涌现模式）
        const { agents: currentAgents } = useStore.getState().sessions[useStore.getState().activeSessionId || ''] || { agents: {} };
        const isAgentToolCall = agentIdForTool && currentAgents[agentIdForTool];
        
        if (isAgentToolCall) {
          // 涌现模式：只添加到 Agent 级别
          addAgentToolCall(agentIdForTool, {
            id: tcStart.tool_call_id,
            toolName: tcStart.tool_call_name,
            skillName: tcStart.tool_call_name,
            status: 'running',
            startedAt: event.timestamp,
          });
        } else {
          // 普通模式：添加到消息流级别
          addStreamToolCall({
            id: tcStart.tool_call_id,
            toolName: tcStart.tool_call_name,
            skillName: tcStart.tool_call_name,
            status: 'running',
            startedAt: event.timestamp,
          });
        }
        break;
      }

      case EventType.TOOL_CALL_ARGS: {
        const tcArgs = event as any;
        const parsedArgs = (() => {
          try { return JSON.parse(tcArgs.delta); } catch { return { raw: tcArgs.delta }; }
        })();
        // 更新工具调用的参数（两个级别都尝试更新）
        updateStreamToolCall(tcArgs.tool_call_id, { arguments: parsedArgs });
        break;
      }

      case EventType.TOOL_CALL_END: {
        // Tool call end is followed by result, no action needed here
        break;
      }

      case EventType.TOOL_CALL_RESULT: {
        const tcResult = event as any;
        let resultData: any = {};
        try {
          resultData = typeof tcResult.result === 'string' ? JSON.parse(tcResult.result) : tcResult.result;
        } catch { /* ignore */ }
        
        const toolCallUpdate = {
          status: (resultData.success ? 'success' : 'error') as 'success' | 'error',
          skillName: resultData.skill_name || '',
          summary: resultData.summary || '',
          resultPreview: resultData.result_preview || '',
          agentName: resultData.agent_name || '',
          finishedAt: event.timestamp,
        };
        
        // 更新 Agent 级别（涌现模式）
        const agentIdForResult = resultData.agent_id || '';
        if (agentIdForResult) {
          updateAgentToolCall(agentIdForResult, tcResult.tool_call_id, toolCallUpdate);
        }
        
        // 更新消息流级别（普通模式）
        updateStreamToolCall(tcResult.tool_call_id, toolCallUpdate);
        break;
      }

      case EventType.AGENT_THINKING: {
        const thinkEvent = event as any;
        // 检查是否是已知 agent 的 thinking（涌现模式）
        const { agents: thinkAgents } = useStore.getState().sessions[useStore.getState().activeSessionId || ''] || { agents: {} };
        const isAgentThinking = thinkEvent.agent_id && thinkAgents[thinkEvent.agent_id];
        
        if (isAgentThinking) {
          // 涌现模式：只更新 Agent 级别
          updateAgentThinking(thinkEvent.agent_id, thinkEvent.thinking);
        } else {
          // 普通模式：更新消息流级别
          appendStreamThinking(thinkEvent.thinking);
        }
        break;
      }

      default:
        console.log('Unknown event:', event);
    }
  }, [
    currentSessionId, createSession, setSessionId, setStoreMode,
    setTask, setStatus, setPlan,
    addAgent, updateAgent, updateAgentThinking,
    addAgentToolCall, updateAgentToolCall,
    addStreamToolCall, updateStreamToolCall, appendStreamThinking,
    addRelayStation, addRelayMessage, closeRelayStation,
    addMessage, appendMessageContent, setFinalReport, setError
  ]);

  /**
   * 执行任务
   * 
   * 重要改动：
   * 1. 可以传入 existingSessionId 来复用已有会话
   * 2. 如果不传，后端会创建新会话并返回 session_id
   * 3. 任务执行期间的所有数据都绑定到该 session
   * 
   * 注意：不要在这里调用 reset()，因为后端会创建新会话。
   * 如果调用 reset()，会导致前端存在两个会话（一个本地空会话 + 一个后端创建的会话）
   */
  const executeTask = useCallback(async (
    task: string,
    provider: string = 'openai',
    model?: string,
    existingSessionId?: string,
    mode: string = 'emergent'
  ) => {
    // 记录当前 mode 到 ref，供 SESSION_CREATED 事件处理时使用
    currentModeRef.current = mode as 'emergent' | 'direct';
    
    // 如果有已存在的会话，切换到该会话
    if (existingSessionId) {
      // 追问场景：重置可视状态但保留消息历史
      const { resetSessionForFollowup } = useStore.getState();
      resetSessionForFollowup(existingSessionId);
      setActiveSession(existingSessionId);
      setTask(task);
      setStoreMode(mode as 'emergent' | 'direct');
    } else {
      // 新任务：后端会创建会话，这里先设置 PLANNING 状态
      // 不要调用 reset()，等待后端 SESSION_CREATED 事件来创建会话
      // 清理当前的临时会话 ID（如果是自动生成的本地会话）
      const { sessionId: currentId, closeSession: closeLocalSession } = useStore.getState();
      if (currentId && currentId.startsWith('session-')) {
        // 这是前端自动生成的临时会话，关闭它
        closeLocalSession(currentId);
      }
    }

    // 关闭之前的连接
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      // 发起 POST 请求获取 SSE 流
      const response = await fetch('/api/task/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader(),
        },
        body: JSON.stringify({ 
          task, 
          provider, 
          model,
          session_id: existingSessionId,  // 可选：传入已有会话 ID
          mode,  // emergent(涌现模式) / direct(普通模式)
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // 解析 SSE 事件
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        let currentData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7);
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEvent && currentData) {
            const event = parseEvent(currentEvent, currentData);
            if (event) {
              handleEvent(event);
            }
            currentEvent = '';
            currentData = '';
          }
        }
      }
    } catch (error) {
      console.error('Task execution error:', error);
      setStatus(AgentStatus.FAILED);
      setError(error instanceof Error ? error.message : 'Unknown error');
    }
  }, [setTask, setStatus, setStoreMode, setError, parseEvent, handleEvent, setActiveSession]);

  /**
   * 人工干预
   * 
   * 重要：必须传入 sessionId，确保只影响指定的会话
   */
  const intervene = useCallback(async (
    sessionId: string,
    interventionType: 'pause' | 'resume' | 'cancel' | 'inject',
    agentId?: string,
    payload?: Record<string, unknown>
  ) => {
    if (!sessionId) {
      throw new Error('sessionId is required for intervention');
    }

    try {
      const response = await fetch('/api/intervention', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader(),
        },
        body: JSON.stringify({
          session_id: sessionId,
          agent_id: agentId,
          intervention_type: interventionType,
          payload,
          broadcast_to_relay: true,
        }),
      });

      const result = await response.json();
      
      // 处理返回的中继消息，更新到前端状态
      if (result.success && result.data?.relay_messages) {
        const relayMessages = result.data.relay_messages;
        console.log(`[useAgui] Intervention relay messages for session ${sessionId}:`, relayMessages);
        
        for (const msg of relayMessages) {
          const stationId = msg.station_id || 'default-intervention-station';
          
          console.log(`[useAgui] Adding relay message to station: ${stationId}`, {
            messageId: msg.message_id,
            relayType: msg.relay_type,
            content: msg.content?.slice(0, 100),
          });
          
          addRelayMessage(stationId, {
            id: msg.message_id,
            stationId: stationId,
            sourceAgentId: msg.source_agent_id,
            sourceAgentName: msg.source_agent_name,
            targetAgentIds: msg.target_agent_ids || [],
            relayType: msg.relay_type,
            content: msg.content,
            importance: msg.importance,
            timestamp: msg.timestamp || new Date().toISOString(),
            metadata: msg.metadata || {},
            viewedBy: msg.viewed_by || [],
            acknowledgedBy: msg.acknowledged_by || [],
            viewedTimestamps: msg.viewed_timestamps || {},
          });
        }
      } else {
        console.warn('[useAgui] No relay messages in intervention response:', result);
      }
      
      return result;
    } catch (error) {
      console.error('Intervention error:', error);
      throw error;
    }
  }, [addRelayMessage]);

  /**
   * 获取中继历史（指定会话）
   */
  const getRelayHistory = useCallback(async (sessionId: string, limit: number = 20) => {
    if (!sessionId) {
      throw new Error('sessionId is required');
    }

    try {
      const response = await fetch(`/api/relay/${sessionId}/history?limit=${limit}`, {
        headers: { ...getAuthHeader() },
      });
      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Get relay history error:', error);
      throw error;
    }
  }, []);

  /**
   * 关闭会话
   */
  const closeSession = useCallback(async (sessionId: string) => {
    if (!sessionId) {
      throw new Error('sessionId is required');
    }

    try {
      const response = await fetch(`/api/session/${sessionId}`, {
        method: 'DELETE',
        headers: { ...getAuthHeader() },
      });
      const result = await response.json();
      
      // 更新前端状态
      const { closeSession: closeLocalSession } = useStore.getState();
      closeLocalSession(sessionId);
      
      return result;
    } catch (error) {
      console.error('Close session error:', error);
      throw error;
    }
  }, []);

  /**
   * 处理状态快照事件 - 从快照恢复会话状态
   */
  const handleStateSnapshot = useCallback((event: StateSnapshotEvent) => {
    const { snapshot, session_id } = event;
    console.log(`[useAgui] Processing STATE_SNAPSHOT for session: ${session_id}`, snapshot);
    
    // 创建或激活会话
    createSession(session_id);
    
    const { 
      setTask: storeSetTask, 
      setStatus: storeSetStatus, 
      setPlan: storeSetPlan,
      addAgent: storeAddAgent, 
      addRelayStation: storeAddRelayStation, 
      addRelayMessage: storeAddRelayMessage,
      addMessage: storeAddMessage,
    } = useStore.getState();
    
    // 设置任务
    if (snapshot.task) {
      storeSetTask(snapshot.task);
    }
    
    // 设置状态
    if (snapshot.status) {
      const statusMap: Record<string, AgentStatus> = {
        active: snapshot.is_live ? AgentStatus.RUNNING : AgentStatus.COMPLETED,
        running: AgentStatus.RUNNING,
        completed: AgentStatus.COMPLETED,
        error: AgentStatus.FAILED,
        expired: AgentStatus.FAILED,
        planning: AgentStatus.PLANNING,
      };
      storeSetStatus(statusMap[snapshot.status] || AgentStatus.COMPLETED);
    }
    
    // 设置计划
    if (snapshot.plan) {
      storeSetPlan(snapshot.plan);
    }
    
    // 添加所有 Agent
    for (const agentData of snapshot.agents) {
      const agentStatus = snapshot.is_live 
        ? (agentData.status as AgentStatus)
        : AgentStatus.COMPLETED;
      
      storeAddAgent({
        id: agentData.agent_id,
        name: agentData.name,
        roleName: agentData.role_name,
        roleDescription: agentData.role_description,
        capabilities: agentData.capabilities,
        taskSegment: agentData.task_segment,
        status: agentStatus,
        progress: snapshot.is_live ? agentData.progress : 100,
        currentStep: agentData.current_step,
        iterations: agentData.iterations,
        thinking: agentData.thinking,
        workObjective: agentData.work_objective,
        deliverables: agentData.deliverables,
        methodology: agentData.methodology,
      });
    }
    
    // 添加所有中继站和消息
    for (const stationData of snapshot.relay_stations) {
      storeAddRelayStation({
        id: stationData.station_id,
        name: stationData.name,
        phase: stationData.phase,
        participatingAgents: stationData.participating_agents,
        messages: [],
        isActive: stationData.is_active,
      });
      
      // 添加该中继站的消息
      for (const msgData of stationData.messages || []) {
        storeAddRelayMessage(stationData.station_id, {
          id: msgData.message_id,
          stationId: msgData.station_id,
          sourceAgentId: msgData.source_agent_id,
          sourceAgentName: msgData.source_agent_name,
          targetAgentIds: msgData.target_agent_ids,
          relayType: msgData.relay_type,
          content: msgData.content,
          importance: msgData.importance,
          timestamp: msgData.timestamp,
          metadata: {},
          viewedBy: msgData.viewed_by,
          acknowledgedBy: msgData.acknowledged_by,
          viewedTimestamps: {},
        });
      }
    }
    
    // 【新增】加载消息历史（Master Agent 的任务分析、思考过程等）
    if (snapshot.messages && snapshot.messages.length > 0) {
      console.log(`[useAgui] Loading ${snapshot.messages.length} messages from snapshot`);
      for (const msgData of snapshot.messages) {
        storeAddMessage({
          id: msgData.message_id,
          role: msgData.role as 'user' | 'assistant' | 'system',
          content: msgData.content,
          timestamp: msgData.timestamp,
        });
      }
    }
    
    console.log(`[useAgui] STATE_SNAPSHOT processed: ${snapshot.agents.length} agents, ${snapshot.relay_stations.length} stations`);
  }, [createSession]);

  /**
   * 订阅已有会话的实时事件流
   * 
   * 用途：
   * 1. 跨浏览器访问同一会话
   * 2. 重新连接到进行中的任务
   * 3. 多客户端同时查看同一会话
   * 
   * 工作流程：
   * 1. 连接到 /api/session/{session_id}/subscribe
   * 2. 首先收到 STATE_SNAPSHOT 事件，恢复当前状态
   * 3. 持续接收后续的实时事件（AGENT_SPAWNED, AGENT_STATUS_CHANGED 等）
   * 4. 定期收到 HEARTBEAT 保持连接
   */
  const subscribeToSession = useCallback(async (sessionId: string): Promise<void> => {
    if (!sessionId) {
      throw new Error('sessionId is required for subscription');
    }
    
    // 关闭之前的订阅连接
    if (subscriptionEventSourceRef.current) {
      console.log('[useAgui] Closing previous subscription');
      subscriptionEventSourceRef.current.close();
      subscriptionEventSourceRef.current = null;
    }
    
    setIsSubscribing(true);
    
    return new Promise((resolve, reject) => {
      console.log(`[useAgui] Subscribing to session: ${sessionId}`);
      
      // 使用 EventSource 连接到订阅端点
      // EventSource 不支持自定义 Header，通过 query 参数传递 token
      const token = getStoredToken();
      const subscribeUrl = token
        ? `/api/session/${sessionId}/subscribe?token=${encodeURIComponent(token)}`
        : `/api/session/${sessionId}/subscribe`;
      const eventSource = new EventSource(subscribeUrl);
      subscriptionEventSourceRef.current = eventSource;
      
      let hasReceivedSnapshot = false;
      
      // 处理打开事件
      eventSource.onopen = () => {
        console.log(`[useAgui] Subscription connection opened for session: ${sessionId}`);
      };
      
      // 处理错误
      eventSource.onerror = (error) => {
        console.error('[useAgui] Subscription error:', error);
        setIsSubscribing(false);
        
        if (!hasReceivedSnapshot) {
          eventSource.close();
          subscriptionEventSourceRef.current = null;
          reject(new Error('Failed to subscribe to session'));
        }
        // 如果已经收到快照，错误可能是连接断开，不需要 reject
      };
      
      // 处理 STATE_SNAPSHOT 事件 - 初始状态
      eventSource.addEventListener('STATE_SNAPSHOT', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          const event: StateSnapshotEvent = {
            type: EventType.STATE_SNAPSHOT,
            timestamp: new Date().toISOString(),
            session_id: sessionId,
            snapshot: data.snapshot || data,
          };
          
          handleStateSnapshot(event);
          hasReceivedSnapshot = true;
          setIsSubscribing(false);
          resolve();
        } catch (err) {
          console.error('[useAgui] Failed to parse STATE_SNAPSHOT:', err);
        }
      });
      
      // 处理心跳事件
      eventSource.addEventListener('HEARTBEAT', (e: MessageEvent) => {
        console.debug(`[useAgui] Heartbeat received for session: ${sessionId}`);
      });
      
      // 处理会话状态变更事件
      eventSource.addEventListener('SESSION_STATE_CHANGED', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          console.log(`[useAgui] Session state changed:`, data);
          
          // 根据变更类型处理
          if (data.change_type === 'completed') {
            setStatus(AgentStatus.COMPLETED);
          } else if (data.change_type === 'error') {
            setStatus(AgentStatus.FAILED);
            if (data.summary?.error) {
              setError(data.summary.error);
            }
          }
        } catch (err) {
          console.error('[useAgui] Failed to parse SESSION_STATE_CHANGED:', err);
        }
      });
      
      // 处理后续的实时事件（复用现有的 handleEvent 逻辑）
      // 【重要】所有事件都传递 sessionId，确保会话隔离
      
      // Agent 涌现
      eventSource.addEventListener('AGENT_SPAWNED', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.AGENT_SPAWNED }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse AGENT_SPAWNED:', err);
        }
      });
      
      // Agent 状态变更
      eventSource.addEventListener('AGENT_STATUS_CHANGED', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.AGENT_STATUS_CHANGED }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse AGENT_STATUS_CHANGED:', err);
        }
      });
      
      // Agent 进度
      eventSource.addEventListener('AGENT_PROGRESS', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.AGENT_PROGRESS }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse AGENT_PROGRESS:', err);
        }
      });
      
      // Agent 思考
      eventSource.addEventListener('AGENT_THINKING', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.AGENT_THINKING }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse AGENT_THINKING:', err);
        }
      });
      
      // 中继站相关事件
      eventSource.addEventListener('RELAY_STATION_OPENED', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.RELAY_STATION_OPENED }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse RELAY_STATION_OPENED:', err);
        }
      });
      
      eventSource.addEventListener('RELAY_MESSAGE_SENT', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.RELAY_MESSAGE_SENT }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse RELAY_MESSAGE_SENT:', err);
        }
      });
      
      eventSource.addEventListener('RELAY_STATION_CLOSED', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.RELAY_STATION_CLOSED }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse RELAY_STATION_CLOSED:', err);
        }
      });
      
      // 任务相关事件
      eventSource.addEventListener('RUN_STARTED', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.RUN_STARTED }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse RUN_STARTED:', err);
        }
      });
      
      eventSource.addEventListener('RUN_FINISHED', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.RUN_FINISHED }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse RUN_FINISHED:', err);
        }
      });
      
      eventSource.addEventListener('RUN_ERROR', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.RUN_ERROR }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse RUN_ERROR:', err);
        }
      });
      
      // 计划生成
      eventSource.addEventListener('PLAN_GENERATED', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.PLAN_GENERATED }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse PLAN_GENERATED:', err);
        }
      });
      
      // Tool Call 事件
      eventSource.addEventListener('TOOL_CALL_START', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.TOOL_CALL_START }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse TOOL_CALL_START:', err);
        }
      });
      
      eventSource.addEventListener('TOOL_CALL_ARGS', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.TOOL_CALL_ARGS }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse TOOL_CALL_ARGS:', err);
        }
      });
      
      eventSource.addEventListener('TOOL_CALL_END', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.TOOL_CALL_END }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse TOOL_CALL_END:', err);
        }
      });
      
      eventSource.addEventListener('TOOL_CALL_RESULT', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.TOOL_CALL_RESULT }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse TOOL_CALL_RESULT:', err);
        }
      });
      
      // Agent 思考事件
      eventSource.addEventListener('AGENT_THINKING', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          handleEvent({ ...data, type: EventType.AGENT_THINKING }, sessionId);
        } catch (err) {
          console.error('[useAgui] Failed to parse AGENT_THINKING:', err);
        }
      });
      
      // 设置超时（30秒内没收到快照则认为失败）
      const timeoutId = setTimeout(() => {
        if (!hasReceivedSnapshot) {
          console.error('[useAgui] Subscription timeout - no snapshot received');
          eventSource.close();
          subscriptionEventSourceRef.current = null;
          setIsSubscribing(false);
          reject(new Error('Subscription timeout'));
        }
      }, 30000);
      
      // 清理超时
      eventSource.addEventListener('STATE_SNAPSHOT', () => {
        clearTimeout(timeoutId);
      }, { once: true });
    });
  }, [handleEvent, handleStateSnapshot, setStatus, setError]);

  /**
   * 取消订阅会话
   */
  const unsubscribeFromSession = useCallback(() => {
    if (subscriptionEventSourceRef.current) {
      console.log('[useAgui] Unsubscribing from session');
      subscriptionEventSourceRef.current.close();
      subscriptionEventSourceRef.current = null;
      setIsSubscribing(false);
    }
  }, []);

  /**
   * 停止执行
   */
  const stop = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    // 同时关闭订阅连接
    unsubscribeFromSession();
  }, [unsubscribeFromSession]);

  return {
    executeTask,
    intervene,
    getRelayHistory,
    closeSession,
    subscribeToSession,
    unsubscribeFromSession,
    isSubscribing,
    stop,
    currentSessionId,
  };
}
