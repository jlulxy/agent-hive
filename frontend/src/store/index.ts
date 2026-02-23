/**
 * Zustand Store - 全局状态管理
 * 
 * 重构：支持多会话（Session）数据隔离
 * 
 * 设计原则：
 * 1. 每个 sessionId 对应独立的状态（SessionState）
 * 2. 不同会话之间的 Agent、中继站、消息完全隔离
 * 3. 支持切换活跃会话
 * 4. 旧会话数据保留，可以查看历史
 */

import { create } from 'zustand';
import {
  TaskPlan,
  Agent,
  AgentToolCall,
  RelayStation,
  RelayMessage,
  Message,
  AgentStatus,
} from '../types/agui';


// ========== 单个会话的状态 ==========

export interface SessionState {
  id: string;
  task: string;
  status: AgentStatus;
  mode: 'emergent' | 'direct';  // 涌现模式 / 普通模式
  plan: TaskPlan | null;
  agents: Record<string, Agent>;
  relayStations: Record<string, RelayStation>;
  messages: Message[];
  finalReport: string | null;
  error: string | null;
  createdAt: string;
  lastActiveAt: string;
  // 消息流级别的 tool call 追踪（普通模式用）
  streamToolCalls: AgentToolCall[];
  // 消息流级别的 thinking 追踪（普通模式用）
  streamThinking: string;
}

const createEmptySessionState = (sessionId: string, mode: 'emergent' | 'direct' = 'emergent'): SessionState => ({
  id: sessionId,
  task: '',
  status: AgentStatus.PENDING,
  mode,
  plan: null,
  agents: {},
  relayStations: {},
  messages: [],
  finalReport: null,
  error: null,
  createdAt: new Date().toISOString(),
  lastActiveAt: new Date().toISOString(),
  streamToolCalls: [],
  streamThinking: '',
});


// ========== 全局应用状态 ==========

interface AppState {
  // 多会话管理
  sessions: Record<string, SessionState>;
  activeSessionId: string | null;
  
  // 会话管理操作
  createSession: (sessionId: string) => void;
  setActiveSession: (sessionId: string) => void;
  getSession: (sessionId: string) => SessionState | undefined;
  getActiveSession: () => SessionState | undefined;
  closeSession: (sessionId: string) => void;
  
  // 兼容旧 API 的快捷方法（操作活跃会话）
  sessionId: string | null;  // 当前活跃会话 ID
  task: string;
  status: AgentStatus;
  mode: 'emergent' | 'direct';
  plan: TaskPlan | null;
  agents: Record<string, Agent>;
  relayStations: Record<string, RelayStation>;
  messages: Message[];
  finalReport: string | null;
  error: string | null;
  streamToolCalls: AgentToolCall[];
  streamThinking: string;

  // 原有操作（自动作用于活跃会话）
  setSessionId: (id: string) => void;
  setTask: (task: string) => void;
  setStatus: (status: AgentStatus) => void;
  setMode: (mode: 'emergent' | 'direct') => void;
  setPlan: (plan: TaskPlan) => void;
  
  addAgent: (agent: Agent) => void;
  updateAgent: (id: string, updates: Partial<Agent>) => void;
  updateAgentThinking: (id: string, thinking: string) => void;
  addAgentToolCall: (agentId: string, toolCall: AgentToolCall) => void;
  updateAgentToolCall: (agentId: string, toolCallId: string, updates: Partial<AgentToolCall>) => void;
  
  addRelayStation: (station: RelayStation) => void;
  addRelayMessage: (stationId: string, message: RelayMessage) => void;
  updateRelayMessage: (messageId: string, updates: Partial<RelayMessage>) => void;
  closeRelayStation: (stationId: string, summary: string) => void;
  
  addMessage: (message: Message) => void;
  appendMessageContent: (messageId: string, content: string) => void;
  
  // 消息流级别的 tool call / thinking（普通模式用）
  addStreamToolCall: (toolCall: AgentToolCall) => void;
  updateStreamToolCall: (toolCallId: string, updates: Partial<AgentToolCall>) => void;
  appendStreamThinking: (thinking: string) => void;
  
  setFinalReport: (report: string) => void;
  setError: (error: string) => void;
  
  reset: () => void;
  resetSession: (sessionId: string) => void;
  resetSessionForFollowup: (sessionId: string) => void;
  clearAllSessions: () => void;
}


// 计算活跃会话的 derived 状态
const getActiveSessionData = (sessions: Record<string, SessionState>, activeSessionId: string | null) => {
  if (!activeSessionId || !sessions[activeSessionId]) {
    return {
      task: '',
      status: AgentStatus.PENDING,
      mode: 'emergent' as const,
      plan: null,
      agents: {},
      relayStations: {},
      messages: [],
      finalReport: null,
      error: null,
      streamToolCalls: [] as AgentToolCall[],
      streamThinking: '',
    };
  }
  const session = sessions[activeSessionId];
  return {
    task: session.task,
    status: session.status,
    mode: session.mode,
    plan: session.plan,
    agents: session.agents,
    relayStations: session.relayStations,
    messages: session.messages,
    finalReport: session.finalReport,
    error: session.error,
    streamToolCalls: session.streamToolCalls,
    streamThinking: session.streamThinking,
  };
};


export const useStore = create<AppState>((set, get) => ({
  // 多会话管理
  sessions: {},
  activeSessionId: null,
  
  // 兼容属性（从活跃会话派生）
  sessionId: null,
  task: '',
  status: AgentStatus.PENDING,
  mode: 'emergent',
  plan: null,
  agents: {},
  relayStations: {},
  messages: [],
  finalReport: null,
  error: null,
  streamToolCalls: [],
  streamThinking: '',

  // ========== 会话管理操作 ==========
  
  createSession: (sessionId) => set((state) => {
    // 如果会话已存在，只更新活跃时间
    if (state.sessions[sessionId]) {
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...state.sessions[sessionId],
            lastActiveAt: new Date().toISOString(),
          },
        },
        activeSessionId: sessionId,
        sessionId: sessionId,
        ...getActiveSessionData(state.sessions, sessionId),
      };
    }
    
    // 创建新会话
    const newSession = createEmptySessionState(sessionId);
    const newSessions = {
      ...state.sessions,
      [sessionId]: newSession,
    };
    
    return {
      sessions: newSessions,
      activeSessionId: sessionId,
      sessionId: sessionId,
      ...getActiveSessionData(newSessions, sessionId),
    };
  }),

  setActiveSession: (sessionId) => set((state) => {
    if (!state.sessions[sessionId]) {
      console.warn(`[Store] Session ${sessionId} not found`);
      return state;
    }
    
    // 更新活跃时间
    const updatedSessions = {
      ...state.sessions,
      [sessionId]: {
        ...state.sessions[sessionId],
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      activeSessionId: sessionId,
      sessionId: sessionId,
      ...getActiveSessionData(updatedSessions, sessionId),
    };
  }),

  getSession: (sessionId) => get().sessions[sessionId],

  getActiveSession: () => {
    const state = get();
    if (!state.activeSessionId) return undefined;
    return state.sessions[state.activeSessionId];
  },

  closeSession: (sessionId) => set((state) => {
    const { [sessionId]: removed, ...remaining } = state.sessions;
    
    // 如果关闭的是活跃会话，切换到其他会话或清空
    let newActiveId = state.activeSessionId;
    if (state.activeSessionId === sessionId) {
      const remainingIds = Object.keys(remaining);
      newActiveId = remainingIds.length > 0 ? remainingIds[0] : null;
    }
    
    return {
      sessions: remaining,
      activeSessionId: newActiveId,
      sessionId: newActiveId,
      ...getActiveSessionData(remaining, newActiveId),
    };
  }),

  // ========== 原有操作（作用于活跃会话） ==========

  setSessionId: (id) => set((state) => {
    // 如果会话不存在，先创建
    if (!state.sessions[id]) {
      const newSession = createEmptySessionState(id);
      const newSessions = {
        ...state.sessions,
        [id]: newSession,
      };
      return {
        sessions: newSessions,
        activeSessionId: id,
        sessionId: id,
        ...getActiveSessionData(newSessions, id),
      };
    }
    
    return {
      activeSessionId: id,
      sessionId: id,
      ...getActiveSessionData(state.sessions, id),
    };
  }),
  
  setTask: (task) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...state.sessions[state.activeSessionId],
        task,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      task,
    };
  }),
  
  setStatus: (status) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...state.sessions[state.activeSessionId],
        status,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      status,
    };
  }),

  setMode: (mode) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...state.sessions[state.activeSessionId],
        mode,
      },
    };
    
    return {
      sessions: updatedSessions,
      mode,
    };
  }),
  
  setPlan: (plan) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...state.sessions[state.activeSessionId],
        plan,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      plan,
    };
  }),

  addAgent: (agent) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    const updatedAgents = { ...currentSession.agents, [agent.id]: agent };
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        agents: updatedAgents,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      agents: updatedAgents,
    };
  }),

  updateAgent: (id, updates) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    if (!currentSession.agents[id]) return state;
    
    const updatedAgents = {
      ...currentSession.agents,
      [id]: { ...currentSession.agents[id], ...updates },
    };
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        agents: updatedAgents,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      agents: updatedAgents,
    };
  }),

  updateAgentThinking: (id, thinking) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    if (!currentSession.agents[id]) return state;
    
    const updatedAgents = {
      ...currentSession.agents,
      [id]: {
        ...currentSession.agents[id],
        thinking: currentSession.agents[id].thinking + thinking,
      },
    };
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        agents: updatedAgents,
      },
    };
    
    return {
      sessions: updatedSessions,
      agents: updatedAgents,
    };
  }),

  addAgentToolCall: (agentId, toolCall) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    const agent = currentSession.agents[agentId];
    if (!agent) return state;
    
    const existingCalls = agent.toolCalls || [];
    const updatedAgents = {
      ...currentSession.agents,
      [agentId]: {
        ...agent,
        toolCalls: [...existingCalls, toolCall],
      },
    };
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        agents: updatedAgents,
      },
    };
    
    return {
      sessions: updatedSessions,
      agents: updatedAgents,
    };
  }),

  updateAgentToolCall: (agentId, toolCallId, updates) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    const agent = currentSession.agents[agentId];
    if (!agent || !agent.toolCalls) return state;
    
    const updatedToolCalls = agent.toolCalls.map((tc) =>
      tc.id === toolCallId ? { ...tc, ...updates } : tc
    );
    
    const updatedAgents = {
      ...currentSession.agents,
      [agentId]: {
        ...agent,
        toolCalls: updatedToolCalls,
      },
    };
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        agents: updatedAgents,
      },
    };
    
    return {
      sessions: updatedSessions,
      agents: updatedAgents,
    };
  }),

  addRelayStation: (station) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    const updatedStations = { ...currentSession.relayStations, [station.id]: station };
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        relayStations: updatedStations,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      relayStations: updatedStations,
    };
  }),

  addRelayMessage: (stationId, message) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    const effectiveStationId = stationId || 'default-intervention-station';
    
    let station = currentSession.relayStations[effectiveStationId];
    
    // 如果中继站不存在，自动创建
    if (!station) {
      console.log(`[Store] Creating temporary relay station for message: ${effectiveStationId}`);
      station = {
        id: effectiveStationId,
        name: '干预消息站',
        phase: 0,
        participatingAgents: [],
        messages: [],
        isActive: true,
      };
    }
    
    // 检查消息是否已存在
    if (station.messages.some(m => m.id === message.id)) {
      console.log(`[Store] Message ${message.id} already exists, skipping`);
      return state;
    }
    
    const updatedStations = {
      ...currentSession.relayStations,
      [effectiveStationId]: {
        ...station,
        messages: [...station.messages, message],
      },
    };
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        relayStations: updatedStations,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      relayStations: updatedStations,
    };
  }),

  updateRelayMessage: (messageId, updates) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    let hasUpdates = false;
    
    const updatedStations: Record<string, RelayStation> = {};
    
    Object.entries(currentSession.relayStations).forEach(([stationId, station]) => {
      const updatedMessages = station.messages.map((msg) => {
        if (msg.id === messageId) {
          hasUpdates = true;
          return { ...msg, ...updates };
        }
        return msg;
      });
      
      updatedStations[stationId] = {
        ...station,
        messages: updatedMessages,
      };
    });
    
    if (!hasUpdates) return state;
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        relayStations: updatedStations,
      },
    };
    
    return {
      sessions: updatedSessions,
      relayStations: updatedStations,
    };
  }),

  closeRelayStation: (stationId, _summary) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    const station = currentSession.relayStations[stationId];
    if (!station) return state;
    
    const updatedStations = {
      ...currentSession.relayStations,
      [stationId]: { ...station, isActive: false },
    };
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        relayStations: updatedStations,
      },
    };
    
    return {
      sessions: updatedSessions,
      relayStations: updatedStations,
    };
  }),

  addMessage: (message) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    
    // 【去重检查】如果消息 ID 已存在，则跳过
    if (currentSession.messages.some(m => m.id === message.id)) {
      console.log(`[Store] Message ${message.id} already exists, skipping`);
      return state;
    }
    
    // 【按轮次收割】如果是 assistant 消息，把当前 streaming 缓冲区的 thinking/toolCalls 关联到这条消息
    let enrichedMessage = message;
    let updatedStreamThinking = currentSession.streamThinking;
    let updatedStreamToolCalls = currentSession.streamToolCalls;
    
    if (message.role === 'assistant') {
      enrichedMessage = {
        ...message,
        turnThinking: currentSession.streamThinking || undefined,
        turnToolCalls: currentSession.streamToolCalls.length > 0 ? [...currentSession.streamToolCalls] : undefined,
      };
      // 收割后清空 streaming 缓冲区，为下一轮准备
      updatedStreamThinking = '';
      updatedStreamToolCalls = [];
    }
    
    const updatedMessages = [...currentSession.messages, enrichedMessage];
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        messages: updatedMessages,
        streamThinking: updatedStreamThinking,
        streamToolCalls: updatedStreamToolCalls,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      messages: updatedMessages,
      streamThinking: updatedStreamThinking,
      streamToolCalls: updatedStreamToolCalls,
    };
  }),

  appendMessageContent: (messageId, content) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    const updatedMessages = currentSession.messages.map((msg) =>
      msg.id === messageId
        ? { ...msg, content: msg.content + content }
        : msg
    );
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        messages: updatedMessages,
      },
    };
    
    return {
      sessions: updatedSessions,
      messages: updatedMessages,
    };
  }),

  // === 消息流级别 tool call / thinking（普通模式用）===
  
  addStreamToolCall: (toolCall) => set((state) => {
    if (!state.activeSessionId) {
      console.warn('[Store] addStreamToolCall: no activeSessionId, dropping tool call:', toolCall.toolName);
      return state;
    }
    
    const currentSession = state.sessions[state.activeSessionId];
    if (!currentSession) {
      console.warn('[Store] addStreamToolCall: session not found for id:', state.activeSessionId);
      return state;
    }
    
    const updatedToolCalls = [...currentSession.streamToolCalls, toolCall];
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        streamToolCalls: updatedToolCalls,
      },
    };
    
    console.log(`[Store] addStreamToolCall: tool=${toolCall.toolName}, total=${updatedToolCalls.length}, sessionId=${state.activeSessionId}`);
    
    return {
      sessions: updatedSessions,
      streamToolCalls: updatedToolCalls,
    };
  }),

  updateStreamToolCall: (toolCallId, updates) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    if (!currentSession) return state;
    
    const updatedToolCalls = currentSession.streamToolCalls.map((tc) =>
      tc.id === toolCallId ? { ...tc, ...updates } : tc
    );
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        streamToolCalls: updatedToolCalls,
      },
    };
    
    return {
      sessions: updatedSessions,
      streamToolCalls: updatedToolCalls,
    };
  }),

  appendStreamThinking: (thinking) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const currentSession = state.sessions[state.activeSessionId];
    if (!currentSession) return state;
    
    const updatedThinking = currentSession.streamThinking + thinking;
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...currentSession,
        streamThinking: updatedThinking,
      },
    };
    
    return {
      sessions: updatedSessions,
      streamThinking: updatedThinking,
    };
  }),

  setFinalReport: (report) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...state.sessions[state.activeSessionId],
        finalReport: report,
        lastActiveAt: new Date().toISOString(),
      },
    };
    
    return {
      sessions: updatedSessions,
      finalReport: report,
    };
  }),

  setError: (error) => set((state) => {
    if (!state.activeSessionId) return state;
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: {
        ...state.sessions[state.activeSessionId],
        error,
      },
    };
    
    return {
      sessions: updatedSessions,
      error,
    };
  }),

  // 重置当前活跃会话（清空数据但保留会话）
  reset: () => set((state) => {
    if (!state.activeSessionId) {
      return {
        sessions: {},
        activeSessionId: null,
        sessionId: null,
        task: '',
        status: AgentStatus.PENDING,
        plan: null,
        agents: {},
        relayStations: {},
        messages: [],
        finalReport: null,
        error: null,
      };
    }
    
    // 重置当前会话的数据
    const resetSession = createEmptySessionState(state.activeSessionId);
    
    const updatedSessions = {
      ...state.sessions,
      [state.activeSessionId]: resetSession,
    };
    
    return {
      sessions: updatedSessions,
      ...getActiveSessionData(updatedSessions, state.activeSessionId),
    };
  }),

  // 重置指定会话
  resetSession: (sessionId) => set((state) => {
    if (!state.sessions[sessionId]) return state;
    
    const resetSession = createEmptySessionState(sessionId);
    
    const updatedSessions = {
      ...state.sessions,
      [sessionId]: resetSession,
    };
    
    // 如果是活跃会话，也更新派生状态
    if (state.activeSessionId === sessionId) {
      return {
        sessions: updatedSessions,
        ...getActiveSessionData(updatedSessions, sessionId),
      };
    }
    
    return {
      sessions: updatedSessions,
    };
  }),

  // 追问重置：清空 agents/relayStations/plan/error 但保留 messages/finalReport/streamToolCalls/streamThinking
  resetSessionForFollowup: (sessionId) => set((state) => {
    if (!state.sessions[sessionId]) return state;
    
    const currentSession = state.sessions[sessionId];
    const followupSession: SessionState = {
      ...currentSession,
      status: AgentStatus.PLANNING,
      plan: null,
      agents: {},
      relayStations: {},
      error: null,
      lastActiveAt: new Date().toISOString(),
      // 保留 messages/finalReport/streamToolCalls/streamThinking 供 UI 展示历史
    };
    
    const updatedSessions = {
      ...state.sessions,
      [sessionId]: followupSession,
    };
    
    // 如果是活跃会话，也更新派生状态
    if (state.activeSessionId === sessionId) {
      return {
        sessions: updatedSessions,
        ...getActiveSessionData(updatedSessions, sessionId),
      };
    }
    
    return {
      sessions: updatedSessions,
    };
  }),

  // 清空所有会话数据（用户登出时调用）
  clearAllSessions: () => set(() => ({
    sessions: {},
    activeSessionId: null,
    sessionId: null,
    task: '',
    status: AgentStatus.PENDING,
    mode: 'emergent' as const,
    plan: null,
    agents: {},
    relayStations: {},
    messages: [],
    streamToolCalls: [],
    streamThinking: '',
    finalReport: null,
    error: null,
  })),
}));
