/**
 * AG-UI 协议类型定义
 */

// ========== 事件类型 ==========

export enum EventType {
  // Lifecycle Events
  RUN_STARTED = 'RUN_STARTED',
  RUN_FINISHED = 'RUN_FINISHED',
  RUN_ERROR = 'RUN_ERROR',
  SESSION_CREATED = 'SESSION_CREATED',  // 会话创建事件
  SESSION_STATE_CHANGED = 'SESSION_STATE_CHANGED',  // 新增：会话状态变更事件（用于列表刷新）
  
  // Text Message Events
  TEXT_MESSAGE_START = 'TEXT_MESSAGE_START',
  TEXT_MESSAGE_CONTENT = 'TEXT_MESSAGE_CONTENT',
  TEXT_MESSAGE_END = 'TEXT_MESSAGE_END',
  
  // Tool Call Events
  TOOL_CALL_START = 'TOOL_CALL_START',
  TOOL_CALL_ARGS = 'TOOL_CALL_ARGS',
  TOOL_CALL_END = 'TOOL_CALL_END',
  TOOL_CALL_RESULT = 'TOOL_CALL_RESULT',
  
  // State Management Events
  STATE_SNAPSHOT = 'STATE_SNAPSHOT',
  STATE_DELTA = 'STATE_DELTA',
  HEARTBEAT = 'HEARTBEAT',  // 新增：心跳事件
  
  // Custom Agent Events
  AGENT_SPAWNED = 'AGENT_SPAWNED',
  AGENT_STATUS_CHANGED = 'AGENT_STATUS_CHANGED',
  AGENT_PROGRESS = 'AGENT_PROGRESS',
  AGENT_THINKING = 'AGENT_THINKING',
  
  // Relay Station Events
  RELAY_STATION_OPENED = 'RELAY_STATION_OPENED',
  RELAY_MESSAGE_SENT = 'RELAY_MESSAGE_SENT',
  RELAY_STATION_CLOSED = 'RELAY_STATION_CLOSED',
  
  // Planning Events
  PLAN_GENERATED = 'PLAN_GENERATED',
  ROLE_EMERGED = 'ROLE_EMERGED',
  
  // Human Intervention Events
  INTERVENTION_REQUESTED = 'INTERVENTION_REQUESTED',
  INTERVENTION_APPLIED = 'INTERVENTION_APPLIED',
  INTERVENTION_BROADCAST = 'INTERVENTION_BROADCAST',
}

// ========== 干预相关类型 ==========

export enum InterventionType {
  PAUSE = 'pause',
  RESUME = 'resume',
  RESTART = 'restart',
  ADJUST = 'adjust',
  INJECT = 'inject',
  CANCEL = 'cancel',
}

export enum InterventionScope {
  SINGLE = 'single',
  SELECTED = 'selected',
  ALL = 'all',
  BROADCAST = 'broadcast',
}

// ========== Agent 状态 ==========

export enum AgentStatus {
  PENDING = 'pending',
  PLANNING = 'planning',
  RUNNING = 'running',
  WAITING_RELAY = 'waiting_relay',
  RELAYING = 'relaying',
  COMPLETED = 'completed',
  FAILED = 'failed',
  PAUSED = 'paused',
  CANCELLED = 'cancelled',
}

// ========== 技能相关类型 ==========

export interface SkillAssignment {
  skill_name: string;
  skill_display_name: string;
  reason: string;
}

export interface WorkMethodology {
  approach: string;
  steps: string[];
  tools_and_frameworks: string[];
  success_criteria: string[];
  quality_metrics: string[];
}

// ========== 事件数据类型 ==========

export interface BaseEvent {
  type: EventType;
  timestamp: string;
}

export interface RunStartedEvent extends BaseEvent {
  type: EventType.RUN_STARTED;
  thread_id: string;
  run_id: string;
}

export interface RunFinishedEvent extends BaseEvent {
  type: EventType.RUN_FINISHED;
  thread_id: string;
  run_id: string;
}

export interface RunErrorEvent extends BaseEvent {
  type: EventType.RUN_ERROR;
  message: string;
  code?: string;
}

export interface TextMessageStartEvent extends BaseEvent {
  type: EventType.TEXT_MESSAGE_START;
  message_id: string;
  role: string;
}

export interface TextMessageContentEvent extends BaseEvent {
  type: EventType.TEXT_MESSAGE_CONTENT;
  message_id: string;
  delta: string;
}

export interface TextMessageEndEvent extends BaseEvent {
  type: EventType.TEXT_MESSAGE_END;
  message_id: string;
}

export interface AgentSpawnedEvent extends BaseEvent {
  type: EventType.AGENT_SPAWNED;
  agent_id: string;
  agent_name: string;
  role_name: string;
  role_description: string;
  capabilities: string[];
  task_segment: string;
  // 新增字段
  work_objective?: string;
  deliverables?: string[];
  methodology?: WorkMethodology;
  assigned_skills?: SkillAssignment[];
  expertise_level?: string;
  focus_areas?: string[];
}

export interface AgentStatusChangedEvent extends BaseEvent {
  type: EventType.AGENT_STATUS_CHANGED;
  agent_id: string;
  agent_name: string;
  previous_status: string;
  new_status: string;
  reason?: string;
}

export interface AgentProgressEvent extends BaseEvent {
  type: EventType.AGENT_PROGRESS;
  agent_id: string;
  agent_name: string;
  progress: number;
  current_step: string;
  iterations: number;
}

export interface AgentThinkingEvent extends BaseEvent {
  type: EventType.AGENT_THINKING;
  agent_id: string;
  agent_name: string;
  thinking: string;
}

export interface RelayStationOpenedEvent extends BaseEvent {
  type: EventType.RELAY_STATION_OPENED;
  station_id: string;
  station_name: string;
  phase: number;
  participating_agents: Array<{ id: string; name: string }>;
}

export interface RelayMessageSentEvent extends BaseEvent {
  type: EventType.RELAY_MESSAGE_SENT;
  station_id: string;
  message_id: string;
  source_agent_id: string;
  source_agent_name: string;
  target_agent_ids: string[];
  relay_type: string;
  content: string;
  importance: number;
}

export interface RelayStationClosedEvent extends BaseEvent {
  type: EventType.RELAY_STATION_CLOSED;
  station_id: string;
  station_name: string;
  summary: string;
}

export interface PlanGeneratedEvent extends BaseEvent {
  type: EventType.PLAN_GENERATED;
  plan_id: string;
  original_task: string;
  analysis: string;
  phases: Array<{
    phase_number: number;
    name: string;
    description: string;
    participating_roles: string[];
  }>;
  estimated_duration: number;
  total_agents: number;
}

export interface RoleEmergedEvent extends BaseEvent {
  type: EventType.ROLE_EMERGED;
  role_id: string;
  role_name: string;
  description: string;
  capabilities: string[];
  focus_areas: string[];
  reasoning: string;
}

// 新增：会话状态变更事件
export interface SessionStateChangedEvent extends BaseEvent {
  type: EventType.SESSION_STATE_CHANGED;
  session_id: string;
  change_type: string;  // "agent_added", "agent_status_changed", "completed", "error", etc.
  summary: Record<string, any>;
}

// 新增：心跳事件
export interface HeartbeatEvent extends BaseEvent {
  type: EventType.HEARTBEAT;
  session_id: string;
}

// 新增：状态快照事件（用于订阅时恢复状态）
export interface StateSnapshotEvent extends BaseEvent {
  type: EventType.STATE_SNAPSHOT;
  session_id: string;
  snapshot: {
    is_live: boolean;
    task?: string;
    status?: string;
    plan?: any;
    agents: any[];
    relay_stations: any[];
    messages?: Array<{
      message_id: string;
      session_id: string;
      role: string;
      content: string;
      timestamp: string;
    }>;
    subscriber_count?: number;
  };
}

// Session Created Event
export interface SessionCreatedEvent extends BaseEvent {
  type: EventType.SESSION_CREATED;
  session_id: string;
}

// Tool Call Events
export interface ToolCallStartEvent extends BaseEvent {
  type: EventType.TOOL_CALL_START;
  tool_call_id: string;
  tool_call_name: string;
  parent_message_id?: string;  // agent_id
}

export interface ToolCallArgsEvent extends BaseEvent {
  type: EventType.TOOL_CALL_ARGS;
  tool_call_id: string;
  delta: string;
}

export interface ToolCallEndEvent extends BaseEvent {
  type: EventType.TOOL_CALL_END;
  tool_call_id: string;
}

export interface ToolCallResultEvent extends BaseEvent {
  type: EventType.TOOL_CALL_RESULT;
  tool_call_id: string;
  result: string;  // JSON string containing agent_id, skill_name, success, summary, result_preview
}

// Tool call record for Agent state
export interface AgentToolCall {
  id: string;
  toolName: string;
  skillName: string;
  arguments?: Record<string, any>;
  status: 'running' | 'success' | 'error';
  summary?: string;
  resultPreview?: string;
  agentName?: string;
  startedAt: string;
  finishedAt?: string;
}

// 联合类型
export type AguiEvent =
  | RunStartedEvent
  | RunFinishedEvent
  | RunErrorEvent
  | SessionCreatedEvent
  | TextMessageStartEvent
  | TextMessageContentEvent
  | TextMessageEndEvent
  | AgentSpawnedEvent
  | AgentStatusChangedEvent
  | AgentProgressEvent
  | AgentThinkingEvent
  | RelayStationOpenedEvent
  | RelayMessageSentEvent
  | RelayStationClosedEvent
  | PlanGeneratedEvent
  | RoleEmergedEvent
  | ToolCallStartEvent
  | ToolCallArgsEvent
  | ToolCallEndEvent
  | ToolCallResultEvent
  | SessionStateChangedEvent
  | HeartbeatEvent
  | StateSnapshotEvent;

// ========== 应用状态类型 ==========

export interface Agent {
  id: string;
  name: string;
  roleName: string;
  roleDescription: string;
  capabilities: string[];
  taskSegment: string;
  status: AgentStatus;
  progress: number;
  currentStep: string;
  iterations: number;
  thinking: string;
  // 新增字段
  workObjective?: string;
  deliverables?: string[];
  methodology?: WorkMethodology;
  assignedSkills?: SkillAssignment[];
  expertiseLevel?: string;
  focusAreas?: string[];
  toolCalls?: AgentToolCall[];
}

export interface RelayMessage {
  id: string;
  stationId: string;
  sourceAgentId: string;
  sourceAgentName: string;
  targetAgentIds: string[];
  relayType: string;
  content: string;
  importance: number;
  timestamp: string;
  metadata?: Record<string, any>;
  // 消息查看状态
  viewedBy?: string[];          // 已查看的 Agent ID 列表
  acknowledgedBy?: string[];    // 已确认的 Agent ID 列表
  viewedTimestamps?: Record<string, string>;  // Agent ID -> 查看时间
}

export interface RelayStation {
  id: string;
  name: string;
  phase: number;
  participatingAgents: Array<{ id: string; name: string }>;
  messages: RelayMessage[];
  isActive: boolean;
}

export interface TaskPlan {
  id: string;
  originalTask: string;
  analysis: string;
  phases: Array<{
    phaseNumber: number;
    name: string;
    description: string;
    participatingRoles: string[];
  }>;
  estimatedDuration: number;
  totalAgents: number;
}

export interface Message {
  id: string;
  role: 'assistant' | 'user' | 'system';
  content: string;
  timestamp: string;
  // 按轮次关联的思考过程和工具调用（在 TEXT_MESSAGE_START 时从 stream 缓冲区"收割"）
  turnThinking?: string;
  turnToolCalls?: AgentToolCall[];
}

export interface TaskSession {
  id: string;
  task: string;
  status: AgentStatus;
  plan?: TaskPlan;
  agents: Record<string, Agent>;
  relayStations: Record<string, RelayStation>;
  messages: Message[];
  finalReport?: string;
  startedAt?: string;
  finishedAt?: string;
  error?: string;
}
