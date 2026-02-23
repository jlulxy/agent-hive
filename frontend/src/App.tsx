/**
 * 主应用组件
 * 
 * 新增功能：
 * - URL 参数支持：通过 ?session=xxx 直接定位到指定会话
 * - 会话切换时自动更新 URL
 * - 支持跨浏览器分享会话链接
 * - 支持订阅进行中的会话实时事件流
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Users,
  Radio,

  Sparkles,
  Github,
  Plus,
  LogOut,
  Brain,
} from 'lucide-react';
import { useStore } from './store';
import { useAgui } from './hooks/useAgui';
import { useAuth } from './auth/AuthProvider';
import { MemoryPanel } from './auth/MemoryPanel';
import { AgentStatus } from './types/agui';
import { TaskInput } from './components/TaskInput';
import { AgentPanorama } from './components/AgentPanorama';
import { RelayStationView } from './components/RelayStation';
import { StreamMessage } from './components/StreamMessage';
import { SessionSidebar } from './components/SessionSidebar';
import { cn } from './utils/cn';
import { fetchSessionLiveState, checkSessionIsLive } from './services/sessionApi';

type TabType = 'agents' | 'relay';

const tabs: { id: TabType; label: string; icon: React.ElementType }[] = [
  { id: 'agents', label: 'Agent 全景', icon: Users },
  { id: 'relay', label: '协作消息', icon: Radio },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('agents');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isLoadingFromUrl, setIsLoadingFromUrl] = useState(true);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showMemoryPanel, setShowMemoryPanel] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);
  
  const { user, logout } = useAuth();
  
  const { 
    status, 
    plan, 
    mode,
    sessionId, 
    createSession, 
    reset 
  } = useStore();
  
  // 使用 useAgui hook 获取订阅功能
  const { subscribeToSession, unsubscribeFromSession } = useAgui();
  
  // 点击外部关闭用户菜单
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setShowUserMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  // 工具函数：更新 URL 参数
  const updateUrlWithSession = useCallback((newSessionId: string | null) => {
    const url = new URL(window.location.href);
    if (newSessionId) {
      url.searchParams.set('session', newSessionId);
    } else {
      url.searchParams.delete('session');
    }
    window.history.replaceState({}, '', url.toString());
  }, []);
  
  // 工具函数：从 URL 获取 session 参数
  const getSessionFromUrl = useCallback(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('session');
  }, []);
  
  // 页面加载时：检查 URL 参数，优先恢复指定会话
  useEffect(() => {
    const urlSessionId = getSessionFromUrl();
    
    if (urlSessionId) {
      // URL 中有 session 参数，尝试加载该会话
      console.log(`[App] Found session in URL: ${urlSessionId}`);
      handleSelectSession(urlSessionId).finally(() => {
        setIsLoadingFromUrl(false);
      });
    } else if (!sessionId) {
      // URL 中没有 session 参数，且当前没有会话，创建新会话
      const newSessionId = `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      createSession(newSessionId);
      // 新建会话不更新 URL（避免刷新后又创建新会话）
      setIsLoadingFromUrl(false);
    } else {
      setIsLoadingFromUrl(false);
    }
  }, []); // 只在首次加载时执行
  
  // 当 sessionId 变化时，同步更新 URL（用户手动切换会话时）
  useEffect(() => {
    if (!isLoadingFromUrl && sessionId) {
      updateUrlWithSession(sessionId);
    }
  }, [sessionId, isLoadingFromUrl, updateUrlWithSession]);

  // 创建新会话
  const handleNewSession = useCallback(() => {
    // 先取消之前的订阅
    unsubscribeFromSession();
    reset();
    // 清除 URL 参数
    updateUrlWithSession(null);
  }, [reset, updateUrlWithSession, unsubscribeFromSession]);

  // 选择会话 - 支持订阅进行中的会话
  const handleSelectSession = useCallback(async (selectedSessionId: string) => {
    // 如果是当前会话，不需要切换
    if (selectedSessionId === sessionId) {
      return;
    }
    
    // 先取消之前的订阅
    unsubscribeFromSession();
    
    // 检查是否在 store 中已有数据（无论是前端临时 ID 还是后端 UUID）
    const { sessions, setActiveSession: setActive } = useStore.getState();
    
    if (sessions[selectedSessionId]) {
      // 会话数据已在 store 中，直接切换，避免覆盖已有数据
      setActive(selectedSessionId);
      updateUrlWithSession(selectedSessionId);
      
      // 如果是运行中的后端会话（非前端临时 ID），也尝试订阅实时更新
      if (!selectedSessionId.startsWith('session-')) {
        const session = sessions[selectedSessionId];
        if (session.status === AgentStatus.RUNNING || session.status === AgentStatus.PLANNING) {
          try {
            // 先确认后端会话确实还在运行，避免对已完成的 session 发起无意义的 subscribe
            const liveCheck = await checkSessionIsLive(selectedSessionId);
            if (liveCheck.success && liveCheck.is_live && liveCheck.has_agent) {
              await subscribeToSession(selectedSessionId);
            } else {
              // 后端已完成，同步前端状态
              console.log(`[App] Session ${selectedSessionId} is no longer live, skipping subscription`);
            }
          } catch (err) {
            console.warn('[App] Re-subscription failed, using cached data:', err);
          }
        }
      }
      return;
    }
    
    // 从后端加载会话
    try {
      // 首先检查会话是否还在运行中（有活跃的 Agent）
      const liveCheck = await checkSessionIsLive(selectedSessionId);
      console.log(`[App] Live check for ${selectedSessionId}:`, liveCheck);
      
      if (liveCheck.success && liveCheck.is_live && liveCheck.has_agent) {
        // 会话还在运行中，使用 SSE 订阅以获取实时更新
        console.log(`[App] Session ${selectedSessionId} is live with active agent, subscribing...`);
        try {
          await subscribeToSession(selectedSessionId);
          updateUrlWithSession(selectedSessionId);
          console.log(`[App] Successfully subscribed to session: ${selectedSessionId}`);
          return;
        } catch (subscribeErr) {
          console.warn('[App] Subscription failed, falling back to snapshot:', subscribeErr);
          // 订阅失败，降级到快照模式
        }
      }
      
      // 历史会话或订阅失败，使用快照模式加载
      console.log(`[App] Loading session snapshot: ${selectedSessionId}`);
      const response = await fetchSessionLiveState(selectedSessionId);
      
      if (response.success && response.data) {
        console.log(`[App] Loaded session: ${selectedSessionId}, is_live: ${response.data.is_live}`);
        
        // 创建会话
        createSession(selectedSessionId);
        
        const { 
          setTask, setStatus, setMode, setPlan, 
          addAgent, addRelayStation, addRelayMessage,
          addMessage,
        } = useStore.getState();
        
        // 设置模式（必须在其他数据之前设置，决定 UI 布局）
        if (response.data.mode) {
          setMode(response.data.mode as 'emergent' | 'direct');
        }
        
        // 设置任务
        if (response.data.task) {
          setTask(response.data.task);
        }
        
        // 设置状态
        if (response.data.status) {
          const statusMap: Record<string, AgentStatus> = {
            active: response.data.is_live ? AgentStatus.RUNNING : AgentStatus.COMPLETED,
            running: AgentStatus.RUNNING,
            completed: AgentStatus.COMPLETED,
            error: AgentStatus.FAILED,
            expired: AgentStatus.FAILED,
            planning: AgentStatus.PLANNING,
          };
          setStatus(statusMap[response.data.status] || AgentStatus.COMPLETED);
        }
        
        // 设置计划
        if (response.data.plan) {
          setPlan(response.data.plan);
        }
        
        // 添加所有 Agent
        for (const agentData of response.data.agents) {
          // 如果是历史数据，状态显示为已完成
          const agentStatus = response.data.is_live 
            ? (agentData.status as AgentStatus)
            : AgentStatus.COMPLETED;
          
          addAgent({
            id: agentData.agent_id,
            name: agentData.name,
            roleName: agentData.role_name,
            roleDescription: agentData.role_description,
            capabilities: agentData.capabilities,
            taskSegment: agentData.task_segment,
            status: agentStatus,
            progress: response.data.is_live ? agentData.progress : 100,
            currentStep: agentData.current_step,
            iterations: agentData.iterations,
            thinking: agentData.thinking,
            workObjective: agentData.work_objective,
            deliverables: agentData.deliverables,
            methodology: agentData.methodology as any,
          });
        }
        
        // 添加所有中继站和消息
        for (const stationData of response.data.relay_stations) {
          addRelayStation({
            id: stationData.station_id,
            name: stationData.name,
            phase: stationData.phase,
            participatingAgents: stationData.participating_agents as any,
            messages: [],
            isActive: stationData.is_active,
          });
          
          // 添加该中继站的消息
          for (const msgData of stationData.messages) {
            addRelayMessage(stationData.station_id, {
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
        if (response.data.messages && response.data.messages.length > 0) {
          console.log(`[App] Loading ${response.data.messages.length} messages for session: ${selectedSessionId}`);
          for (const msgData of response.data.messages) {
            addMessage({
              id: msgData.message_id,
              role: msgData.role as 'user' | 'assistant' | 'system',
              content: msgData.content,
              timestamp: msgData.timestamp,
            });
          }
        }
        
        // 更新 URL
        updateUrlWithSession(selectedSessionId);
        return;
      }
    } catch (err) {
      console.error('Failed to load session:', err);
      
      // 如果是前端临时 ID 且后端没有数据，不要创建空会话
      // 保持在当前会话，让用户知道切换失败
      if (selectedSessionId.startsWith('session-')) {
        console.warn(`[App] Cannot load temporary session ID: ${selectedSessionId}, ignoring`);
        return;
      }
      
      // 对于后端创建的会话（UUID格式），即使加载失败也创建一个占位
      createSession(selectedSessionId);
      updateUrlWithSession(selectedSessionId);
    }
  }, [sessionId, createSession, updateUrlWithSession, subscribeToSession, unsubscribeFromSession]);

  return (
    <div className="min-h-screen bg-dark-950 text-white flex">
      {/* 左侧会话列表面板 - 固定显示 */}
      <SessionSidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
      />
      
      {/* 主内容区 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 顶部导航 */}
        <header className="h-16 border-b border-dark-800 bg-dark-900/80 backdrop-blur-sm sticky top-0 z-40">
          <div className="h-full px-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-purple-600 
                              flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-lg">Agent Swarm</h1>
                <p className="text-xs text-dark-400">角色涌现 × 3D编排 × AG-UI</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* 当前会话 ID */}
              {sessionId && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-dark-800 border border-dark-700">
                  <span className="text-xs text-dark-500 font-mono">
                    会话: {sessionId.slice(0, 8)}...
                  </span>
                </div>
              )}
              
              {/* 状态指示 */}
              {status !== AgentStatus.PENDING && (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-dark-800 border border-dark-700">
                  <span className={cn(
                    'w-2 h-2 rounded-full',
                    status === AgentStatus.RUNNING && 'bg-green-500 animate-pulse',
                    status === AgentStatus.PLANNING && 'bg-purple-500 animate-pulse',
                    status === AgentStatus.COMPLETED && 'bg-cyan-500',
                    status === AgentStatus.FAILED && 'bg-red-500',
                    status === AgentStatus.CANCELLED && 'bg-orange-500',
                  )} />
                  <span className="text-sm text-dark-300">
                    {status === AgentStatus.RUNNING && '执行中'}
                    {status === AgentStatus.PLANNING && '规划中'}
                    {status === AgentStatus.COMPLETED && '已完成'}
                    {status === AgentStatus.FAILED && '失败'}
                    {status === AgentStatus.CANCELLED && '已终止'}
                  </span>
                </div>
              )}

              {/* 新建会话按钮 */}
              <button 
                onClick={handleNewSession}
                className="p-2 rounded-lg hover:bg-dark-800 transition-colors"
                title="新建会话"
              >
                <Plus className="w-5 h-5 text-dark-400" />
              </button>
              
              <a
                href="https://github.com/jlulxy/agent-swarm"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg hover:bg-dark-800 transition-colors"
              >
                <Github className="w-5 h-5 text-dark-400" />
              </a>

              {/* 用户头像 + 下拉菜单 */}
              {user && (
                <div className="relative" ref={userMenuRef}>
                  <button
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-dark-800 transition-colors cursor-pointer"
                  >
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#6366F1] to-[#8B5CF6] flex items-center justify-center text-xs font-bold text-white shadow-md shadow-[#6366F1]/20">
                      {(user.display_name || user.username).charAt(0).toUpperCase()}
                    </div>
                    <span className="text-sm text-dark-300 hidden lg:block max-w-[100px] truncate">
                      {user.display_name || user.username}
                    </span>
                  </button>
                  
                  <AnimatePresence>
                    {showUserMenu && (
                      <motion.div
                        initial={{ opacity: 0, y: -5, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -5, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 top-full mt-2 w-52 rounded-xl border border-white/[0.08] overflow-hidden z-50"
                        style={{
                          background: 'rgba(30, 41, 59, 0.95)',
                          backdropFilter: 'blur(16px)',
                          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.4)',
                        }}
                      >
                        <div className="px-4 py-3 border-b border-white/[0.06]">
                          <p className="text-sm font-medium text-[#F8FAFC] truncate">{user.display_name || user.username}</p>
                          <p className="text-xs text-[#64748B] truncate">@{user.username}</p>
                        </div>
                        <div className="py-1">
                          <button
                            onClick={() => {
                              setShowUserMenu(false);
                              setShowMemoryPanel(true);
                            }}
                            className="w-full px-4 py-2.5 text-left text-sm text-[#CBD5E1] hover:bg-white/[0.06] transition-colors flex items-center gap-2 cursor-pointer"
                          >
                            <Brain className="w-4 h-4 text-violet-400" />
                            我的记忆
                          </button>
                          <button
                            onClick={() => {
                              setShowUserMenu(false);
                              logout();
                            }}
                            className="w-full px-4 py-2.5 text-left text-sm text-[#EF4444] hover:bg-[#EF4444]/10 transition-colors flex items-center gap-2 cursor-pointer"
                          >
                            <LogOut className="w-4 h-4" />
                            退出登录
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* 主内容区 */}
        <main className="flex-1 flex min-h-0">
          {/* 左侧主区域 */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* 任务信息（涌现模式才显示） */}
            {plan && mode !== 'direct' && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 border-b border-dark-700 bg-dark-900/30 flex-shrink-0"
              >
                <h3 className="text-sm font-medium text-dark-300 mb-2">任务分析</h3>
                <p className="text-sm text-dark-400 line-clamp-2">{plan.analysis}</p>
                <div className="mt-2 flex items-center gap-4 text-xs text-dark-500">
                  <span>{plan.totalAgents} 个 Agent</span>
                  <span>{plan.phases.length} 个阶段</span>
                  <span>预计 {Math.round(plan.estimatedDuration / 60)} 分钟</span>
                </div>
              </motion.div>
            )}

            {/* 消息流（占满中间区域） */}
            <div className="flex-1 overflow-hidden">
              <StreamMessage />
            </div>

            {/* 任务输入（固定在底部） */}
            <TaskInput />
          </div>

          {/* 右侧面板（涌现模式才显示） */}
          {mode !== 'direct' && (
          <div className="w-[400px] border-l border-dark-800 bg-dark-900/50 flex flex-col">
            {/* Tab 栏 */}
            <div className="flex border-b border-dark-800">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={cn(
                      'flex-1 py-3 px-2 text-sm font-medium transition-colors',
                      'flex items-center justify-center gap-1.5',
                      activeTab === tab.id
                        ? 'text-primary-400 border-b-2 border-primary-500 bg-primary-500/5'
                        : 'text-dark-400 hover:text-dark-200 hover:bg-dark-800/50'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="hidden lg:inline">{tab.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Tab 内容 */}
            <div className="flex-1 overflow-hidden">
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.15 }}
                  className="h-full"
                >
                  {activeTab === 'agents' && <AgentPanorama />}
                  {activeTab === 'relay' && <RelayStationView />}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
          )}
        </main>
      </div>

      {/* 记忆面板 */}
      <MemoryPanel open={showMemoryPanel} onClose={() => setShowMemoryPanel(false)} />
    </div>
  );
}
