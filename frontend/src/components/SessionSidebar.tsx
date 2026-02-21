/**
 * Session 侧边栏组件 - 固定面板模式
 * 
 * 显示所有会话列表，支持：
 * - 查看历史会话
 * - 切换到指定会话
 * - 创建新会话
 * - 关闭会话
 * - 折叠/展开
 * 
 * 刷新策略（事件驱动 + 智能轮询）：
 * 1. 监听 store 中的会话状态变化，立即刷新
 * 2. 有活跃会话时：每 15 秒刷新一次
 * 3. 无活跃会话时：每 60 秒刷新一次（低频后备）
 */

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, 
  RefreshCw, 
  ChevronLeft, 
  ChevronRight,
  X,
  Clock,
  Zap,
  CheckCircle,
  AlertCircle,
  MessageSquare,
  Wifi,
  Brain
} from 'lucide-react';
import { useStore } from '../store';
import { fetchSessions, closeSession as closeSessionApi, SessionData } from '../services/sessionApi';
import { cn } from '../utils/cn';

interface SessionSidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  onNewSession: () => void;
  onSelectSession: (sessionId: string) => void;
}

export const SessionSidebar: React.FC<SessionSidebarProps> = ({
  collapsed,
  onToggle,
  onNewSession,
  onSelectSession,
}) => {
  const [sessions, setSessions] = useState<SessionData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [lastRefreshTime, setLastRefreshTime] = useState<Date | null>(null);
  
  // 用于追踪上一次 store 状态，避免不必要的刷新
  const prevStoreStateRef = useRef<string>('');
  
  const { activeSessionId, sessions: storeSessions, status: activeStatus, agents } = useStore();

  // 加载会话列表 - 合并数据库和内存中的会话
  const loadSessions = useCallback(async (silent: boolean = false) => {
    if (!silent) {
      setLoading(true);
    }
    setError(null);
    
    try {
      // 同时从数据库和内存获取
      const [dbResponse, memResponse] = await Promise.all([
        fetchSessions({ source: 'db', status: statusFilter || undefined, limit: 100 }),
        fetchSessions({ source: 'memory', limit: 100 })
      ]);
      
      // 合并会话列表，内存中的会话优先
      const dbSessions = dbResponse.success ? dbResponse.data.sessions : [];
      const memSessions = memResponse.success ? memResponse.data.sessions : [];
      
      // 使用 Map 来合并，内存中的会话覆盖数据库中的
      const sessionMap = new Map<string, SessionData>();
      
      // 先添加数据库中的会话
      for (const session of dbSessions) {
        sessionMap.set(session.session_id, session);
      }
      
      // 再添加/覆盖内存中的会话（内存中的更新，优先显示）
      for (const session of memSessions) {
        const existing = sessionMap.get(session.session_id);
        sessionMap.set(session.session_id, {
          ...session,
          // 关键：has_agent 为 true 才表示真正在运行中
          // is_active_in_memory 可能只是内存缓存还存在，但 Agent 可能已经结束
          is_active_in_memory: session.has_agent === true,
          // 如果数据库中已存在，保留数据库中的创建时间
          created_at: existing?.created_at || session.created_at,
        });
      }
      
      // 转换为数组并按创建时间排序
      let allSessions = Array.from(sessionMap.values());
      allSessions.sort((a, b) => 
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      
      // 应用状态筛选
      if (statusFilter) {
        allSessions = allSessions.filter(s => s.status === statusFilter);
      }
      
      setSessions(allSessions);
      setLastRefreshTime(new Date());
      
      // 使用内存统计更准确
      const stats = memResponse.success ? memResponse.data.stats : dbResponse.data?.stats;
      if (dbResponse.success && dbResponse.data?.stats) {
        stats.db_total_sessions = dbResponse.data.stats.db_total_sessions ?? dbSessions.length;
      }
      setStats(stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions');
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [statusFilter]);

  // 首次加载
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // 事件驱动刷新：监听 store 中的状态变化
  useEffect(() => {
    // 构建当前状态的快照字符串
    const agentCount = Object.keys(agents).length;
    const sessionIds = Object.keys(storeSessions).join(',');
    const currentStateSnapshot = `${activeSessionId}:${activeStatus}:${agentCount}:${sessionIds}`;
    
    // 如果状态发生变化，触发刷新
    if (prevStoreStateRef.current && prevStoreStateRef.current !== currentStateSnapshot) {
      console.log('[SessionSidebar] Store state changed, refreshing...');
      loadSessions(true);  // 静默刷新，不显示 loading
    }
    
    prevStoreStateRef.current = currentStateSnapshot;
  }, [activeSessionId, activeStatus, agents, storeSessions, loadSessions]);

  // 智能轮询：根据是否有活跃会话调整间隔
  useEffect(() => {
    // 判断是否有活跃的运行中会话
    const hasActiveRunningSessions = sessions.some(
      s => s.is_active_in_memory && s.status === 'active'
    );
    
    // 根据状态选择轮询间隔
    // - 有活跃会话时：15 秒
    // - 无活跃会话时：60 秒（低频后备）
    const pollInterval = hasActiveRunningSessions ? 15000 : 60000;
    
    const interval = setInterval(() => {
      loadSessions(true);  // 静默刷新
    }, pollInterval);
    
    return () => clearInterval(interval);
  }, [sessions, loadSessions]);

  // 关闭会话
  const handleCloseSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!confirm('确定要关闭此会话吗？')) {
      return;
    }
    
    try {
      await closeSessionApi(sessionId);
      // 从本地状态移除
      setSessions(prev => prev.filter(s => s.session_id !== sessionId));
      // 如果关闭的是当前会话，清理 store
      if (sessionId === activeSessionId) {
        const { closeSession } = useStore.getState();
        closeSession(sessionId);
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to close session');
    }
  };

  // 获取状态图标
  const getStatusIcon = (status: string, isInMemory: boolean) => {
    if (isInMemory && status === 'active') {
      return <Zap className="w-3 h-3 text-green-400" />;
    }
    switch (status) {
      case 'active':
        // 不在内存中的 active 会话，显示为历史
        return <Clock className="w-3 h-3 text-blue-400" />;
      case 'completed':
        return <CheckCircle className="w-3 h-3 text-blue-400" />;
      case 'error':
      case 'expired':
        return <AlertCircle className="w-3 h-3 text-red-400" />;
      default:
        return <Clock className="w-3 h-3 text-gray-400" />;
    }
  };

  // 获取状态标签
  const getStatusLabel = (status: string, isInMemory: boolean) => {
    if (isInMemory && status === 'active') {
      return '运行中';
    }
    switch (status) {
      case 'active':
        // 不在内存中的 active 会话
        return '历史会话';
      case 'completed':
        return '已完成';
      case 'expired':
        return '已过期';
      case 'error':
        return '错误';
      default:
        return status;
    }
  };

  // 格式化时间
  const formatTime = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
    
    return date.toLocaleDateString('zh-CN', {
      month: 'numeric',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // 截断任务文本
  const truncateTask = (task: string, maxLength: number = 30) => {
    if (!task) return '(无任务)';
    return task.length > maxLength ? task.slice(0, maxLength) + '...' : task;
  };

  // 折叠状态下的简约视图
  if (collapsed) {
    return (
      <div className="w-12 bg-dark-900 border-r border-dark-800 flex flex-col items-center py-4">
        <button
          onClick={onToggle}
          className="p-2 rounded-lg hover:bg-dark-800 transition-colors mb-4"
          title="展开会话列表"
        >
          <ChevronRight className="w-5 h-5 text-dark-400" />
        </button>
        
        <button
          onClick={onNewSession}
          className="p-2 rounded-lg hover:bg-dark-800 transition-colors mb-4"
          title="新建会话"
        >
          <Plus className="w-5 h-5 text-primary-400" />
        </button>
        
        {/* 会话数量指示 */}
        <div className="mt-auto">
          <div className="w-8 h-8 rounded-full bg-dark-800 flex items-center justify-center">
            <span className="text-xs text-dark-300">{sessions.length}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-72 bg-dark-900 border-r border-dark-800 flex flex-col h-screen">
      {/* 头部 */}
      <div className="p-4 border-b border-dark-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white">会话列表</h2>
          <button
            onClick={onToggle}
            className="p-1.5 rounded-lg hover:bg-dark-800 transition-colors"
            title="折叠"
          >
            <ChevronLeft className="w-4 h-4 text-dark-400" />
          </button>
        </div>
        
        {/* 新建会话按钮 */}
        <button
          onClick={onNewSession}
          className="w-full py-2 px-4 bg-primary-600 hover:bg-primary-700 text-white rounded-lg 
                     flex items-center justify-center gap-2 transition-colors text-sm font-medium"
        >
          <Plus className="w-4 h-4" />
          新建会话
        </button>
        
        {/* 状态筛选 */}
        <div className="mt-3">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-full px-3 py-1.5 bg-dark-800 text-dark-200 rounded border border-dark-700 
                       text-xs focus:outline-none focus:border-primary-500"
          >
            <option value="">全部状态</option>
            <option value="active">进行中</option>
            <option value="completed">已完成</option>
            <option value="expired">已过期</option>
          </select>
        </div>
      </div>
      
      {/* 统计信息 */}
      {stats && (
        <div className="px-4 py-2 bg-dark-850 border-b border-dark-800">
          <div className="flex justify-between items-center">
            <div className="flex gap-3 text-xs text-dark-400">
              <span>
                总计: <span className="text-dark-200">{stats.db_total_sessions ?? stats.active_sessions}</span>
              </span>
              <span>
                活跃: <span className="text-green-400">{stats.db_active_sessions ?? stats.active_sessions}</span>
              </span>
            </div>
            {/* 实时同步状态指示 */}
            {sessions.some(s => s.is_active_in_memory) && (
              <div className="flex items-center gap-1 text-[10px] text-green-400">
                <Wifi className="w-3 h-3" />
                <span>已同步</span>
              </div>
            )}
          </div>
          {/* 上次刷新时间 */}
          {lastRefreshTime && (
            <div className="mt-1 text-[10px] text-dark-500">
              更新于 {formatTime(lastRefreshTime.toISOString())}
            </div>
          )}
        </div>
      )}
      
      {/* 会话列表 */}
      <div className="flex-1 overflow-y-auto">
        {loading && sessions.length === 0 && (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-5 h-5 text-primary-400 animate-spin" />
          </div>
        )}
        
        {error && (
          <div className="p-4 text-center">
            <p className="text-red-400 text-xs">{error}</p>
            <button 
              onClick={() => loadSessions()}
              className="mt-2 text-primary-400 hover:text-primary-300 text-xs"
            >
              重试
            </button>
          </div>
        )}
        
        {!loading && !error && sessions.length === 0 && (
          <div className="p-6 text-center">
            <MessageSquare className="w-10 h-10 mx-auto text-dark-600 mb-2" />
            <p className="text-dark-400 text-sm">暂无会话</p>
            <p className="text-dark-500 text-xs mt-1">点击"新建会话"开始</p>
          </div>
        )}
        
        <AnimatePresence>
          {sessions.map((session, index) => {
            const isActive = session.session_id === activeSessionId;
            // 只有后端明确标记为 is_active_in_memory 才表示真正在运行
            // 前端 store 中有数据不代表后端有活跃的 Agent
            const isLiveInBackend = session.is_active_in_memory === true;
            
            return (
              <motion.div
                key={session.session_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ delay: index * 0.02 }}
                onClick={() => onSelectSession(session.session_id)}
                className={cn(
                  'p-3 border-b border-dark-800 cursor-pointer transition-all duration-150',
                  'hover:bg-dark-800/50',
                  isActive && 'bg-primary-900/20 border-l-2 border-l-primary-500'
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    {/* 状态和标签 */}
                    <div className="flex items-center gap-1.5 mb-1">
                      {getStatusIcon(session.status, isLiveInBackend)}
                      <span className={cn(
                        'text-xs',
                        isLiveInBackend && session.status === 'active' ? 'text-green-400' : 'text-dark-400'
                      )}>
                        {getStatusLabel(session.status, isLiveInBackend)}
                      </span>
                      {/* 模式标签 */}
                      {session.mode === 'direct' ? (
                        <span className="text-[10px] px-1 py-0.5 rounded bg-cyan-900/40 text-cyan-400 flex items-center gap-0.5">
                          <Zap className="w-2.5 h-2.5" />
                          普通
                        </span>
                      ) : (
                        <span className="text-[10px] px-1 py-0.5 rounded bg-purple-900/40 text-purple-400 flex items-center gap-0.5">
                          <Brain className="w-2.5 h-2.5" />
                          涌现
                        </span>
                      )}
                      {isLiveInBackend && (
                        <span className="text-[10px] px-1 py-0.5 bg-green-900/50 text-green-400 rounded">
                          实时
                        </span>
                      )}
                    </div>
                    
                    {/* 任务描述 */}
                    <p className="text-sm text-dark-200 truncate">
                      {truncateTask(session.task)}
                    </p>
                    
                    {/* 会话 ID 和时间 */}
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] text-dark-500 font-mono">
                        {session.session_id.slice(0, 8)}
                      </span>
                      <span className="text-[10px] text-dark-500">
                        {formatTime(session.created_at)}
                      </span>
                    </div>
                  </div>
                  
                  {/* 关闭按钮 */}
                  <button
                    onClick={(e) => handleCloseSession(session.session_id, e)}
                    className="p-1 text-dark-500 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                    title="关闭会话"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
      
      {/* 底部刷新按钮 */}
      <div className="p-3 border-t border-dark-800">
        <button
          onClick={() => loadSessions(false)}
          disabled={loading}
          className="w-full py-1.5 px-3 bg-dark-800 hover:bg-dark-700 text-dark-300 rounded-lg 
                     flex items-center justify-center gap-2 transition-colors disabled:opacity-50 text-xs"
        >
          <RefreshCw className={cn('w-3.5 h-3.5', loading && 'animate-spin')} />
          {loading ? '刷新中...' : '手动刷新'}
        </button>
      </div>
    </div>
  );
};

export default SessionSidebar;
