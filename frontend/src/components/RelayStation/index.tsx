/**
 * 中继站可视化组件
 * 
 * 展示 Agent 间的信息交换节点
 * 升级版：支持人工干预消息的特殊展示 + 消息查看状态
 */

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Radio, ArrowRight, Zap, UserCircle, AlertTriangle, Bell, Eye, CheckCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { getAuthHeader } from '../../auth/api';
import { useStore } from '../../store';
import { cn } from '../../utils/cn';

// 消息类型配置
const relayTypeConfig: Record<string, { bg: string; text: string; label: string; icon?: string }> = {
  // 发现类
  discovery: { bg: 'bg-green-500/20', text: 'text-green-400', label: '发现', icon: '🔍' },
  insight: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', label: '洞察', icon: '🎯' },
  
  // 对齐/协作类
  alignment_request: { bg: 'bg-blue-500/20', text: 'text-blue-400', label: '请求对齐', icon: '🔄' },
  alignment_response: { bg: 'bg-cyan-500/20', text: 'text-cyan-400', label: '响应对齐', icon: '✅' },
  alignment: { bg: 'bg-blue-500/20', text: 'text-blue-400', label: '对齐', icon: '🔄' },  // 向后兼容
  
  // 建议/反馈类
  suggestion: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', label: '建议', icon: '💡' },
  question: { bg: 'bg-amber-500/20', text: 'text-amber-400', label: '求助', icon: '❓' },
  confirmation: { bg: 'bg-teal-500/20', text: 'text-teal-400', label: '确认', icon: '✔️' },
  
  // 状态类
  checkpoint: { bg: 'bg-purple-500/20', text: 'text-purple-400', label: '检查点', icon: '📍' },
  correction: { bg: 'bg-orange-500/20', text: 'text-orange-400', label: '纠偏', icon: '⚠️' },
  completion: { bg: 'bg-cyan-500/20', text: 'text-cyan-400', label: '完成', icon: '🏁' },
  
  // 干预类
  human_intervention: { bg: 'bg-red-500/20', text: 'text-red-400', label: '人工干预', icon: '👤' },
};

// 消息查看状态组件
function MessageViewStatus({ viewedBy, acknowledgedBy, agents }: { 
  viewedBy?: string[]; 
  acknowledgedBy?: string[];
  agents: any[];
}) {
  const viewedCount = viewedBy?.length || 0;
  const acknowledgedCount = acknowledgedBy?.length || 0;
  
  if (viewedCount === 0 && acknowledgedCount === 0) {
    return (
      <span className="text-xs text-dark-500 flex items-center gap-1">
        <Eye className="w-3 h-3" />
        未查看
      </span>
    );
  }
  
  // 获取 agent 名称
  const getAgentName = (agentId: string) => {
    const agent = agents.find(a => a.id === agentId);
    return agent?.name || agentId.slice(0, 8);
  };
  
  return (
    <div className="flex flex-wrap items-center gap-2 mt-1">
      {viewedCount > 0 && (
        <div className="flex items-center gap-1">
          <Eye className="w-3 h-3 text-blue-400" />
          <span className="text-xs text-blue-400">
            {viewedCount} 已查看
          </span>
          {viewedCount <= 3 && viewedBy && (
            <span className="text-xs text-dark-500">
              ({viewedBy.map(id => getAgentName(id)).join(', ')})
            </span>
          )}
        </div>
      )}
      {acknowledgedCount > 0 && (
        <div className="flex items-center gap-1">
          <CheckCircle className="w-3 h-3 text-green-400" />
          <span className="text-xs text-green-400">
            {acknowledgedCount} 已确认
          </span>
        </div>
      )}
    </div>
  );
}

// 人工干预消息组件
function InterventionMessage({ msg, agents }: { msg: any; agents: any[] }) {
  // 解析干预类型
  const interventionType = msg.metadata?.intervention_type || '';
  const priority = msg.metadata?.priority || 5;
  const scope = msg.metadata?.scope || 'single';
  
  // 优先级颜色
  const priorityColor = priority >= 8 ? 'text-red-400' : priority >= 5 ? 'text-yellow-400' : 'text-blue-400';
  
  // 查看状态
  const viewedBy = msg.viewedBy || [];
  const acknowledgedBy = msg.acknowledgedBy || [];
  const hasBeenViewed = viewedBy.length > 0;
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        "p-3 rounded-lg border relative overflow-hidden",
        hasBeenViewed
          ? "bg-red-500/5 border-red-500/20"
          : "bg-red-500/10 border-red-500/30"
      )}
    >
      {/* 闪烁指示条 - 只有未被查看时才闪烁 */}
      <div className={cn(
        "absolute top-0 left-0 w-1 h-full bg-red-500",
        !hasBeenViewed && "animate-pulse"
      )} />
      
      {/* 头部 */}
      <div className="flex items-center gap-2 mb-2 pl-2">
        <UserCircle className="w-4 h-4 text-red-400" />
        <span className="text-xs font-medium text-red-300">
          {msg.sourceAgentName}
        </span>
        <div className="flex-1" />
        <span className={cn('text-xs font-bold', priorityColor)}>
          P{priority}
        </span>
        <AlertTriangle className="w-3 h-3 text-red-400" />
      </div>
      
      {/* 干预类型和范围 */}
      <div className="flex items-center gap-2 mb-2 pl-2">
        <span className="px-2 py-0.5 text-xs rounded bg-red-500/30 text-red-300">
          {interventionType || '干预'}
        </span>
        <span className="text-xs text-dark-400">
          范围: {scope === 'all' ? '全部' : scope === 'broadcast' ? '广播' : scope === 'selected' ? '选定' : '单个'}
        </span>
      </div>
      
      {/* 消息内容 */}
      <div className="pl-2 text-xs text-dark-200 whitespace-pre-wrap line-clamp-4">
        {msg.content}
      </div>
      
      {/* 目标 Agent */}
      {msg.targetAgentIds && msg.targetAgentIds.length > 0 && (
        <div className="mt-2 pl-2 flex items-center gap-1">
          <ArrowRight className="w-3 h-3 text-red-400" />
          <span className="text-xs text-dark-400">
            目标: {msg.targetAgentIds.join(', ')}
          </span>
        </div>
      )}
      
      {/* 查看状态 */}
      <div className="mt-2 pl-2 pt-2 border-t border-red-500/20">
        <MessageViewStatus 
          viewedBy={viewedBy} 
          acknowledgedBy={acknowledgedBy}
          agents={agents}
        />
      </div>
    </motion.div>
  );
}

// 普通中继消息组件
function RelayMessageItem({ msg, agents }: { msg: any; agents: any[] }) {
  const config = relayTypeConfig[msg.relayType] || { bg: 'bg-dark-700', text: 'text-dark-300', label: msg.relayType, icon: '📨' };
  const [isExpanded, setIsExpanded] = useState(false);
  
  const viewedBy = msg.viewedBy || [];
  const acknowledgedBy = msg.acknowledgedBy || [];
  
  // 获取 agent 名称
  const getAgentName = (agentId: string) => {
    const agent = agents.find(a => a.id === agentId);
    return agent?.name || agentId.slice(0, 8);
  };
  
  // 判断是否是需要响应的消息类型
  const isRequestType = ['alignment_request', 'question'].includes(msg.relayType);
  const isResponseType = ['alignment_response', 'confirmation'].includes(msg.relayType);
  
  // 检查内容是否较长（超过 100 字符）
  const isLongContent = msg.content && msg.content.length > 100;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className={cn(
        "p-2 rounded-lg bg-dark-900/50 border",
        isRequestType ? "border-blue-500/30" : 
        isResponseType ? "border-green-500/30" : 
        "border-dark-700"
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="text-sm">{config.icon}</span>
        <span className="text-xs font-medium text-dark-300">
          {msg.sourceAgentName}
        </span>
        {msg.targetAgentIds && msg.targetAgentIds.length > 0 ? (
          <>
            <ArrowRight className="w-3 h-3 text-dark-500" />
            <span className="text-xs text-dark-400">
              {msg.targetAgentIds.map((id: string) => getAgentName(id)).join(', ')}
            </span>
          </>
        ) : (
          <span className="text-xs text-dark-500">广播</span>
        )}
        {msg.importance > 0.7 && (
          <Zap className="w-3 h-3 text-yellow-500" />
        )}
      </div>
      
      {/* 请求对齐消息 - 结构化展示 */}
      {isRequestType && (
        <div className="mb-1 p-1.5 rounded bg-blue-500/10 border border-blue-500/20">
          <div className="flex items-center gap-1 text-xs text-blue-400 mb-1">
            <span>📋 对齐目标:</span>
            {msg.targetAgentIds && msg.targetAgentIds.length > 0 ? (
              <span className="font-medium">
                {msg.targetAgentIds.map((id: string) => getAgentName(id)).join(', ')}
              </span>
            ) : (
              <span className="font-medium">全部 Agent ({agents.length} 个)</span>
            )}
          </div>
        </div>
      )}
      
      {/* 消息内容 - 支持展开/收起 */}
      <div 
        className={cn(
          "text-xs text-dark-300 whitespace-pre-wrap",
          !isExpanded && isLongContent && "line-clamp-3"
        )}
      >
        {msg.content}
      </div>
      
      {/* 展开/收起按钮 */}
      {isLongContent && (
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-dark-500 hover:text-dark-300 mt-1"
        >
          {isExpanded ? '收起 ▲' : '展开全部 ▼'}
        </button>
      )}
      
      <div className="mt-1 flex items-center justify-between gap-2 flex-wrap">
        <span className={cn('px-1.5 py-0.5 text-xs rounded flex items-center gap-1', config.bg, config.text)}>
          {config.label}
          {isRequestType && <span className="text-[10px] opacity-70">待响应</span>}
        </span>
        {/* 查看状态 - 始终显示谁查看了 */}
        {viewedBy.length > 0 && (
          <div className="flex items-center gap-1 flex-wrap">
            <Eye className="w-3 h-3 text-blue-400" />
            <span className="text-xs text-dark-400">
              {viewedBy.length} 已查看
            </span>
            <span className="text-xs text-dark-500">
              ({viewedBy.slice(0, 5).map((id: string) => getAgentName(id)).join(', ')}
              {viewedBy.length > 5 && ` +${viewedBy.length - 5}`})
            </span>
          </div>
        )}
        {/* 响应状态 */}
        {acknowledgedBy.length > 0 && (
          <div className="flex items-center gap-1">
            <CheckCircle className="w-3 h-3 text-green-400" />
            <span className="text-xs text-green-400">
              {acknowledgedBy.length} 已响应
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// 普通消息列表组件 - 支持折叠展开
function RegularMessagesSection({ messages, agents }: { 
  messages: any[]; 
  agents: any[];
  stationId: string;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // 显示最新5条消息，旧消息可折叠展开
  const VISIBLE_COUNT = 5;
  const totalCount = messages.length;
  const hasOlderMessages = totalCount > VISIBLE_COUNT;
  
  // 最新的消息（始终显示）
  const latestMessages = messages.slice(-VISIBLE_COUNT);
  // 旧消息（可折叠）
  const olderMessages = messages.slice(0, -VISIBLE_COUNT);
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-dark-400">
          信息交换: 
          <span className="ml-1 text-dark-500">
            共 {totalCount} 条
          </span>
        </p>
      </div>
      
      {/* 折叠的旧消息区域 */}
      {hasOlderMessages && (
        <div className="mb-2">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={cn(
              "w-full flex items-center justify-center gap-2 py-1.5 px-3 rounded-lg text-xs transition-all",
              "bg-dark-800/50 hover:bg-dark-700/50 border border-dark-700/50",
              isExpanded ? "text-orange-400" : "text-dark-400 hover:text-dark-300"
            )}
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-3 h-3" />
                收起历史消息
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3" />
                展开 {olderMessages.length} 条历史消息
              </>
            )}
          </button>
          
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="space-y-2 mt-2 pl-2 border-l-2 border-dark-700/50">
                  {olderMessages.map((msg) => (
                    <RelayMessageItem key={msg.id} msg={msg} agents={agents} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
      
      {/* 最新消息（始终显示） */}
      <div className="space-y-2">
        {latestMessages.map((msg) => (
          <RelayMessageItem key={msg.id} msg={msg} agents={agents} />
        ))}
      </div>
    </div>
  );
}

export function RelayStationView() {
  const { relayStations, agents, updateRelayMessage, sessionId } = useStore();

  const stationList = Object.values(relayStations);
  const activeStation = stationList.find(s => s.isActive);
  const agentList = Object.values(agents);
  
  // 定期拉取中继站消息查看状态
  // 【会话隔离】只拉取当前会话的中继历史
  useEffect(() => {
    if (!sessionId) return;
    
    let isMounted = true;
    
    const fetchRelayHistory = async () => {
      try {
        // 【重要】添加 session_id 参数，确保只获取当前会话的中继消息
        const response = await fetch(`/api/relay/${sessionId}/history?limit=100`, {
          headers: { ...getAuthHeader() },
        });
        if (!response.ok) return;
        const result = await response.json();
        const messages = result?.data?.messages || [];
        
        if (!isMounted) return;
        
        messages.forEach((msg: any) => {
          updateRelayMessage(msg.id, {
            viewedBy: msg.viewed_by || [],
            acknowledgedBy: msg.acknowledged_by || [],
            viewedTimestamps: msg.viewed_timestamps || {},
          });
        });
      } catch (error) {
        // 忽略网络错误
      }
    };
    
    fetchRelayHistory();
    const intervalId = window.setInterval(fetchRelayHistory, 3000);
    
    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [updateRelayMessage, sessionId]);  // 添加 sessionId 依赖
  
  // 计算人工干预消息数量
  const interventionCount = stationList.reduce(
    (count, station) => count + station.messages.filter(m => m.relayType === 'human_intervention').length,
    0
  );
  
  // 计算未查看的干预消息数量
  const unviewedInterventionCount = stationList.reduce(
    (count, station) => count + station.messages.filter(
      m => m.relayType === 'human_intervention' && (!m.viewedBy || m.viewedBy.length === 0)
    ).length,
    0
  );

  return (
    <div className="h-full flex flex-col">
      {/* 头部 */}
      <div className="flex-shrink-0 p-4 border-b border-dark-700">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Radio className="w-5 h-5 text-orange-400" />
          中继站
          {activeStation && (
            <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-orange-500/20 text-orange-400">
              活跃
            </span>
          )}
          {interventionCount > 0 && (
            <span className={cn(
              "ml-auto flex items-center gap-1 px-2 py-0.5 text-xs rounded-full",
              unviewedInterventionCount > 0 
                ? "bg-red-500/30 text-red-400 animate-pulse" 
                : "bg-red-500/20 text-red-400"
            )}>
              <Bell className="w-3 h-3" />
              {interventionCount} 干预
              {unviewedInterventionCount > 0 && (
                <span className="ml-1">({unviewedInterventionCount} 未读)</span>
              )}
            </span>
          )}
        </h2>
      </div>

      {/* 中继站列表 */}
      <div className="flex-1 overflow-y-auto p-4">
        {stationList.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Radio className="w-12 h-12 text-dark-600 mx-auto mb-3" />
              <p className="text-dark-400">等待中继站开启...</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <AnimatePresence mode="popLayout">
              {stationList.map((station) => {
                // 分离人工干预消息和普通消息
                const interventionMsgs = station.messages.filter(m => m.relayType === 'human_intervention');
                const regularMsgs = station.messages.filter(m => m.relayType !== 'human_intervention');
                
                return (
                  <motion.div
                    key={station.id}
                    layout
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className={cn(
                      'p-4 rounded-xl border transition-all',
                      station.isActive
                        ? 'bg-orange-500/10 border-orange-500/30'
                        : 'bg-dark-800/50 border-dark-700'
                    )}
                  >
                    {/* 中继站头部 */}
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <div className={cn(
                          'w-2 h-2 rounded-full',
                          station.isActive ? 'bg-orange-500 animate-pulse' : 'bg-dark-500'
                        )} />
                        <h3 className="font-medium text-white">{station.name}</h3>
                      </div>
                      <div className="flex items-center gap-2">
                        {interventionMsgs.length > 0 && (
                          <span className="px-1.5 py-0.5 text-xs rounded bg-red-500/20 text-red-400">
                            {interventionMsgs.length} 干预
                          </span>
                        )}
                        <span className="text-xs text-dark-400">
                          阶段 {station.phase}
                        </span>
                      </div>
                    </div>

                    {/* 参与的 Agent - 仅当有参与 Agent 时显示 */}
                    {station.participatingAgents && station.participatingAgents.length > 0 && (
                      <div className="mb-3">
                        <p className="text-xs text-dark-400 mb-2">参与 Agent:</p>
                        <div className="flex flex-wrap gap-1">
                          {station.participatingAgents.map((agent) => (
                            <span
                              key={agent.id}
                              className="px-2 py-0.5 text-xs rounded-full bg-dark-700 text-dark-300"
                            >
                              {agent.name}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* 人工干预消息 - 突出显示 */}
                    {interventionMsgs.length > 0 && (
                      <div className="mb-3 space-y-2">
                        <p className="text-xs text-red-400 font-medium flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          人工干预:
                        </p>
                        {interventionMsgs.slice(-3).map((msg) => (
                          <InterventionMessage key={msg.id} msg={msg} agents={agentList} />
                        ))}
                      </div>
                    )}

                    {/* 普通消息列表 - 支持折叠展开 */}
                    {regularMsgs.length > 0 && (
                      <RegularMessagesSection 
                        messages={regularMsgs} 
                        agents={agentList} 
                        stationId={station.id}
                      />
                    )}
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
