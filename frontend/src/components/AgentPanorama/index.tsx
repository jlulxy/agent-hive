/**
 * Agent 全景图组件
 * 
 * 实时显示所有 Subagent 的状态、进度
 */

import { useState, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import { Users, Activity, Zap } from 'lucide-react';
import { useStore } from '../../store';
import { useAgui } from '../../hooks/useAgui';
import { AgentCard } from './AgentCard';
import { AgentStatus } from '../../types/agui';

export function AgentPanorama() {
  const { agents, sessionId } = useStore();
  const { intervene } = useAgui();
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  const agentList = Object.values(agents);
  
  // 【会话隔离】当会话切换时，清除选中的 Agent
  useEffect(() => {
    if (selectedAgentId && !agents[selectedAgentId]) {
      console.log(`[AgentPanorama] Clearing selected agent ${selectedAgentId} - not in current session`);
      setSelectedAgentId(null);
    }
  }, [sessionId, agents, selectedAgentId]);
  
  // 统计
  const runningCount = agentList.filter(a => a.status === AgentStatus.RUNNING).length;
  const completedCount = agentList.filter(a => a.status === AgentStatus.COMPLETED).length;
  const avgProgress = agentList.length > 0
    ? agentList.reduce((sum, a) => sum + a.progress, 0) / agentList.length
    : 0;

  const handlePause = async (agentId: string) => {
    if (sessionId) {
      await intervene(sessionId, 'pause', agentId);
    }
  };

  const handleResume = async (agentId: string) => {
    if (sessionId) {
      await intervene(sessionId, 'resume', agentId);
    }
  };

  const handleCancel = async (agentId: string) => {
    if (sessionId) {
      await intervene(sessionId, 'cancel', agentId);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* 头部统计 */}
      <div className="flex-shrink-0 p-4 border-b border-dark-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Users className="w-5 h-5 text-primary-400" />
            Agent 全景图
          </h2>
          <span className="text-sm text-dark-400">
            {agentList.length} 个 Agent
          </span>
        </div>

        {/* 统计卡片 */}
        <div className="grid grid-cols-3 gap-3">
          <div className="p-3 rounded-lg bg-dark-800/50 border border-dark-700">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="w-4 h-4 text-green-400" />
              <span className="text-xs text-dark-400">运行中</span>
            </div>
            <span className="text-xl font-semibold text-white">{runningCount}</span>
          </div>
          <div className="p-3 rounded-lg bg-dark-800/50 border border-dark-700">
            <div className="flex items-center gap-2 mb-1">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span className="text-xs text-dark-400">已完成</span>
            </div>
            <span className="text-xl font-semibold text-white">{completedCount}</span>
          </div>
          <div className="p-3 rounded-lg bg-dark-800/50 border border-dark-700">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs text-dark-400">平均进度</span>
            </div>
            <span className="text-xl font-semibold text-white">{Math.round(avgProgress)}%</span>
          </div>
        </div>
      </div>

      {/* Agent 列表 */}
      <div className="flex-1 overflow-y-auto p-4">
        {agentList.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Users className="w-12 h-12 text-dark-600 mx-auto mb-3" />
              <p className="text-dark-400">等待 Agent 涌现...</p>
            </div>
          </div>
        ) : (
          <div className="grid gap-4">
            <AnimatePresence mode="sync">
              {agentList.map((agent) => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                  isSelected={selectedAgentId === agent.id}
                  onSelect={() => setSelectedAgentId(
                    selectedAgentId === agent.id ? null : agent.id
                  )}
                  onPause={() => handlePause(agent.id)}
                  onResume={() => handleResume(agent.id)}
                  onCancel={() => handleCancel(agent.id)}
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
