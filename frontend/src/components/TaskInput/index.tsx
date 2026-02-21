/**
 * 任务输入组件
 * 
 * 双态设计：
 * - 空闲态：任务输入 + 模式/模型选择 + 开始任务按钮
 * - 运行态：指令输入 + 停止任务 / 下发指令按钮（融合人工干预能力）
 */

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2, Sparkles, Zap, Brain, Square, MessageCircle } from 'lucide-react';
import { useAgui } from '../../hooks/useAgui';
import { useStore } from '../../store';
import { AgentStatus } from '../../types/agui';
import { cn } from '../../utils/cn';

const exampleTasks = [
  "分析《盗梦空间》的叙事结构和视觉语言",
  "深度解读 Attention is All You Need 论文",
  "分析 React 和 Vue 的设计哲学差异",
  "设计一个电商推荐系统的技术方案",
];

export function TaskInput() {
  const [task, setTask] = useState('');
  const [instruction, setInstruction] = useState('');
  const [provider] = useState<'openai' | 'claude'>('openai');
  const [mode, setMode] = useState<'emergent' | 'direct'>('emergent');
  const [isStopping, setIsStopping] = useState(false);
  const [isSendingInstruction, setIsSendingInstruction] = useState(false);
  const { executeTask, intervene, stop } = useAgui();
  const { status, sessionId, setStatus, setMode: setStoreMode, agents, finalReport, mode: storeMode } = useStore();

  const isRunning = status === AgentStatus.RUNNING || status === AgentStatus.PLANNING;
  const hasAgents = Object.keys(agents).length > 0;
  const isDirectMode = storeMode === 'direct';
  // 追问检测：已完成/取消/失败且有历史报告
  const isFollowupReady = !isRunning && !!finalReport && 
    (status === AgentStatus.COMPLETED || status === AgentStatus.CANCELLED || status === AgentStatus.FAILED);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim() || isRunning) return;
    
    // 同步 mode 到 store
    setStoreMode(mode);
    
    if (isFollowupReady && sessionId) {
      await executeTask(task.trim(), provider, undefined, sessionId, mode);
    } else {
      await executeTask(task.trim(), provider, undefined, undefined, mode);
    }
  };

  const handleExampleClick = (example: string) => {
    setTask(example);
  };

  // 停止任务：取消所有 Agent + 关闭 SSE + 状态变为 CANCELLED
  const handleStopTask = async () => {
    if (!sessionId || isStopping) return;
    setIsStopping(true);
    try {
      // 1. 后端取消所有 Agent（广播到中继站）
      await intervene(sessionId, 'cancel', undefined, {
        information: '用户主动停止任务',
      });
      // 2. 关闭 SSE 连接
      stop();
      // 3. 前端状态变为已取消
      setStatus(AgentStatus.CANCELLED);
    } catch (err) {
      console.error('[TaskInput] Stop task error:', err);
    } finally {
      setIsStopping(false);
    }
  };

  // 下发指令：走 inject 广播到所有 Agent（人工干预能力）
  const handleSendInstruction = async () => {
    if (!sessionId || !instruction.trim() || isSendingInstruction) return;
    setIsSendingInstruction(true);
    try {
      // 走 broadcast scope 的 inject，广播给所有 Agent
      await intervene(sessionId, 'inject', undefined, {
        information: instruction.trim(),
      });
      setInstruction('');
    } catch (err) {
      console.error('[TaskInput] Send instruction error:', err);
    } finally {
      setIsSendingInstruction(false);
    }
  };

  // 输入框回车发送指令（运行态）
  const handleInstructionKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && isRunning) {
      e.preventDefault();
      handleSendInstruction();
    }
  };

  return (
    <div className="p-6 border-b border-dark-700 bg-dark-900/50">
      {isRunning ? (
        /* ========== 运行态：停止按钮始终显示，指令输入+下发按钮在 Agent 涌现后出现 ========== */
        <div className="space-y-4">
          {/* 普通模式：显示禁用的输入框（保留对话框但不可输入） */}
          {isDirectMode && (
            <div className="relative">
              <textarea
                disabled
                placeholder="任务执行中，完成后可继续对话..."
                rows={3}
                className={cn(
                  'w-full px-4 py-3 rounded-xl resize-none',
                  'bg-dark-800/50 border border-dark-600/50',
                  'text-dark-500 placeholder-dark-500',
                  'cursor-not-allowed opacity-60',
                  'transition-all duration-200'
                )}
              />
            </div>
          )}

          {/* 涌现模式：指令输入框，仅在 Agent 涌现后显示 */}
          {!isDirectMode && hasAgents && (
            <div className="relative">
              <textarea
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
                onKeyDown={handleInstructionKeyDown}
                placeholder="输入指令，实时干预正在运行的 Agent..."
                rows={2}
                className={cn(
                  'w-full px-4 py-3 rounded-xl resize-none',
                  'bg-dark-800 border border-dark-600',
                  'text-white placeholder-dark-400',
                  'focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500/20',
                  'transition-all duration-200'
                )}
              />
            </div>
          )}

          {/* 运行态按钮栏 */}
          <div className="flex items-center justify-between">
            {/* 左侧状态提示 */}
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm text-dark-300">
                {isDirectMode ? '任务执行中...' : (!hasAgents ? '规划中，等待 Agent 涌现...' : '任务执行中...')}
              </span>
            </div>

            {/* 右侧按钮组 */}
            <div className="flex items-center gap-3">
              {/* 停止任务按钮 — 始终可用 */}
              <button
                type="button"
                onClick={handleStopTask}
                disabled={isStopping}
                className={cn(
                  'px-5 py-2 rounded-lg font-medium',
                  'flex items-center gap-2 transition-all duration-200',
                  'bg-red-500/20 text-red-400 border border-red-500/30',
                  'hover:bg-red-500/30 hover:border-red-500/50',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                {isStopping ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Square className="w-4 h-4" />
                )}
                停止任务
              </button>

              {/* 下发指令按钮 — 仅在涌现模式 Agent 涌现后显示 */}
              {!isDirectMode && hasAgents && (
                <button
                  type="button"
                  onClick={handleSendInstruction}
                  disabled={!instruction.trim() || isSendingInstruction}
                  className={cn(
                    'px-5 py-2 rounded-lg font-medium',
                    'flex items-center gap-2 transition-all duration-200',
                    instruction.trim() && !isSendingInstruction
                      ? 'bg-amber-500 text-white hover:bg-amber-600 hover:shadow-lg hover:shadow-amber-500/20'
                      : 'bg-dark-700 text-dark-500 cursor-not-allowed'
                  )}
                >
                  {isSendingInstruction ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <MessageCircle className="w-4 h-4" />
                  )}
                  下发指令
                </button>
              )}
            </div>
          </div>
        </div>
      ) : (
        /* ========== 空闲态：任务输入 + 开始任务 ========== */
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* 输入框 */}
          <div className="relative">
            <textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder={isFollowupReady ? "基于上轮结果继续追问..." : "输入你的任务，让 Agent 集群帮你完成..."}
              rows={3}
              className={cn(
                'w-full px-4 py-3 rounded-xl resize-none',
                'bg-dark-800 border border-dark-600',
                'text-white placeholder-dark-400',
                'focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500/20',
                'transition-all duration-200'
              )}
            />
          </div>

          {/* 底部操作栏 */}
          <div className="flex items-center justify-between">
            {/* 左侧选择器 */}
            <div className="flex items-center gap-4">
              {/* 模式选择 */}
              <div className="flex items-center gap-2">
                <div className="flex rounded-lg overflow-hidden border border-dark-600">
                  <button
                    type="button"
                    onClick={() => setMode('emergent')}
                    className={cn(
                      'px-3 py-1.5 text-xs font-medium transition-colors flex items-center gap-1',
                      mode === 'emergent'
                        ? 'bg-gradient-to-r from-purple-500 to-primary-500 text-white'
                        : 'bg-dark-800 text-dark-400 hover:text-white'
                    )}
                  >
                    <Brain className="w-3 h-3" />
                    涌现模式
                  </button>
                  <button
                    type="button"
                    onClick={() => setMode('direct')}
                    className={cn(
                      'px-3 py-1.5 text-xs font-medium transition-colors flex items-center gap-1',
                      mode === 'direct'
                        ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white'
                        : 'bg-dark-800 text-dark-400 hover:text-white'
                    )}
                  >
                    <Zap className="w-3 h-3" />
                    普通模式
                  </button>
                </div>
              </div>


            </div>

            {/* 提交按钮 */}
            <button
              type="submit"
              disabled={!task.trim()}
              className={cn(
                'px-6 py-2 rounded-lg font-medium',
                'flex items-center gap-2 transition-all duration-200',
                task.trim()
                  ? 'bg-primary-500 text-white hover:bg-primary-600 hover:shadow-lg hover:shadow-primary-500/20'
                  : 'bg-dark-700 text-dark-500 cursor-not-allowed'
              )}
            >
              <Send className="w-4 h-4" />
              {isFollowupReady ? '继续追问' : '开始任务'}
            </button>
          </div>

          {/* 示例任务 */}
          <div className="pt-2">
            <p className="text-xs text-dark-500 mb-2 flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              示例任务:
            </p>
            <div className="flex flex-wrap gap-2">
              {exampleTasks.map((example, index) => (
                <motion.button
                  key={index}
                  type="button"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handleExampleClick(example)}
                  className="px-3 py-1.5 rounded-full text-xs
                             bg-dark-800 border border-dark-600
                             text-dark-300 hover:text-white hover:border-dark-500
                             transition-colors"
                >
                  {example.length > 30 ? example.slice(0, 30) + '...' : example}
                </motion.button>
              ))}
            </div>
          </div>
        </form>
      )}
    </div>
  );
}
