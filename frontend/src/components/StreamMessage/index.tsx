/**
 * 流式消息组件
 * 
 * 实时显示消息、工具调用过程和模型思考过程
 */

import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Bot, ChevronDown, ChevronUp, Wrench, CheckCircle, XCircle, Loader2, Brain } from 'lucide-react';
import { useStore } from '../../store';
import { cn } from '../../utils/cn';
import { AgentToolCall } from '../../types/agui';

// 技能名到图标/颜色的映射
const skillIconMap: Record<string, { icon: string; color: string; label: string }> = {
  'web-search': { icon: '🔍', color: 'text-blue-400', label: '网络搜索' },
  'data-analysis': { icon: '📊', color: 'text-green-400', label: '数据分析' },
  'code-execution': { icon: '💻', color: 'text-yellow-400', label: '代码执行' },
  'document-summary': { icon: '📄', color: 'text-purple-400', label: '文档摘要' },
  'reasoning': { icon: '🧠', color: 'text-pink-400', label: '深度推理' },
};

function ToolCallCard({ toolCall, index }: { toolCall: AgentToolCall; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const skillInfo = skillIconMap[toolCall.skillName || toolCall.toolName] || { icon: '🔧', color: 'text-dark-300', label: toolCall.skillName || toolCall.toolName };
  
  const isRunning = toolCall.status === 'running';
  const isSuccess = toolCall.status === 'success';
  
  const statusIcon = isRunning ? (
    <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />
  ) : isSuccess ? (
    <CheckCircle className="w-3.5 h-3.5 text-green-400" />
  ) : (
    <XCircle className="w-3.5 h-3.5 text-red-400" />
  );

  const hasDetails = toolCall.summary || toolCall.resultPreview || toolCall.arguments;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className={cn(
        "rounded-lg border overflow-hidden transition-colors",
        isRunning && "bg-blue-500/5 border-blue-500/20",
        isSuccess && "bg-emerald-500/5 border-emerald-500/20",
        !isRunning && !isSuccess && "bg-red-500/5 border-red-500/20",
      )}
    >
      <button
        onClick={() => hasDetails && setExpanded(!expanded)}
        className={cn(
          "w-full px-3 py-2 flex items-center gap-2 transition-colors",
          hasDetails && "hover:bg-white/5 cursor-pointer",
          !hasDetails && "cursor-default",
        )}
      >
        <span className="text-sm flex-shrink-0">{skillInfo.icon}</span>
        <div className="flex-1 min-w-0 text-left">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-dark-200">
              {skillInfo.label}
            </span>
            {toolCall.agentName && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-dark-700 text-dark-400">
                {toolCall.agentName}
              </span>
            )}
          </div>
          {/* 行内摘要预览 */}
          {toolCall.summary && !expanded && (
            <p className="text-[11px] text-dark-400 truncate mt-0.5">{toolCall.summary}</p>
          )}
        </div>
        <span className="flex items-center gap-1 flex-shrink-0">
          {statusIcon}
        </span>
        {hasDetails && (
          expanded ? <ChevronUp className="w-3 h-3 text-dark-500 flex-shrink-0" /> : <ChevronDown className="w-3 h-3 text-dark-500 flex-shrink-0" />
        )}
      </button>
      
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="border-t border-dark-700/50"
          >
            <div className="px-3 py-2 space-y-2 max-h-64 overflow-y-auto">
              {toolCall.arguments && (
                <div>
                  <span className="text-[10px] uppercase text-dark-500 font-medium">技能参数</span>
                  <pre className="text-xs text-dark-300 font-mono whitespace-pre-wrap mt-0.5 bg-dark-900/50 rounded p-1.5">
                    {typeof toolCall.arguments === 'string' 
                      ? toolCall.arguments 
                      : JSON.stringify(toolCall.arguments, null, 2)}
                  </pre>
                </div>
              )}
              {toolCall.summary && (
                <div>
                  <span className="text-[10px] uppercase text-dark-500 font-medium">执行摘要</span>
                  <p className="text-xs text-dark-300 mt-0.5 leading-relaxed">{toolCall.summary}</p>
                </div>
              )}
              {toolCall.resultPreview && (
                <div>
                  <span className="text-[10px] uppercase text-dark-500 font-medium">结果预览</span>
                  <pre className="text-xs text-dark-400 font-mono whitespace-pre-wrap mt-0.5 bg-dark-900/50 rounded p-1.5 max-h-40 overflow-y-auto">
                    {toolCall.resultPreview}
                  </pre>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export function StreamMessage() {
  const { messages, agents, streamToolCalls, streamThinking, mode } = useStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set());
  const [thinkingExpanded, setThinkingExpanded] = useState(true);

  // 自动滚动到底部
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamToolCalls, streamThinking, autoScroll]);

  const handleScroll = () => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
      setAutoScroll(scrollHeight - scrollTop - clientHeight < 50);
    }
  };

  const toggleAgentExpand = (agentId: string) => {
    setExpandedAgents(prev => {
      const next = new Set(prev);
      if (next.has(agentId)) {
        next.delete(agentId);
      } else {
        next.add(agentId);
      }
      return next;
    });
  };

  const agentList = Object.values(agents);
  const isDirectMode = mode === 'direct';

  // 调试日志
  console.log(`[StreamMessage] Render: mode=${mode}, streamToolCalls.length=${streamToolCalls.length}, streamThinking.length=${streamThinking.length}, agentList.length=${agentList.length}, messages.length=${messages.length}`);

  return (
    <div className="h-full flex flex-col">
      {/* 头部 */}
      <div className="flex-shrink-0 p-4 border-b border-dark-700">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-blue-400" />
          实时消息
        </h2>
      </div>

      {/* 消息区域 */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
      {/* 思考过程（没有 agent 时显示，即 direct 模式和 emergent 模式无 subagent 时均显示） */}
        {streamThinking && agentList.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl bg-purple-500/5 border border-purple-500/20 overflow-hidden"
          >
            <button
              onClick={() => setThinkingExpanded(!thinkingExpanded)}
              className="w-full p-3 flex items-center justify-between hover:bg-purple-500/5 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-purple-300">模型思考过程</span>
                <span className="text-xs text-dark-400">
                  {streamThinking.length} 字符
                </span>
              </div>
              {thinkingExpanded ? (
                <ChevronUp className="w-4 h-4 text-dark-400" />
              ) : (
                <ChevronDown className="w-4 h-4 text-dark-400" />
              )}
            </button>
            <AnimatePresence>
              {thinkingExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="border-t border-purple-500/20"
                >
                  <div className="p-3 max-h-64 overflow-y-auto">
                    <pre className="text-xs text-purple-200/70 whitespace-pre-wrap font-mono">
                      {streamThinking}
                    </pre>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}

        {/* Skills 使用过程（没有 agent 时显示，即 direct 模式和 emergent 模式无 subagent 时均显示） */}
        {streamToolCalls.length > 0 && agentList.length === 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 mb-1">
              <Wrench className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-xs font-medium text-dark-400">
                Skills 使用 ({streamToolCalls.length})
              </span>
            </div>
            {streamToolCalls.map((tc, i) => (
              <ToolCallCard key={tc.id} toolCall={tc} index={i} />
            ))}
          </div>
        )}

        {/* 系统消息 */}
        <AnimatePresence mode="popLayout">
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className={cn(
                'p-4 rounded-xl',
                msg.role === 'assistant' && 'bg-dark-800/50 border border-dark-700',
                msg.role === 'user' && 'bg-primary-500/10 border border-primary-500/30',
              )}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className={cn(
                  'w-6 h-6 rounded-full flex items-center justify-center',
                  msg.role === 'assistant' ? 'bg-dark-700' : 'bg-primary-500/20'
                )}>
                  {msg.role === 'assistant' ? (
                    <Bot className="w-3.5 h-3.5 text-dark-300" />
                  ) : (
                    <span className="text-xs text-primary-400">U</span>
                  )}
                </div>
                <span className="text-sm font-medium text-dark-300">
                  {msg.role === 'assistant' ? 'Master Agent' : 'You'}
                </span>
              </div>
              <div className="text-sm text-dark-200 whitespace-pre-wrap">
                {msg.content}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Agent 工作过程（涌现模式，普通模式下隐藏） */}
        {agentList.length > 0 && !isDirectMode && (
          <div className="mt-4 space-y-3">
            <h3 className="text-sm font-medium text-dark-400 flex items-center gap-2">
              <Bot className="w-4 h-4" />
              Agent 工作过程 ({agentList.length})
            </h3>
            {agentList.map((agent) => {
              const agentToolCalls = agent.toolCalls || [];
              const hasContent = agent.thinking || agentToolCalls.length > 0;
              
              return (
                <motion.div
                  key={agent.id}
                  layout
                  className="rounded-xl bg-dark-800/30 border border-dark-700 overflow-hidden"
                >
                  {/* Agent 头部 */}
                  <button
                    onClick={() => toggleAgentExpand(agent.id)}
                    className="w-full p-3 flex items-center justify-between hover:bg-dark-700/30 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <div className={cn(
                        "w-2 h-2 rounded-full",
                        agent.status === 'running' ? 'bg-blue-500 animate-pulse' :
                        agent.status === 'completed' ? 'bg-green-500' :
                        agent.status === 'failed' ? 'bg-red-500' : 'bg-dark-500'
                      )} />
                      <span className="text-sm font-medium text-white">{agent.name}</span>
                      {agent.roleName && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-dark-700 text-dark-400">
                          {agent.roleName}
                        </span>
                      )}
                      {agentToolCalls.length > 0 && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400">
                          {agentToolCalls.length} 次 Skills 调用
                        </span>
                      )}
                    </div>
                    {hasContent && (
                      expandedAgents.has(agent.id) ? (
                        <ChevronUp className="w-4 h-4 text-dark-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-dark-400" />
                      )
                    )}
                  </button>

                  {/* Agent 详细内容 */}
                  <AnimatePresence>
                    {expandedAgents.has(agent.id) && hasContent && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="border-t border-dark-700"
                      >
                        <div className="p-3 space-y-3 max-h-96 overflow-y-auto">
                          {/* Agent 的工具调用 */}
                          {agentToolCalls.length > 0 && (
                            <div className="space-y-1.5">
                              <span className="text-[10px] uppercase text-dark-500 font-medium flex items-center gap-1">
                                <Wrench className="w-3 h-3" /> Skills 使用
                              </span>
                              {agentToolCalls.map((tc, i) => (
                                <ToolCallCard key={tc.id} toolCall={tc} index={i} />
                              ))}
                            </div>
                          )}
                          
                          {/* Agent 的思考过程 */}
                          {agent.thinking && (
                            <div>
                              <span className="text-[10px] uppercase text-dark-500 font-medium flex items-center gap-1">
                                <Brain className="w-3 h-3" /> 思考过程
                              </span>
                              <pre className="text-xs text-dark-300 whitespace-pre-wrap font-mono mt-1 max-h-48 overflow-y-auto">
                                {agent.thinking}
                              </pre>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>

      {/* 自动滚动指示器 */}
      {!autoScroll && (
        <button
          onClick={() => {
            setAutoScroll(true);
            if (scrollRef.current) {
              scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }
          }}
          className="absolute bottom-20 right-4 px-3 py-1.5 rounded-full 
                     bg-dark-700 border border-dark-600 text-xs text-dark-300
                     hover:bg-dark-600 transition-colors"
        >
          滚动到底部
        </button>
      )}
    </div>
  );
}
