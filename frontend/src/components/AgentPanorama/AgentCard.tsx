/**
 * Agent 卡片组件 - 增强版
 * 
 * 展示完整的角色信息：身份、目标、方法论、技能等
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bot,
  Play,
  Pause,
  Square,
  RefreshCw,
  Zap,
  CheckCircle,
  XCircle,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Target,
  Wrench,
  BookOpen,
  Star,
  Lightbulb,
  Film,
  Pen,
  Palette,
  Search,
  Brain,
  Code,
  FileText,
  BarChart,
} from 'lucide-react';
import { Agent, AgentStatus, SkillAssignment } from '../../types/agui';
import { cn } from '../../utils/cn';

interface AgentCardProps {
  agent: Agent;
  isSelected: boolean;
  onSelect: () => void;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
}

const statusConfig: Record<AgentStatus, { color: string; icon: React.ElementType; label: string }> = {
  [AgentStatus.PENDING]: { color: 'bg-gray-500', icon: AlertCircle, label: '等待中' },
  [AgentStatus.PLANNING]: { color: 'bg-purple-500', icon: RefreshCw, label: '规划中' },
  [AgentStatus.RUNNING]: { color: 'bg-green-500', icon: Zap, label: '执行中' },
  [AgentStatus.WAITING_RELAY]: { color: 'bg-yellow-500', icon: RefreshCw, label: '等待中继' },
  [AgentStatus.RELAYING]: { color: 'bg-orange-500', icon: RefreshCw, label: '中继中' },
  [AgentStatus.COMPLETED]: { color: 'bg-cyan-500', icon: CheckCircle, label: '已完成' },
  [AgentStatus.FAILED]: { color: 'bg-red-500', icon: XCircle, label: '失败' },
  [AgentStatus.PAUSED]: { color: 'bg-gray-500', icon: Pause, label: '已暂停' },
  [AgentStatus.CANCELLED]: { color: 'bg-gray-500', icon: Square, label: '已取消' },
};

const expertiseLevelConfig: Record<string, { color: string; label: string }> = {
  'novice': { color: 'text-gray-400', label: '初级' },
  'intermediate': { color: 'text-blue-400', label: '中级' },
  'expert': { color: 'text-purple-400', label: '专家' },
  'master': { color: 'text-yellow-400', label: '大师' },
};

// 技能配置 - 包含图标、颜色和能力描述
const skillConfig: Record<string, {
  icon: React.ElementType;
  color: string;
  bgColor: string;
  capabilities: string[];
}> = {
  'director': {
    icon: Film,
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
    capabilities: ['创意愿景', '视觉风格', '叙事把控', '团队协调', '质量监督']
  },
  'screenwriter': {
    icon: Pen,
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10',
    capabilities: ['故事创作', '剧本撰写', '角色塑造', '对白创作', '结构设计']
  },
  'visual_designer': {
    icon: Palette,
    color: 'text-pink-400',
    bgColor: 'bg-pink-500/10',
    capabilities: ['视觉风格', '画面构图', '色彩方案', '光影设计', '美术指导']
  },
  'web_search': {
    icon: Search,
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10',
    capabilities: ['信息检索', '实时数据', '多源聚合']
  },
  'reasoning': {
    icon: Brain,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10',
    capabilities: ['逻辑推理', '因果分析', '问题解决']
  },
  'code_execution': {
    icon: Code,
    color: 'text-green-400',
    bgColor: 'bg-green-500/10',
    capabilities: ['代码执行', '数据计算', '自动化处理']
  },
  'document_summary': {
    icon: FileText,
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-500/10',
    capabilities: ['文档摘要', '关键提取', '内容整理']
  },
  'data_analysis': {
    icon: BarChart,
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10',
    capabilities: ['数据分析', '趋势发现', '统计处理']
  },
};

// 获取技能配置
function getSkillConfig(skillName: string) {
  return skillConfig[skillName] || {
    icon: Wrench,
    color: 'text-gray-400',
    bgColor: 'bg-gray-500/10',
    capabilities: []
  };
}

// 渲染技能标签（带图标）
function SkillTag({ skill, showCapabilities = false }: { skill: SkillAssignment; showCapabilities?: boolean }) {
  const config = getSkillConfig(skill.skill_name);
  const Icon = config.icon;
  
  return (
    <div className={cn(
      'rounded-lg p-2',
      config.bgColor,
      showCapabilities ? 'space-y-2' : ''
    )}>
      <div className="flex items-center gap-1.5">
        <Icon className={cn('w-3.5 h-3.5', config.color)} />
        <span className={cn('text-xs font-medium', config.color)}>
          {skill.skill_display_name}
        </span>
      </div>
      
      {showCapabilities && config.capabilities.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {config.capabilities.map((cap, idx) => (
            <span
              key={idx}
              className={cn(
                'px-1.5 py-0.5 text-[10px] rounded',
                'bg-dark-800/50',
                config.color
              )}
            >
              {cap}
            </span>
          ))}
        </div>
      )}
      
      {showCapabilities && skill.reason && (
        <p className="text-[10px] text-dark-400 mt-1 leading-relaxed">
          {skill.reason}
        </p>
      )}
    </div>
  );
}

export function AgentCard({
  agent,
  isSelected,
  onSelect,
  onPause,
  onResume,
  onCancel,
}: AgentCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const status = statusConfig[agent.status] || statusConfig[AgentStatus.PENDING];
  const StatusIcon = status.icon;
  const expertiseLevel = expertiseLevelConfig[agent.expertiseLevel || 'expert'] || expertiseLevelConfig['expert'];

  const isRunning = agent.status === AgentStatus.RUNNING;
  const isPaused = agent.status === AgentStatus.PAUSED;
  const isActive = isRunning || isPaused;

  return (
    <motion.div
      layout="position"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className={cn(
        'relative rounded-xl transition-all duration-200',
        'bg-dark-800/50 backdrop-blur-sm border',
        isSelected
          ? 'border-primary-500 ring-2 ring-primary-500/20'
          : 'border-dark-700 hover:border-dark-600'
      )}
    >
      {/* 主卡片区域 */}
      <div 
        className="p-4 cursor-pointer"
        onClick={onSelect}
      >
        {/* 状态指示灯 */}
        <div className="absolute top-3 right-3 flex items-center gap-2">
          <span className={cn(expertiseLevel.color, 'text-xs font-medium')}>
            {expertiseLevel.label}
          </span>
          <span
            className={cn(
              'w-2 h-2 rounded-full',
              status.color,
              isRunning && 'animate-pulse'
            )}
          />
        </div>

        {/* Agent 头部 */}
        <div className="flex items-start gap-3 mb-3">
          <div className={cn(
            'p-2 rounded-lg',
            status.color.replace('bg-', 'bg-opacity-20 bg-')
          )}>
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-white truncate">{agent.name}</h3>
            <p className="text-xs text-dark-400 truncate">{agent.roleName}</p>
          </div>
        </div>

        {/* 工作目标 */}
        {agent.workObjective && (
          <div className="mb-3 p-2 rounded-lg bg-dark-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <Target className="w-3 h-3 text-primary-400" />
              <span className="text-xs text-primary-400 font-medium">工作目标</span>
            </div>
            <p className="text-xs text-dark-300 line-clamp-2">
              {agent.workObjective}
            </p>
          </div>
        )}

        {/* 进度条 */}
        <div className="mb-3">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-dark-400">{agent.currentStep || '准备中...'}</span>
            <span className="text-dark-400">{Math.round(agent.progress)}%</span>
          </div>
          <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
            <motion.div
              className={cn('h-full rounded-full', status.color)}
              initial={{ width: 0 }}
              animate={{ width: `${agent.progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        {/* 技能标签 - 增强版 */}
        {agent.assignedSkills && agent.assignedSkills.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {agent.assignedSkills.slice(0, 4).map((skill, idx) => {
              const config = getSkillConfig(skill.skill_name);
              const Icon = config.icon;
              return (
                <span
                  key={idx}
                  className={cn(
                    'px-2 py-1 text-xs rounded-md flex items-center gap-1',
                    config.bgColor,
                    config.color
                  )}
                  title={skill.reason || skill.skill_display_name}
                >
                  <Icon className="w-3 h-3" />
                  {skill.skill_display_name}
                </span>
              );
            })}
            {agent.assignedSkills.length > 4 && (
              <span className="px-2 py-1 text-xs rounded-md bg-dark-700 text-dark-400">
                +{agent.assignedSkills.length - 4}
              </span>
            )}
          </div>
        )}

        {/* 状态和迭代 */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <StatusIcon className={cn('w-3.5 h-3.5', status.color.replace('bg-', 'text-').replace('-500', '-400'))} />
            <span className="text-xs text-dark-400">{status.label}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-dark-500">迭代 {agent.iterations}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsExpanded(!isExpanded);
              }}
              className="p-1 rounded hover:bg-dark-700 transition-colors"
            >
              {isExpanded ? (
                <ChevronUp className="w-4 h-4 text-dark-400" />
              ) : (
                <ChevronDown className="w-4 h-4 text-dark-400" />
              )}
            </button>
          </div>
        </div>

        {/* Skill 调用实时状态 */}
        {agent.toolCalls && agent.toolCalls.length > 0 && (
          <div className="mt-3 space-y-1.5">
            {agent.toolCalls.slice(-3).map((tc) => {
              const skillCfg = getSkillConfig(tc.skillName.replace(/-/g, '_'));
              const SkillIcon = skillCfg.icon;
              const isRunning = tc.status === 'running';
              const isSuccess = tc.status === 'success';
              const isError = tc.status === 'error';
              
              return (
                <div
                  key={tc.id}
                  className={cn(
                    'flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs',
                    isRunning && 'bg-blue-500/10 border border-blue-500/20',
                    isSuccess && 'bg-emerald-500/10 border border-emerald-500/20',
                    isError && 'bg-red-500/10 border border-red-500/20',
                  )}
                >
                  <SkillIcon className={cn(
                    'w-3.5 h-3.5 flex-shrink-0',
                    isRunning && 'text-blue-400 animate-pulse',
                    isSuccess && 'text-emerald-400',
                    isError && 'text-red-400',
                  )} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className={cn(
                        'font-medium truncate',
                        isRunning && 'text-blue-300',
                        isSuccess && 'text-emerald-300',
                        isError && 'text-red-300',
                      )}>
                        {tc.skillName}
                      </span>
                      {isRunning && (
                        <span className="text-blue-400 animate-pulse">...</span>
                      )}
                      {isSuccess && (
                        <CheckCircle className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                      )}
                      {isError && (
                        <XCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
                      )}
                    </div>
                    {tc.summary && (
                      <p className="text-[10px] text-dark-400 truncate mt-0.5">
                        {tc.summary}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
            {agent.toolCalls.length > 3 && (
              <p className="text-[10px] text-dark-500 text-center">
                + {agent.toolCalls.length - 3} 次更早的调用
              </p>
            )}
          </div>
        )}
      </div>

      {/* 展开的详细信息 */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-t border-dark-700"
          >
            <div className="p-4 space-y-4">
              {/* 角色描述 */}
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <Star className="w-3.5 h-3.5 text-yellow-400" />
                  <span className="text-xs text-dark-400 font-medium">角色简介</span>
                </div>
                <p className="text-xs text-dark-300 leading-relaxed">
                  {agent.roleDescription}
                </p>
              </div>

              {/* 核心能力 */}
              {agent.capabilities && agent.capabilities.length > 0 && (
                <div>
                  <div className="flex items-center gap-1.5 mb-2">
                    <Lightbulb className="w-3.5 h-3.5 text-blue-400" />
                    <span className="text-xs text-dark-400 font-medium">核心能力</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {agent.capabilities.map((cap, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 text-xs rounded bg-blue-500/10 text-blue-400"
                      >
                        {cap}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* 关注领域 */}
              {agent.focusAreas && agent.focusAreas.length > 0 && (
                <div>
                  <div className="flex items-center gap-1.5 mb-2">
                    <Target className="w-3.5 h-3.5 text-green-400" />
                    <span className="text-xs text-dark-400 font-medium">关注领域</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {agent.focusAreas.map((area, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 text-xs rounded bg-green-500/10 text-green-400"
                      >
                        {area}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* 工作方法论 */}
              {agent.methodology && (
                <div>
                  <div className="flex items-center gap-1.5 mb-2">
                    <BookOpen className="w-3.5 h-3.5 text-purple-400" />
                    <span className="text-xs text-dark-400 font-medium">工作方法论</span>
                  </div>
                  <div className="space-y-2 text-xs text-dark-300">
                    <p><span className="text-dark-400">方法：</span>{agent.methodology.approach}</p>
                    {agent.methodology.steps && agent.methodology.steps.length > 0 && (
                      <div>
                        <span className="text-dark-400">步骤：</span>
                        <ol className="list-decimal list-inside mt-1 space-y-0.5">
                          {agent.methodology.steps.map((step, idx) => (
                            <li key={idx} className="text-dark-300">{step}</li>
                          ))}
                        </ol>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* 预期交付物 */}
              {agent.deliverables && agent.deliverables.length > 0 && (
                <div>
                  <div className="flex items-center gap-1.5 mb-2">
                    <CheckCircle className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-xs text-dark-400 font-medium">预期交付物</span>
                  </div>
                  <ul className="space-y-1">
                    {agent.deliverables.map((item, idx) => (
                      <li key={idx} className="text-xs text-dark-300 flex items-center gap-1.5">
                        <span className="w-1 h-1 rounded-full bg-cyan-400" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 技能详情 - 增强版 */}
              {agent.assignedSkills && agent.assignedSkills.length > 0 && (
                <div>
                  <div className="flex items-center gap-1.5 mb-2">
                    <Zap className="w-3.5 h-3.5 text-orange-400" />
                    <span className="text-xs text-dark-400 font-medium">可用技能 & 能力</span>
                  </div>
                  <div className="space-y-2">
                    {agent.assignedSkills.map((skill, idx) => (
                      <SkillTag key={idx} skill={skill} showCapabilities={true} />
                    ))}
                  </div>
                </div>
              )}

              {/* 当前任务 */}
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <Zap className="w-3.5 h-3.5 text-yellow-400" />
                  <span className="text-xs text-dark-400 font-medium">当前任务</span>
                </div>
                <p className="text-xs text-dark-300 leading-relaxed">
                  {agent.taskSegment}
                </p>
              </div>

              {/* Skill 调用历史 */}
              {agent.toolCalls && agent.toolCalls.length > 0 && (
                <div>
                  <div className="flex items-center gap-1.5 mb-2">
                    <Search className="w-3.5 h-3.5 text-blue-400" />
                    <span className="text-xs text-dark-400 font-medium">
                      技能调用记录 ({agent.toolCalls.length})
                    </span>
                  </div>
                  <div className="space-y-2">
                    {agent.toolCalls.map((tc) => {
                      const skillCfg = getSkillConfig(tc.skillName.replace(/-/g, '_'));
                      const SkillIcon = skillCfg.icon;
                      return (
                        <div
                          key={tc.id}
                          className={cn(
                            'p-2 rounded-lg border text-xs',
                            tc.status === 'running' && 'bg-blue-500/5 border-blue-500/20',
                            tc.status === 'success' && 'bg-emerald-500/5 border-emerald-500/20',
                            tc.status === 'error' && 'bg-red-500/5 border-red-500/20',
                          )}
                        >
                          <div className="flex items-center gap-1.5 mb-1">
                            <SkillIcon className={cn('w-3.5 h-3.5', skillCfg.color)} />
                            <span className={cn('font-medium', skillCfg.color)}>
                              {tc.skillName}
                            </span>
                            <span className={cn(
                              'ml-auto px-1.5 py-0.5 rounded text-[10px]',
                              tc.status === 'running' && 'bg-blue-500/20 text-blue-300',
                              tc.status === 'success' && 'bg-emerald-500/20 text-emerald-300',
                              tc.status === 'error' && 'bg-red-500/20 text-red-300',
                            )}>
                              {tc.status === 'running' ? '执行中' : tc.status === 'success' ? '成功' : '失败'}
                            </span>
                          </div>
                          {tc.summary && (
                            <p className="text-dark-400 leading-relaxed">{tc.summary}</p>
                          )}
                          {tc.resultPreview && (
                            <p className="text-dark-500 leading-relaxed mt-1 line-clamp-3">
                              {tc.resultPreview}
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 控制按钮 */}
      {isActive && (
        <div className="px-4 pb-4 pt-0 flex gap-2">
          {isRunning ? (
            <button
              onClick={(e) => { e.stopPropagation(); onPause(); }}
              className="flex-1 py-1.5 px-3 rounded-lg bg-dark-700 hover:bg-dark-600 
                         text-xs text-dark-300 hover:text-white transition-colors
                         flex items-center justify-center gap-1"
            >
              <Pause className="w-3 h-3" />
              暂停
            </button>
          ) : (
            <button
              onClick={(e) => { e.stopPropagation(); onResume(); }}
              className="flex-1 py-1.5 px-3 rounded-lg bg-dark-700 hover:bg-dark-600 
                         text-xs text-dark-300 hover:text-white transition-colors
                         flex items-center justify-center gap-1"
            >
              <Play className="w-3 h-3" />
              继续
            </button>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); onCancel(); }}
            className="py-1.5 px-3 rounded-lg bg-red-500/10 hover:bg-red-500/20 
                       text-xs text-red-400 hover:text-red-300 transition-colors
                       flex items-center justify-center gap-1"
          >
            <Square className="w-3 h-3" />
            取消
          </button>
        </div>
      )}
    </motion.div>
  );
}
