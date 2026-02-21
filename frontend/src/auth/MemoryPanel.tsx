/**
 * 用户长期记忆面板
 * 
 * 以 Drawer 形式从右侧滑出，展示用户记忆列表和记忆系统状态
 */

import { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Brain, Search, RefreshCw, AlertCircle, Zap, Database, Trash2 } from 'lucide-react';
import { getAuthHeader } from './api';

interface MemoryItem {
  id?: string;
  content?: string;
  text?: string;
  category?: string;
  score?: number;
  [key: string]: any;
}

interface MemoryCategory {
  name?: string;
  [key: string]: any;
}

interface MemoryData {
  enabled: boolean;
  mode: string;
  items: MemoryItem[];
  categories: MemoryCategory[];
  status: string;
  message?: string;
}

interface MemoryPanelProps {
  open: boolean;
  onClose: () => void;
}

export function MemoryPanel({ open, onClose }: MemoryPanelProps) {
  const [data, setData] = useState<MemoryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchMemories = useCallback(async (query?: string) => {
    setLoading(true);
    setError(null);
    try {
      const params = query ? `?query=${encodeURIComponent(query)}` : '';
      const response = await fetch(`/api/auth/memories${params}`, {
        headers: { ...getAuthHeader() },
      });
      if (!response.ok) {
        throw new Error('获取记忆失败');
      }
      const result = await response.json();
      setData(result);
    } catch (e: any) {
      setError(e.message || '未知错误');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      fetchMemories();
    }
  }, [open, fetchMemories]);

  const handleSearch = () => {
    if (searchQuery.trim()) {
      fetchMemories(searchQuery.trim());
    } else {
      fetchMemories();
    }
  };

  const handleDelete = async (memoryId: string) => {
    if (!memoryId || deletingId) return;
    setDeletingId(memoryId);
    try {
      const response = await fetch(`/api/auth/memories/${memoryId}`, {
        method: 'DELETE',
        headers: { ...getAuthHeader() },
      });
      if (!response.ok) {
        throw new Error('删除失败');
      }
      // 从本地状态中移除
      if (data) {
        setData({
          ...data,
          items: data.items.filter((item) => item.id !== memoryId),
        });
      }
    } catch (e: any) {
      setError(e.message || '删除失败');
    } finally {
      setDeletingId(null);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ok': return 'text-emerald-400';
      case 'disabled': return 'text-amber-400';
      case 'timeout': return 'text-orange-400';
      case 'error': return 'text-red-400';
      default: return 'text-[#64748B]';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'ok': return '运行中';
      case 'disabled': return '未启用';
      case 'timeout': return '超时';
      case 'error': return '错误';
      default: return status;
    }
  };

  const getModeLabel = (mode: string) => {
    switch (mode) {
      case 'local': return '本地 SDK';
      case 'cloud': return '云端 API';
      case 'disabled': return '已关闭';
      default: return mode;
    }
  };

  const getItemContent = (item: MemoryItem): string => {
    if (typeof item === 'string') return item;
    return item.content || item.text || JSON.stringify(item);
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* 遮罩 */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[60]"
            onClick={onClose}
          />

          {/* 抽屉 */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-[420px] max-w-[90vw] z-[70] flex flex-col"
            style={{
              background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(15, 23, 42, 0.95) 100%)',
              backdropFilter: 'blur(20px)',
              borderLeft: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            {/* 头部 */}
            <div className="px-6 py-5 border-b border-white/[0.06] flex items-center justify-between shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500/20 to-purple-500/20 border border-violet-500/20 flex items-center justify-center">
                  <Brain className="w-5 h-5 text-violet-400" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-[#F8FAFC]">长期记忆</h2>
                  <p className="text-xs text-[#64748B]">用户偏好与行为特征</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-white/[0.06] transition-colors cursor-pointer"
              >
                <X className="w-5 h-5 text-[#64748B]" />
              </button>
            </div>

            {/* 系统状态卡片 */}
            {data && (
              <div className="mx-6 mt-4 p-4 rounded-xl border border-white/[0.06]"
                style={{ background: 'rgba(30, 41, 59, 0.5)' }}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-medium text-[#94A3B8] uppercase tracking-wider">记忆系统状态</span>
                  <span className={`text-xs font-medium ${getStatusColor(data.status)}`}>
                    {getStatusLabel(data.status)}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex items-center gap-2">
                    <Database className="w-3.5 h-3.5 text-[#64748B]" />
                    <span className="text-xs text-[#94A3B8]">模式：</span>
                    <span className="text-xs text-[#CBD5E1]">{getModeLabel(data.mode)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="w-3.5 h-3.5 text-[#64748B]" />
                    <span className="text-xs text-[#94A3B8]">记忆数：</span>
                    <span className="text-xs text-[#CBD5E1]">{data.items.length}</span>
                  </div>
                </div>

                {/* 注入链路说明 */}
                <div className="mt-3 pt-3 border-t border-white/[0.04]">
                  <p className="text-[11px] text-[#64748B] leading-relaxed">
                    {data.enabled
                      ? '✅ 记忆已启用 — 每次任务执行时，MasterAgent 会检索你的记忆注入到角色涌现上下文，每个 SubAgent 的 system prompt 末尾也会注入你的偏好。'
                      : data.status === 'degraded'
                        ? `⚠️ 记忆服务初始化失败 — ${data.message || '可能缺少 memu 包，请运行 pip install memu 安装后重启服务。'}`
                        : '⚠️ 记忆系统当前未启用 — 在 .env 中设置 MEMU_MODE=local 或 MEMU_MODE=cloud 即可激活。激活后，你的对话偏好将自动被记忆并注入到所有 Agent 的决策中。'
                    }
                  </p>
                </div>
              </div>
            )}

            {/* 搜索框 */}
            <div className="px-6 py-4 shrink-0">
              <div className="flex gap-2">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#64748B]" />
                  <input
                    type="text"
                    placeholder="搜索记忆关键词..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.08] text-sm text-[#F8FAFC] placeholder-[#475569] focus:outline-none focus:border-violet-500/40 focus:ring-1 focus:ring-violet-500/20 transition-all"
                  />
                </div>
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="p-2.5 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] transition-colors disabled:opacity-50 cursor-pointer"
                >
                  <RefreshCw className={`w-4 h-4 text-[#94A3B8] ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>
            </div>

            {/* 记忆列表 */}
            <div className="flex-1 overflow-y-auto px-6 pb-6">
              {loading && !data && (
                <div className="flex flex-col items-center justify-center py-16 gap-3">
                  <RefreshCw className="w-6 h-6 text-violet-400 animate-spin" />
                  <p className="text-sm text-[#64748B]">加载记忆中...</p>
                </div>
              )}

              {error && (
                <div className="flex items-center gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                  <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                  <p className="text-sm text-red-300">{error}</p>
                </div>
              )}

              {data && !data.enabled && (
                <div className="flex flex-col items-center justify-center py-12 gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-amber-500/10 to-orange-500/10 border border-amber-500/10 flex items-center justify-center">
                    <Brain className="w-8 h-8 text-amber-400/60" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-[#CBD5E1] mb-1">记忆系统未启用</p>
                    <p className="text-xs text-[#64748B] max-w-[260px] leading-relaxed">
                      在后端 .env 中配置 MEMU_MODE=local 或 MEMU_MODE=cloud 以启用记忆功能，让 Agent 记住你的偏好和习惯。
                    </p>
                  </div>
                </div>
              )}

              {data && data.enabled && data.items.length === 0 && (
                <div className="flex flex-col items-center justify-center py-12 gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/10 to-purple-500/10 border border-violet-500/10 flex items-center justify-center">
                    <Brain className="w-8 h-8 text-violet-400/60" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-[#CBD5E1] mb-1">暂无记忆</p>
                    <p className="text-xs text-[#64748B] max-w-[260px] leading-relaxed">
                      开始执行任务后，系统会自动记忆你的偏好和交互特点。
                    </p>
                  </div>
                </div>
              )}

              {data && data.items.length > 0 && (
                <div className="space-y-3">
                  {/* 分类标签 */}
                  {data.categories.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-4">
                      {data.categories.map((cat, i) => {
                        const catName = typeof cat === 'string' ? cat : cat.name || String(cat);
                        return (
                          <span
                            key={i}
                            className="px-2.5 py-1 rounded-lg text-xs font-medium bg-violet-500/10 text-violet-300 border border-violet-500/15"
                          >
                            {catName}
                          </span>
                        );
                      })}
                    </div>
                  )}

                  {/* 记忆条目 */}
                  {data.items.map((item, i) => (
                    <motion.div
                      key={item.id || i}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="p-4 rounded-xl border border-white/[0.06] group hover:border-violet-500/20 transition-all relative"
                      style={{ background: 'rgba(30, 41, 59, 0.4)' }}
                    >
                      {/* 删除按钮 */}
                      {item.id && (
                        <button
                          onClick={() => handleDelete(item.id!)}
                          disabled={deletingId === item.id}
                          className="absolute top-3 right-3 p-1.5 rounded-lg opacity-0 group-hover:opacity-100 hover:bg-red-500/20 transition-all cursor-pointer disabled:opacity-50"
                          title="删除此记忆"
                        >
                          <Trash2 className={`w-3.5 h-3.5 text-red-400 ${deletingId === item.id ? 'animate-spin' : ''}`} />
                        </button>
                      )}
                      <p className="text-sm text-[#CBD5E1] leading-relaxed whitespace-pre-wrap pr-8">
                        {getItemContent(item)}
                      </p>
                      {item.category && (
                        <span className="inline-block mt-2 px-2 py-0.5 rounded text-[10px] font-medium bg-white/[0.04] text-[#64748B]">
                          {item.category}
                        </span>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            {/* 底部注入链路信息 */}
            <div className="px-6 py-4 border-t border-white/[0.06] shrink-0">
              <p className="text-[11px] text-[#475569] leading-relaxed">
                <span className="text-[#64748B] font-medium">注入链路：</span>
                任务开始 → MemoryService.retrieve → 格式化为文本 → 注入 MasterAgent 角色涌现上下文 + 每个 SubAgent system prompt 末尾
              </p>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
