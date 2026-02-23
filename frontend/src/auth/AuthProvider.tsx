/**
 * 认证 Provider
 *
 * 管理用户认证状态，未登录时显示 LoginPage
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import {
  AuthUser,
  login as apiLogin,
  register as apiRegister,
  getMe,
  getStoredToken,
  storeToken,
  clearToken,
} from './api';
import { LoginPage } from './LoginPage';
import { useStore } from '../store';

interface AuthContextValue {
  user: AuthUser | null;
  token: string | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, displayName?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return ctx;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // 初始化：从 localStorage 恢复 token 并验证
  useEffect(() => {
    const savedToken = getStoredToken();
    if (savedToken) {
      getMe(savedToken)
        .then((res) => {
          setUser(res);
          setToken(savedToken);
        })
        .catch(() => {
          clearToken();
        })
        .finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const res = await apiLogin(username, password);
    storeToken(res.token);
    setToken(res.token);
    setUser(res.user);
  }, []);

  const register = useCallback(async (username: string, password: string, displayName?: string) => {
    const res = await apiRegister(username, password, displayName);
    storeToken(res.token);
    setToken(res.token);
    setUser(res.user);
  }, []);

  const logout = useCallback(() => {
    // 清空 Zustand store 中所有会话数据，防止切换用户后残留上一个用户的数据
    useStore.getState().clearAllSessions();
    // 清除 URL 中的 session 参数
    const url = new URL(window.location.href);
    url.searchParams.delete('session');
    window.history.replaceState({}, '', url.toString());
    // 清除认证状态
    clearToken();
    setToken(null);
    setUser(null);
  }, []);

  const value: AuthContextValue = { user, token, isLoading, login, register, logout };

  // 加载中显示加载动画
  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0F172A] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#6366F1] to-[#8B5CF6] flex items-center justify-center animate-pulse">
            <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <p className="text-[#64748B] text-sm">正在验证身份...</p>
        </div>
      </div>
    );
  }

  // 未登录显示登录页
  if (!user || !token) {
    return (
      <AuthContext.Provider value={value}>
        <LoginPage />
      </AuthContext.Provider>
    );
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}
