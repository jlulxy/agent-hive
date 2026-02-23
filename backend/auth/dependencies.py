"""
FastAPI 依赖注入函数

提供认证依赖和会话归属校验：
- get_current_user: 强制认证，未登录返回 401
- get_optional_user: 可选认证，未登录返回 None（仅用于兼容场景）
- verify_token_from_query: 从 URL query 参数验证 token（用于 SSE/EventSource）
- verify_session_owner: 校验 session 归属当前用户（防 IDOR）
"""

import os
from typing import Optional
from fastapi import Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from auth.provider import AuthProvider, LocalAuthProvider
from storage.factory import get_repository


security = HTTPBearer(auto_error=False)

_auth_provider: Optional[AuthProvider] = None


def get_auth_provider() -> AuthProvider:
    """获取全局 AuthProvider 单例"""
    global _auth_provider
    if _auth_provider is None:
        secret_key = os.getenv("JWT_SECRET", "agent-swarm-default-secret-change-me")
        token_expire_hours = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
        repo = get_repository()
        _auth_provider = LocalAuthProvider(
            secret_key=secret_key,
            user_repository=repo,
            token_expire_hours=token_expire_hours,
        )
    return _auth_provider


def reset_auth_provider():
    """重置 AuthProvider（用于测试）"""
    global _auth_provider
    _auth_provider = None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth: AuthProvider = Depends(get_auth_provider),
) -> str:
    """强制认证依赖 - 返回 user_id，未登录返回 401"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = auth.verify_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_id


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth: AuthProvider = Depends(get_auth_provider),
) -> Optional[str]:
    """可选认证依赖 - 返回 user_id 或 None（兼容匿名访问）
    
    注意：仅用于少数确实需要兼容匿名的场景（如健康检查）。
    数据端点应使用 get_current_user。
    """
    if credentials is None:
        return None
    
    return auth.verify_token(credentials.credentials)


def verify_token_from_query(
    token: Optional[str],
    auth: Optional[AuthProvider] = None,
) -> str:
    """从 URL query 参数验证 JWT token（用于 SSE EventSource 等无法设置 Header 的场景）
    
    Args:
        token: URL query 参数中的 JWT token
        auth: AuthProvider 实例（为 None 时自动获取）
        
    Returns:
        user_id
        
    Raises:
        HTTPException 401 如果 token 无效或缺失
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated (token query param required)",
        )
    
    if auth is None:
        auth = get_auth_provider()
    
    user_id = auth.verify_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    return user_id


async def verify_session_owner(session_id: str, user_id: str) -> None:
    """校验 session 归属当前用户（防 IDOR 越权访问）
    
    查询逻辑：
    1. 先查内存缓存（活跃会话）
    2. 缓存未命中则查数据库
    3. session 不存在 → 404
    4. session.user_id != user_id → 404（不暴露 session 存在性）
    
    Raises:
        HTTPException 404 如果 session 不存在或不属于当前用户
    """
    from core.session_manager import get_session_manager
    
    session_manager = get_session_manager()
    
    # 先查内存
    session_info = session_manager.get_session_info(session_id)
    if session_info:
        if session_info.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        return
    
    # 再查数据库
    session_data = await session_manager.get_session_info_from_db(session_id)
    if not session_data or session_data.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Session not found")
