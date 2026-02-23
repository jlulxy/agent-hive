"""
LLM Provider - 统一的 LLM 接口抽象

支持 OpenAI 和 Claude API
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List, Dict, Any
from pydantic import BaseModel
import os


class LLMMessage(BaseModel):
    """LLM 消息"""
    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_api_dict(self) -> Dict[str, Any]:
        """序列化为 API 请求格式，处理空 content 兼容性问题。
        
        Venus/OpenAI 兼容 API 要求每条消息 content 字段必须存在且非空。
        当 assistant 消息有 tool_calls 但 content 为空时，
        需要填充一个占位文本以通过 API 校验。
        """
        d = self.model_dump(exclude_none=True)
        # assistant 消息有 tool_calls 时，确保 content 非空
        # Venus API 要求 content 字段必须存在且非空（不同于 OpenAI 原生 API 接受 null）
        if self.role == "assistant" and self.tool_calls and not self.content:
            d["content"] = "Calling tools..."
        return d


class LLMConfig(BaseModel):
    """LLM 配置"""
    model: str
    temperature: float = 0.7
    max_tokens: int = 16384  # 增大到 16K，支持复杂输出
    top_p: Optional[float] = None  # 默认不设置，避免与 temperature 冲突
    stream: bool = True


class LLMProvider(ABC):
    """LLM 提供者抽象基类"""
    
    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """流式聊天"""
        pass
    
    @abstractmethod
    async def chat_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """非流式聊天，返回完整响应"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI Provider"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import AsyncOpenAI
        import httpx
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=httpx.Timeout(120.0, connect=10.0),  # 总超时 120s，连接超时 10s
        )
    
    async def chat(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """流式聊天"""
        request_params = {
            "model": config.model,
            "messages": [m.to_api_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True,
        }
        
        # 只有明确设置了 top_p 时才传递，避免与 temperature 冲突
        if config.top_p is not None:
            request_params["top_p"] = config.top_p
        
        if tools:
            request_params["tools"] = tools
        
        try:
            stream = await self.client.chat.completions.create(**request_params)
        except Exception as e:
            print(f"[OpenAIProvider] Stream chat error: {type(e).__name__}: {e}")
            raise
        
        async for chunk in stream:
            # 检查 choices 是否为空
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
    
    async def chat_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """非流式聊天"""
        request_params = {
            "model": config.model,
            "messages": [m.to_api_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": False,
        }
        
        # 只有明确设置了 top_p 时才传递，避免与 temperature 冲突
        if config.top_p is not None:
            request_params["top_p"] = config.top_p
        
        if tools:
            request_params["tools"] = tools
        
        print(f"[OpenAIProvider] chat_complete request: model={config.model}, temp={config.temperature}")
        
        # 调试：打印消息概要
        api_messages = request_params["messages"]
        for idx, m in enumerate(api_messages):
            role = m.get("role", "?")
            has_content = "content" in m
            content_val = m.get("content")
            content_info = f"None" if content_val is None else f"len={len(content_val)}" if content_val else "empty"
            has_tc = "tool_calls" in m
            print(f"  [{idx}] role={role}, content={content_info}, has_tc={has_tc}")
        
        try:
            response = await self.client.chat.completions.create(**request_params)
            print(f"[OpenAIProvider] Got response, choices count: {len(response.choices) if response.choices else 0}")
        except Exception as e:
            print(f"[OpenAIProvider] API Error: {type(e).__name__}: {e}")
            raise
        
        # 检查 choices 是否有效
        if not response.choices or len(response.choices) == 0:
            print(f"[OpenAIProvider] Warning: No choices in response!")
            return {
                "content": "",
                "tool_calls": None,
                "finish_reason": "error"
            }
        
        message = response.choices[0].message
        content = message.content or ""
        print(f"[OpenAIProvider] Response content length: {len(content)}")
        
        result = {
            "content": content,
            "tool_calls": None,
            "finish_reason": response.choices[0].finish_reason
        }
        
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        return result


class ClaudeProvider(LLMProvider):
    """Claude Provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
    
    def _build_claude_messages(self, messages: List[LLMMessage]):
        """将 LLMMessage 列表转换为 Claude API 格式
        
        处理：
        - system 消息提取为独立字段
        - assistant(tool_calls) → assistant content blocks (tool_use)
        - tool 消息 → user content blocks (tool_result)
        - 合并连续的 tool_result 消息到同一个 user message
        
        Returns:
            (system_content, chat_messages) 元组
        """
        system_content = ""
        chat_messages = []
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            if msg.role == "system":
                system_content = msg.content
                i += 1
                continue
            
            if msg.role == "assistant" and msg.tool_calls:
                # assistant 消息包含 tool_calls → 转换为 Claude 格式的 content blocks
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    import json as _json
                    tool_input = tc["function"]["arguments"]
                    if isinstance(tool_input, str):
                        try:
                            tool_input = _json.loads(tool_input)
                        except _json.JSONDecodeError:
                            tool_input = {"raw": tool_input}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": tool_input,
                    })
                chat_messages.append({"role": "assistant", "content": content_blocks})
                i += 1
                continue
            
            if msg.role == "tool":
                # tool 消息 → 转换为 user message 的 tool_result content blocks
                # 合并连续的 tool 消息
                tool_results = []
                while i < len(messages) and messages[i].role == "tool":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": messages[i].tool_call_id,
                        "content": messages[i].content,
                    })
                    i += 1
                chat_messages.append({"role": "user", "content": tool_results})
                continue
            
            # 普通 user/assistant 消息
            chat_messages.append({
                "role": msg.role,
                "content": msg.content
            })
            i += 1
        
        return system_content, chat_messages
    
    async def chat(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """流式聊天"""
        system_content, chat_messages = self._build_claude_messages(messages)
        
        request_params = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": chat_messages,
        }
        
        if system_content:
            request_params["system"] = system_content
        
        if tools:
            claude_tools = self._convert_tools(tools)
            request_params["tools"] = claude_tools
        
        async with self.client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text
    
    async def chat_complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """非流式聊天"""
        system_content, chat_messages = self._build_claude_messages(messages)
        
        request_params = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": chat_messages,
        }
        
        if system_content:
            request_params["system"] = system_content
        
        if tools:
            claude_tools = self._convert_tools(tools)
            request_params["tools"] = claude_tools
        
        response = await self.client.messages.create(**request_params)
        
        result = {
            "content": "",
            "tool_calls": None,
            "finish_reason": response.stop_reason
        }
        
        for block in response.content:
            if block.type == "text":
                result["content"] = block.text
            elif block.type == "tool_use":
                if result["tool_calls"] is None:
                    result["tool_calls"] = []
                result["tool_calls"].append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": str(block.input)
                    }
                })
        
        return result
    
    def _convert_tools(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将 OpenAI 格式的 tools 转换为 Claude 格式"""
        claude_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                claude_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        return claude_tools


class LLMProviderFactory:
    """LLM Provider 工厂"""
    
    _providers: Dict[str, LLMProvider] = {}
    
    @classmethod
    def get_provider(cls, provider_type: str = "openai") -> LLMProvider:
        """获取 LLM Provider"""
        if provider_type not in cls._providers:
            if provider_type == "openai":
                cls._providers[provider_type] = OpenAIProvider()
            elif provider_type == "claude":
                cls._providers[provider_type] = ClaudeProvider()
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")
        
        return cls._providers[provider_type]
    
    @classmethod
    def get_default_config(cls, provider_type: str = "openai") -> LLMConfig:
        """获取默认配置"""
        if provider_type == "openai":
            # 使用兼容性更好的模型名称
            return LLMConfig(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
        elif provider_type == "claude":
            return LLMConfig(model=os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229"))
        else:
            return LLMConfig(model="gpt-4o")
