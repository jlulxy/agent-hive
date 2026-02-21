"""
Skills - 技能执行器

负责：
1. 执行技能（将技能指令注入 LLM 上下文）
2. 运行辅助脚本
3. 管理执行上下文
4. 结果处理和错误处理
"""

import asyncio
import subprocess
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .models import (
    Skill,
    SkillExecutionContext,
    SkillExecutionResult,
    SkillResourceType,
)
from .registry import SkillRegistry, get_global_registry

logger = logging.getLogger(__name__)


class SkillExecutor:
    """
    技能执行器
    
    执行模式：
    1. Prompt 注入模式 - 将技能指令注入 LLM 系统提示
    2. 脚本执行模式 - 运行技能附带的脚本
    3. 混合模式 - 结合 LLM 推理和脚本执行
    """
    
    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        timeout_seconds: float = 120.0,
        max_script_output: int = 10000
    ):
        """
        Args:
            registry: 技能注册表
            timeout_seconds: 执行超时时间
            max_script_output: 脚本最大输出长度
        """
        self.registry = registry or get_global_registry()
        self.timeout_seconds = timeout_seconds
        self.max_script_output = max_script_output
        
        # 执行历史
        self.execution_history: List[SkillExecutionResult] = []
    
    async def prepare_context(
        self,
        skill_name: str,
        task: str,
        agent_id: str,
        agent_name: str,
        load_resources: bool = False,
        resource_types: Optional[List[SkillResourceType]] = None,
        **kwargs
    ) -> Optional[SkillExecutionContext]:
        """
        准备执行上下文
        
        渐进式披露：
        - 默认只加载核心指令
        - 按需加载资源内容
        
        Args:
            skill_name: 技能名
            task: 任务描述
            agent_id: Agent ID
            agent_name: Agent 名称
            load_resources: 是否加载资源
            resource_types: 要加载的资源类型
            
        Returns:
            执行上下文，技能不存在返回 None
        """
        skill = self.registry.get(skill_name)
        if not skill:
            logger.error(f"技能不存在: {skill_name}")
            return None
        
        # 创建上下文
        context = SkillExecutionContext(
            skill_name=skill_name,
            agent_id=agent_id,
            agent_name=agent_name,
            task=task,
            options=kwargs
        )
        
        # 按需加载资源
        if load_resources:
            context.loaded_resources = self.registry.load_skill_resources(
                skill_name,
                resource_types=resource_types
            )
        
        return context
    
    def generate_prompt_injection(
        self,
        skill_name: str,
        context: Optional[SkillExecutionContext] = None,
        include_resources: bool = False
    ) -> str:
        """
        生成要注入 LLM 系统提示的内容
        
        这是技能的核心执行方式：将技能的指令、工作流、指南
        注入到 LLM 的上下文中，引导 LLM 按技能定义执行任务
        
        Args:
            skill_name: 技能名
            context: 执行上下文
            include_resources: 是否包含资源内容
            
        Returns:
            要注入的提示内容
        """
        skill = self.registry.get(skill_name)
        if not skill:
            return ""
        
        parts = []
        
        # 技能标题和描述
        parts.append(f"## 技能: {skill.display_name}")
        parts.append(f"\n{skill.description}\n")
        
        # 核心指令
        instruction = skill.instruction
        
        # 工作流程（SOP）
        if instruction.workflow:
            parts.append("\n### 工作流程\n")
            for step in instruction.workflow:
                step_num = step.get('step', '')
                action = step.get('action', '')
                parts.append(f"{step_num}. {action}")
                
                # 子步骤
                sub_steps = step.get('sub_steps')
                if sub_steps:
                    for sub in sub_steps:
                        parts.append(f"   - {sub}")
        
        # 指导原则
        if instruction.guidelines:
            parts.append("\n### 指导原则\n")
            for guideline in instruction.guidelines:
                parts.append(f"- {guideline}")
        
        # 安全检查
        if instruction.safety_checks:
            parts.append("\n### 安全检查\n")
            for check in instruction.safety_checks:
                parts.append(f"- {check}")
        
        # 成功标准
        if instruction.success_criteria:
            parts.append("\n### 成功标准\n")
            for criteria in instruction.success_criteria:
                parts.append(f"- {criteria}")
        
        # 示例
        if instruction.examples:
            parts.append("\n### 示例\n")
            for example in instruction.examples[:3]:  # 限制示例数量
                if example.get('type') == 'code':
                    lang = example.get('language', '')
                    code = example.get('content', '')
                    parts.append(f"```{lang}\n{code}\n```")
                else:
                    parts.append(f"- {example.get('content', '')}")
        
        # 可选：包含资源内容
        if include_resources and context and context.loaded_resources:
            parts.append("\n### 参考资料\n")
            for name, content in context.loaded_resources.items():
                # 截断过长的资源
                if len(content) > 2000:
                    content = content[:2000] + "\n...[内容已截断]"
                parts.append(f"\n#### {name}\n{content}")
        
        return "\n".join(parts)
    
    async def execute_script(
        self,
        skill_name: str,
        script_name: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None
    ) -> SkillExecutionResult:
        """
        执行技能脚本
        
        Args:
            skill_name: 技能名
            script_name: 脚本名（不含路径）
            args: 脚本参数
            env: 环境变量
            cwd: 工作目录
            
        Returns:
            执行结果
        """
        start_time = time.time()
        
        skill = self.registry.get(skill_name)
        if not skill or not skill.skill_path:
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"技能不存在或路径未知: {skill_name}",
                error_code="SKILL_NOT_FOUND"
            )
        
        # 查找脚本（跳过 macOS ._* 元数据文件）
        scripts = skill.get_scripts()
        script_resource = None
        for s in scripts:
            if s.name.startswith('._'):
                continue
            if script_name in s.path or s.name == script_name:
                script_resource = s
                break
        
        if not script_resource:
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"脚本不存在: {script_name}",
                error_code="SCRIPT_NOT_FOUND"
            )
        
        script_path = Path(skill.skill_path) / script_resource.path
        
        if not script_path.exists():
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"脚本文件不存在: {script_path}",
                error_code="SCRIPT_FILE_NOT_FOUND"
            )
        
        try:
            # 构建命令
            if script_path.suffix == '.py':
                import shutil
                python_bin = shutil.which('python3') or shutil.which('python') or 'python3'
                cmd = [python_bin, str(script_path)] + (args or [])
            elif script_path.suffix == '.sh':
                cmd = ['bash', str(script_path)] + (args or [])
            elif script_path.suffix == '.js':
                cmd = ['node', str(script_path)] + (args or [])
            else:
                cmd = [str(script_path)] + (args or [])
            
            # 执行脚本
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or skill.skill_path,
                env=env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                return SkillExecutionResult(
                    skill_name=skill_name,
                    success=False,
                    error=f"脚本执行超时 ({self.timeout_seconds}s)",
                    error_code="SCRIPT_TIMEOUT",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            stdout_str = stdout.decode('utf-8', errors='replace')[:self.max_script_output]
            stderr_str = stderr.decode('utf-8', errors='replace')[:self.max_script_output]
            
            success = process.returncode == 0
            
            result = SkillExecutionResult(
                skill_name=skill_name,
                success=success,
                result=stdout_str if success else None,
                result_type="text",
                summary=f"脚本 {script_name} 执行{'成功' if success else '失败'}",
                error=stderr_str if not success else None,
                error_code="SCRIPT_ERROR" if not success else None,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "script": script_name,
                    "return_code": process.returncode,
                    "stdout_length": len(stdout_str),
                    "stderr_length": len(stderr_str)
                }
            )
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                error=str(e),
                error_code="SCRIPT_EXECUTION_ERROR",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def execute(
        self,
        skill_name: str,
        task: str,
        agent_id: str,
        agent_name: str,
        mode: str = "prompt",  # prompt, script, hybrid
        script_name: Optional[str] = None,
        script_args: Optional[List[str]] = None,
        **kwargs
    ) -> SkillExecutionResult:
        """
        执行技能
        
        执行模式：
        - prompt: 生成提示注入（主要模式）
        - script: 执行脚本
        - hybrid: 先执行脚本，结果作为上下文
        
        Args:
            skill_name: 技能名
            task: 任务描述
            agent_id: Agent ID
            agent_name: Agent 名称
            mode: 执行模式
            script_name: 脚本名（script/hybrid 模式需要）
            script_args: 脚本参数
            
        Returns:
            执行结果
        """
        start_time = time.time()
        
        skill = self.registry.get(skill_name)
        if not skill:
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"技能不存在: {skill_name}",
                error_code="SKILL_NOT_FOUND"
            )
        
        # 准备上下文
        context = await self.prepare_context(
            skill_name=skill_name,
            task=task,
            agent_id=agent_id,
            agent_name=agent_name,
            load_resources=kwargs.get('load_resources', False),
            **kwargs
        )
        
        if mode == "prompt":
            # Prompt 注入模式
            prompt = self.generate_prompt_injection(
                skill_name,
                context,
                include_resources=kwargs.get('include_resources', False)
            )
            
            result = SkillExecutionResult(
                skill_name=skill_name,
                success=True,
                result=prompt,
                result_type="markdown",
                summary=f"技能 {skill.display_name} 的指令已准备就绪",
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "mode": "prompt",
                    "prompt_length": len(prompt),
                    "workflow_steps": len(skill.instruction.workflow)
                }
            )
            
        elif mode == "script" and script_name:
            # 脚本执行模式
            result = await self.execute_script(
                skill_name,
                script_name,
                args=script_args
            )
            
        elif mode == "hybrid" and script_name:
            # 混合模式：先执行脚本，再生成提示
            script_result = await self.execute_script(
                skill_name,
                script_name,
                args=script_args
            )
            
            if script_result.success:
                # 将脚本结果加入上下文
                if context:
                    context.loaded_resources['script_output'] = script_result.result
                
                prompt = self.generate_prompt_injection(
                    skill_name,
                    context,
                    include_resources=True
                )
                
                result = SkillExecutionResult(
                    skill_name=skill_name,
                    success=True,
                    result={
                        "prompt": prompt,
                        "script_output": script_result.result
                    },
                    result_type="json",
                    summary=f"技能 {skill.display_name} 混合执行完成",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    metadata={
                        "mode": "hybrid",
                        "script": script_name
                    }
                )
            else:
                result = script_result
        else:
            result = SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"无效的执行模式或缺少参数: {mode}",
                error_code="INVALID_MODE"
            )
        
        self.execution_history.append(result)
        return result
    
    def get_tool_definitions(
        self,
        skill_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """获取技能的 Tool 定义"""
        return self.registry.get_tool_definitions(skill_names)
    
    def get_system_prompt_for_skills(
        self,
        skill_names: Optional[List[str]] = None,
        include_resources: bool = False
    ) -> str:
        """获取技能的系统提示"""
        return self.registry.get_system_prompt_for_skills(
            skill_names,
            include_resources
        )
    
    def clear_history(self):
        """清空执行历史"""
        self.execution_history.clear()


class AgentSkillSet:
    """
    Agent 技能集
    
    管理单个 Agent 被分配的技能
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        executor: Optional[SkillExecutor] = None
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.executor = executor or SkillExecutor()
        
        # 分配的技能
        self._assigned_skills: List[str] = []
        self._skill_configs: Dict[str, Dict[str, Any]] = {}
    
    def assign_skill(
        self,
        skill_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """分配技能"""
        if not self.executor.registry.has(skill_name):
            logger.warning(f"技能不存在: {skill_name}")
            return False
        
        if skill_name not in self._assigned_skills:
            self._assigned_skills.append(skill_name)
        
        if config:
            self._skill_configs[skill_name] = config
        
        return True
    
    def assign_skills(self, skill_names: List[str]) -> int:
        """批量分配技能"""
        return sum(1 for name in skill_names if self.assign_skill(name))
    
    def remove_skill(self, skill_name: str) -> bool:
        """移除技能"""
        if skill_name in self._assigned_skills:
            self._assigned_skills.remove(skill_name)
            self._skill_configs.pop(skill_name, None)
            return True
        return False
    
    def has_skill(self, skill_name: str) -> bool:
        """检查是否有某技能"""
        return skill_name in self._assigned_skills
    
    def list_skills(self) -> List[str]:
        """列出所有技能"""
        return self._assigned_skills.copy()
    
    def get_skills(self) -> List[Skill]:
        """获取所有技能对象"""
        return self.executor.registry.get_by_names(self._assigned_skills)
    
    async def execute_skill(
        self,
        skill_name: str,
        task: str,
        mode: str = "prompt",
        **kwargs
    ) -> SkillExecutionResult:
        """执行技能"""
        if not self.has_skill(skill_name):
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"Agent '{self.agent_name}' 未分配技能 '{skill_name}'",
                error_code="SKILL_NOT_ASSIGNED"
            )
        
        # 合并配置
        config = self._skill_configs.get(skill_name, {})
        merged_kwargs = {**config, **kwargs}
        
        return await self.executor.execute(
            skill_name=skill_name,
            task=task,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            mode=mode,
            **merged_kwargs
        )
    
    def get_system_prompt(self, include_resources: bool = False) -> str:
        """获取技能系统提示"""
        return self.executor.get_system_prompt_for_skills(
            self._assigned_skills,
            include_resources
        )
    
    def get_system_prompt_injection(self, include_resources: bool = False) -> str:
        """
        获取技能系统提示（注入格式）
        
        这是 get_system_prompt 的别名，保持与 v1 API 的兼容性
        """
        return self.get_system_prompt(include_resources)
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取 Tool 定义"""
        return self.executor.get_tool_definitions(self._assigned_skills)
