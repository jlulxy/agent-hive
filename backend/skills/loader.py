"""
Skills - SKILL.md 加载器

负责：
1. 解析 SKILL.md 文件（YAML Front Matter + Markdown）
2. 扫描和加载资源文件
3. 构建完整的 Skill 对象

参照 Anthropic Agent Skills 标准：
- SKILL.md 是唯一必需文件
- 以 --- 标记之间的 YAML 元数据开头
- 后跟 Markdown 指令内容
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .models import (
    SkillMetadata,
    SkillInstruction,
    SkillResource,
    SkillResourceType,
    Skill,
)

logger = logging.getLogger(__name__)


class SkillParseError(Exception):
    """技能解析错误"""
    pass


class SkillLoader:
    """
    技能加载器
    
    支持：
    1. 从单个目录加载技能
    2. 批量扫描目录加载所有技能
    3. 热重载（监听文件变化）
    """
    
    SKILL_FILE = "SKILL.md"
    
    # 资源文件模式
    RESOURCE_PATTERNS = {
        SkillResourceType.REFERENCE: [
            "REFERENCE.md", "GUIDE.md", "DOCS.md", 
            "README.md", "*_GUIDE.md", "*_REFERENCE.md"
        ],
        SkillResourceType.EXAMPLE: [
            "EXAMPLES.md", "EXAMPLE.md", "examples/*"
        ],
        SkillResourceType.SCRIPT: [
            "scripts/*.py", "scripts/*.sh", "scripts/*.js"
        ],
        SkillResourceType.TEMPLATE: [
            "templates/*", "template/*"
        ],
        SkillResourceType.ASSET: [
            "assets/*", "images/*", "data/*"
        ]
    }
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Args:
            base_path: 技能库基础路径
        """
        self.base_path = Path(base_path) if base_path else None
        self._cache: Dict[str, Skill] = {}
    
    def load_skill(self, skill_path: str | Path) -> Skill:
        """
        从目录加载技能
        
        Args:
            skill_path: 技能目录路径
            
        Returns:
            Skill 对象
            
        Raises:
            SkillParseError: 解析失败
        """
        path = Path(skill_path)
        
        if not path.exists():
            raise SkillParseError(f"技能目录不存在: {path}")
        
        skill_file = path / self.SKILL_FILE
        if not skill_file.exists():
            raise SkillParseError(f"缺少必需文件 {self.SKILL_FILE}: {path}")
        
        try:
            # 1. 读取并解析 SKILL.md
            content = skill_file.read_text(encoding='utf-8')
            metadata, instruction = self._parse_skill_md(content)
            
            # 2. 扫描资源文件
            resources = self._scan_resources(path)
            
            # 3. 构建 Skill 对象
            skill = Skill(
                metadata=metadata,
                instruction=instruction,
                resources=resources,
                skill_path=str(path),
                is_loaded=True,
                load_time=datetime.now()
            )
            
            # 4. 缓存
            self._cache[skill.name] = skill
            
            logger.info(f"成功加载技能: {skill.display_name} ({skill.name}) from {path}")
            return skill
            
        except Exception as e:
            raise SkillParseError(f"解析技能失败 {path}: {e}")
    
    def _parse_skill_md(self, content: str) -> Tuple[SkillMetadata, SkillInstruction]:
        """
        解析 SKILL.md 内容
        
        格式：
        ```
        ---
        name: skill-name
        description: Skill description...
        ---
        
        # Skill Title
        
        Markdown instruction content...
        ```
        """
        # 分离 YAML Front Matter 和 Markdown 内容
        yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        
        if not yaml_match:
            raise SkillParseError("无效的 SKILL.md 格式：缺少 YAML Front Matter")
        
        yaml_content = yaml_match.group(1)
        markdown_content = yaml_match.group(2).strip()
        
        # 解析 YAML 元数据
        try:
            yaml_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise SkillParseError(f"YAML 解析错误: {e}")
        
        if not yaml_data:
            raise SkillParseError("YAML 元数据为空")
        
        # 验证必需字段
        if 'name' not in yaml_data:
            raise SkillParseError("缺少必需字段: name")
        if 'description' not in yaml_data:
            raise SkillParseError("缺少必需字段: description")
        
        # 构建 SkillMetadata
        metadata = SkillMetadata(
            name=yaml_data['name'],
            description=yaml_data['description'],
            version=yaml_data.get('version', '1.0.0'),
            author=yaml_data.get('author', 'system'),
            tags=yaml_data.get('tags', []),
            dependencies=yaml_data.get('dependencies', []),
            requires_packages=yaml_data.get('requires_packages', []),
            trigger_keywords=yaml_data.get('trigger_keywords', []),
            display_name=yaml_data.get('display_name'),
            icon=yaml_data.get('icon'),
            category=yaml_data.get('category'),
        )
        
        # 解析 Markdown 内容
        instruction = self._parse_markdown_instruction(markdown_content)
        
        return metadata, instruction
    
    def _parse_markdown_instruction(self, content: str) -> SkillInstruction:
        """
        解析 Markdown 指令内容
        
        提取：
        - 标题
        - 工作流程（Workflow/Instructions 章节）
        - 指导原则（Guidelines 章节）
        - 示例（Examples 章节）
        - 安全检查（Safety Checks 章节）
        """
        instruction = SkillInstruction(raw_content=content)
        
        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            instruction.title = title_match.group(1).strip()
        
        # 分割章节
        sections = self._split_sections(content)
        instruction.sections = sections
        
        # 提取工作流程
        workflow_content = sections.get('workflow') or sections.get('instructions') or sections.get('quick start')
        if workflow_content:
            instruction.workflow = self._extract_workflow(workflow_content)
        
        # 提取指导原则
        guidelines_content = sections.get('guidelines') or sections.get('principles')
        if guidelines_content:
            instruction.guidelines = self._extract_list_items(guidelines_content)
        
        # 提取示例
        examples_content = sections.get('examples') or sections.get('example')
        if examples_content:
            instruction.examples = self._extract_examples(examples_content)
        
        # 提取安全检查
        safety_content = sections.get('safety checks') or sections.get('safety')
        if safety_content:
            instruction.safety_checks = self._extract_list_items(safety_content)
        
        # 提取成功标准
        criteria_content = sections.get('success criteria') or sections.get('criteria')
        if criteria_content:
            instruction.success_criteria = self._extract_list_items(criteria_content)
        
        # 概述：第一段内容
        first_para_match = re.search(r'^#.+?\n\n(.+?)(?=\n\n|\n#|$)', content, re.DOTALL)
        if first_para_match:
            instruction.overview = first_para_match.group(1).strip()
        
        return instruction
    
    def _split_sections(self, content: str) -> Dict[str, str]:
        """分割 Markdown 章节"""
        sections = {}
        
        # 匹配 ## 开头的章节
        pattern = r'^##\s+(.+?)\s*\n(.*?)(?=^##\s+|\Z)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for title, body in matches:
            key = title.lower().strip()
            sections[key] = body.strip()
        
        return sections
    
    def _extract_workflow(self, content: str) -> List[Dict[str, Any]]:
        """提取工作流程步骤"""
        workflow = []
        
        # 匹配有序列表
        pattern = r'^\d+\.\s+(.+?)(?=^\d+\.|$)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for i, step in enumerate(matches, 1):
            step_text = step.strip()
            
            # 检查是否有子步骤
            sub_steps = re.findall(r'^\s*[-*]\s+(.+)$', step_text, re.MULTILINE)
            main_text = re.sub(r'\n\s*[-*]\s+.+', '', step_text).strip()
            
            workflow.append({
                "step": i,
                "action": main_text,
                "sub_steps": sub_steps if sub_steps else None
            })
        
        # 如果没有有序列表，尝试无序列表
        if not workflow:
            items = self._extract_list_items(content)
            workflow = [
                {"step": i, "action": item, "sub_steps": None}
                for i, item in enumerate(items, 1)
            ]
        
        return workflow
    
    def _extract_list_items(self, content: str) -> List[str]:
        """提取列表项"""
        items = []
        
        # 匹配有序和无序列表
        pattern = r'^[\d.]*\s*[-*]\s+(.+)$|^\d+\.\s+(.+)$'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for match in matches:
            item = match[0] or match[1]
            if item:
                items.append(item.strip())
        
        return items
    
    def _extract_examples(self, content: str) -> List[Dict[str, str]]:
        """提取示例"""
        examples = []
        
        # 匹配代码块示例
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', content, re.DOTALL)
        for lang, code in code_blocks:
            examples.append({
                "type": "code",
                "language": lang or "text",
                "content": code.strip()
            })
        
        # 匹配文字示例（列表项）
        items = self._extract_list_items(content)
        for item in items:
            examples.append({
                "type": "text",
                "content": item
            })
        
        return examples
    
    def _scan_resources(self, skill_path: Path) -> List[SkillResource]:
        """扫描技能目录中的资源文件"""
        resources = []
        
        for resource_type, patterns in self.RESOURCE_PATTERNS.items():
            for pattern in patterns:
                if '*' in pattern:
                    # 通配符模式
                    matched_files = list(skill_path.glob(pattern))
                else:
                    # 精确匹配
                    file_path = skill_path / pattern
                    matched_files = [file_path] if file_path.exists() else []
                
                for file_path in matched_files:
                    if file_path.is_file() and file_path.name != self.SKILL_FILE and not file_path.name.startswith('._'):
                        relative_path = str(file_path.relative_to(skill_path))
                        resource = SkillResource(
                            name=file_path.stem,
                            path=relative_path,
                            resource_type=resource_type,
                            description=f"{resource_type.value}: {file_path.name}"
                        )
                        resources.append(resource)
        
        return resources
    
    def load_all_skills(self, skills_dir: str | Path) -> List[Skill]:
        """
        批量加载目录下所有技能
        
        Args:
            skills_dir: 技能库目录
            
        Returns:
            加载成功的技能列表
        """
        skills_path = Path(skills_dir)
        loaded_skills = []
        
        if not skills_path.exists():
            logger.warning(f"技能目录不存在: {skills_path}")
            return loaded_skills
        
        # 遍历子目录
        for subdir in skills_path.iterdir():
            if subdir.is_dir() and (subdir / self.SKILL_FILE).exists():
                try:
                    skill = self.load_skill(subdir)
                    loaded_skills.append(skill)
                except SkillParseError as e:
                    logger.error(f"加载技能失败: {e}")
        
        logger.info(f"批量加载完成: {len(loaded_skills)} 个技能")
        return loaded_skills
    
    def get_cached_skill(self, name: str) -> Optional[Skill]:
        """获取缓存的技能"""
        return self._cache.get(name)
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def reload_skill(self, skill_name: str) -> Optional[Skill]:
        """重新加载技能"""
        skill = self._cache.get(skill_name)
        if skill and skill.skill_path:
            self._cache.pop(skill_name, None)
            return self.load_skill(skill.skill_path)
        return None


def load_skill_from_path(skill_path: str | Path) -> Skill:
    """
    便捷函数：从路径加载单个技能
    
    Args:
        skill_path: 技能目录路径
        
        Returns:
        Skill 对象
    """
    loader = SkillLoader()
    return loader.load_skill(skill_path)


def create_skill_template(
    skill_name: str,
    description: str,
    output_dir: str | Path,
    with_scripts: bool = False,
    with_examples: bool = False
) -> Path:
    """
    创建技能模板目录
    
    Args:
        skill_name: 技能名称（小写，连字符分隔）
        description: 技能描述
        output_dir: 输出目录
        with_scripts: 是否包含 scripts 目录
        with_examples: 是否包含 examples 目录
        
    Returns:
        创建的技能目录路径
    """
    output_path = Path(output_dir)
    skill_path = output_path / skill_name
    
    # 创建目录
    skill_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 SKILL.md
    skill_md_content = f'''---
name: {skill_name}
description: {description}
version: "1.0.0"
author: system
tags: []
trigger_keywords: []
---

# {skill_name.replace('-', ' ').title()}

{description}

## Workflow

1. **分析任务**: 理解用户需求和上下文
2. **规划方案**: 制定执行计划
3. **执行操作**: 按照计划执行
4. **验证结果**: 检查输出是否符合预期
5. **总结反馈**: 提供执行总结

## Guidelines

- 遵循最佳实践
- 确保输出质量
- 处理异常情况

## Examples

- 示例用法 1
- 示例用法 2

## Safety Checks

- 验证输入数据
- 检查权限
- 确认操作安全
'''
    
    (skill_path / "SKILL.md").write_text(skill_md_content, encoding='utf-8')
    
    # 可选：创建 scripts 目录
    if with_scripts:
        scripts_dir = skill_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        helper_py = '''"""
辅助脚本
"""

def main():
    print("Helper script executed")

if __name__ == "__main__":
    main()
'''
        (scripts_dir / "helper.py").write_text(helper_py, encoding='utf-8')
    
    # 可选：创建 examples 目录
    if with_examples:
        examples_dir = skill_path / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        example_md = f'''# {skill_name.replace('-', ' ').title()} Examples

## Example 1

Description of example 1...

## Example 2

Description of example 2...
'''
        (examples_dir / "EXAMPLES.md").write_text(example_md, encoding='utf-8')
    
    logger.info(f"创建技能模板: {skill_path}")
    return skill_path
