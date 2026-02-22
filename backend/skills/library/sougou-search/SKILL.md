---
name: sougou-search
description: 使用搜狗AI搜索接口搜索互联网实时信息。通过内部 API (api.tianji.woa.com) 调用搜狗专业搜索，返回高质量搜索结果（标题、URL、正文段落、来源站点）。适用于需要高质量中文搜索结果的场景。
version: "1.0.0"
author: system
category: search
tags:
  - search
  - internet
  - information
  - real-time
  - sougou
  - chinese
trigger_keywords:
  - 搜索
  - 查找
  - search
  - find
  - look up
  - latest
  - 最新
  - 查询
  - 网上搜
  - 搜狗
  - sougou
requires_packages:
  - httpx
display_name: 搜狗AI搜索
icon: 🔎
---

# Sougou AI Search 搜狗AI搜索

通过搜狗AI专业搜索接口搜索互联网实时信息，返回高质量搜索结果。适用于需要最新数据、时事动态或超出训练数据范围的查询，尤其擅长中文搜索场景。

## 核心能力

本技能通过搜狗AI搜索接口提供**高质量网络搜索能力**：
- 返回真实的搜索结果（标题、URL、正文段落、来源站点）
- 结果包含相关性评分，自动排序
- 支持图片和 Favicon 信息
- 内部 API，稳定可靠

## Scripts 可用脚本

本技能在 `scripts/` 目录下提供以下可执行脚本：

### search.py - 搜狗AI搜索脚本

**路径**: `scripts/search.py`

**功能**: 执行搜狗AI网络搜索，返回高质量中文搜索结果。

**使用方法**:

```bash
# 基础搜索
python scripts/search.py --query "Python 异步编程最佳实践" --max-results 5

# 限制结果数量
python scripts/search.py --query "AI 最新进展" --max-results 15

# 输出 Markdown 格式
python scripts/search.py --query "Rust 教程" --format markdown

# 自定义超时
python scripts/search.py --query "大模型最新进展" --timeout 20
```

**参数说明**:

| 参数 | 缩写 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| --query | -q | ✅ | - | 搜索关键词 |
| --max-results | -n | ❌ | 10 | 返回结果数量 (最大30) |
| --format | -f | ❌ | json | 输出格式: json, markdown |
| --timeout | - | ❌ | 20 | 搜索超时时间（秒） |

**输出格式** (JSON):

```json
{
  "success": true,
  "query": "Python asyncio",
  "type": "web",
  "count": 5,
  "results": [
    {
      "title": "Python asyncio documentation",
      "url": "https://docs.python.org/3/library/asyncio.html",
      "snippet": "asyncio is a library to write concurrent code...",
      "source": "sougou",
      "score": 0.95,
      "site": "docs.python.org",
      "date": "2024-01-15"
    }
  ]
}
```

## Workflow

1. **分析搜索意图**: 理解用户的查询需求
   - 识别核心搜索关键词
   - 确定时间范围要求（最新/历史）
   - 判断结果数量需求

2. **构建搜索查询**: 优化搜索关键词
   - 使用精准的搜索词组
   - 添加必要的限定词
   - 对于中文搜索场景，搜狗AI搜索效果更佳

3. **执行搜索**: 运行 search.py 脚本
   ```bash
   python scripts/search.py --query "<优化后的关键词>" --max-results <数量>
   ```

4. **解析结果**: 处理搜索返回的 JSON
   - 提取每个结果的标题、URL、段落内容
   - 利用 score 字段判断相关性
   - 过滤低质量或不相关内容

5. **整合呈现**: 组织和总结搜索结果
   - 提取关键信息和要点
   - **必须标注信息来源和链接**
   - 多个来源时进行信息整合
   - 对矛盾信息进行说明

## Examples 使用示例

### 示例 1: 技术调研

用户问："帮我搜索 2024 年最流行的 Python Web 框架"

执行:
```bash
python scripts/search.py --query "2024 年最流行 Python Web 框架对比" --max-results 10
```

### 示例 2: 时事查询

用户问："最近 AI 行业有什么重大新闻"

执行:
```bash
python scripts/search.py --query "AI 人工智能 重大突破 最新" --max-results 15
```

### 示例 3: 产品信息

用户问："搜索 Claude 模型的最新能力"

执行:
```bash
python scripts/search.py --query "Claude AI model latest capabilities 2024" --max-results 8
```

## Response Format 响应格式

搜索结果应按以下格式呈现：

```markdown
## 搜索结果

### 1. [结果标题](URL)
> 结果正文段落内容...
*来源: site | 相关度: score*

### 2. [结果标题](URL)
> 结果正文段落内容...
*来源: site*

---

### 信息整合

基于以上搜索结果，总结关键信息...
```

## Guidelines 指导原则

- **优先使用精准关键词**: 避免过长的查询语句
- **合理设置结果数量**: 通常 5-10 个结果足够，复杂问题可增加到 15-20
- **交叉验证**: 对于重要信息，建议从多个结果中验证
- **标注来源**: 使用搜索结果时必须标注出处和链接
- **处理搜索失败**: 如果搜索失败，尝试调整关键词或简化查询
- **利用评分**: 搜狗搜索返回相关性评分，优先使用高评分结果

## Error Handling 错误处理

脚本会返回 JSON 格式的错误信息：

```json
{
  "success": false,
  "error": "错误描述",
  "error_code": "SEARCH_ERROR|MISSING_DEPENDENCY|AUTH_ERROR|UNKNOWN_ERROR",
  "query": "原始查询"
}
```

| 错误类型 | error_code | 处理方式 |
|----------|------------|----------|
| 网络超时 | SEARCH_ERROR | 重试搜索，或提示用户稍后重试 |
| 缺少依赖 | MISSING_DEPENDENCY | 运行 `pip install httpx` 安装依赖 |
| 认证失败 | AUTH_ERROR | 检查 SOUGOU_APPID 和 SOUGOU_SECRET 环境变量 |
| 无结果 | - | 调整关键词，尝试更宽泛的查询 |

## Safety Checks 安全检查

- 验证搜索结果来源的可信度
- 注意区分事实与观点
- 对于敏感话题保持中立客观
- 不传播未经验证的信息
- 标注信息的时效性

## Success Criteria 成功标准

- ✅ 搜索结果与查询意图高度相关
- ✅ 返回真实可访问的 URL 链接
- ✅ 信息来源可靠且可追溯
- ✅ 结果呈现清晰有条理
- ✅ 时效性要求得到满足
