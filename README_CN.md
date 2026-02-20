<p align="center">
  <h1 align="center">🐝 Agent Swarm</h1>
  <p align="center"><strong>让 AI 像顶尖咨询团队一样协作</strong></p>
  <p align="center">动态涌现角色 · 认知对齐通讯 · 专业技能注入 · 交付物导向</p>
  <p align="center"><a href="README.md">English</a> | 中文</p>
</p>

<p align="center">

  <img src="https://img.shields.io/badge/Architecture-Multi--Agent%20Hive-blue" />
  <img src="https://img.shields.io/badge/Protocol-AG--UI-green" />
  <img src="https://img.shields.io/badge/Backend-Python%20FastAPI-yellow" />
  <img src="https://img.shields.io/badge/Frontend-React%20TypeScript-cyan" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" />
</p>

---

## 核心思想

大多数 Multi-Agent 系统沿袭了"固定岗位"模式——开发者预先定义好角色和流水线，Agent 按流程图执行。这种方式面对开放式的复杂任务时，会遇到**角色不匹配、协作靠拼接、产出难交付**三大结构性问题。

Agent Swarm 换了一种思路：**借鉴顶尖咨询公司的项目制团队运作模式**。面对每一个新任务，由 LLM 从零分析需要什么样的专家、如何分工、如何协作，动态"涌现"一支最合适的团队——任务结束后团队解散，能力回归池中。

这不是工程上的微调，而是范式上的跃迁：从"预定义流水线"到"自组织专家蜂巢"。

---

## 四大核心能力

### 1. 动态角色涌现

传统框架需要开发者预设 `Agent(role="researcher")` 等固定角色。Agent Swarm 的 **角色涌现引擎** 让 LLM 根据任务本质自主规划：

```
用户输入: "分析《花样年华》的镜头语言"

→ 自动涌现:
  🎬 镜头分析师  — 视觉叙事语言、构图、色彩
  📖 叙事结构师  — 故事线、时间结构、留白技法
  🎭 视觉符号学家 — 视觉隐喻、文化符号解读
  🎵 配乐解读者  — 声画关系、音乐叙事功能
```

每个涌现角色都不是简单的名字标签，而是包含完整的**工作目标、预期交付物、方法论、成功标准和协作触发器**的专家画像。这意味着无论任务是分析电影、策划广告还是做行业研究，系统都能"组建"出最合适的专家团队，而非在预设角色里"凑合"。

### 2. 中继站认知对齐

多数并发方案是"放羊式"的——Agent 各干各的，最后堆砌合并。Agent Swarm 引入了 **中继站（Relay Station）** 机制，实现实时认知同步：

```
传统 2D 并发:                    Agent Swarm 3D 编排:
                                        ┌──────────────┐
Agent A ──▶ 结果A ─┐                    │   中继站      │
Agent B ──▶ 结果B ─┼─▶ 堆砌       ┌────┤  (战情室)    ├────┐
Agent C ──▶ 结果C ─┘             ↕    ↕              ↕    ↕
                              Agent A  Agent B     Agent C
各干各的,互不知情              实时共享发现、对齐认知、校准方向
```

中继站支持 **10 种消息类型**（发现广播、对齐请求/响应、建议、检查点、人工干预等），Agent 通过**自适应触发机制**自主判断何时需要与他人同步——发现关键信息时广播、遇到不确定时求证、到达进度节点时同步。

用户也可以随时发起**人工干预**，干预信息通过中继站广播给所有相关 Agent，立即影响整个团队的认知方向。

### 3. 专业技能注入

纯 LLM 推理存在输出不稳定和能力天花板两大问题。Agent Swarm 采用 **SKILL.md 驱动的技能注入机制**（借鉴 Claude Skills 模式）同时解决：

每个技能定义为一个 `SKILL.md` 文件——包含完整的专业工作流、指导原则、成功标准和安全检查的结构化 Markdown 文档。角色涌现时，相关技能被**直接注入 Agent 的系统提示**，将通用 LLM 转化为内化了方法论的领域专家。

```
SKILL.md（如"导演"技能）
├── 元数据        → 名称、标签、触发关键词
├── 工作流程       → 分步骤的专业流程
├── 指导原则       → 行业最佳实践
├── 参考示例       → 案例模板
└── 成功标准       → 质量检查点
         ↓
    注入 Agent 的 System Prompt
         ↓
    LLM 像专业导演一样"思考"
```

对于需要真实执行力的技能（如联网搜索），`SKILL.md` 可搭配**可执行脚本**——Agent 调用后获得真实数据（搜索结果、分析输出），再纳入推理链条。

技能在角色涌现时自动匹配——"创意总监"获得导演技能、"内容策划"获得编剧技能、需要实时信息的角色获得搜索技能。技能系统支持灵活扩展：在 `backend/skills/library/` 下添加 `SKILL.md`（及可选脚本）即可注册新技能。

### 4. 交付物导向产出

LLM 最常见的问题是"形式漂亮、内容空洞"。Agent Swarm 在角色涌现时就锚定每个 Agent 必须交付什么：

```python
# 角色涌现时自动定义
{
    "name": "创意总监",
    "work_objective": "制定广告创意方向，确保创意与品牌调性一致",
    "deliverables": ["创意方向文档", "视觉风格指南", "最终创意审核"],
    "methodology": {
        "approach": "以品牌核心价值为出发点，结合目标受众特点",
        "steps": ["分析品牌调性", "确定创意方向", "定义视觉风格", ...],
        "success_criteria": ["创意与品牌一致", "视觉风格统一", "符合目标受众"]
    }
}
```

这套**"目标锚定 → 产出锚定 → 过程锚定 → 质量锚定"**的机制，将 Agent 从"接到指令就开始絮絮叨叨"的对话机器，变成"明确产出物、遵循方法论、交付专业成果"的价值创造者。

---

## 系统架构

```
┌────────────────────────────────────────────────────┐
│                  React Frontend                     │
│   Agent 全景视图 · 中继站面板 · 流式消息 · 人工干预   │
└──────────────────────┬─────────────────────────────┘
                       │ AG-UI Protocol (SSE)
┌──────────────────────┴─────────────────────────────┐
│                  Python Backend                      │
│                                                      │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ Master Agent │──│ 角色涌现引擎 │──│  中继站      │ │
│  └──────┬──────┘  └────────────┘  └──────┬───────┘ │
│         │                                 │         │
│  ┌──────┴──────────────────────────────────┴──────┐ │
│  │          Dynamic Subagents (2-5)               │ │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐      │ │
│  │   │ Agent 1 │  │ Agent 2 │  │ Agent N │      │ │
│  │   └────┬────┘  └────┬────┘  └────┬────┘      │ │
│  └────────┼─────────────┼───────────┼────────────┘ │
│           └─────────────┼───────────┘               │
│                    ┌────┴────┐                       │
│                    │  技能系统 │                      │
│                    └─────────┘                       │
│          reasoning · director · web_search           │
│        screenwriter · visual_designer · ...          │
└──────────────────────┬─────────────────────────────┘
                       │
              ┌────────┴────────┐
              │   LLM Provider   │
              │  OpenAI / Claude │
              └─────────────────┘
```

| 模块 | 路径 | 职责 |
|------|------|------|
| Master Agent | `core/master_agent.py` | 任务分析、角色涌现调度、结果整合 |
| 角色涌现引擎 | `core/role_emergence.py` | LLM 驱动的动态角色生成与技能分配 |
| 中继站 | `core/relay_station.py` | 消息广播、认知对齐、人工干预处理 |
| Subagent 运行时 | `core/subagent.py` | 独立执行单元、技能调用、中继触发 |
| 技能系统 | `skills/` | 技能定义、自动注册、双通道注入与执行 |
| AG-UI 协议 | `agui/` | SSE 事件流，前后端实时通信 |
| 记忆系统 | `memory/` | 用户偏好与知识的持久化记忆 |
| 会话管理 | `core/session_manager.py` | 多会话隔离与历史管理 |

---

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 18+
- OpenAI API Key 或兼容的 API 端点

### 方式一：一键启动

```bash
git clone https://github.com/jlulxy/agent-swarm.git
cd agent-swarm

# 配置 API Key
cp backend/.env.example backend/.env
# 编辑 backend/.env，填入你的 API Key

chmod +x start.sh
./start.sh
```

### 方式二：Docker

```bash
# 配置 API Key
cp backend/.env.example backend/.env
# 编辑 backend/.env

docker compose up --build
```

### 方式三：分别启动

```bash
# 后端
cd backend
pip install -r requirements.txt
python main.py

# 前端（新终端）
cd frontend
npm install
npm run dev
```

### 访问

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:3000 |
| 后端 API | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |

---

## 使用示例

**电影深度分析：**

输入："深度分析电影《盗梦空间》的叙事结构和视觉语言"

系统自动涌现镜头分析师、叙事结构师、音效解读者、整合分析师四个专家角色，通过中继站实时共享发现（如"大量对称构图"与"宿命主题"的关联），最终产出跨维度交叉印证的深度分析报告。

**品牌广告创意：**

输入："为新能源汽车品牌创作30秒广告创意方案"

系统涌现创意总监、内容策划、视觉设计师，分别负责创意方向把控、脚本文案创作、视觉风格设计。交付物包括创意方向文档、分镜脚本、视觉风格指南，而非泛泛而谈的"建议"。

---

## 与主流方案的定位差异

| 维度 | 单体 Agent (Claude Code 等) | 预定义 Multi-Agent (AutoGen/CrewAI) | **Agent Swarm** |
|------|---------------------------|--------------------------------------|----------------|
| 角色 | 固定角色 | 开发者预设 | **LLM 动态涌现** |
| 协作 | 主-子调度 | 预设流程图 | **中继站实时对齐** |
| 能力 | 领域专精 | 依赖工具定义 | **双通道技能注入** |
| 产出 | 代码/对话 | 各自独立输出 | **交付物导向 + 成功标准** |
| 最佳场景 | 线性任务 | 固定流程任务 | **开放式复杂协作任务** |

Agent Swarm 不是要取代单体 Agent 或固定流程框架——当任务复杂到需要"组建专家团队"而非"找一个高手"时，Agent Swarm 是更优选择。

---

## 技术栈

**后端：** Python 3.9+ · FastAPI · Uvicorn · SQLite · OpenAI SDK · bcrypt + JWT

**前端：** React 18 · TypeScript · Vite · Tailwind CSS · Zustand · Lucide Icons

---

## 配置说明

复制 `backend/.env.example` 为 `backend/.env` 后编辑：

```bash
# 必须配置
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1    # 或其他兼容端点
OPENAI_MODEL=gpt-4o                           # 推荐 GPT-4 级别模型

# 可选配置
HOST=0.0.0.0
PORT=8000
DEBUG=false                                    # true 启用热重载
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
JWT_SECRET=change-this-to-a-random-string
```

完整配置项参见 `backend/.env.example`。

---

## 项目结构

```
agent-swarm/
├── backend/
│   ├── main.py              # 入口
│   ├── core/                # 核心引擎
│   │   ├── master_agent.py  # Master Agent
│   │   ├── role_emergence.py# 角色涌现
│   │   ├── relay_station.py # 中继站
│   │   ├── subagent.py      # Subagent 运行时
│   │   └── models.py        # 数据模型
│   ├── skills/              # 技能系统
│   │   ├── library/         # 技能库（可扩展）
│   │   ├── registry.py      # 技能注册
│   │   ├── executor.py      # 技能执行
│   │   └── loader.py        # 技能加载
│   ├── agui/                # AG-UI 协议
│   ├── memory/              # 记忆系统
│   ├── auth/                # 认证
│   └── api/                 # API 路由
├── frontend/
│   └── src/
│       ├── App.tsx           # 主界面
│       ├── components/       # UI 组件
│       ├── hooks/            # AG-UI Hook
│       └── store/            # 状态管理
├── start.sh                  # 一键启动
├── docker-compose.yml        # Docker 部署
└── Dockerfile
```

---

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发流程和规范。

## 设计哲学

> **"不是让 AI 更聪明，而是让 AI 更专业地协作。"**

- **角色涌现** 解决 "谁来做" — 让最合适的专家来处理任务
- **中继通讯** 解决 "怎么协作" — 让专家之间真正对齐认知
- **技能注入** 解决 "做得好" — 让专家拥有专业工具和方法论
- **交付物导向** 解决 "交得出" — 让产出真正可交付、可使用

这四者相互增强，形成一个自组织、自协调的智能协作蜂巢。

## License

[MIT](LICENSE)
