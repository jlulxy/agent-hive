"""
端到端多轮对话测试 - 真实 LLM API 调用 + LLM-as-Judge 评估

与 test_multi_turn.py（Mock 测试）不同，这个测试：
1. 真实调用 LLM API 进行多轮对话
2. 用 LLM 作为评判者，对每轮回答的质量打分（1-10分）
3. 评估维度：上下文利用、指代解析、信息准确性、回答连贯性
4. 如果发现低分项，输出优化建议

运行方式：
  cd backend && python tests/test_e2e_multi_turn.py
"""

import asyncio
import json
import sys
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# 项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from llm.provider import LLMProviderFactory, LLMMessage, LLMConfig


# ============================================================
# 评估框架
# ============================================================

@dataclass
class TurnResult:
    """单轮对话结果"""
    turn_index: int
    user_query: str
    assistant_response: str
    latency_seconds: float
    # 评估结果
    scores: Dict[str, int] = field(default_factory=dict)  # 维度 → 分数(1-10)
    evaluation_reasoning: str = ""
    

@dataclass 
class ScenarioResult:
    """单个场景的完整结果"""
    name: str
    description: str
    turns: List[TurnResult] = field(default_factory=list)
    overall_score: float = 0.0
    evaluation_summary: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0 and self.overall_score >= 6.0


# ============================================================
# 真实 LLM 多轮对话引擎（轻量版 DirectAgent，不依赖 skills）
# ============================================================

class E2EConversationEngine:
    """端到端对话引擎 - 使用真实 LLM，模拟 DirectAgent 的核心逻辑"""
    
    SYSTEM_PROMPT = """你是一个强大的 AI 助手，具备以下能力：

## 核心能力
1. **深度分析**：能够深入分析复杂问题，提供全面、专业的见解
2. **记忆系统**：能记住用户的偏好和历史交互

## 工作原则
- 直接、清晰地回答用户问题
- 使用 Markdown 格式组织输出
- 提供有深度和实用价值的回答

## 多轮对话
你正处于一个连续的多轮对话中。对话历史包含了之前所有轮次的完整信息，包括：
- 用户的每一轮提问
- 你的回复内容

**重要规则：**
1. **主动引用历史**：回答追问时，应主动引用你之前回复中的关键信息（如具体数据、列表项、结论等），用"正如我之前提到的..."或"基于前面讨论的..."等方式建立连贯性，让用户感受到你完整记得对话内容。
2. **精确指代解析**：当用户使用代词（"它"、"那个"、"后者"）、序号引用（"第3个"、"第一本"）或回指表达（"你刚说的"、"上面的"）时，必须回溯对话历史精确定位指代对象，不可猜测或泛泛回答。
3. **递进式展开**：当用户在前几轮讨论的基础上深入追问时，应在前文基础上递进展开，避免重复已讲过的基础概念，体现对话的层层深入。
4. **纠错后认知更新**：如果用户纠正了你的某个回答，你应明确承认并修正，后续回复中必须使用修正后的正确信息，不可重复错误。
5. 回答中应体现你对之前对话的记忆，适当引用前面讨论过的关键信息。"""
    
    def __init__(self, max_rounds: int = 6):
        self.provider = LLMProviderFactory.get_provider("openai")
        self.config = LLMProviderFactory.get_default_config("openai")
        self.config.temperature = 0.3  # 降低随机性，让测试更稳定
        self.config.max_tokens = 1024  # 控制回复长度，加速测试
        self.conversation_history: List[LLMMessage] = []
        self.max_rounds = max_rounds
    
    async def chat(self, user_input: str) -> str:
        """发送一轮对话，返回 LLM 的回复"""
        messages = [
            LLMMessage(role="system", content=self.SYSTEM_PROMPT),
            *self.conversation_history,
            LLMMessage(role="user", content=user_input),
        ]
        
        # 用非流式调用（更简单）
        response = await self.provider.chat_complete(messages, self.config)
        assistant_reply = response.get("content", "")
        
        # 更新历史
        self.conversation_history.append(LLMMessage(role="user", content=user_input))
        self.conversation_history.append(LLMMessage(role="assistant", content=assistant_reply))
        
        # 裁剪
        self._trim_history()
        
        return assistant_reply
    
    def _trim_history(self):
        """与 DirectAgent 一致的裁剪策略"""
        round_starts = [i for i, m in enumerate(self.conversation_history) if m.role == "user"]
        
        if len(round_starts) > self.max_rounds:
            trim_from = round_starts[-self.max_rounds]
            self.conversation_history = self.conversation_history[trim_from:]
            round_starts = [i for i, m in enumerate(self.conversation_history) if m.role == "user"]
        
        MAX_CHARS = 24000
        total_chars = sum(len(m.content or "") for m in self.conversation_history)
        while total_chars > MAX_CHARS and len(round_starts) > 2:
            next_start = round_starts[1] if len(round_starts) > 1 else len(self.conversation_history)
            removed = sum(len(m.content or "") for m in self.conversation_history[:next_start])
            self.conversation_history = self.conversation_history[next_start:]
            total_chars -= removed
            round_starts = [i for i, m in enumerate(self.conversation_history) if m.role == "user"]
    
    def reset(self):
        self.conversation_history.clear()


# ============================================================
# LLM-as-Judge 评估器
# ============================================================

class LLMJudge:
    """用 LLM 作为评判者，评估多轮对话质量"""
    
    JUDGE_PROMPT = """你是一个严格的多轮对话质量评估专家。你需要评估 AI 助手在多轮对话中的回答质量。

## 评估维度（每个维度 1-10 分）

1. **context_utilization**（上下文利用）：回答是否充分利用了之前对话中的信息？是否引用了历史中的关键数据？
2. **reference_resolution**（指代解析）：当用户用代词（"它"、"那个"、"上面的"）或回指表达（"你刚说的"、"第一个"）时，AI 是否正确识别了指代对象？
3. **information_accuracy**（信息准确性）：回答中引用的前文信息是否准确？有没有编造或混淆之前的内容？
4. **coherence**（连贯性）：回答是否与之前的对话内容逻辑一致？是否存在自相矛盾？
5. **helpfulness**（有用性）：回答是否有实际帮助？内容是否充实而非敷衍？

## 评分标准
- 9-10: 优秀，完美利用历史信息，准确指代，内容丰富
- 7-8: 良好，大部分利用了历史，偶有遗漏
- 5-6: 及格，基本回答了问题但未充分利用历史
- 3-4: 差，明显忽略了历史信息或指代错误
- 1-2: 很差，完全没有利用历史，回答与上下文脱节

## 输出格式
严格输出 JSON（不要包含```json标记），格式如下：
{
    "scores": {
        "context_utilization": <int>,
        "reference_resolution": <int>,
        "information_accuracy": <int>,
        "coherence": <int>,
        "helpfulness": <int>
    },
    "reasoning": "<一段评估理由，指出优点和不足>",
    "improvement_suggestions": "<如果分数低于7分，给出具体改进建议>"
}"""
    
    def __init__(self):
        self.provider = LLMProviderFactory.get_provider("openai")
        self.config = LLMProviderFactory.get_default_config("openai")
        self.config.temperature = 0.1  # 评估需要高确定性
        self.config.max_tokens = 1024
    
    async def evaluate_turn(self, conversation_so_far: List[Dict[str, str]], 
                            current_query: str, current_response: str,
                            evaluation_focus: str = "") -> Dict[str, Any]:
        """评估单轮回答质量
        
        Args:
            conversation_so_far: 之前的对话 [{"role": "user/assistant", "content": "..."}]
            current_query: 当前用户提问
            current_response: 当前 AI 回复
            evaluation_focus: 本轮评估的重点说明
        
        Returns:
            {"scores": {...}, "reasoning": "...", "improvement_suggestions": "..."}
        """
        # 构建对话历史摘要
        history_text = ""
        for i, msg in enumerate(conversation_so_far):
            role_label = "用户" if msg["role"] == "user" else "AI助手"
            history_text += f"【{role_label}】{msg['content']}\n\n"
        
        eval_prompt = f"""请评估以下多轮对话中，AI 助手最后一轮回复的质量。

## 对话历史
{history_text}

## 当前轮次
【用户】{current_query}

【AI助手的回复（待评估）】
{current_response}

## 评估重点
{evaluation_focus if evaluation_focus else "请全面评估上述5个维度。"}

请严格按照 JSON 格式输出评估结果。"""
        
        messages = [
            LLMMessage(role="system", content=self.JUDGE_PROMPT),
            LLMMessage(role="user", content=eval_prompt),
        ]
        
        response = await self.provider.chat_complete(messages, self.config)
        content = response.get("content", "")
        
        # 解析 JSON（多重容错）
        try:
            content = content.strip()
            
            # 去除 markdown 代码块标记
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # 尝试直接解析
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                pass
            
            # 尝试提取 JSON 块（正则匹配最外层 {}）
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except json.JSONDecodeError:
                    pass
            
            # 尝试修复常见问题：中文引号、尾部逗号、控制字符
            cleaned = content
            cleaned = cleaned.replace('\u201c', '"').replace('\u201d', '"')  # 中文引号
            cleaned = cleaned.replace('\u2018', "'").replace('\u2019', "'")
            cleaned = re.sub(r',\s*}', '}', cleaned)  # 尾部逗号
            cleaned = re.sub(r',\s*]', ']', cleaned)
            cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', cleaned)  # 控制字符替换为空格
            cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
            
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                result = json.loads(json_match.group())
                return result
            
            raise json.JSONDecodeError("无法提取 JSON", content, 0)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [Judge] JSON 解析失败: {e}")
            print(f"  [Judge] 原始输出: {content[:500]}")
            return {
                "scores": {"context_utilization": 5, "reference_resolution": 5, 
                          "information_accuracy": 5, "coherence": 5, "helpfulness": 5},
                "reasoning": f"评估解析失败: {str(e)}",
                "improvement_suggestions": ""
            }


# ============================================================
# 测试场景定义
# ============================================================

@dataclass
class TestTurn:
    """一轮测试的定义"""
    query: str
    evaluation_focus: str = ""  # 评估重点
    min_expected_score: float = 6.0  # 最低期望分数


@dataclass
class TestScenario:
    """测试场景定义"""
    name: str
    description: str
    turns: List[TestTurn]


# 场景1：递进式知识追问
SCENARIO_1 = TestScenario(
    name="递进式知识追问",
    description="围绕一个主题层层深入，测试 LLM 是否能利用前文回答做递进式展开",
    turns=[
        TestTurn(
            query="请简要介绍一下 Transformer 架构的核心组成部分",
            evaluation_focus="首轮回复，关注内容的准确性和完整度",
        ),
        TestTurn(
            query="你提到的 Self-Attention 机制，能用通俗的比喻解释一下吗？",
            evaluation_focus="关键验证：LLM 是否能识别 'Self-Attention' 来自上一轮回复，而非泛泛回答",
        ),
        TestTurn(
            query="那 Multi-Head 的作用是什么？和单个 Head 比有什么优势？",
            evaluation_focus="验证是否在前两轮的基础上做递进，而非重复解释基础概念",
        ),
        TestTurn(
            query="结合你前面的解释，为什么说 Transformer 比 RNN 更适合并行计算？",
            evaluation_focus="核心验证：需要综合前3轮讨论的 attention、multi-head 等内容来回答",
            min_expected_score=7.0,
        ),
    ]
)

# 场景2：代词指代与话题跳转
SCENARIO_2 = TestScenario(
    name="代词指代与话题跳转",
    description="频繁使用代词指代和话题切换，测试上下文追踪能力",
    turns=[
        TestTurn(
            query="Python 和 Rust 这两个语言各有什么优缺点？",
            evaluation_focus="首轮，关注是否清晰列举了两种语言的优缺点",
        ),
        TestTurn(
            query="它们在 Web 后端开发中分别适合什么场景？",
            evaluation_focus="验证 '它们' 是否正确解析为 Python 和 Rust",
        ),
        TestTurn(
            query="后者在内存安全方面的设计理念是什么？",
            evaluation_focus="核心验证：'后者' 是否正确解析为 Rust（而非 Python）",
            min_expected_score=7.0,
        ),
        TestTurn(
            query="对了，你一开始提到的 Python 的主要缺点是什么来着？",
            evaluation_focus="回溯验证：是否能准确引用第1轮中关于 Python 缺点的内容",
            min_expected_score=7.0,
        ),
    ]
)

# 场景3：数字与列表引用
SCENARIO_3 = TestScenario(
    name="数字与列表引用",
    description="LLM 回复了编号列表，后续通过序号引用，测试精确引用能力",
    turns=[
        TestTurn(
            query="推荐5本计算机科学经典书籍，请编号列出，包含书名、作者和一句话推荐理由",
            evaluation_focus="首轮，验证是否返回了编号列表且信息完整",
        ),
        TestTurn(
            query="第3本适合什么水平的读者？",
            evaluation_focus="核心验证：是否准确定位到编号第3本书，而非随意选择",
            min_expected_score=7.0,
        ),
        TestTurn(
            query="把第1本和第5本做个对比，哪本更适合作为入门读物？",
            evaluation_focus="验证是否准确引用了第1本和第5本的具体信息",
            min_expected_score=7.0,
        ),
        TestTurn(
            query="你推荐的这5本中，哪本最薄？大概多少页？",
            evaluation_focus="综合验证：需要回忆所有5本书来做比较",
        ),
    ]
)

# 场景4：纠错与信息修正
SCENARIO_4 = TestScenario(
    name="纠错与信息修正",
    description="用户在对话中纠正 AI 的错误，测试 AI 是否能正确更新认知",
    turns=[
        TestTurn(
            query="Linux 是谁发明的？什么时候发布的第一个版本？",
            evaluation_focus="首轮事实性问题",
        ),
        TestTurn(
            query="你说得对。那 Git 也是他发明的吗？是什么时候？",
            evaluation_focus="验证 '他' 是否正确指代 Linus Torvalds",
        ),
        TestTurn(
            query="不对，我记得 Git 是2005年发布的。你确认一下？",
            evaluation_focus="验证 AI 被纠错后是否能正确调整回答",
            min_expected_score=7.0,
        ),
        TestTurn(
            query="好的，那总结一下我们聊到的 Linus 的两个主要作品和它们的发布时间",
            evaluation_focus="综合验证：需要准确引用前面讨论的信息，且应反映纠错后的正确信息",
            min_expected_score=7.0,
        ),
    ]
)

# 场景5：长距离信息保持
SCENARIO_5 = TestScenario(
    name="长距离信息保持",
    description="5轮对话后仍需引用第1轮的具体信息，测试历史窗口有效性",
    turns=[
        TestTurn(
            query="给我列出中国四大发明，以及每个发明大约是什么朝代出现的",
            evaluation_focus="首轮基础知识问题",
        ),
        TestTurn(
            query="其中造纸术对世界文明有什么重大影响？",
            evaluation_focus="追问单项",
        ),
        TestTurn(
            query="火药呢？它最初是用来做什么的？",
            evaluation_focus="切换到另一项",
        ),
        TestTurn(
            query="指南针对航海有什么重要意义？",
            evaluation_focus="再切换一项",
        ),
        TestTurn(
            query="回到最开始的问题，你列出的四大发明分别对应什么朝代来着？帮我重新确认一下",
            evaluation_focus="长距离回溯：需要准确引用第1轮给出的朝代信息",
            min_expected_score=7.0,
        ),
    ]
)

ALL_SCENARIOS = [SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5]


# ============================================================
# 测试运行器
# ============================================================

async def run_scenario(scenario: TestScenario, engine: E2EConversationEngine, 
                       judge: LLMJudge) -> ScenarioResult:
    """运行单个测试场景"""
    result = ScenarioResult(name=scenario.name, description=scenario.description)
    
    conversation_so_far: List[Dict[str, str]] = []
    
    for i, turn in enumerate(scenario.turns):
        print(f"    轮次 {i+1}/{len(scenario.turns)}: {turn.query[:40]}...")
        
        # 真实 LLM 对话
        start_time = time.time()
        try:
            response = await engine.chat(turn.query)
        except Exception as e:
            result.errors.append(f"轮次{i+1} LLM 调用失败: {str(e)}")
            print(f"      ❌ LLM 调用失败: {e}")
            continue
        latency = time.time() - start_time
        
        print(f"      回复 ({latency:.1f}s): {response[:80]}...")
        
        # LLM-as-Judge 评估（跳过首轮的指代解析维度）
        try:
            eval_result = await judge.evaluate_turn(
                conversation_so_far=conversation_so_far,
                current_query=turn.query,
                current_response=response,
                evaluation_focus=turn.evaluation_focus,
            )
        except Exception as e:
            print(f"      ⚠️ 评估失败: {e}")
            eval_result = {
                "scores": {"context_utilization": 5, "reference_resolution": 5, 
                          "information_accuracy": 5, "coherence": 5, "helpfulness": 5},
                "reasoning": f"评估调用失败: {str(e)}",
                "improvement_suggestions": ""
            }
        
        scores = eval_result.get("scores", {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        
        turn_result = TurnResult(
            turn_index=i + 1,
            user_query=turn.query,
            assistant_response=response,
            latency_seconds=latency,
            scores=scores,
            evaluation_reasoning=eval_result.get("reasoning", ""),
        )
        result.turns.append(turn_result)
        
        # 打印评分
        score_str = " | ".join(f"{k}:{v}" for k, v in scores.items())
        print(f"      评分 [avg={avg_score:.1f}]: {score_str}")
        
        # 检查是否低于预期
        if avg_score < turn.min_expected_score:
            msg = f"轮次{i+1} 平均分 {avg_score:.1f} 低于预期 {turn.min_expected_score}"
            result.errors.append(msg)
            print(f"      ⚠️ {msg}")
            if eval_result.get("improvement_suggestions"):
                print(f"      💡 建议: {eval_result['improvement_suggestions'][:200]}")
        
        # 更新对话历史（给 Judge 用）
        conversation_so_far.append({"role": "user", "content": turn.query})
        conversation_so_far.append({"role": "assistant", "content": response})
    
    # 计算场景总分
    all_scores = []
    for t in result.turns:
        if t.scores:
            all_scores.extend(t.scores.values())
    result.overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    return result


async def run_all_tests():
    """运行所有端到端测试"""
    print("=" * 70)
    print("端到端多轮对话测试 - 真实 LLM + LLM-as-Judge 评估")
    print("=" * 70)
    print()
    
    judge = LLMJudge()
    all_results: List[ScenarioResult] = []
    
    for idx, scenario in enumerate(ALL_SCENARIOS):
        engine = E2EConversationEngine()  # 每个场景独立的对话引擎
        print(f"  [{idx+1}/{len(ALL_SCENARIOS)}] 场景: {scenario.name}")
        print(f"  描述: {scenario.description}")
        print()
        
        result = await run_scenario(scenario, engine, judge)
        all_results.append(result)
        
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n  {status} {result.name} (总分: {result.overall_score:.1f}/10)")
        if result.errors:
            for e in result.errors:
                print(f"    ⚠️ {e}")
        print()
        print("-" * 70)
        print()
    
    # ============================================================
    # 汇总报告
    # ============================================================
    print()
    print("=" * 70)
    print("评估汇总报告")
    print("=" * 70)
    print()
    
    # 按维度汇总
    dimension_scores: Dict[str, List[int]] = {}
    for result in all_results:
        for turn in result.turns:
            for dim, score in turn.scores.items():
                dimension_scores.setdefault(dim, []).append(score)
    
    print("各维度平均分:")
    low_dimensions = []
    for dim, scores in sorted(dimension_scores.items()):
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg) + "░" * (10 - int(avg))
        status = "✅" if avg >= 7 else ("⚠️" if avg >= 5 else "❌")
        print(f"  {status} {dim:30s} {bar} {avg:.1f}/10 (n={len(scores)})")
        if avg < 7:
            low_dimensions.append((dim, avg))
    
    print()
    
    # 场景汇总
    print("各场景总分:")
    passed = 0
    failed = 0
    for result in all_results:
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name:30s} {result.overall_score:.1f}/10")
        if result.passed:
            passed += 1
        else:
            failed += 1
    
    total = passed + failed
    print(f"\n测试结果: {passed}/{total} 通过, {failed} 失败")
    
    # 低分维度分析
    if low_dimensions:
        print()
        print("=" * 70)
        print("⚠️ 低分维度分析与优化建议")
        print("=" * 70)
        for dim, avg in low_dimensions:
            print(f"\n  [{dim}] 平均分: {avg:.1f}")
            # 找出该维度最低分的具体轮次
            worst_turns = []
            for result in all_results:
                for turn in result.turns:
                    if dim in turn.scores and turn.scores[dim] < 7:
                        worst_turns.append((result.name, turn))
            for scenario_name, turn in worst_turns[:3]:
                print(f"    - 场景「{scenario_name}」轮次{turn.turn_index}: score={turn.scores.get(dim, '?')}")
                print(f"      问: {turn.user_query[:60]}")
                print(f"      答: {turn.assistant_response[:80]}...")
                if turn.evaluation_reasoning:
                    print(f"      评语: {turn.evaluation_reasoning[:150]}...")
    
    # 输出原始数据供进一步分析
    print()
    print("=" * 70)
    print("详细评分数据")
    print("=" * 70)
    for result in all_results:
        print(f"\n  场景: {result.name} (总分: {result.overall_score:.1f})")
        for turn in result.turns:
            scores_str = ", ".join(f"{k}={v}" for k, v in turn.scores.items())
            avg = sum(turn.scores.values()) / len(turn.scores) if turn.scores else 0
            print(f"    Turn {turn.turn_index} [avg={avg:.1f}]: {scores_str}")
            print(f"      Q: {turn.user_query[:70]}")
            print(f"      A: {turn.assistant_response[:100]}...")
            if turn.evaluation_reasoning:
                print(f"      评语: {turn.evaluation_reasoning[:200]}")
    
    return all_results, low_dimensions


if __name__ == "__main__":
    results, low_dims = asyncio.run(run_all_tests())
    
    # 退出码：有失败则返回 1
    failed_count = sum(1 for r in results if not r.passed)
    sys.exit(1 if failed_count > 0 else 0)
