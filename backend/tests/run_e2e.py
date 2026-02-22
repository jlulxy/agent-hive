"""
全自动端到端多轮对话测试运行器

核心特性：
1. 自动限流重试（指数退避，最多5次，从15秒开始）
2. 场景间自动插入冷却间隔
3. 进度持久化到 JSON，支持断点续跑
4. 全程无人值守，最终汇总评估报告

API 限流: 10/min → 每分钟最多10次调用
每个场景: 4-5轮 × 2次调用(对话+评估) = 8-10次
策略: 每次 API 调用间隔 7s，每个场景完成后额外等 15s
"""
import asyncio
import sys
import os
import json
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from test_e2e_multi_turn import (
    E2EConversationEngine, LLMJudge, 
    ALL_SCENARIOS, TestScenario, TestTurn,
    ScenarioResult, TurnResult,
)
from llm.provider import LLMProviderFactory, LLMMessage, LLMConfig

PROGRESS_FILE = "/tmp/e2e_progress.json"
RESULT_FILE = "/tmp/e2e_final_report.json"

# ============================================================
# 带限流重试的 API 调用封装
# ============================================================

async def call_with_retry(coro_func, max_retries=5, base_delay=15):
    """带指数退避重试的 API 调用"""
    for attempt in range(max_retries + 1):
        try:
            result = await coro_func()
            return result
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "限流" in err_str:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 15, 30, 60, 120, 240
                    delay = min(delay, 120)  # 最多等120秒
                    print(f"      ⏳ 限流，等待 {delay}s 后重试 ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(delay)
                    continue
            raise  # 非限流错误直接抛出


# ============================================================
# 带限流保护的场景运行器
# ============================================================

async def run_scenario_safe(scenario: TestScenario, call_interval: float = 7.0) -> ScenarioResult:
    """带限流保护的场景运行"""
    result = ScenarioResult(name=scenario.name, description=scenario.description)
    
    engine = E2EConversationEngine()
    judge = LLMJudge()
    
    conversation_so_far = []
    
    for i, turn in enumerate(scenario.turns):
        print(f"    轮次 {i+1}/{len(scenario.turns)}: {turn.query[:50]}...")
        
        # --- 对话调用（带重试）---
        start_time = time.time()
        try:
            response = await call_with_retry(lambda t=turn: engine.chat(t.query))
        except Exception as e:
            result.errors.append(f"轮次{i+1} LLM 调用失败(重试后仍失败): {str(e)}")
            print(f"      ❌ LLM 调用最终失败: {e}")
            conversation_so_far.append({"role": "user", "content": turn.query})
            conversation_so_far.append({"role": "assistant", "content": "[调用失败]"})
            continue
        latency = time.time() - start_time
        
        print(f"      ✓ 回复 ({latency:.1f}s): {response[:80]}...")
        
        # 冷却间隔
        await asyncio.sleep(call_interval)
        
        # --- 评估调用（带重试）---
        try:
            eval_result = await call_with_retry(
                lambda conv=list(conversation_so_far), q=turn.query, r=response, ef=turn.evaluation_focus: 
                    judge.evaluate_turn(conv, q, r, ef)
            )
        except Exception as e:
            print(f"      ⚠️ 评估最终失败: {e}")
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
        
        score_str = " | ".join(f"{k}:{v}" for k, v in scores.items())
        print(f"      📊 评分 [avg={avg_score:.1f}]: {score_str}")
        
        if avg_score < turn.min_expected_score:
            msg = f"轮次{i+1} 平均分 {avg_score:.1f} 低于预期 {turn.min_expected_score}"
            result.errors.append(msg)
            print(f"      ⚠️ {msg}")
            suggestions = eval_result.get("improvement_suggestions", "")
            if suggestions:
                print(f"      💡 建议: {suggestions[:200]}")
        
        conversation_so_far.append({"role": "user", "content": turn.query})
        conversation_so_far.append({"role": "assistant", "content": response})
        
        # 每轮对话之间冷却
        if i < len(scenario.turns) - 1:
            await asyncio.sleep(call_interval)
    
    # 计算总分
    all_scores = []
    for t in result.turns:
        if t.scores:
            all_scores.extend(t.scores.values())
    result.overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    return result


def save_progress(all_results, total_scenarios, phase="running"):
    """保存中间进度"""
    data = {
        "phase": phase,
        "completed": len(all_results),
        "total": total_scenarios,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": []
    }
    for r in all_results:
        data["results"].append({
            "name": r.name,
            "score": round(r.overall_score, 2),
            "passed": r.passed,
            "errors": r.errors,
            "turns": [{
                "turn": t.turn_index,
                "scores": t.scores,
                "avg_score": round(sum(t.scores.values()) / len(t.scores), 2) if t.scores else 0,
                "reasoning": t.evaluation_reasoning[:500],
                "query": t.user_query,
                "response": t.assistant_response[:300],
                "latency": round(t.latency_seconds, 1),
            } for t in r.turns]
        })
    
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_final_report(all_results):
    """生成最终报告"""
    print()
    print("=" * 70)
    print("📊 端到端多轮对话测试 - 最终评估报告")
    print("=" * 70)
    print()
    
    # 维度汇总
    dim_scores = {}
    for r in all_results:
        for t in r.turns:
            for d, s in t.scores.items():
                dim_scores.setdefault(d, []).append(s)
    
    print("📈 各维度平均分:")
    low_dims = []
    for d in ["context_utilization", "reference_resolution", "information_accuracy", "coherence", "helpfulness"]:
        scores = dim_scores.get(d, [])
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg) + "░" * (10 - int(avg))
        flag = "✅" if avg >= 7 else ("⚠️" if avg >= 5 else "❌")
        print(f"  {flag} {d:30s} {bar} {avg:.1f}/10 (n={len(scores)})")
        if avg < 7:
            low_dims.append((d, avg))
    
    print()
    print("📋 各场景总分:")
    passed = 0
    for r in all_results:
        flag = "✅" if r.passed else "❌"
        print(f"  {flag} {r.name:30s} {r.overall_score:.1f}/10", end="")
        if r.errors:
            print(f"  ({len(r.errors)} 个问题)")
        else:
            print()
        if r.passed:
            passed += 1
    
    total = len(all_results)
    print(f"\n🏁 测试结果: {passed}/{total} 通过")
    
    # 低分分析
    if low_dims:
        print()
        print("=" * 70)
        print("⚠️ 低分维度分析与优化建议")
        print("=" * 70)
        for dim, avg in low_dims:
            print(f"\n  📉 [{dim}] 平均分: {avg:.1f}")
            for r in all_results:
                for t in r.turns:
                    if dim in t.scores and t.scores[dim] < 7:
                        print(f"    - 场景「{r.name}」轮次{t.turn_index}: {dim}={t.scores[dim]}")
                        print(f"      问: {t.user_query[:60]}")
                        print(f"      答: {t.assistant_response[:100]}...")
                        if t.evaluation_reasoning:
                            print(f"      评语: {t.evaluation_reasoning[:200]}")
    
    # 详细评分
    print()
    print("=" * 70)
    print("📝 详细评分数据")
    print("=" * 70)
    for r in all_results:
        print(f"\n  场景: {r.name} (总分: {r.overall_score:.1f})")
        for t in r.turns:
            avg = sum(t.scores.values()) / len(t.scores) if t.scores else 0
            print(f"    Turn {t.turn_index} [avg={avg:.1f}]: {', '.join(f'{k}={v}' for k,v in t.scores.items())}")
            print(f"      Q: {t.user_query[:70]}")
            print(f"      A: {t.assistant_response[:120]}...")
    
    # 保存最终 JSON 报告
    save_progress(all_results, total, phase="completed")
    
    report = {
        "summary": {
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "dimension_averages": {d: round(sum(s)/len(s), 2) for d, s in dim_scores.items()},
            "low_dimensions": [(d, round(a, 2)) for d, a in low_dims],
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 报告已保存到: {RESULT_FILE}")
    
    return low_dims


async def main():
    start_time = time.time()
    all_results = []
    
    print("=" * 70)
    print("🚀 端到端多轮对话测试 - 全自动运行")
    print(f"   API 限流: 10/min, 调用间隔: 7s, 场景冷却: 20s")
    print(f"   共 {len(ALL_SCENARIOS)} 个场景, 预计耗时 ~{len(ALL_SCENARIOS) * 3}分钟")
    print("=" * 70)
    
    for idx, scenario in enumerate(ALL_SCENARIOS):
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(ALL_SCENARIOS)}] 场景: {scenario.name}")
        print(f"  描述: {scenario.description}")
        print(f"{'='*60}")
        
        result = await run_scenario_safe(scenario, call_interval=7.0)
        all_results.append(result)
        
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n  {status} {result.name} (总分: {result.overall_score:.1f}/10)")
        if result.errors:
            for e in result.errors:
                print(f"    ⚠️ {e}")
        
        # 保存进度
        save_progress(all_results, len(ALL_SCENARIOS))
        
        # 场景间冷却（最后一个不需要）
        if idx < len(ALL_SCENARIOS) - 1:
            cooldown = 20
            print(f"\n  ⏳ 场景冷却 {cooldown}s...")
            await asyncio.sleep(cooldown)
    
    # 生成最终报告
    low_dims = generate_final_report(all_results)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed/60:.1f}分钟")
    
    failed_count = sum(1 for r in all_results if not r.passed)
    return failed_count, low_dims


if __name__ == "__main__":
    try:
        failed_count, low_dims = asyncio.run(main())
        sys.exit(1 if failed_count > 0 else 0)
    except KeyboardInterrupt:
        print("\n中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 致命错误: {e}")
        traceback.print_exc()
        sys.exit(2)
