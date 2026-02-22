#!/usr/bin/env python3
"""
Sougou AI Search Skill - 搜狗AI搜索脚本

通过搜狗AI搜索接口 (api.tianji.woa.com) 执行网络搜索。
返回高质量搜索结果，包含标题、URL、正文段落、来源站点和相关性评分。

环境变量:
    SOUGOU_APPID: 搜狗搜索 AppID
    SOUGOU_SECRET: 搜狗搜索 Secret

依赖: pip install httpx

使用方式:
    # 基础搜索
    python search.py --query "Python 异步编程" --max-results 5

    # Markdown 格式输出
    python search.py --query "AI 最新进展" --format markdown --max-results 10
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
from typing import Any, Dict, List, Optional

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

if not _HAS_HTTPX:
    print(json.dumps({
        "success": False,
        "error": "缺少依赖: 请运行 pip install httpx",
        "error_code": "MISSING_DEPENDENCY"
    }))
    sys.exit(1)


class SearchError(Exception):
    """搜索错误"""
    pass


class SougouSearcher:
    """搜狗AI搜索封装

    通过 api.tianji.woa.com 调用搜狗专业搜索接口。
    接口路径: /rsrc/i/prosearch
    认证方式: appid + md5(secret + timestamp) 签名
    """

    BASE_URL = "http://api.tianji.woa.com"

    def __init__(
        self,
        appid: Optional[str] = None,
        secret: Optional[str] = None,
        timeout: int = 20,
    ):
        self.appid = appid or os.getenv("SOUGOU_APPID", "")
        self.secret = secret or os.getenv("SOUGOU_SECRET", "")
        self.timeout = timeout

        if not self.appid or not self.secret:
            raise SearchError(
                "搜狗搜索凭证未配置: 请设置 SOUGOU_APPID 和 SOUGOU_SECRET 环境变量"
            )

    def _make_auth_params(self) -> Dict[str, str]:
        """生成认证参数: timestamp + md5(secret + timestamp) 签名"""
        ts = str(int(time.time()))
        tk = hashlib.md5((self.secret + ts).encode()).hexdigest()
        return {
            "appid": self.appid,
            "timestamp": ts,
            "tk": tk,
        }

    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """执行搜狗AI搜索

        Args:
            query: 搜索关键词
            max_results: 返回结果数量 (最大30)

        Returns:
            搜索结果列表, 每个结果包含:
            - title: 标题
            - url: 链接
            - snippet: 正文段落
            - score: 相关性评分
            - date: 日期
            - site: 来源站点
            - images: 图片列表
            - favicon: 站点图标
            - source: 固定 "sougou"
        """
        if not query:
            return []

        max_results = min(max_results, 30)
        auth_params = self._make_auth_params()

        params = {
            "keyword": query,
            "open_wx": "1",
            **auth_params,
        }

        path = "/rsrc/i/prosearch"
        url = f"{self.BASE_URL}{path}"

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, verify=False) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            raise SearchError(f"搜狗搜索超时 ({self.timeout}s)")
        except httpx.HTTPStatusError as e:
            raise SearchError(f"搜狗搜索 HTTP 错误: {e.response.status_code}")
        except Exception as e:
            raise SearchError(f"搜狗搜索请求失败: {e}")

        # 解析响应
        if data.get("code") != 0:
            raise SearchError(
                f"搜狗搜索接口错误: code={data.get('code')}, msg={data.get('msg', 'unknown')}"
            )

        docs = (
            data.get("data", {})
            .get("response_data", {})
            .get("docs", [])
        )

        results = []
        for doc in docs[:max_results]:
            results.append({
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "snippet": doc.get("passage", ""),
                "score": doc.get("score", 0.0),
                "date": doc.get("date", ""),
                "site": doc.get("site", ""),
                "images": doc.get("images", []),
                "favicon": doc.get("favicon", ""),
                "source": "sougou",
            })

        return results


def format_results_markdown(results: List[Dict[str, Any]]) -> str:
    """将搜索结果格式化为 Markdown"""
    if not results:
        return "未找到相关结果。"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        site = r.get("site", "")
        score = r.get("score", 0)
        date = r.get("date", "")

        lines.append(f"### {i}. [{title}]({url})")

        meta_parts = []
        if site:
            meta_parts.append(f"来源: {site}")
        if date:
            meta_parts.append(f"日期: {date}")
        if score:
            meta_parts.append(f"相关度: {score:.2f}")
        if meta_parts:
            lines.append(f"*{' | '.join(meta_parts)}*")

        if snippet:
            lines.append(f"> {snippet}")

        lines.append("")

    return "\n".join(lines)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Sougou AI Search - 搜狗AI搜索"
    )

    parser.add_argument(
        "--query", "-q",
        required=True,
        help="搜索关键词",
    )

    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=10,
        help="返回结果数量 (默认10, 最大30)",
    )

    parser.add_argument(
        "--format", "-f",
        choices=["json", "markdown"],
        default="json",
        help="输出格式 (默认: json)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="搜索超时时间（秒）",
    )

    args = parser.parse_args()

    try:
        searcher = SougouSearcher(timeout=args.timeout)
        results = searcher.search(
            query=args.query,
            max_results=args.max_results,
        )

        response = {
            "success": True,
            "query": args.query,
            "type": "web",
            "count": len(results),
            "results": results,
        }

        if args.format == "markdown":
            response["markdown"] = format_results_markdown(results)

        print(json.dumps(response, ensure_ascii=False, indent=2))

    except SearchError as e:
        error_code = "AUTH_ERROR" if "凭证" in str(e) else "SEARCH_ERROR"
        print(json.dumps({
            "success": False,
            "error": str(e),
            "error_code": error_code,
            "query": args.query,
        }, ensure_ascii=False))
        sys.exit(1)

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"未知错误: {e}",
            "error_code": "UNKNOWN_ERROR",
            "query": args.query,
        }, ensure_ascii=False))
        sys.exit(1)


# === 供其他 Python 代码直接调用的函数 ===

def sougou_search(
    query: str,
    max_results: int = 10,
    timeout: int = 20,
) -> Dict[str, Any]:
    """搜狗AI搜索（供直接调用）

    Args:
        query: 搜索关键词
        max_results: 返回结果数量
        timeout: 超时时间（秒）

    Returns:
        包含搜索结果的字典
    """
    try:
        searcher = SougouSearcher(timeout=timeout)
        results = searcher.search(query=query, max_results=max_results)
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }


if __name__ == "__main__":
    main()
