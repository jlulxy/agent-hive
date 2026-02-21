#!/usr/bin/env python3
"""
Web Search Skill - 搜索脚本

使用 DuckDuckGo 实现真实的网络搜索功能。
支持通用搜索和新闻搜索。

依赖: pip install duckduckgo-search

使用方式:
    # 通用搜索
    python search.py --query "Python 异步编程" --max-results 5
    
    # 新闻搜索
    python search.py --query "AI 最新进展" --type news --max-results 10
    
    # 限定时间范围
    python search.py --query "React 19" --time-range w
"""

import argparse
import json
import sys
import asyncio
from typing import List, Dict, Any, Optional

import re
import html as html_module

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    from ddgs import DDGS
    DuckDuckGoSearchException = Exception
    _HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import DuckDuckGoSearchException
        _HAS_DDGS = True
    except ImportError:
        _HAS_DDGS = False
        DuckDuckGoSearchException = Exception

if not _HAS_DDGS and not _HAS_HTTPX:
    print(json.dumps({
        "success": False,
        "error": "缺少依赖: 请运行 pip install ddgs httpx",
        "error_code": "MISSING_DEPENDENCY"
    }))
    sys.exit(1)


class BingFallbackSearcher:
    """Bing 搜索 fallback — 使用 httpx 直接抓取 Bing HTML 结果"""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        url = "https://www.bing.com/search"
        params = {"q": query, "count": str(min(max_results * 2, 30))}
        with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=self.HEADERS) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return self._parse_html(resp.text, max_results)

    def search_news(self, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        url = "https://www.bing.com/news/search"
        params = {"q": query, "count": str(min(max_results * 2, 30))}
        with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=self.HEADERS) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return self._parse_news_html(resp.text, max_results)

    @staticmethod
    def _parse_html(body: str, max_results: int) -> List[Dict[str, Any]]:
        results = []
        li_pattern = re.compile(r'<li class="b_algo"[^>]*>(.*?)</li>', re.DOTALL)
        for m in li_pattern.finditer(body):
            block = m.group(1)
            href_m = re.search(r'<a\s+href="(https?://[^"]+)"', block)
            title_m = re.search(r'<a[^>]*>(.*?)</a>', block, re.DOTALL)
            snippet_m = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
            if href_m:
                title_raw = title_m.group(1) if title_m else ""
                title_clean = re.sub(r'<[^>]+>', '', title_raw).strip()
                snippet_raw = snippet_m.group(1) if snippet_m else ""
                snippet_clean = re.sub(r'<[^>]+>', '', snippet_raw).strip()
                results.append({
                    "title": html_module.unescape(title_clean),
                    "url": href_m.group(1),
                    "snippet": html_module.unescape(snippet_clean),
                    "source": "bing",
                })
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _parse_news_html(body: str, max_results: int) -> List[Dict[str, Any]]:
        results = []
        card_pattern = re.compile(r'<a[^>]+class="[^"]*title[^"]*"[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        for m in card_pattern.finditer(body):
            title_clean = re.sub(r'<[^>]+>', '', m.group(2)).strip()
            results.append({
                "title": html_module.unescape(title_clean),
                "url": m.group(1),
                "snippet": "",
                "date": "",
                "source": "bing-news",
            })
            if len(results) >= max_results:
                break
        return results


class WebSearcher:
    """Web 搜索 — DuckDuckGo 优先，Bing 兜底"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._bing = BingFallbackSearcher(timeout=min(timeout, 20)) if _HAS_HTTPX else None

    def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
        time_range: Optional[str] = None,
        safesearch: str = "moderate"
    ) -> List[Dict[str, Any]]:
        max_results = min(max_results, 20)

        # 尝试 DuckDuckGo
        if _HAS_DDGS:
            try:
                ddgs = DDGS(timeout=self.timeout)
                results = ddgs.text(
                    query, region=region, safesearch=safesearch,
                    timelimit=time_range, max_results=max_results
                )
                standardized = []
                for r in results:
                    standardized.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", r.get("snippet", "")),
                        "source": "duckduckgo"
                    })
                if standardized:
                    return standardized
            except Exception as e:
                print(f"[WebSearch] DuckDuckGo failed: {e}, falling back to Bing", file=sys.stderr)

        # Fallback: Bing
        if self._bing:
            try:
                return self._bing.search(query, max_results)
            except Exception as e:
                raise SearchError(f"Bing fallback 也失败了: {e}")

        raise SearchError("DuckDuckGo 和 Bing 均不可用")

    def search_news(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
        time_range: Optional[str] = None,
        safesearch: str = "moderate"
    ) -> List[Dict[str, Any]]:
        max_results = min(max_results, 20)

        if _HAS_DDGS:
            try:
                ddgs = DDGS(timeout=self.timeout)
                results = ddgs.news(
                    query, region=region, safesearch=safesearch,
                    timelimit=time_range, max_results=max_results
                )
                standardized = []
                for r in results:
                    standardized.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", r.get("link", "")),
                        "snippet": r.get("body", r.get("excerpt", "")),
                        "date": r.get("date", ""),
                        "source": r.get("source", "unknown"),
                        "image": r.get("image", "")
                    })
                if standardized:
                    return standardized
            except Exception as e:
                print(f"[WebSearch] DuckDuckGo news failed: {e}, falling back to Bing", file=sys.stderr)

        if self._bing:
            try:
                return self._bing.search_news(query, max_results)
            except Exception as e:
                raise SearchError(f"Bing news fallback 也失败了: {e}")

        raise SearchError("新闻搜索：DuckDuckGo 和 Bing 均不可用")

    def instant_answer(self, query: str) -> Optional[Dict[str, Any]]:
        if not _HAS_DDGS:
            return None
        try:
            ddgs = DDGS(timeout=self.timeout)
            results = ddgs.answers(query)
            if results:
                return {
                    "type": "instant_answer",
                    "text": results[0].get("text", ""),
                    "url": results[0].get("url", "")
                }
            return None
        except Exception:
            return None


class SearchError(Exception):
    """搜索错误"""
    pass


def format_results_markdown(
    results: List[Dict[str, Any]],
    search_type: str = "web"
) -> str:
    """
    将搜索结果格式化为 Markdown
    
    Args:
        results: 搜索结果列表
        search_type: 搜索类型 (web/news)
        
    Returns:
        Markdown 格式的结果
    """
    if not results:
        return "未找到相关结果。"
    
    lines = []
    
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        
        lines.append(f"### {i}. [{title}]({url})")
        
        if search_type == "news":
            date = r.get("date", "")
            source = r.get("source", "")
            if date or source:
                lines.append(f"*{source} - {date}*")
        
        if snippet:
            lines.append(f"> {snippet}")
        
        lines.append("")
    
    return "\n".join(lines)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Web Search - 使用 DuckDuckGo 搜索网络信息"
    )
    
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="搜索关键词"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["web", "news", "instant"],
        default="web",
        help="搜索类型: web(网页), news(新闻), instant(即时答案)"
    )
    
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=5,
        help="返回结果数量 (默认5, 最大20)"
    )
    
    parser.add_argument(
        "--region", "-r",
        default="wt-wt",
        help="搜索区域 (默认: wt-wt 全球, cn-zh 中国, us-en 美国)"
    )
    
    parser.add_argument(
        "--time-range",
        choices=["d", "w", "m", "y"],
        help="时间范围: d(天), w(周), m(月), y(年)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "markdown"],
        default="json",
        help="输出格式 (默认: json)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="搜索超时时间（秒）"
    )
    
    args = parser.parse_args()
    
    searcher = WebSearcher(timeout=args.timeout)
    
    try:
        if args.type == "web":
            results = searcher.search(
                query=args.query,
                max_results=args.max_results,
                region=args.region,
                time_range=args.time_range
            )
        elif args.type == "news":
            results = searcher.search_news(
                query=args.query,
                max_results=args.max_results,
                region=args.region,
                time_range=args.time_range
            )
        elif args.type == "instant":
            result = searcher.instant_answer(args.query)
            results = [result] if result else []
        
        # 构建响应
        response = {
            "success": True,
            "query": args.query,
            "type": args.type,
            "count": len(results),
            "results": results
        }
        
        if args.format == "markdown":
            response["markdown"] = format_results_markdown(results, args.type)
        
        print(json.dumps(response, ensure_ascii=False, indent=2))
        
    except SearchError as e:
        print(json.dumps({
            "success": False,
            "error": str(e),
            "error_code": "SEARCH_ERROR",
            "query": args.query
        }, ensure_ascii=False))
        sys.exit(1)
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"未知错误: {e}",
            "error_code": "UNKNOWN_ERROR",
            "query": args.query
        }, ensure_ascii=False))
        sys.exit(1)


# === 供其他 Python 代码直接调用的函数 ===

def web_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    time_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    网页搜索（供直接调用）
    
    Args:
        query: 搜索关键词
        max_results: 返回结果数量
        region: 搜索区域
        time_range: 时间范围 (d/w/m/y)
        
    Returns:
        包含搜索结果的字典
    """
    searcher = WebSearcher()
    try:
        results = searcher.search(
            query=query,
            max_results=max_results,
            region=region,
            time_range=time_range
        )
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


def web_search_news(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    time_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    新闻搜索（供直接调用）
    
    Args:
        query: 搜索关键词
        max_results: 返回结果数量
        region: 搜索区域
        time_range: 时间范围 (d/w/m)
        
    Returns:
        包含新闻搜索结果的字典
    """
    searcher = WebSearcher()
    try:
        results = searcher.search_news(
            query=query,
            max_results=max_results,
            region=region,
            time_range=time_range
        )
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


if __name__ == "__main__":
    main()
