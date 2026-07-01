#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import datetime
import urllib.request
import json
import requests

def fetch_trending_repos(query, days=7):
    date_n_days_ago = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://api.github.com/search/repositories?q={query}+created:>{date_n_days_ago}&sort=stars&order=desc&per_page=15"
    
    req = urllib.request.Request(
        url, 
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("items", [])
    except Exception as e:
        print(f"Error fetching for query '{query}': {e}", file=sys.stderr)
        return []

def curate_with_gemini(api_key, raw_data):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    prompt = f"""你是一位资深的量化交易员、软件架构师和 AI 研究员。
这里有一份最近一周 GitHub 上热门的开源项目数据（按类别分类）：
{json.dumps(raw_data, indent=2, ensure_ascii=False)}

请为每个类别精选出“最值得关注、对我们项目最有帮助或能促进我们系统进化”的 3 个项目。
我们的当前项目背景：
- 主攻 A 股中证1000指数，采用机器学习/GBDT（LightGBM, XGBoost, CatBoost）进行多因子建模与凸优化再平衡。
- 工作流中涉及 AI 智能体开发、音视频转录与分析（如 Whisper 提取关键帧与会议纪要等）。

请输出一份精美的中文 Markdown 报告，标题为“GitHub 每周热门开源项目精选”。报告格式要求：
1. 每个类别下清晰列出 3 个精选项目。
2. 列出项目名称（带超链接跳转到项目 URL）、Star 数、Fork 数和主语言。
3. **重点**：详细解释为什么挑选这个项目（推荐理由），以及它如何能优化我们当前的项目或改善我们的工作流。

请直接输出 Markdown 内容，不要包含任何 markdown 块标记（如 ```markdown）包裹。
"""

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.2
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            # Extract text from Gemini response structure
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            print(f"Error: Unexpected response structure from Gemini API: {result}", file=sys.stderr)
        else:
            print(f"Gemini API request failed with status {response.status_code}: {response.text}", file=sys.stderr)
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        
    return None

def main():
    print("Starting weekly GitHub trends scan...")
    
    # 1. Check API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    queries = {
        "量化交易与投资组合 (Quant & Portfolio)": "quant+OR+backtest+OR+portfolio+OR+trading",
        "音视频与AI转录 (Audio/Video & Whisper)": "whisper+OR+audio+OR+transcription+OR+voice",
        "智能体生态与工具 (AI Agents & MCP)": "agent+OR+mcp+OR+llm-tool"
    }
    
    raw_results = {}
    total_found = 0
    
    for category, query in queries.items():
        print(f"Scanning category: {category}...")
        repos = fetch_trending_repos(query, days=7)
        category_repos = []
        
        for repo in repos:
            category_repos.append({
                "name": repo.get("full_name"),
                "url": repo.get("html_url"),
                "description": repo.get("description") or "无项目描述。",
                "stars": repo.get("stargazers_count"),
                "forks": repo.get("forks_count"),
                "language": repo.get("language") or "未知"
            })
            total_found += 1
            
        raw_results[category] = category_repos
        
    if total_found == 0:
        print("No repositories found in any category.", file=sys.stderr)
        sys.exit(1)
        
    # Save raw scan for backup
    raw_output_path = r"D:\lean-quant-vibe\reports\.github_raw_scan.json"
    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=4, ensure_ascii=False)
        
    print("Raw scan data saved. Starting Gemini curation...")
    
    # 2. Call Gemini for curation
    report_content = curate_with_gemini(api_key, raw_results)
    
    if not report_content:
        print("Error: Failed to generate curated report from Gemini.", file=sys.stderr)
        sys.exit(1)
        
    # Clean up markdown wrapping if Gemini ignored the prompt instruction
    report_content = report_content.strip()
    if report_content.startswith("```markdown"):
        report_content = report_content[11:]
    if report_content.startswith("```"):
        report_content = report_content[3:]
    if report_content.endswith("```"):
        report_content = report_content[:-3]
    report_content = report_content.strip()
    
    # Add metadata header
    header = f"---\ngenerated_at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n---\n\n"
    final_report = header + report_content
    
    # 3. Write final report
    report_path = r"D:\lean-quant-vibe\reports\github_weekly_trends.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)
        
    print(f"Curated weekly report successfully generated at {report_path}!")

if __name__ == "__main__":
    main()
