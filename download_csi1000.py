import os
import zipfile
import urllib.request
import ssl
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 强行在全局禁用 Registry 代理，确保 Python 底层连接干净
urllib.request.getproxies = lambda: {}

# 2. 全局禁用 SSL 证书验证
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 3. 核心补丁 A：重写 urllib.request.urlopen
# 当 pandas 或其他库直接通过 urllib 访问数据时，强制重写 HTTPS 为 HTTP
original_urlopen = urllib.request.urlopen
def patched_urlopen(url, *args, **kwargs):
    # 处理字符串格式的 URL
    if isinstance(url, str):
        url = url.replace('https://oss-ch.csindex.com.cn', 'http://oss-ch.csindex.com.cn')
        url = url.replace('https://push2his.eastmoney.com', 'http://push2his.eastmoney.com')
    # 处理 urllib.request.Request 对象
    elif hasattr(url, 'full_url'):
        url.full_url = url.full_url.replace('https://oss-ch.csindex.com.cn', 'http://oss-ch.csindex.com.cn')
        url.full_url = url.full_url.replace('https://push2his.eastmoney.com', 'http://push2his.eastmoney.com')
    
    # 移除可能存在的 context 参数以防止 SSL 问题，强制不校验
    kwargs['context'] = ssl._create_unverified_context()
    return original_urlopen(url, *args, **kwargs)

urllib.request.urlopen = patched_urlopen

# 4. 核心补丁 B：重写 requests 的 Session.request
import requests
requests.packages.urllib3.disable_warnings()

original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    if isinstance(url, str):
        if 'push2his.eastmoney.com' in url:
            url = url.replace('https://', 'http://')
        elif 'oss-ch.csindex.com.cn' in url:
            url = url.replace('https://', 'http://')
    print("REWRITING URL:", url)
    kwargs['verify'] = False
    kwargs['proxies'] = {}
    return original_request(self, method, url, **kwargs)

requests.Session.request = patched_request

# 导入 akshare (必须在上述补丁都注入完毕后导入)
import akshare as ak

# 目标输出目录
OUTPUT_DIR = os.path.join("data", "equity", "usa", "daily")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_csi1000_constituents():
    """
    获取中证1000指数当前所有的成分股代码和交易所信息
    """
    print("正在获取中证1000成分股列表...")
    try:
        df = ak.index_stock_cons_csindex(symbol="000852")
        # 兼容列名与列位置索引，确保接口更新时列错位不引发静默错误
        code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[4]
        exchange_col = "交易所" if "交易所" in df.columns else df.columns[7]
        constituents = []
        for _, row in df.iterrows():
            code = row[code_col]
            exchange = row[exchange_col]
            if "上海" in exchange:
                prefix = "sh"
            elif "深圳" in exchange:
                prefix = "sz"
            else:
                prefix = "bj"
            constituents.append({"code": code, "symbol": f"{prefix}{code}"})
        print(f"获取成功，共计 {len(constituents)} 只成分股。")
        return constituents
    except Exception as e:
        print("获取成分股列表失败：", e)
        raise e

def download_and_format_stock(stock_info, start_date="20100101", end_date="20260528"):
    """
    下载单只股票的历史日线数据，并转换为 LEAN 兼容的 CSV ZIP 格式
    LEAN 格式: Date (yyyyMMdd 00:00), Open, High, Low, Close, Volume
    价格需放大 10,000 倍保存为整数。
    """
    code = stock_info["code"]
    symbol = stock_info["symbol"]
    zip_path = os.path.join(OUTPUT_DIR, f"{code}.zip")
    csv_filename = f"{code}.csv"
    
    # 支持断点续传，如果文件已存在且大小正常，直接跳过
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000:
        return code, "EXIST"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 双重保险：确保线程级别 getproxies 被正确覆盖
            urllib.request.getproxies = lambda: {}
            
            # 使用新浪接口下载数据
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权数据
            )
            
            if df is None or df.empty:
                return code, "EMPTY"
            
            # 格式化日期：将 YYYY-MM-DD 转换为 YYYYMMDD 00:00
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d 00:00")
            
            # 缩放价格并转换为整数（LEAN 规范：价格乘以 10,000）
            df["open"] = (df["open"] * 10000).round().astype(int)
            df["high"] = (df["high"] * 10000).round().astype(int)
            df["low"] = (df["low"] * 10000).round().astype(int)
            df["close"] = (df["close"] * 10000).round().astype(int)
            
            # 成交量转换为整数
            df["volume"] = df["volume"].round().astype(int)
            
            # 确保时间严格排序，过滤无效零价数据，去除空值
            df = df.sort_values("date")
            df = df[(df["open"] > 0) & (df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]

            # 筛选出 LEAN 格式所需的 6 列
            lean_df = df[["date", "open", "high", "low", "close", "volume"]]
            
            # 去除包含空值或非法数据的行
            lean_df = lean_df.dropna()
            
            # 转换为 CSV 字符串
            csv_data = lean_df.to_csv(index=False, header=False, lineterminator="\n")
            
            # 压缩保存为 ZIP 包，内部文件名必须是 <code>.csv
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(csv_filename, csv_data)
                
            return code, "SUCCESS"
        except Exception as e:
            if attempt == max_retries - 1:
                return code, f"FAILED: {str(e)}"
            time.sleep(1 + attempt)  # 逐渐增加重试间隔

def main():
    start_time = time.time()
    try:
        constituents = get_csi1000_constituents()
    except Exception:
        return

    print("开始下载历史数据（支持断点续传）...")
    success_count = 0
    exist_count = 0
    fail_count = 0
    empty_count = 0
    
    # 限制最大并发数（10个线程），多线程更适合Windows下的网络并发，且该日线接口无需进程隔离
    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_format_stock, stock): stock for stock in constituents}
        
        for i, future in enumerate(as_completed(futures)):
            stock = futures[future]
            try:
                code, status = future.result()
                if status == "SUCCESS":
                    success_count += 1
                elif status == "EXIST":
                    exist_count += 1
                elif status == "EMPTY":
                    empty_count += 1
                else:
                    fail_count += 1
                    print(f"\n下载失败 {code}: {status}")
            except Exception as e:
                fail_count += 1
                print(f"\n执行异常 {stock['code']}: {e}")
                
            # 每下载 50 只股票输出一次进度进度
            if (i + 1) % 50 == 0 or (i + 1) == len(constituents):
                print(f"进度: {i+1}/{len(constituents)} | 成功: {success_count} | 已存在: {exist_count} | 空数据: {empty_count} | 失败: {fail_count}")

    duration = time.time() - start_time
    print(f"\n数据准备完毕！耗时: {duration:.2f} 秒")
    print(f"总计成分股: {len(constituents)} | 成功下载: {success_count} | 本地已存在: {exist_count} | 空白无数据: {empty_count} | 失败: {fail_count}")

if __name__ == "__main__":
    main()
