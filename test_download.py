import urllib.request
# Mock getproxies to bypass dead system proxy
urllib.request.getproxies = lambda: {}

import requests

# Test 1: requests.get with raw URL string containing parameters
print("Test 1: requests.get with raw URL string...")
raw_url = "http://push2his.eastmoney.com/api/qt/stock/kline/get?fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61%2Cf116&ut=7eea3edcaed734bea9cbfc24409ed989&klt=101&fqt=1&secid=1.600519&beg=20240101&end=20240105"
try:
    r = requests.get(raw_url)
    print("Test 1 Status:", r.status_code)
    print("Test 1 Data:", r.text[:200])
except Exception as e:
    print("Test 1 Failed with:", type(e), e)

# Test 2: urllib.request.urlopen with raw URL string
print("\nTest 2: urllib.request.urlopen with raw URL string...")
try:
    with urllib.request.urlopen(raw_url) as response:
        html = response.read().decode('utf-8')
        print("Test 2 Success!")
        print("Test 2 Data:", html[:200])
except Exception as e:
    print("Test 2 Failed with:", type(e), e)
