---
title: 使用yfinance获取美股数据的时报错
tags:
  - 智能投顾
categories:
  - 实验室
abbrlink: ce066d14
date: 2022-07-13 10:42:19
---

报错信息:
`No data found for this date range, symbol may be delisted`

开始以为是股票信息出了问题，之后找到原因是大陆不能访问雅虎财经了，所以使用 yf.download时需要添加代理访问。

作者给出了添加代理的方法

```python
import yfinance as yf

msft = yf.Ticker("MSFT")

msft.history(..., proxy="PROXY_SERVER")
msft.get_actions(proxy="PROXY_SERVER")
msft.get_dividends(proxy="PROXY_SERVER")
msft.get_splits(proxy="PROXY_SERVER")
msft.get_balance_sheet(proxy="PROXY_SERVER")
msft.get_cashflow(proxy="PROXY_SERVER")
msft.option_chain(..., proxy="PROXY_SERVER")
```

具体代码

```python
stock_price = yf.download("AAPL", start="2017-01-01", end="2017-04-30", proxy="http://127.0.0.1:7890")
```

这个代理地址是我的vpn的端口，问题解决
