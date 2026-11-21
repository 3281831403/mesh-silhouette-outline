# mesh-silhouette-outline

## 题面

给角色加卡通描边后，线条粗细不均、背面还描不出来：视线方向未归一，阈值判定失真；轮廓判定用 `>` 而不是 `>=`，背向面漏描；边去重用有序对，重复边未合并；外扩方向用顶点法线而不是屏幕法线，宽度不均；描边网格索引顺序反了，背面剔除后不可见；深度偏移沿视线，斜面渗色；退化面不剔除，法线 NaN；索引越界不校验，越界读；批量共用边集，网格之间串；`width=0` 不报错，输出空描边。outline/silhouette.py 是核心，tests/test_silhouette.py 15个红。要修的是：归一化视线、补齐阈值边界、按无序边去重、按屏幕法线外扩、修正描边三角形绕序、把深度偏移限制在深度分量、剔除退化面、校验索引范围、隔离批量边集，并让 `width=0` 抛 `OutlineError`。

## 运行方式

在仓库根目录执行：

```console
python -m pytest tests -q
```

初始失败环境必须恰好为：

```text
15 failed
```

不允许出现 collection error、import error 或 syntax error。修好 `outline/silhouette.py` 后，上述 15 个测试应全部通过。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_silhouette.py` 不可修改。实现只使用 Python 标准库与 pytest，全部计算离线、无随机性；网格统一使用 `{"positions": [...], "indices": [...]}` 结构，长度与宽度均按屏幕空间像素处理，浮点比较容差为 `1e-6`。
