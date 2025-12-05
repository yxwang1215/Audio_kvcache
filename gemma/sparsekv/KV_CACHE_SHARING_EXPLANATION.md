# Gemma KV Cache 复用机制详解

## 核心概念

**不是所有层都复用最后一层的 KV cache！** 而是**按层类型（layer_types）匹配复用**。

## 复用规则

### 1. 共享层的起始位置
```python
first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
```

**示例**：
- 总层数：27
- `num_kv_shared_layers`：15
- `first_kv_shared_layer_idx` = 27 - 15 = **12**
- **从第12层开始（索引12-26），后面的15层都是共享层**

### 2. 复用逻辑（关键！）

**共享层复用的是：前面非共享层中，最后一个相同类型的层**

```python
prev_layers = config.layer_types[:first_kv_shared_layer_idx]  # 前12层的类型列表

# 对于共享层（layer_idx >= 12）
kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
```

**这个公式的含义**：
- `prev_layers[::-1]`：反向列表
- `.index(config.layer_types[layer_idx])`：找到当前层类型在反向列表中的**第一个位置**（即最后一个相同类型）
- `len(prev_layers) - 1 - index`：转换为正向索引

## 具体示例

### 示例1：简单情况

假设：
- 总层数：10
- `num_kv_shared_layers`：4
- `first_kv_shared_layer_idx`：10 - 4 = **6**
- `layer_types`：`['A', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A']`

**前6层（非共享层，索引0-5）**：
```
Layer 0: type='A' → store_full_length_kv = False (不是最后一个'A')
Layer 1: type='A' → store_full_length_kv = False
Layer 2: type='B' → store_full_length_kv = False (不是最后一个'B')
Layer 3: type='A' → store_full_length_kv = False
Layer 4: type='B' → store_full_length_kv = False
Layer 5: type='A' → store_full_length_kv = True  ✅ (最后一个'A'，存储KV)
```

**prev_layers = ['A', 'A', 'B', 'A', 'B', 'A']** (索引0-5)

**后4层（共享层，索引6-9）**：
```
Layer 6: type='B' → is_kv_shared_layer = True
         计算 kv_shared_layer_index:
         - prev_layers[::-1] = ['A', 'B', 'A', 'B', 'A', 'A'] (反向)
         - prev_layers[::-1].index('B') = 1 (反向列表第一个'B'的位置)
         - kv_shared_layer_index = 6 - 1 - 1 = 4 ✅
         → 复用 Layer 4 的 KV cache (最后一个'B'类型)

Layer 7: type='A' → 复用 Layer 5 的 KV cache (最后一个'A'类型)
Layer 8: type='B' → 复用 Layer 4 的 KV cache (最后一个'B'类型)
Layer 9: type='A' → 复用 Layer 5 的 KV cache (最后一个'A'类型)
```

### 示例2：Gemma 实际配置

假设 Gemma-3n 配置：
- 总层数：27
- `num_kv_shared_layers`：15
- `first_kv_shared_layer_idx`：27 - 15 = **12**
- `layer_types` 模式：每5层一个 `full_attention`，其余是 `sliding_attention`

**前12层（非共享层，索引0-11）**：
```
Layer 0-3:  sliding_attention → store_full_length_kv = False
Layer 4:    full_attention    → store_full_length_kv = False (不是最后一个full)
Layer 5-8:  sliding_attention → store_full_length_kv = False
Layer 9:    full_attention    → store_full_length_kv = False (不是最后一个full)
Layer 10:   sliding_attention → store_full_length_kv = True  ✅ (最后一个sliding)
Layer 11:   sliding_attention → store_full_length_kv = False
```

**后15层（共享层，索引12-26）**：
```
Layer 12-14: sliding_attention → 复用 Layer 10 的 KV cache
Layer 15:    full_attention    → 复用 Layer 9 的 KV cache
Layer 16-19: sliding_attention → 复用 Layer 10 的 KV cache
Layer 20:    full_attention    → 复用 Layer 9 的 KV cache
Layer 21-25: sliding_attention → 复用 Layer 10 的 KV cache
Layer 26:    full_attention    → 复用 Layer 9 的 KV cache
```

**关键点**：
- `sliding_attention` 类型的共享层 → 复用最后一个 `sliding_attention` 非共享层（Layer 10）
- `full_attention` 类型的共享层 → 复用最后一个 `full_attention` 非共享层（Layer 9）

## 总结

1. **不是复用最后一层**，而是复用**前面非共享层中最后一个相同类型的层**

2. **存储机制**：
   - 每个类型的最后一个非共享层（`store_full_length_kv = True`）会存储完整KV到 `shared_layers[layer_idx]`
   - 共享层通过 `kv_shared_layer_index` 找到对应的源层，从 `shared_layers[kv_shared_layer_index]` 读取KV

3. **类型匹配**：
   - `sliding_attention` 类型的共享层 → 复用最后一个 `sliding_attention` 非共享层
   - `full_attention` 类型的共享层 → 复用最后一个 `full_attention` 非共享层

4. **Decode阶段更新**：
   - `store_full_length_kv` 层在decode阶段会追加新token的KV到 `shared_layers`
   - 共享层在decode阶段会从更新后的 `shared_layers` 读取KV

## 代码位置

- **初始化逻辑**：`modeling_gemma3n.py` 第1297-1309行
- **Prefill阶段存储**：第1372-1375行（标准路径），第1496-1500行（SparseMM路径）
- **Decode阶段更新**：第1639-1658行（SparseMM路径）
- **共享层读取**：第1332-1338行（标准路径），第1476-1483行（SparseMM路径）

