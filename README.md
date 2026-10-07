# llama.cpp · Hygon DCU 移植版（gfx928 / gfx936）

[llama.cpp](https://github.com/ggml-org/llama.cpp) 的**海光 DCU（Hygon DCU）移植版**。

> ✅ **本移植版已可直接运行当前最新的 Qwen3.8-Flash-Next（qwen4exp）模型** —— 48 层、其中 36 层 Gated-Delta-Net（SSM 线性注意力）、512 个 MoE 专家、UD-Q4_K_XL 量化约 104 GiB，已在海光 **K100_AI（gfx928）** 上跑通并完成调优（见下文跑分）。

- 源码基线：上游 `master` @ `26394b4`
- 目标架构：`gfx926 / gfx928 / gfx936 / gfx938`（`GGML_CUDA_CC_IS_HYGON_DCU` 宏统一覆盖，**同代海光可直接用**）
- 后端：HIP（DTK 26.04 / clang 17）
- 矩阵核：**MMAC**（`v_mmac_*`），**不是** AMD MFMA

## 1. 亮点

| 特性 | 说明 |
|---|---|
| **Qwen3.8-Flash-Next / qwen4exp** | 支持这种「SSM + MoE 混合」最新架构，`llama-server` 可正常起 OpenAI 兼容服务 |
| **Hygon DCU HIP 移植** | `ggml/src/ggml-cuda/vendors/hip.h`、`common.cuh`、`ggml-hip/CMakeLists.txt` 等已适配海光 |
| **自写 MMAC F16 稠密 GEMM** | `ggml/src/ggml-cuda/mmac-gemm.{cu,cuh}`。gfx928/gfx936 上 MFMA 被屏蔽，但 MMAC 可用（实测 FP16 178.1 / BF16 180.7 TFLOPS、INT8 361.6 TOPS）；接入后 prefill 提升约 **1.32×**（`GGML_CUDA_MMAC_GEMM=0` 可回退对比） |
| **预编译 fat binary** | `llama.cpp-dcu-bin-gfx928-gfx936.tar.gz` 同时含 gfx928 + gfx936 code object，运行时由 HIP 自动选择 |
| **MoE → MMQ + argsort 修复** | 见 `dcu-patches/` |

## 2. 仓库内容

```
.
├── <llama.cpp 源码>            # 上游 master @ 26394b4 + 海光 DCU 适配补丁（已应用）
├── dcu-patches/                # DCU 适配补丁（git diff 形态，便于重打 / 提交上游）
│   ├── hygon_dcu_llamacpp_full.patch        # 全量：14 文件 = Hygon 移植 + MMAC，相对 26394b4
│   ├── template_mmac_delta.patch            # MMAC 增量（4 文件）
│   ├── gfx928_moe_mmq_and_argsort.patch     # MoE→MMQ + argsort 修复
│   └── gfx928_all_uncommitted.patch         # gfx928 适配面（hip.h / common.cuh / norm ...）
├── llama.cpp-dcu-bin-gfx928-gfx936.tar.gz     # ★ 预编译产物（bin/ 目录，gfx928;gfx936 fat）
├── README-upstream.md          # 上游原始 README
└── README.md                   # 本文件
```

## 3. 使用预编译产物（免编译）

```sh
tar xzf llama.cpp-dcu-bin-gfx928-gfx936.tar.gz          # 解出 build-gfx928/bin/
source /opt/dtk/env.sh
export LD_LIBRARY_PATH=$PWD/build-gfx928/bin:$LD_LIBRARY_PATH

./build-gfx928/bin/llama-server --help   # OpenAI 兼容服务
./build-gfx928/bin/llama-cli   --help
```

`build-gfx928/bin/` 内含：`llama-server`、`llama-cli`、`llama-bench`、`test-backend-ops`，以及
`libggml-hip.so`（含 HIP kernel）、`libllama.so`、`libllama-common.so`、`libmtmd.so` 等运行库。

## 4. 从源码编译

依赖：海光 DTK（`/opt/dtk`，含 `hipcc` / clang / rocBLAS）。默认编 gfx928 + gfx936 的 fat binary：

```sh
source /opt/dtk/env.sh
# DTK 的 hsa.h 位于 /opt/dtk/.hyhal
export CMAKE_PREFIX_PATH=/opt/dtk/.hyhal:${CMAKE_PREFIX_PATH:-}

cmake -B build-gfx928 -S . -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="gfx928;gfx936" -DGPU_TARGETS="gfx928;gfx936" \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang

cmake --build build-gfx928 -j16 --target llama-cli llama-server test-backend-ops llama-bench
```

> `GGML_HIP_GRAPHS=ON` 配合 `GGML_HIP_NO_VMM=ON` 是海光 DCU 上的可用组合。

## 5. 运行 Qwen3.8-Flash-Next 的推荐参数（K100_AI / gfx928 实测）

| 参数 | 值 | 说明 |
|---|---|---|
| `-ncmoe` | **14** | 让 MoE 部分专家留在 CPU；`-ub 4096` 比 2048 多要 ~4.43 GiB 计算缓冲，12 会 OOM |
| `-b` / `-ub` | **4096 / 4096** | 实测 **+6.8%**；`-b` 必须同步改，否则被静默钳回 2048 |
| `-t` | **8** | 配 NUMA 钉核；更多线程无收益 |
| `-C 0xffff0000` / `--cpu-strict 1` | | NUMA 钉到权重所在节点（decode 最大杠杆） |
| `-fa` | **on** | **必须**；off 会让 qwen4exp 退化成复读 `/` |
| `GGML_CUDA_MMAC_GEMM` | 默认开 | MMAC F16 GEMM；`=0` 回退对比 |

## 6. 跑分（K100_AI，ctx=65536，每长度起新 server 发 1 个请求）

| prompt (tok) | prefill (tok/s) | 首字延迟 (s) | decode (tok/s) |
|---:|---:|---:|---:|
| 517 | 220.4 | 2.35 | 14.09 |
| 1030 | 266.1 | 3.88 | 14.15 |
| 2054 | 344.2 | 5.98 | 14.37 |
| 4102 | 376.6 | 10.91 | 13.15 |
| 10247 | **376.8** | **27.23** | 13.70 |

- prefill 随 prompt 变长而升高并趋平；decode 与长度基本无关（≈13.7 tok/s）。
- 真实 Agent 场景（固定前缀 ≈14980 tok）：冷首字 ≈40 s，**前缀缓存命中后 0.32–0.33 s**。

## 7. 注意 / 适用范围

- 矩阵核是 **MMAC（`v_mmac_*`）不是 MFMA**：MFMA 在 gfx928/gfx936 上被封，但 MMAC 默认可用。
- **gfx936（BW150）与 gfx928（K100_AI）的调优结论不同**（NUMA 钉核、带宽/计算比都相反），**不要互相套用参数**。
- 若 `-fa off` 或 chat 模板不当，qwen4exp 会退化，请用上述参数。

---

上游 llama.cpp 的原始文档见 [`README-upstream.md`](README-upstream.md)。
