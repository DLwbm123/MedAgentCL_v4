# Med-PRISM v2 TPM — 工作交接（2026-09-07）

## 最新完成：A–E composition 归因（2026-09-08）

Task1/Task2各256条dev、五个variant全部PASS，两个GPU进程已退出。详见 `COMPOSITION_DIAGNOSTIC_REPORT.md`。
关闭P3使KL下降99.34%/99.64%；Task2accuracy从40.625%恢复至64.453%。恢复S2或TPM均未改善Task2accuracy。
结论支持当前Task2退化以P3干扰为主；存在约5–6%相对L2的非加性交互，但不是主导来源。Task1dev无总体accuracy退化，不能把KL恢复等同于accuracy提升。
所有模型/源文件不变性检查PASS。没有训练、重新repair、调参或新checkpoint，也没有自动启动下一轮实验。下方条目为历史。

## 最新工作：retention evaluation 路径审计（已完成，未启动正式评估）

用户最新 repair 为 `MedPRISM_v2_TPM_repair_backtrack_20260907_110405_19036`，144/144 commit。
已核对 pre_tpm/accepted：仅2243个历史A变化，其他适配器完全一致，全模型原始不变量PASS。
新增独立 `scripts/medicalskill_v2_tpm/evaluate_retention.py`，只适配state加载并复用旧generation/metric；无算法修改。
全部17个CPU测试PASS。精确执行命令、加载路径和限制见 `RETENTION_EVALUATION_AUDIT.md`。
未启动正式评估/Stage0/oracle/训练/repair。后续需用户执行或授权；本次审计结束时两套eval输出尚不存在。

## 最新补丁：BF16 gamma backtracking（已完成）

本次只修改 safety.py 和测试；新增2个回归测试，服务器 CPU-only 共14个测试全部 PASS（6.020秒）。仅在唯一失败原因为 post_cast:edit_cap 时回退 gamma，每次重新 cast/measure/gate，最多16次；rho和原容差不变。未重跑64+64或任何训练。详见同目录 `BF16_BACKTRACKING_REPORT.md`；下方是初始实现的历史记录。后续实验须使用新输出目录，保留用户已完成的旧64+64结果。

本轮实现已完成。不是停在计划或 import 阶段；没有待补的核心代码，也没有启动正式训练。

## 本轮修改

- 新增 `med_prism/transport/` 13个 Python 模块：参考解析求解器、配置、bank映射、确定性校准、安全门限、Qwen原生分块前向、诊断、完整accepted checkpoint、任务边界流程和pilot。
- 新增 `scripts/medicalskill_v2_tpm/` 两个启动脚本。
- 新增 `tests/med_prism_tpm/test_transport.py` 和本目录文档。
- 未改旧v1训练器/脚本、正式checkpoint/结果、源数据、论文参考文件和评测解析器。15个审计原文件SHA256不变。无commit/push。

## 已验证

- 12个测试用例全部通过，其中包含原始15项数值检查。
- fake小型Qwen：完整任务边界、A-only更新、B/shared/current不变、strict回滚、checkpoint重载、下一任务key引用、unbudgeted/shuffle/off控制均通过。
- 最终模型所有目标投影输入哈希与当时solver输入逐项一致，检测并排除了陈旧输入更新路径。
- GPU0真实烟测：2fit+2holdout、每样本4token；36block、72q/v、144历史bank候选全部接受，2240个历史A张量变化；非目标参数不变；原生/分块前向含DeepStack逐比特一致。
- solver D有效秩最大2；BF16差分秩最高16仅作为诊断，绝不宣称BF16精确rank2；最大相对编辑0.0371805。
- Task3适配参数量前后均38,338,560。off模式4个组件权重哈希与原checkpoint完全相同。
- 每任务2000train/256development分组划分通过，跨任务训练/开发分组无重叠；Concept额外16条同开发组样本不进入训练。

## 已生成但不能当作正式实验的输出

- `/remote-home/wangbomin/MedPRISM_v2_TPM_implementation_smoke_20260907_final`
- `/remote-home/wangbomin/MedPRISM_v2_TPM_implementation_off_20260907_final`
- `/remote-home/wangbomin/MedPRISM_v2_TPM_implementation_plan_20260907_final`（只准备，无训练）

早期 `_a` 网络文件扩展属性复制失败、`_b` 有5个保守秩诊断误拒绝，均保留诊断。已修复：只复制权重字节并校验；rank门限改为直接检查solver返回的D，另报舍入后差分秩。参考solver本身未改动。

## 下一次从哪里开始

1. 阅读同目录 `IMPLEMENTATION_REPORT.md`，里面有详细审计、逐文件说明和全部命令。
2. 连接服务器后先检查GPU与当前进程；本轮烟测已退出，不要停止其他任务。
3. 若用户决定开始实验，先跑固定64fit/64holdout checkpoint repair，使用新的包含 `MedPRISM_v2_TPM` 的输出根目录。off、unbudgeted、shuffled都不需重训。
4. 三任务pilot默认只准备，加 `--run` 才训练。Task1共用一次，Task2/3各分支分别训练；每次边界后accepted的全部历史A都会进入下一阶段。不能拿旧no-geo Task3当作真正TPM链的Task3。
5. 完成阶段与边界可验证后复用。边界失败可从已有pre-TPM重试，不重训；中断的梯度阶段保留原目录，必须先检查再明确归档/重启，不会自动覆盖。

## 不要误读的限制

- 未跑正式64+64修复研究、三任务训练、五任务训练、测试矩阵、生成或多种子。
- 小样本读取误差下降不是accuracy/NLL/retention证据。
- 历史库之间RMS key cosine约2.76e-5→2.71e-4，相对增幅明显、绝对值仍小；须后续监测，不做自动重新正交化。
- 校准存在真实CPU缓存/模型hash/额外前向开销。最终4样本烟测外层共282.501秒，不能把前向计时当作全部TPM成本。
- 本地更完整历史账本：`D:/postgraduate/med_prism/.work/tpm_v2/HANDOFF.md`；测试与诊断在同目录 `verification/`。

实际工程目录为 `/root/MedAgentCL_v4`；新增源文件在本地 `D:/postgraduate/med_prism` 同步保留。密码未写入代码或交接文件。
