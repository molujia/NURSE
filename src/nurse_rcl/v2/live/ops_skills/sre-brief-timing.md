# 时序

问题：已测起点能否区分先后？compare_timing 只接受 onsets 目录中的 source_onset_id 和 target_onset_id；缺起点时可显式 measure_signal，不传模型估计时间。
反馈：核对真实边、区间重叠、覆盖和时钟条件。严格先后仅在原条件支持时成立；unknown 不按点估计或卡片发布时间补判。可检查缺失端点，不能把方向合法当因果成立。
停止：无法得到更可靠时序时说明缺口，不宣称唯一原因。
