1.开发一个可以测评模型写闭源cuda算子的triton实现的benchmark。
2.每一个overload层级的算子，就是一个独立的测试，我们就是做的overload层级。
3.重点是baseline，ut，也就是流程都要确保正确！triton的prompt也要对，就是我们的测试环境和指令都要正确，triton算子写不对无所谓，这只能说明是模型写算子的能力不足，就是我们benchmark测评的意义。
4.debug的时候，你先看哪些算子是一个类型的，或者你认为有共同点的，这种为一批。你一次运行一批算子，然后修这批算子，确保这批都可以通过，通过后计入pass.md。
5.遇到testfunc参数出问题，不妨自己构建一个最简单脚本，来一个一个的测试这个baseline可以使用的参数，然后再在testfunc里应用。
6.cublas和cupy是两个！cupy可以做流程参考！


base rules
rule1：中文回答
rule2：收到指令先用自然语言回答，先解释思路，不要直接动手，等我确定没问题下达指令再改代码
rule3:没有要求的情况下不要写md，直接输出回答就行
rule4：实验环境：flagbench工作树用 source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench；flagbench_cublas工作树用 source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench_cublas
rule5:就在指定的分支完成任务，千万不要干扰其他分支，其他已经提交pr的，或者还在开发的分支，都不要影响到！保持每个分支的干净独立！
rule6:以下是api使用方法，curl -X POST https://kspmas.ksyun.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer 8407460c-9a3d-4a32-bb0d-43e91a74304f" -d '{
    "model": "mog-1",
    "messages": [
        {"role": "system", "content": "你是一位中文诗人，请严格遵循五言绝句格式。"},
        {"role": "user",   "content": "写一首关于春天的五言绝句"}
    ]
}'
rule7:跑测试默认放在后台跑。

useful：
1.  DISPATCH_TORCH_LIB=0 python test/test_accuracy_ut.py --test-file flagbench.accuracy.在的文件夹名 --name 算子类型名::算子名                      

2. gkav (generate_kernel_and_verify.py) 使用方法：
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench && KSYUN_API_KEY=8407460c-9a3d-4a32-bb0d-43e91a74304f python scripts/generate_kernel_and_verify.py --dataset KernelGenBench --max-rounds 5 --device-count 1 --num-workers 50 --server-type ksyun --model-name mog-1
关键点：
    - --dataset: 数据集名称 (200ops, vllm13, cublas等)
    - --op-name: 单独测试某个算子 (如 cublas::cublasSgemm_v2)
    - --max-rounds: Pass@K的K值
    - --device-count: GPU数量，设为实际卡数
    - --num-workers: 生成并发数
    - --server-type ksyun + KSYUN_API_KEY 环境变量