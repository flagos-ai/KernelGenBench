<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

## Verification Tool

Use the following command to verify your implementation:

```bash
{{DEVICE_ENV}}={{GPU_ID}} python {{VERIFY_SCRIPT}} --code kernel.py --op {{OPERATOR}} --dataset {{DATASET}} --output-json
```

The verification result is output as JSON containing:
- `passed`: whether verification passed
- `total_tests`: total number of tests
- `passed_tests`: number of tests passed
- `error`: error message (if any)

## Workflow

Follow this process:

1. **Write initial implementation**: implement the Triton kernel based on the operator spec, save to `kernel.py`
2. **Run verification**: check correctness using the verification command above
3. **Analyze errors**: if verification fails, read the error message carefully
4. **Fix the code**: update `kernel.py` based on the errors
5. **Re-verify**: run verification again until it passes or you have exhausted your best effort

**Important**:
- Re-verify after every change
- Handle edge cases (empty tensors, different dtypes, etc.)

## Performance Optimization

After passing all correctness tests, optimize kernel performance:

- Tune hyperparameters such as `BLOCK_SIZE`
- Reduce unnecessary memory accesses and data copies
- Use Triton auto-tuning (`@triton.autotune`)
- For float16/bfloat16 inputs, leverage low-precision computation while maintaining accuracy
- Avoid unnecessary `.contiguous()` calls
- Fuse multiple kernel calls into one where possible

Re-run verification after optimization to ensure correctness is preserved.

## Output Requirements

**After optimization**, output your final code at the end of your reply:

```python
import torch
import triton
import triton.language as tl

# Your final implementation...
```

Ensure the code block is complete and runnable, including all necessary imports and function definitions.
