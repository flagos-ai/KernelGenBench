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

# Agent Track

Agent Track evaluates coding agents that iteratively generate, verify, and optimize kernels.

```{toctree}
:maxdepth: 1

setup
methods
commands
cost-analysis
```

## What It Tests

Autonomous debugging and optimization capability with execution feedback.

## When to Use

- Testing agent frameworks (Claude Code, OpenCode)
- Evaluating kernel-specialized agents (AutoKernel, AKO4ALL)
- Production-ready kernel generation

## Quick Start

```bash
cd agent_bench
bash test_ops.sh add --device-count 1
```
