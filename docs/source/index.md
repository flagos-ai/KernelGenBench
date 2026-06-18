# KernelGenBench Documentation

A benchmark framework for evaluating LLM and agent-based Triton kernel generation across multiple hardware platforms.

```{button-ref} getting-started/index
:ref-type: doc
:color: primary
:class: sd-btn-lg sd-px-4 sd-py-2 sd-fw-bold

[Getting Started »](getting-started/index)
```

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`browser;1.5em;sd-mr-1` Overview
:link: overview/index
:link-type: doc

Learn what KernelGenBench is, why it matters, and what it can do for you.

+++
[Learn more »](overview/index)
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Features
:link: features/index
:link-type: doc

Explore multi-source operators, multi-chip support, anti-hack validation, and evaluation metrics.

+++
[Learn more »](features/index)
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` LLM Track
:link: operation-guide/llm-track/index
:link-type: doc

Evaluate LLMs on generating Triton kernels with Pass@K metric.

+++
[Learn more »](operation-guide/llm-track/index)
:::

:::{grid-item-card} {octicon}`hubot;1.5em;sd-mr-1` Agent Track
:link: operation-guide/agent-track/index
:link-type: doc

Evaluate coding agents that iteratively generate, verify, and optimize kernels.

+++
[Learn more »](operation-guide/agent-track/index)
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc

Datasets, operators, hardware platforms, and technical specifications.

+++
[Learn more »](reference/index)
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Development
:link: development/index
:link-type: doc

Contributing guides, custom operators, and extending the framework.

+++
[Learn more »](development/index)
:::

::::

---

```{toctree}
:caption: 📚 Overview
:maxdepth: 2
:hidden:

overview/index.md
```

```{toctree}
:caption: 🚀 Getting Started
:maxdepth: 2
:hidden:

getting-started/index.md
```

```{toctree}
:caption: 🔬 Features
:maxdepth: 2
:hidden:

features/index.md
```

```{toctree}
:caption: 📖 Operation Guide
:maxdepth: 3
:hidden:

operation-guide/index.md
```

```{toctree}
:caption: 📑 Reference
:maxdepth: 2
:hidden:

reference/index.md
```

```{toctree}
:caption: 🔧 Development
:maxdepth: 2
:hidden:

development/index.md
```

```{toctree}
:caption: ❓ FAQ & Glossary
:maxdepth: 2
:hidden:

faq/index.md
glossary/index.md
```
