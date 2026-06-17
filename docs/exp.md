\section{Experiments and Evaluation}
\label{sec:experiments}

In this section, we deploy KernelGenBench to evaluate LLMs and agentic frameworks across diverse operator sources and heterogeneous hardware platforms.

\subsection{Experimental Setup}
\label{subsec:setup}

We deploy the evaluation suite across six dedicated full-node hardware platforms: NVIDIA A100 and five alternative hardware platforms (Platform A--E). The full 210-operator problem set is evaluated on the NVIDIA baseline, while the 110-operator ATen subset is used to benchmark cross-platform portability. For all language models, we use \texttt{temperature}=0 for Pass@1 generation and \texttt{temperature}=0.8 for Pass@5, with a context window limit of \texttt{max\_tokens}=16384 and a unified 30-minute wall-clock timeout per operator task. Evaluated paradigms are detailed in Section~\ref{subsec:pipeline_and_antihack}. Results for legacy models are deferred to Appendix~\ref{app:legacy_baselines}.

\subsection{KernelGenBench-MS: Multi-Source Evaluation}
\label{subsec:nvidia_analysis}

We evaluate all methods on NVIDIA A100 across the full 210-operator suite. Table~\ref{tab:nvidia_main_ops} presents the comprehensive results.

% \begin{table}[htbp]
% \centering
% \setlength{\tabcolsep}{4pt}
% \caption{NVIDIA A100 evaluation across 210 operators from three sources (ATen, vLLM, cuBLAS), showing accuracy and speedup by operator source across all generation paradigms.}
% \label{tab:nvidia_main_ops}
% \resizebox{\textwidth}{!}{%
% \begin{tabular}{l | cc | cc | cc | cc}
% \toprule
% \multirow{2}{*}{\textbf{Method \& Setup}} & \multicolumn{2}{c|}{\textbf{Overall (210)}} & \multicolumn{2}{c|}{\textbf{ATen (110)}} & \multicolumn{2}{c|}{\textbf{vLLM (50)}} & \multicolumn{2}{c}{\textbf{cuBLAS (50)}} \\
% \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
% & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) \\
% \midrule
% % --- LLM Sampling Methods ---
% \multicolumn{9}{l}{\textbf{LLM Sampling Methods}} \\
% \midrule
% Pass@1 (Opus-4.6)       & 41 & 0.70 & 39 & 0.90 & 20 & 0.76 & 68 & 0.49 \\
% Pass@1 (GLM-5.0)        & 21 & 0.68 & 21 & 0.49 & 24 & 1.24 & 20 & 0.73 \\
% Pass@1 (Qwen3.5-27b)    & 7  & 0.85 & 8  & 0.83 & 2  & 2.05 & 8  & 0.71 \\
% Pass@1 (MiniMax M-2.5)  & 2  & 0.88 & 4  & 0.88 & 0  & 0.00 & 0  & 0.00 \\
% \midrule
% Pass@5 (Opus-4.6)       & 57 & 0.68 & 62 & 0.79 & 28 & 0.71 & 74 & 0.49 \\
% Pass@5 (GLM-5.0)        & 36 & 0.77 & 45 & 0.64 & 32 & 1.28 & 20 & 0.76 \\
% Pass@5 (Qwen3.5-27b)    & 11 & 1.01 & 13 & 1.04 & 12 & 0.70 & 8  & 0.68 \\
% Pass@5 (MiniMax M-2.5)  & 17 & 0.69 & 21 & 0.76 & 18 & 1.27 & 2  & 0.46 \\
% \midrule
% % --- Vanilla Agentic Frameworks ---
% \multicolumn{9}{l}{\textbf{Vanilla Agentic Frameworks}} \\
% \midrule
% Claude Code (Opus-4.6)     & 87 & 0.78 & 92 & 0.86 & 68 & 1.02 & 94 & 0.51 \\
% Claude Code (GLM-5.0)      & 67 & 0.83 & 72 & 0.88 & 52 & 1.23 & 72 & 0.53 \\
% Claude Code (Qwen3.5-27b)  & 62 & 0.70 & 80 & 0.68 & 38 & 1.17 & 48 & 0.50 \\
% Claude Code (MiniMax M-2.5)& 49 & 0.69 & 69 & 0.78 & 26 & 0.46 & 26 & 0.58 \\
% \midrule
% OpenCode (Opus-4.6)        & 81 & 0.73 & 92 & 0.82 & 46 & 0.97 & 92 & 0.50 \\
% OpenCode (GLM-5.0)         & 72 & 0.69 & 87 & 0.71 & 42 & 0.94 & 70 & 0.51 \\
% OpenCode (Qwen3.5-27b)     & 53 & 0.78 & 58 & 0.75 & 44 & 1.31 & 52 & 0.58 \\
% OpenCode (MiniMax M-2.5)   & 41 & 0.62 & 50 & 0.77 & 26 & 0.44 & 36 & 0.42 \\
% \midrule
% % --- Kernel-Specialized Agents ---
% \multicolumn{9}{l}{\textbf{Kernel-Specialized Agents}} \\
% \midrule
% AKO4all (Opus-4.6)              & 83 & 0.97 & 91 & 1.00 & 64 & 1.62 & 84 & 0.61 \\
% CUDA Opt. Skill (MiniMax M-2.5) & 45 & 0.80 & 63 & 0.81 & 24 & 0.92 & 28 & 0.45 \\
% \midrule
% AutoKernel (GLM-5.0)            & 71 & 0.99 & 87 & 1.00 & 43 & 1.40 & 66 & 0.75 \\
% AutoKernel (Qwen3.5-27b)        & 47 & 1.02 & 69 & 1.00 & 16 & 1.63 & 30 & 0.80 \\
% AutoKernel (MiniMax M-2.5)      & 43 & 0.89 & 66 & 0.87 & 20 & 1.52 & 16 & 0.46 \\
% \bottomrule
% \end{tabular}%
% }
% \end{table}

\begin{table}[htbp]
\centering
\setlength{\tabcolsep}{4pt}
\caption{NVIDIA A100 evaluation across 210 operators from three sources (ATen, vLLM, cuBLAS), showing accuracy and speedup by operator source across all generation paradigms.}
\label{tab:nvidia_main_ops}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | cc | cc | cc | cc}
\toprule
\multirow{2}{*}{\textbf{Method \& Setup}} & \multicolumn{2}{c|}{\textbf{Overall (210)}} & \multicolumn{2}{c|}{\textbf{ATen (110)}} & \multicolumn{2}{c|}{\textbf{vLLM (50)}} & \multicolumn{2}{c}{\textbf{cuBLAS (50)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
& Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) \\
\midrule
% --- LLM Sampling Methods ---
\multicolumn{9}{l}{\textbf{LLM Sampling Methods}} \\
\midrule
Pass@1 (Opus-4.6)       & 41 & 0.70 & 39 & 0.90 & 20 & 0.76 & 68 & 0.49 \\
Pass@1 (GLM-5.0)        & 21 & 0.68 & 21 & 0.49 & 24 & 1.24 & 20 & 0.73 \\
Pass@1 (Qwen3.5-27b)    & 7  & 0.85 & 8  & 0.83 & 2  & \textbf{2.05} & 8  & 0.71 \\
Pass@1 (MiniMax M-2.5)  & 2  & 0.88 & 4  & 0.88 & 0  & 0.00 & 0  & 0.00 \\
\midrule
Pass@5 (Opus-4.6)       & 57 & 0.68 & 62 & 0.79 & 28 & 0.71 & 74 & 0.49 \\
Pass@5 (GLM-5.0)        & 36 & 0.77 & 45 & 0.64 & 32 & 1.28 & 20 & \underline{0.76} \\
Pass@5 (Qwen3.5-27b)    & 11 & \underline{1.01} & 13 & \textbf{1.04} & 12 & 0.70 & 8  & 0.68 \\
Pass@5 (MiniMax M-2.5)  & 17 & 0.69 & 21 & 0.76 & 18 & 1.27 & 2  & 0.46 \\
\midrule
% --- Vanilla Agentic Frameworks ---
\multicolumn{9}{l}{\textbf{Vanilla Agentic Frameworks}} \\
\midrule
Claude Code (Opus-4.6)     & 87 & 0.78 & 92 & 0.86 & 68 & 1.02 & 94 & 0.51 \\
Claude Code (GLM-5.0)      & 67 & 0.83 & 72 & 0.88 & 52 & 1.23 & 72 & 0.53 \\
Claude Code (Qwen3.5-27b)  & 62 & 0.70 & 80 & 0.68 & 38 & 1.17 & 48 & 0.50 \\
Claude Code (MiniMax M-2.5)& 49 & 0.69 & 69 & 0.78 & 26 & 0.46 & 26 & 0.58 \\
\midrule
OpenCode (Opus-4.6)        & 81 & 0.73 & 92 & 0.82 & 46 & 0.97 & 92 & 0.50 \\
OpenCode (GLM-5.0)         & 72 & 0.69 & 87 & 0.71 & 42 & 0.94 & 70 & 0.51 \\
OpenCode (Qwen3.5-27b)     & 53 & 0.78 & 58 & 0.75 & 44 & 1.31 & 52 & 0.58 \\
OpenCode (MiniMax M-2.5)   & 41 & 0.62 & 50 & 0.77 & 26 & 0.44 & 36 & 0.42 \\
\midrule
% --- Kernel-Specialized Agents ---
\multicolumn{9}{l}{\textbf{Kernel-Specialized Agents}} \\
\midrule
AKO4all (Opus-4.6)              & 83 & 0.97 & 91 & \underline{1.00} & 64 & 1.62 & 84 & 0.61 \\
CUDA Opt. Skill (MiniMax M-2.5) & 45 & 0.80 & 63 & 0.81 & 24 & 0.92 & 28 & 0.45 \\
\midrule
AutoKernel (GLM-5.0)            & 71 & 0.99 & 87 & \underline{1.00} & 43 & 1.40 & 66 & 0.75 \\
AutoKernel (Qwen3.5-27b)        & 47 & \textbf{1.02} & 69 & \underline{1.00} & 16 & \underline{1.63} & 30 & \textbf{0.80} \\
AutoKernel (MiniMax M-2.5)      & 43 & 0.89 & 66 & 0.87 & 20 & 1.52 & 16 & 0.46 \\
\bottomrule
\end{tabular}%
}
\end{table}

\textbf{Finding 1: Model and Method Capabilities.}
Claude Code (Opus-4.6) achieves the highest overall accuracy at 87\%, while AutoKernel (Qwen3.5-27b) achieves the highest overall speedup at 1.02$\times$. Comparing the two top performers reveals a fundamental trade-off: Claude Code outperforms the kernel-specialized AKO4all (Opus-4.6) at 83\% accuracy by 4~pp, but AKO4all achieves 0.97$\times$ speedup versus Claude Code's 0.78$\times$. Kernel-specialized agents prioritize performance optimization over correctness, sacrificing functional correctness on edge cases to maximize speedup. By contrast, vanilla agentic frameworks allocate more iterations to debugging and correctness verification, yielding higher pass rates at the cost of performance. Speedup distribution metrics (fast$_p$) for all configurations are detailed in Appendix~\ref{app:fastp_results}.

Breaking down by operator source reveals where specialization pays off. On ATen operators, both Claude Code and OpenCode with Opus-4.6 reach 92\% accuracy, while kernel-specialized agents (AKO4all, AutoKernel with GLM-5.0/Qwen3.5-27b) achieve 1.00$\times$ speedup, perfectly matching the baseline. On vLLM operators, Claude Code (Opus-4.6) leads in accuracy at 68\%, but AutoKernel (Qwen3.5-27b) delivers 1.63$\times$ speedup---the highest across all sources---demonstrating that specialized agents excel at performance optimization on complex operators. On cuBLAS operators, Claude Code (Opus-4.6) achieves 94\% accuracy, yet even the best speedup (AutoKernel Qwen at 0.80$\times$) remains below the proprietary baseline. Model-specific generation bottlenecks and anti-hack interception patterns are detailed in Appendix~\ref{app:generation_behaviors}. 

\textbf{Finding 2: Operator-Source Difficulty Hierarchy.}
A strict, model-agnostic difficulty hierarchy emerges across the three operator sources. ATen operators are the most tractable: even weaker models achieve reasonable accuracy (e.g., Claude Code with MiniMax reaches 69\%), and speedup consistently clusters around 0.8--1.0$\times$, matching the framework baseline. This reflects ATen's design as a high-level framework API with well-defined semantics and moderate performance requirements.

vLLM operators present the opposite profile---functional correctness is extremely challenging (accuracy drops sharply, e.g., Claude Code with MiniMax falls to 26\%), yet when successfully generated, they deliver genuine acceleration potential (AKO4all achieves 1.62$\times$, AutoKernel with Qwen reaches 1.63$\times$). This difficulty stems from vLLM's complex inference-specific operators (paged attention, KV cache management, mixed-precision quantization), which require intricate memory layouts and algorithmic understanding that LLMs struggle to implement correctly. However, the baseline implementations are not heavily hand-tuned, leaving room for optimization when correctness is achieved.

cuBLAS operators occupy the middle ground: moderate accuracy is achievable (Claude Code with Opus reaches 94\%), but speedup is universally capped---virtually all configurations cluster tightly around 0.50$\times$, unable to surpass the hand-tuned proprietary baseline. This performance ceiling reflects cuBLAS's status as a closed-source, heavily optimized library representing decades of expert engineering. The baseline directly loads \texttt{libcublas.so} via \texttt{ctypes}, bypassing all high-level wrappers, making it nearly impossible for LLM-generated Triton kernels to match proprietary BLAS performance. This hierarchy reveals that ATen serves as the tractable baseline, vLLM tests optimization capability on complex kernels, and cuBLAS exposes the fundamental difficulty of matching closed-source performance.


\subsection{KernelGenBench-MC: Cross-Platform Evaluation}
\label{subsec:cross_platform_analysis}

We extend the evaluation of the 110 ATen operators across six hardware platforms. Table~\ref{tab:cross_platform_main} presents the comprehensive results.

% \begin{table}[htbp]
% \centering
% \setlength{\tabcolsep}{3.5pt}
% \caption{Cross-platform evaluation on 110 ATen operators across six hardware platforms, showing whether correctness and speedup transfer across heterogeneous hardware backends.}
% \label{tab:cross_platform_main}
% \resizebox{\textwidth}{!}{%
% \begin{tabular}{l | cc | cc | cc | cc | cc | cc}
% \toprule
% \multirow{2}{*}{\textbf{Method \& Setup}} & \multicolumn{2}{c|}{\textbf{NVIDIA}} & \multicolumn{2}{c|}{\textbf{Platform A}} & \multicolumn{2}{c|}{\textbf{Platform B}} & \multicolumn{2}{c|}{\textbf{Platform C}} & \multicolumn{2}{c|}{\textbf{Platform D}} & \multicolumn{2}{c}{\textbf{Platform E}} \\
% \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}
% & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) \\
% \midrule
% % --- LLM Sampling Methods ---
% \multicolumn{13}{l}{\textbf{LLM Sampling Methods}} \\
% \midrule
% Pass@1 (Opus-4.6)           & 39 & 0.90 & 46 & 0.19 & 44 & 0.69 & 37 & 0.98 & 38 & 0.89 & 38 & 0.88 \\
% Pass@1 (Qwen3.5-27b)        & 8  & 0.83 & 9  & 0.09 & 3  & 1.02 & 7  & 0.90 & 7  & 0.98 & 10 & 1.03 \\
% Pass@1 (MiniMax M-2.5)      & 4  & 0.88 & 4  & 0.25 & 6  & 1.33 & 4  & 1.05 & 5  & 0.77 & 4  & 1.15 \\
% \midrule
% Pass@5 (Opus-4.6)           & 62 & 0.79 & 63 & 0.15 & 60 & 0.74 & 54 & 0.92 & 65 & 0.68 & 57 & 0.83 \\
% Pass@5 (Qwen3.5-27b)        & 13 & 1.04 & 16 & 0.18 & 11 & 1.10 & 15 & 0.72 & 10 & 0.99 & 17 & 1.02 \\
% Pass@5 (MiniMax M-2.5)      & 21 & 0.76 & 17 & 0.20 & 15 & 0.53 & 12 & 1.05 & 8  & 0.33 & 9  & 0.76 \\
% \midrule
% % --- Vanilla Agentic Frameworks ---
% \multicolumn{13}{l}{\textbf{Vanilla Agentic Frameworks}} \\
% \midrule
% Claude Code (Opus-4.6)      & 92 & 0.86 & 89 & 0.18 & 93 & 0.80 & 88 & 0.87 & 96 & 0.89 & 83 & 0.83 \\
% Claude Code (GLM-5.0)       & 67 & 0.83 & 65 & 0.16 & 65 & 0.96 & 65 & 0.81 & 59 & 0.90 & 37 & 0.77 \\
% Claude Code (Qwen3.5-27b)   & 80 & 0.68 & 78 & 0.25 & 75 & 0.61 & 75 & 0.85 & 82 & 0.77 & 23 & 0.81 \\
% Claude Code (MiniMax M-2.5) & 69 & 0.78 & 69 & 0.16 & 74 & 0.59 & 73 & 0.72 & 83 & 0.63 & 69 & 0.58 \\
% \midrule
% % --- Kernel-Specialized Agents ---
% \multicolumn{13}{l}{\textbf{Kernel-Specialized Agents}} \\
% \midrule
% AKO4all (Opus-4.6)              & 89 & 1.00 & 84 & 0.30 & 88 & 1.09 & 88 & 1.08 & 86 & 1.12 & 80 & 1.07 \\
% CUDA Opt. Skill (MiniMax M-2.5) & 63 & 0.81 & 53 & 0.21 & 64 & 0.77 & 65 & 0.81 & 67 & 0.77 & 58 & 0.79 \\
% \midrule
% AutoKernel (GLM-5.0)            & 87 & 1.00 & 53 & 0.82 & 56 & 1.01 & 64 & 0.99 & 59 & 1.00 & 25 & 1.01 \\
% AutoKernel (Qwen3.5-27b)        & 69 & 1.00 & 40 & 0.37 & 75 & 1.03 & 65 & 1.00 & 74 & 1.04 & 21 & 1.01 \\
% AutoKernel (MiniMax M-2.5)      & 66 & 0.87 & 61 & 0.66 & 71 & 1.36 & 66 & 0.99 & 71 & 1.04 & 50 & 1.02 \\
% \bottomrule
% \end{tabular}%
% }
% \end{table}

\begin{table}[htbp]
\centering
\setlength{\tabcolsep}{3.5pt}
\caption{Cross-platform evaluation on 110 ATen operators across six hardware platforms, showing whether correctness and speedup transfer across heterogeneous hardware backends.}
\label{tab:cross_platform_main}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | cc | cc | cc | cc | cc | cc}
\toprule
\multirow{2}{*}{\textbf{Method \& Setup}} & \multicolumn{2}{c|}{\textbf{NVIDIA}} & \multicolumn{2}{c|}{\textbf{Platform A}} & \multicolumn{2}{c|}{\textbf{Platform B}} & \multicolumn{2}{c|}{\textbf{Platform C}} & \multicolumn{2}{c|}{\textbf{Platform D}} & \multicolumn{2}{c}{\textbf{Platform E}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}
& Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) & Acc (\%) & Spd ($\times$) \\
\midrule
% --- LLM Sampling Methods ---
\multicolumn{13}{l}{\textbf{LLM Sampling Methods}} \\
\midrule
Pass@1 (Opus-4.6)           & 39 & 0.90 & 46 & 0.19 & 44 & 0.69 & 37 & 0.98 & 38 & 0.89 & 38 & 0.88 \\
Pass@1 (Qwen3.5-27b)        & 8  & 0.83 & 9  & 0.09 & 3  & 1.02 & 7  & 0.90 & 7  & 0.98 & 10 & 1.03 \\
Pass@1 (MiniMax M-2.5)      & 4  & 0.88 & 4  & 0.25 & 6  & \underline{1.33} & 4  & \underline{1.05} & 5  & 0.77 & 4  & \textbf{1.15} \\
\midrule
Pass@5 (Opus-4.6)           & 62 & 0.79 & 63 & 0.15 & 60 & 0.74 & 54 & 0.92 & 65 & 0.68 & 57 & 0.83 \\
Pass@5 (Qwen3.5-27b)        & 13 & \textbf{1.04} & 16 & 0.18 & 11 & 1.10 & 15 & 0.72 & 10 & 0.99 & 17 & 1.02 \\
Pass@5 (MiniMax M-2.5)      & 21 & 0.76 & 17 & 0.20 & 15 & 0.53 & 12 & \underline{1.05} & 8  & 0.33 & 9  & 0.76 \\
\midrule
% --- Vanilla Agentic Frameworks ---
\multicolumn{13}{l}{\textbf{Vanilla Agentic Frameworks}} \\
\midrule
Claude Code (Opus-4.6)      & 92 & 0.86 & 89 & 0.18 & 93 & 0.80 & 88 & 0.87 & 96 & 0.89 & 83 & 0.83 \\
Claude Code (GLM-5.0)       & 67 & 0.83 & 65 & 0.16 & 65 & 0.96 & 65 & 0.81 & 59 & 0.90 & 37 & 0.77 \\
Claude Code (Qwen3.5-27b)   & 80 & 0.68 & 78 & 0.25 & 75 & 0.61 & 75 & 0.85 & 82 & 0.77 & 23 & 0.81 \\
Claude Code (MiniMax M-2.5) & 69 & 0.78 & 69 & 0.16 & 74 & 0.59 & 73 & 0.72 & 83 & 0.63 & 69 & 0.58 \\
\midrule
% --- Kernel-Specialized Agents ---
\multicolumn{13}{l}{\textbf{Kernel-Specialized Agents}} \\
\midrule
AKO4all (Opus-4.6)              & 89 & \underline{1.00} & 84 & 0.30 & 88 & 1.09 & 88 & \textbf{1.08} & 86 & \textbf{1.12} & 80 & \underline{1.07} \\
CUDA Opt. Skill (MiniMax M-2.5) & 63 & 0.81 & 53 & 0.21 & 64 & 0.77 & 65 & 0.81 & 67 & 0.77 & 58 & 0.79 \\
\midrule
AutoKernel (GLM-5.0)            & 87 & \underline{1.00} & 53 & \textbf{0.82} & 56 & 1.01 & 64 & 0.99 & 59 & 1.00 & 25 & 1.01 \\
AutoKernel (Qwen3.5-27b)        & 69 & \underline{1.00} & 40 & 0.37 & 75 & 1.03 & 65 & \underline{1.00} & 74 & \underline{1.04} & 21 & 1.01 \\
AutoKernel (MiniMax M-2.5)      & 66 & 0.87 & 61 & \underline{0.66} & 71 & \textbf{1.36} & 66 & 0.99 & 71 & \underline{1.04} & 50 & 1.02 \\
\bottomrule
\end{tabular}%
}
\end{table}

\textbf{Finding 1: Model and Method Capabilities Across Platforms.}
For accuracy, Claude Code (Opus-4.6) consistently achieves the highest rates across most platforms: 92\% on NVIDIA, 89\% on Platform A, 93\% on Platform B, 88\% on Platform C, and 96\% on Platform D; Platform E is the exception, where Claude Code (MiniMax M-2.5) reaches 69\%. For speedup, kernel-specialized agents dominate: AKO4all (Opus-4.6) achieves the best overall performance with 1.00$\times$ on NVIDIA, 1.08$\times$ on Platform C, 1.12$\times$ on Platform D, and 1.07$\times$ on Platform E, while AutoKernel (MiniMax M-2.5) reaches the highest single-platform speedup at 1.36$\times$ on Platform B. However, kernel-specialized agents exhibit severe accuracy variance across platforms: AKO4all ranges from 89\% on NVIDIA to 80\% on Platform E, while AutoKernel (Qwen3.5-27b) spans 75\% on Platform B down to 21\% on Platform E—revealing that even state-of-the-art specialized methods struggle with cross-platform portability.

This divergence stems from how methods utilize platform-specific information. Vanilla agentic frameworks like Claude Code provide minimal initial context but include hardware-specific constraints (e.g., API limitations, type strictness) in the prompt; agents actively leverage this information during debugging, adapting their implementations to platform quirks. Kernel-specialized agents, by contrast, focus heavily on performance tuning—profiling, block-size search, memory-access optimization—and often overlook the provided platform constraints, leading to compilation failures or runtime errors on non-NVIDIA backends despite achieving superior speedup when kernels do compile successfully. 

\textbf{Finding 2: Platform-Specific Performance Divergence.}
Cross-platform evaluation exposes severe performance heterogeneity. Platform A suffers from a catastrophic speedup collapse: Claude Code (Opus-4.6) maintains 89\% accuracy but achieves only 0.18$\times$ speedup, the lowest across all platforms. This 4.8$\times$ degradation relative to NVIDIA (0.86$\times$) reveals unoptimized backend implementations despite functional correctness. Platform E exhibits the opposite failure mode---accuracy collapse: Claude Code with Qwen3.5-27b drops to 23\%, and AutoKernel variants fall to 21--25\%, significantly lower than other platforms (typically 60--90\%). This reflects immature vendor compilers that frequently hang or crash when processing unstructured LLM-generated code, leading to compilation timeouts. Furthermore, non-NVIDIA platforms incur massive compilation overheads: Platform A requires 2.1$\times$ tokens and 2.0$\times$ time relative to NVIDIA (Figure~\ref{fig:radar_cross_platform}), forcing agents to burn iteration budgets on compilation debugging rather than kernel optimization.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.75\textwidth]{figures/fig2_radar_cross_platform.pdf}
\caption{Platform A collapses speedup to 0.18$\times$ despite high accuracy, while non-NVIDIA platforms incur up to 2$\times$ compilation overhead. Left: radar chart showing accuracy and speedup across six platforms (110 ATen, Claude Code Opus-4.6). Right: token and time overhead relative to NVIDIA.}
\label{fig:radar_cross_platform}
\end{figure}

\textbf{Finding 3: Cross-Platform Cost Overhead.}
Non-NVIDIA platforms incur massive compilation overheads, as quantified in Figure~\ref{fig:radar_cross_platform} (right panel). Platform A exhibits the most severe overhead: 173M total tokens (2.06$\times$ NVIDIA's 84M baseline) and 18 hours (2.00$\times$ NVIDIA's 9 hours). Platform B requires 128M tokens (1.52$\times$) and 16 hours (1.78$\times$), while Platforms C, D, and E consume 107--123M tokens (1.27--1.46$\times$) and 15--16 hours (1.67--1.78$\times$). This overhead is not algorithmic—it is purely ecosystem friction from immature vendor compilers and incomplete Triton backend support.

The root cause is that models lack prior exposure to heterogeneous hardware constraints during pretraining. When generating kernels for alternative platforms, agents must iteratively discover platform-specific limitations through trial and error: API availability (e.g., missing \texttt{tl.acosh} or \texttt{tl.math.tanh}), type system strictness (mixed int32/int64 loops that compile silently on CUDA but raise hard errors elsewhere), pointer addressing modes (32-bit vs 64-bit), and LLVM IR compatibility gaps. Each compilation failure forces the agent to burn tokens diagnosing opaque backend errors, adjusting code to satisfy undocumented constraints, and re-attempting compilation—consuming iteration budgets that would otherwise be spent on functional correctness or performance optimization. Platform A's 2$\times$ overhead directly reflects its backend's fragility: frequent compilation hangs and cryptic error messages force agents into extended debugging cycles, while Platform E's compiler instability (leading to the 21--25\% accuracy collapse noted in Finding 2) similarly inflates token costs as agents repeatedly retry failed compilations before timing out.

\subsection{Accuracy-Speedup Gap}
\label{subsec:accuracy_speedup_gap}

Figure~\ref{fig:dumbbell} plots per-operator accuracy against speedup on the NVIDIA baseline across 16 configurations. A systematic pattern emerges: accuracy spans the full range (2\% for MiniMax Pass@1 to 87\% for Claude Code Opus-4.6), while speedup clusters tightly in a narrow band (0.62--1.01$\times$, with 14 of 16 configurations falling within 0.68--0.83$\times$). The two outliers are Qwen3.5 Pass@5 at 1.01$\times$ (11\% accuracy) and MiniMax OpenCode at 0.62$\times$ (41\% accuracy). This divergence reveals survivorship bias: operators that weaker models fail to solve are disproportionately the computationally complex ones, where the baseline is heavily hand-tuned and high speedup is hardest to achieve. Weaker configurations report higher average speedup not because they optimize better, but because they fail the complex tasks and avoid their severe performance penalties (e.g., the 0.50$\times$ ceiling of cuBLAS). They only survive the simpler operators where matching the baseline (achieving $\sim$0.8--1.0$\times$) is relatively easy. Consequently, speedup comparisons across methods are only meaningful when conditioned on matched operator subsets.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig3_dumbbell.pdf}
\caption{Accuracy--speedup divergence: accuracy spans the full range while speedup clusters in a narrow band---``accuracy leaps, speedup stalls.''}
\label{fig:dumbbell}
\end{figure}


\subsection{Trajectory Analysis}
\label{subsec:trajectory}

Analysis of hundreds of complete LLM and agent trajectories uncovers two distinct failure layers. \textbf{Universal algorithmic failures} occur across all platforms: infinite dispatch recursion (calling the overridden ATen operator internally triggers unbounded recursion), hallucinated Triton APIs (models generate calls to non-existent functions such as \texttt{tl.pow}, \texttt{tl.einsum}, \texttt{tl.gather}), and algorithmically hard operators (\texttt{matmul}, \texttt{sort}, \texttt{cumsum} require cross-block parallel algorithms that agents rarely converge to).

\textbf{Heterogeneous-platform failures} directly explain the accuracy collapse on Platform E: LLVM IR incompatibility triggers \texttt{PassManager::run failed} errors (31 observed occurrences on one platform), 32-bit pointer addressing causes memory errors on large-tensor operators, and missing math APIs (\texttt{tl.acosh}, \texttt{tl.math.tanh}) must be manually reimplemented. Further failure patterns—8 universal and 6 platform-specific—are provided in Appendix~\ref{app:platform_failures}.

\subsection{Agentic Cost Efficiency}
\label{subsec:cost}

The introduction of closed-loop execution brings significant economic and time overhead. Table~\ref{tab:nvidia_cost_ops} quantifies this cost on the NVIDIA A100 baseline. Kernel-specialized agents universally consume far more tokens and time than vanilla frameworks---their extended iteration budgets are spent optimizing kernel performance. AKO4all is the most extreme case, requiring 904M tokens and 83 hours---over 3$\times$ the token cost of Claude Code (Opus-4.6) at 263M---achieving 5.19M tokens per successful operator. Across all kernel-specialized methods (AKO4all, CUDA Optimized Skill, AutoKernel variants), the average tokens per successful operator is 5.11M, orders of magnitude higher than vanilla agentic frameworks (1.45--3.30M) and simple LLM sampling approaches. These results highlight the need for more cost-efficient agentic methods that can close the performance gap without prohibitive overhead. Ablation studies isolating the value of execution feedback are detailed in Appendix~\ref{app:ablation_full}.

\begin{table}[htbp]
\centering
\caption{Agentic cost on NVIDIA A100 (210 operators). Total Tokens (M), Tokens per Successful Operator (M), and Total Time (h).}
\label{tab:nvidia_cost_ops}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{l | c c c}
\toprule
\textbf{Method} & \textbf{Total Tokens (M)} & \textbf{Tokens per Success (M)} & \textbf{Total Time (h)} \\
\midrule
Claude Code (Opus-4.6)     & 263 & 1.45 & 33 \\
Claude Code (GLM-5.0)      & 243 & 1.67 & 45 \\
Claude Code (Qwen3.5-27b)  & 381 & 2.93 & 48 \\
Claude Code (MiniMax M-2.5)& 340 & 3.30 & 50 \\
\midrule
AKO4all (Opus-4.6)       & 904 & 5.19 & 83 \\
CUDA Opt. Skill (MiniMax M-2.5) & 594 & 6.75 & 97 \\
AutoKernel (GLM-5.0)      & 471 & 3.16 & 102 \\
AutoKernel (Qwen3.5-27b)  & 475 & 4.80 & 102 \\
AutoKernel (MiniMax M-2.5)& 508 & 5.64 & 105 \\
\bottomrule
\end{tabular}
\end{table}