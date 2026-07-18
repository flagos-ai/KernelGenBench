# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rich.logging import RichHandler
from rich._log_render import LogRender, FormatTimeCallable
from rich.text import Text, TextType
from rich.console import Console, ConsoleRenderable, RenderableType
from typing import Iterable, List, Optional, TYPE_CHECKING, Union, Callable
from datetime import datetime
import re
import importlib
import itertools
from rich.containers import Renderables
from rich.table import Table


def add_register_decorator(code: str, operator: str, namespace: str = None, api: str = None) -> str:
    op_func_name = operator.replace(".", "_")
    pattern = f"def {op_func_name}("
    api = api if api else operator
    parts = code.rsplit(pattern, 1)
    code = f'{pattern}'.join([
        parts[0] + f'@register("{api}", "{operator}", False)\n', parts[-1]
    ]) if namespace is None else f'{pattern}'.join([
        parts[0] + f'@register("{api}", "{operator}", False, namespace="{namespace}")\n', parts[-1]
    ])
    return code

def expand_params(slots):
    # Each slot's values are bindings like [(v1,...), (v2,...)]
    # Single-parameter slots are stored as [v1, v2], so convert to tuples for consistency
    slot_values = []
    for slot in slots:
        names = slot["names"]
        vals = slot["values"]
        if len(names) == 1:
            vals = [(v,) for v in vals]  # convert single value to tuple
        slot_values.append((names, vals))

    # Cartesian product across slots
    results = []
    for combo in itertools.product(*[vals for _, vals in slot_values]):
        merged = {}
        for (names, _), vals in zip(slot_values, combo):
            merged.update(dict(zip(names, vals)))
        results.append(merged)

    return results

def generate_speedup_html(speedup_data, title="Performance Comparison Results", language="zh_CN"):
    console = Console(record=True)
    print(speedup_data)
    speedup_str = "Speedup"
    match language:
        case "zh_CN":
            speedup_str = "Speedup"
        case "en_US":
            speedup_str = "Speedup"
        case _:
            speedup_str = "Speedup"

    # Create main table
    table = Table(
        title="",
        show_header=True,
        header_style="bold bright_magenta",
        border_style="bright_blue",
        title_style="bold bright_cyan"
    )

    # Dynamically analyze parameter structure and get all parameter keys
    all_param_keys = set()
    for item in speedup_data:
        if item.get("params") != "avg" and isinstance(item.get("params"), dict):
            all_param_keys.update(item["params"].keys())

    # Sort parameter keys to ensure consistent display order
    param_keys = sorted(list(all_param_keys))

    # Add columns: speedup + parameter columns + time columns
    table.add_column(speedup_str, style="bold bright_white", justify="right", width=10)
    
    # Add one column per parameter key
    for key in param_keys:
        table.add_column(key, style="bold bright_yellow", width=10)
    
    table.add_column("PyTorch (ms)", style="bold bright_green", justify="right", width=14)
    table.add_column("Triton (ms)", style="bold bright_green", justify="right", width=14)

    # Add data rows
    for item in speedup_data:
        if item.get("params") == "avg":
            continue  # skip average row, handle separately later

        params = item.get("params", {})
        
        # Convert time to milliseconds
        ref_time_ms = item["ref_time"] * 1000
        res_time_ms = item["res_time"] * 1000
        speedup = item["speedup"]

        # Set color based on speedup value
        if speedup > 1:
            speedup_color = "bold bright_green"
        else:
            speedup_color = "bold bright_red"

        # Build row data
        row_data = [Text(f"{speedup:.3f}", style=speedup_color)]  # speedup

        # Add parameter values
        for key in param_keys:
            if isinstance(params, dict):
                value = params.get(key, "—")
                # Handle torch type display
                if "torch." in str(value):
                    value = str(value).replace("torch.", "")
                row_data.append(str(value))
            else:
                row_data.append("—")

        # Add time data
        row_data.extend([
            f"{ref_time_ms:.3f}",
            f"{res_time_ms:.3f}",
        ])

        table.add_row(*row_data)

    # Add average row
    avg_data = next((item for item in speedup_data if item.get("params") == "avg"), None)
    if avg_data:
        table.add_section()  # add separator line
        avg_speedup = avg_data["speedup"]
        avg_ref_time_ms = avg_data["ref_time"] * 1000
        avg_res_time_ms = avg_data["res_time"] * 1000

        if avg_speedup > 1:
            avg_color = "bold bright_green"
        else:
            avg_color = "bold bright_red"

        # Build average row
        avg_row = [Text(f"{avg_speedup:.3f}", style=avg_color)]  # speedup
        avg_row.extend([Text("—", style="dim")] * len(param_keys))  # fill parameter columns with dashes
        avg_row.extend([
            f"{avg_ref_time_ms:.3f}",
            f"{avg_res_time_ms:.3f}",
        ])

        table.add_row(*avg_row)

    # Record table with console
    console.print(table)

    # Export HTML
    html_content = console.export_html(inline_styles=True, clear=False)


    # Wrap into a complete HTML page
    html_page = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title></title>
        <style>
            body {{
                background-color: #0a0a0a;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', 'Lucida Console', monospace;
                padding: 20px;
                line-height: 1.6;
            }}
            pre {{
                background-color: #1a1a1a;
                padding: 20px;
                border-radius: 12px;
                overflow-x: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                border: 1px solid #333;
            }}
            /* Enhance color contrast */
            .ansi-bright-green-fg {{ color: #00ff00 !important; }}
            .ansi-bright-red-fg {{ color: #ff4444 !important; }}
            .ansi-bright-cyan-fg {{ color: #44ffff !important; }}
            .ansi-bright-yellow-fg {{ color: #ffff44 !important; }}
            .ansi-bright-blue-fg {{ color: #4444ff !important; }}
            .ansi-bright-magenta-fg {{ color: #ff44ff !important; }}
            .ansi-bright-white-fg {{ color: #ffffff !important; }}
        </style>
    </head>
    <body>
        <h1></h1>
        {html_content}
    </body>
    </html>
    """
    return html_page

class CustomLogRender(LogRender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(
        self,
        console: "Console",
        renderables: Iterable["ConsoleRenderable"],
        log_time: Optional[datetime] = None,
        time_format: Optional[Union[str, FormatTimeCallable]] = None,
        level: TextType = "",
        path: Optional[str] = None,
        line_no: Optional[int] = None,
        link_path: Optional[str] = None,
    ) -> "Table":
        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_time:
            output.add_column(style="log.time")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)
        output.add_column(ratio=1, style="log.message", overflow="fold")
        if self.show_path and path:
            output.add_column(style="log.path")
        row: List["RenderableType"] = []
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        if self.show_level:
            row.append(level)

        row.append(Renderables(renderables))
        if self.show_path and path:
            path_text = Text()
            path_text.append(
                path, style=f"link {link_path}" if link_path else ""
            )
            if line_no:
                path_text.append(":")
                path_text.append(
                    f"{line_no}",
                    style=f"link {link_path}#{line_no}" if link_path else "",
                )
            row.append(path_text)

        output.add_row(*row)
        return output


class CustomRichHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_render = CustomLogRender(
            show_time=True,
            show_level=True,
            show_path=True,
            time_format="[%x %X]",
            omit_repeated_times=True,
            level_width=None,
        )


def save_benchmark_result(bench_result: "BenchmarkResult", save_path: Optional[str] = None):
    """
    save benchmark result to txt file
    """
    # add mode to save_path
    if save_path is None:
        return
    with open(save_path, "a") as f:
        f.write("\n")
        f.write(str(bench_result))
    if save_path:
        print(f"Benchmark result saved to {save_path}")
