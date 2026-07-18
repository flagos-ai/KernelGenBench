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

"""
Cheat kernel #3: Forbidden import attack.
Attempts to import os, socket, ctypes, subprocess.

Should be caught by Layer 3 (Import Hook).
"""
import os
import socket
import subprocess
import ctypes

def forward(x, weight):
    # Try system calls
    os.system("echo hacked")
    # Try socket
    s = socket.socket()
    return x @ weight.T
