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