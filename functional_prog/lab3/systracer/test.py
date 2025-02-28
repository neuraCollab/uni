import ctypes
import os
import signal
import sys
import time

# Load the libc library
libc = ctypes.CDLL("libc.so.6")

# Define the ptrace constants
PTRACE_ATTACH = 16
PTRACE_DETACH = 17
PTRACE_SYSCALL = 24
PTRACE_CONT = 7
PTRACE_GETREGS = 12
PTRACE_SETREGS = 13

# Define the user_regs_struct
class user_regs_struct(ctypes.Structure):
    _fields_ = [
        ("r15", ctypes.c_ulonglong),
        ("r14", ctypes.c_ulonglong),
        ("r13", ctypes.c_ulonglong),
        ("r12", ctypes.c_ulonglong),
        ("rbp", ctypes.c_ulonglong),
        ("rbx", ctypes.c_ulonglong),
        ("r11", ctypes.c_ulonglong),
        ("r10", ctypes.c_ulonglong),
        ("r9", ctypes.c_ulonglong),
        ("r8", ctypes.c_ulonglong),
        ("rax", ctypes.c_ulonglong),
        ("rcx", ctypes.c_ulonglong),
        ("rdx", ctypes.c_ulonglong),
        ("rsi", ctypes.c_ulonglong),
        ("rdi", ctypes.c_ulonglong),
        ("orig_rax", ctypes.c_ulonglong),
        ("rip", ctypes.c_ulonglong),
        ("cs", ctypes.c_ulonglong),
        ("eflags", ctypes.c_ulonglong),
        ("rsp", ctypes.c_ulonglong),
        ("ss", ctypes.c_ulonglong),
        ("fs_base", ctypes.c_ulonglong),
        ("gs_base", ctypes.c_ulonglong),
        ("ds", ctypes.c_ulonglong),
        ("es", ctypes.c_ulonglong),
        ("fs", ctypes.c_ulonglong),
        ("gs", ctypes.c_ulonglong),
    ]

def ptrace(request, pid, addr, data):
    return libc.ptrace(request, pid, addr, data)

def attach_to_process(pid):
    if ptrace(PTRACE_ATTACH, pid, None, None) < 0:
        print(f"Failed to attach to process {pid}")
        return False
    print(f"Attached to process {pid}")
    return True

def detach_from_process(pid):
    if ptrace(PTRACE_DETACH, pid, None, None) < 0:
        print(f"Failed to detach from process {pid}")
        return False
    print(f"Detached from process {pid}")
    return True

def wait_for_syscall(pid):
    while True:
        if ptrace(PTRACE_SYSCALL, pid, None, None) < 0:
            print(f"Failed to wait for syscall in process {pid}")
            return False

        _, status = os.waitpid(pid, 0)
        if os.WIFSTOPPED(status) and os.WSTOPSIG(status) & 0x80:
            break
    return True

def get_registers(pid):
    regs = user_regs_struct()
    if ptrace(PTRACE_GETREGS, pid, None, ctypes.byref(regs)) < 0:
        print(f"Failed to get registers from process {pid}")
        return None
    return regs

def get_process_title(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").strip().decode('utf-8')
        return cmdline
    except Exception as e:
        print(f"Failed to get process title for PID {pid}: {e}")
        return "Unknown"

def get_process_status(pid):
    try:
        info = []
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f.readlines():
                if "Name" in line or "Pid" in line or "State" in line:
                    info.append(line.replace("\t", "").replace("\n", ""))
            # info = f.readlines()
        
        # status = status.split("\n")
        return info
    except Exception as e:
        print(f"Failed to get process status for PID {pid}: {e}")
        return "Unknown"

def monitor_process(pid):
    process_title = get_process_title(pid)
    process_status = get_process_status(pid)
    print(process_status)
    print(f"PID: {pid}, Title: {process_title}, Status: {process_status}")

    if not attach_to_process(pid):
        return

    try:
        while True:
            if not wait_for_syscall(pid):
                break
            # regs = get_registers(pid)
            # if regs:
            #     print(f"PID {pid} Syscall: {regs.orig_rax}")
            ptrace(PTRACE_CONT, pid, None, None)
    except KeyboardInterrupt:
        print(f"Monitoring stopped for PID {pid}.")
    finally:
        detach_from_process(pid)

def get_all_pids():
    pids = []
    for pid in os.listdir('/proc'):
        if pid.isdigit():
            pids.append(int(pid))
    return pids

def main():
    pids = get_all_pids()
    print(f"Monitoring {len(pids)} processes.")

    for pid in pids:
        monitor_process(pid)

if __name__ == "__main__":
    main()
