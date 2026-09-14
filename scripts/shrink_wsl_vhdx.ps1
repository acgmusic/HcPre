# Shrink WSL ext4.vhdx (D:\wsl\ext4.vhdx)
#
# Step 1 (WSL 内, 需要 sudo 密码): fstrim /   —— 把空闲块返还给虚拟磁盘
# Step 2 (PowerShell 管理员): wsl --shutdown 后用 diskpart/Optimize-VHD 收缩 vhdx
#
# 逐条执行（不要整个脚本跑，Step1 需要交互输密码）:

# === Step 1: WSL 内执行 trim ===
# wsl -d Ubuntu-22.04 -- sudo fstrim -v /
#   （会提示输入密码；若 sudoers 允许 NOPASSWD 则直接成功）
#
# === Step 2: PowerShell（管理员）===
# wsl --shutdown
# # 确认没有 WSL 进程占用:
# Get-Process *vmmem*,*wsl* -ErrorAction SilentlyContinue
#
# # 若系统有 Hyper-V 模块（专业版）:
# Optimize-VHD -Path D:\wsl\ext4.vhdx -Mode Full
#
# # 否则用 diskpart（所有版本可用）:
# diskpart
#   select vdisk file="D:\wsl\ext4.vhdx"
#   attach vdisk readonly
#   compact vdisk
#   detach vdisk
#   exit
#
# === 验证 ===
# Get-Item D:\wsl\ext4.vhdx | Select-Object Length
Write-Host "这是一个操作说明脚本，请按注释逐步执行。Step1 需要交互输入 sudo 密码。"
