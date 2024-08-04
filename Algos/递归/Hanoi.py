def hanoi(n, source, target, auxiliary):
    """递归解决汉诺塔问题，并打印每一步的移动过程"""
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    hanoi(n - 1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    hanoi(n - 1, auxiliary, target, source)

# 示例
n = 3  # 盘子数量
hanoi(n, 'A', 'C', 'B')