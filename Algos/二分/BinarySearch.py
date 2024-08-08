
def BinarySearch(nums,target):
    l = 0
    r = len(nums) - 1
    while l <= r:
        p = l + (r - l) // 2
        if nums[p] == target: # 找到Target
            return p
        elif nums[p] > target: # 左移r
            r = p - 1
        elif nums[p] < target: # 右移l
            l = p + 1
    return -1

def FindLeftOne(nums,target):
    # 找到大于等于元素的最靠左的下标
    l = 0
    r = len(nums) - 1
    res = []
    while l <= r:
        p = l + (r - l) // 2
        if nums[p] == target: # 找到Target
            res.append(p)
            r = p - 1
        elif nums[p] > target: # 左移r
            r = p - 1
        elif nums[p] < target: # 右移l
            l = p + 1
    return min(res)

def FindLocalMinimum(nums):
    # 局部最小值
    def localMinimum(i,n):
        if(i == 0):
            return n[i] < n[i+1]
        elif i == len(n) - 1:
            return n[i - 1] > n[i]
        else:
            return n[i-1] > n[i] and n[i] < n[i+1]

    l = 0
    r = len(nums) - 1
    while l < r:
        p = l + (r - l) // 2
        if (localMinimum(p,nums)):
            return p
        elif nums[p - 1] < nums[p]:
            r = p - 1
        elif nums[p + 1] < nums[p]:
            l = p + 1


if __name__ == '__main__':
    print(BinarySearch([1,2,3,4,5,6,7,8,9,10],5))
    print(FindLeftOne([1,2,2,2,2,3,3,3,3,4,4,4,5,5,5],3))
    print(FindLocalMinimum([9,8,7,6,5,6,7,8,9]))