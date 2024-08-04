

# 插入排序

def insertionSort(nums):
    """
    Insertion Sort
    :param nums:
    :return:
    """
    if not nums: return
    if len(nums) < 2: return nums
    for i in range(1, len(nums)):
        for j in range(i - 1,-1,-1):
            if nums[j] > nums[j+1]:
                nums[j],nums[j+1] = nums[j+1],nums[j]
    return nums

def swap(i,j,nums):
    temp = nums[i]
    nums[i] = nums[j]
    nums[j] = temp

if __name__ == '__main__':
    print(insertionSort([3,2,4,6,1,7,10,12]))