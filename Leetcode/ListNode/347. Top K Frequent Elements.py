'''
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]
'''


def topKFrequent(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """
    # 实现数组元素与出现次数的映射哈希表
    hashmap = {}
    for num in nums:
        if(num in hashmap.keys()):
            hashmap[num] += 1
        else:
            hashmap[num] = 1
    
    


a = [1, 2, 3, 3, 3, 4, 5, 6, 6, 7]
print(topKFrequent(a, 1))
