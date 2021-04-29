'''
Given a string s, find the length of the longest substring without repeating characters.
'''
# 暴力解法 Time O(n^2) Space O(n)
def lengthOfLongestSubstring(s):
    maxlen = 0
    for i in range(len(s)):
        curlen = 0
        lookup = set()
        for j in range(i, len(s)):
            if s[j] in lookup:
                break
            lookup.add(s[j])
            curlen += 1
                
        maxlen = max(curlen, maxlen)
    return maxlen


# 滑动窗口 Time O(n) Space O(gamma) gamma为字符可能出现的个数
def lengthOfLongestSubstring_(s):
    start,end = 0,-1
    lookup = set()
    for i in range(len(s)):
        if(i!=0):
            lookup.remove(s[i-1])
            

    # return max_len


if __name__ == "__main__":
    print(lengthOfLongestSubstring("abcabcbb"))
