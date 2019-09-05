[1.Two Sum](https://leetcode.com/problems/two-sum/)

Use a map to record <number, index>.



[167. Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted)

Binary Search



[3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

Use one int array map[256] to record the last index of each character, the length will be `current index - last index`.
```java
    public int lengthOfLongestSubstring(String s) {
        int[] lastIndex = new int[256];
        Arrays.fill(lastIndex, -1); // fill -1
        int res = 0;
        for(int hi = 0, lo = 0; hi < s.length(); hi++){
            char c = s.charAt(hi);
            if(lastIndex[c] >= lo){
                lo = lastIndex[c] + 1; // remember +1
            }
            lastIndex[c] = hi;
            res = Math.max(res, hi - lo + 1);
        }
        
        return res;
    }

```


[5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

```java
private int start, maxLen;

public String longestPalindrome(String s) {
    if(s.length() == 0)
        return "";
    if(s.length() < 2)
        return s;
    for(int i = 0; i < s.length() - 1; i++){
        findPalindrome(s, i, i); // even
        findPalindrome(s, i, i + 1); // odd
    }
    return s.substring(start, start + maxLen);
}
private void findPalindrome(String s, int left, int right){
    while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)){
        left--;
        right++;
    }
    if(maxLen < right - left - 1){
        start = left + 1;
        maxLen = right - left - 1;
    }
}

```



[7. Reverse Integer](https://leetcode.com/problems/reverse-integer/)

```java
public int reverse(int x) {
    long res = 0;
    while(x != 0){
        res = res * 10 + x % 10;
        x /= 10;
    }
    return (int)res == res ? (int)res : 0; // deal with overflow. 
    //exp: 1534236469 -> 9646324351
}
```



[8. String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/)

Use one pointer `index`. Be attention about overflow. 

```java
public int myAtoi(String str) {
    if(str == null || str.length() == 0) return 0;
    
    int sign = 1, total = 0, index = 0;
    
    //remove space
    while(index < str.length() && str.charAt(index) == ' ')
        index++;
    
    //judge sign
    if(index < str.length() && (str.charAt(index) == '+' || str.charAt(index) == '-')){
        sign = str.charAt(index) == '+' ? 1 : -1;
        index++;
    }
    
    //iterate number or chars
    while(index < str.length()){
        int cur = str.charAt(index) - '0';
        if(cur < 0 || cur > 9) //judge whether number or not
            break;
        
        if((Integer.MAX_VALUE - cur)/10 < total){
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        
        total = total * 10 + cur;
        index++;
    }
    
    return total * sign;
}
```



[9. Palindrome Number](https://leetcode.com/problems/palindrome-number/)

reverse the number, then compare whether the `reverse` equals to the original one.

```java
public boolean isPalindrome(int x) {
    if(x < 0) return false;        
    int num = x, reverse = 0;
    while(num != 0) {
        reverse = reverse * 10 + num % 10;
        num /= 10;
    }
    return x - reverse == 0;
}
```



[11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

Two pointers, `i`  points to head and `j` points to tail.  The area is `(j - i) * Math.min(height[i], height[j])`. And every time compare two side heights. Move the minor one.

```java
public int maxArea(int[] height) {
    int i = 0, j = height.length - 1;
    int res = Integer.MIN_VALUE;
    while (i < j) {
        res = Math.max(res, (j - i) * Math.min(height[i], height[j]));
        if (height[i] < height[j]) 
            i++;
        else                       
            j--;
    }
    return res;
}
```



[12. Integer to Roman](https://leetcode.com/problems/integer-to-roman/)

全文背诵

```java
public String intToRoman(int num) {
    if(num < 1 || num > 3999) return "";
    int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    String[] roman = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
    
    StringBuilder res = new StringBuilder();
    
    int i = 0;
    while(num > 0){
        while(num >= values[i]){
            num -= values[i];
            res.append(roman[i]);
        }
        i++;
    }
    return res.toString();
}
```



[13. Roman to Integer](https://leetcode.com/problems/roman-to-integer/)

全文背诵

```java
public int romanToInt(String s) {
    Map<Character, Integer> map = new HashMap<>();
    map.put('I', 1);
    map.put('V', 5);
    map.put('X', 10);
    map.put('L', 50);
    map.put('C', 100);
    map.put('D', 500);
    map.put('M', 1000);
    
	int pre = 0;
    int res = 0;
    for (int i = s.length() - 1; i >= 0; i--){
        char ch = s.charAt(i);
        int cur = map.get(ch);
        if(cur >= pre)
            res += cur;
        else
            res -= cur;
        pre = cur;
    }
    return res;
    
}
```



[14. Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)

Use the first string to match others. 

```java
public String longestCommonPrefix(String[] strs) {
    if(strs.length == 0 || strs == null) return "";
    StringBuilder res = new StringBuilder();
    for(int i = 0; i < strs[0].length(); i++){
        char c = strs[0].charAt(i);
        for(String str : strs){
            if(str.length() < i+1 || c != str.charAt(i)) 
                return res.toString();
        }
        res.append(c);
    }
    return res.toString();
}
```

Or every time cut the last character while current prefix doesn't match.

```java
public String longestCommonPrefix(String[] strs) {
    if(strs == null || strs.length == 0)  return "";
    String pre = strs[0];
    int i = 1;
    while(i < strs.length){
        while(strs[i].indexOf(pre) != 0)
            pre = pre.substring(0, pre.length()-1);
        i++;
    }
    return pre;
}
```



[15. 3Sum](https://leetcode.com/problems/3sum/)

Convert to 2Sum question for each element in array. Remember to skip the multiple same value in array.

```java
public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(nums);
    for(int i = 0; i < nums.length; i++){
        if(i > 0 && nums[i] == nums[i-1])
            continue;
        int sum = -nums[i];
        int low = i + 1, high = nums.length - 1;
        twoSum(nums, res, low, high, sum);
    }
    return res;
}
private void twoSum(int[] nums, List<List<Integer>> res, int low, int high, int sum){
    while(low < high){
        if(nums[low] + nums[high] == sum){
            res.add(Arrays.asList(-sum, nums[low], nums[high]));
            while(low < high && nums[low] == nums[low + 1])
                low++;
            while(low < high && nums[high] == nums[high - 1])
                high--;
            low++;
            high--;
        }
        else if(nums[low] + nums[high] < sum)
            low++;
        else 
            high--;
    }
}
```

[16. 3Sum Closest](https://leetcode.com/problems/3sum-closest/)

Same as 3Sum and there is only solution, so there no need to skip the same.

```java
public int threeSumClosest(int[] nums, int target) {
    if (nums == null || nums.length < 3) 
        throw new IllegalArgumentException();
    
    Arrays.sort(nums);
    int n = nums.length;
    int res = nums[0] + nums[1] + nums[2];
    for (int i = 0; i < n - 2; i++) {
        int lo = i + 1;
        int hi = n - 1;
        while (lo < hi) {
            int sum = nums[i] + nums[lo] + nums[hi];
            if (sum == target) return sum;
            if (Math.abs(sum - target) < Math.abs(res - target))
                res = sum;
            if (sum > target) hi--;
            else lo++;
        }
    }
    
    return res;
}
```

[17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

dfs

```java
public List<String> letterCombinations(String digits) {
    LinkedList<String> res = new LinkedList<>();
    if(digits.length() == 0)
        return res;
    String[] map = new String[]{"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    dfs(res, map, digits.toCharArray(), "", 0);
    return res;
}

private void dfs(List<String> res, String[] map, char[] digits, String temp, int start){
    if(start == digits.length){
        res.add(temp);
        return;
    }
    for(int i = 0; i < map[digits[start]-'0'].length(); i++){
        dfs(res, map, digits, temp + map[digits[start]-'0'].charAt(i), start +1);
    }
}
```



[18. 4Sum](https://leetcode.com/problems/4sum/)

```java
public List<List<Integer>> fourSum(int[] nums, int target) {
    List<List<Integer>> res = new ArrayList<>();
    if(nums.length < 4) return res;
    
    Arrays.sort(nums);
    
    int n = nums.length;
    
    for(int a = 0; a < n - 3; a++){
        if(a > 0 && nums[a] == nums[a-1]) continue;
        
        if(nums[a] * 4 > target) break;
        if(nums[a] + nums[n - 1] * 3 < target) continue;
        
        for(int b = a + 1; b < n - 2; b++){
            if(b > a + 1 && nums[b] == nums[b - 1]) continue;
            
            if(nums[a] + nums[b] * 3 > target) break;
            if(nums[a] + nums[b] + nums[n - 1] * 2 < target) continue;
            
            int c = b + 1;
            int d = n - 1;
            
            while(c < d){
                int sum = nums[a] + nums[b] + nums[c] + nums[d];
                if(sum < target){
                    c++;
                }else if(sum > target){
                    d--;
                }else{
                    res.add(Arrays.asList(nums[a], nums[b], nums[c], nums[d]));
                    while(c < d && nums[c] == nums[c+1]) c++;
                    while(c < d && nums[d] == nums[d-1]) d--;
                    c++;
                    d--;
                }
            }
        }
    }
    return res;
}
```

[19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

`slow` and `fast` pointers. `fast` move `n` steps first, then slow and fast move together until fast reach tail.

```java
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode fast = head, slow = head;
    while(n-- > 0) fast = fast.next;
    if(fast == null) return head.next;
    while(fast.next != null){
        slow = slow.next;
        fast = fast.next;
    }
    slow.next = slow.next.next;
    return head;
}
```



[20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

stack

```java
public boolean isValid(String s) {
    if(s == null || s.length() == 0) return true;
    Stack<Character> stack = new Stack<>();
    for(char c : s.toCharArray()){
        if(c == '(' || c == '[' || c == '{') stack.push(c);
        if(stack.isEmpty()) return false;
        if(c == ')' && stack.pop() != '(') return false;
        else if(c == ']' && stack.pop() != '[') return false;
        else if(c == '}' && stack.pop() != '{') return false;
    }
    return stack.isEmpty();
}
```



[21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if(l1 == null){
        return l2;
    }
    if(l2 == null){
        return l1;
    }
    if(l1.val < l2.val){
        l1.next = mergeTwoLists(l1.next, l2);
        return l1;
    }else{
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
    }
}
```



[22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

Backtrack. Generate`(` first, then generate the equal number`)`. Once `temp`'s length equals to n*2, then add this result.

```java
public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();
    backtrack(res, "", 0, 0, n);
    return res;
}
private void backtrack(List<String> res, String temp, int left, int right, int n){
    if(temp.length() == n * 2){
        res.add(temp);
        return;
    }
    if(left < n)
        backtrack(res, temp + "(", left + 1, right, n);
    if(right < left)
        backtrack(res, temp + ")", left, right + 1, n);
}
```



[23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

Use sort and merge. 

In sort, `left` and `right` two pointers. `left` to begin and `right` to end. Then like binary search.

In merge, just like [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) solution.

```java
public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0) 
        return null;
    return sort(lists, 0, lists.length - 1);
}
private ListNode sort(ListNode[] lists, int left, int right) {
    if (left >= right) 
        return lists[left];

    int mid = left + (right - left) / 2;
    ListNode node1 = sort(lists, left, mid);
    ListNode node2 = sort(lists, mid + 1, right);
    return merge(node1, node2);
}
private ListNode merge(ListNode node1, ListNode node2) {
    if (node1 == null) 
        return node2;
    if (node2 == null) 
        return node1;

    if (node1.val < node2.val) {
        node1.next = merge(node1.next, node2);
        return node1;
    }
    node2.next = merge(node1, node2.next);
    return node2;
}
```



[24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

Use dummy node. In while loop, use two node `first`and `second` to manipulate the linked list.

```java
public ListNode swapPairs(ListNode head) {
    ListNode dummy = new ListNode(-1);
    dummy.next = head;
    ListNode cur = dummy;
    while(cur.next != null && cur.next.next != null){
        ListNode first = cur.next;
        ListNode second = cur.next.next;
        first.next = second.next;
        cur.next = second;
        second.next = first;
        
        cur = cur.next.next;
    }
    return dummy.next;
}
```



[26. Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

Find the next value that is not equal to current one, then put this next value to the index next to  current.

```java
public int removeDuplicates(int[] nums) {
    if(nums.length < 2)
        return nums.length;
    
    int start = 0;
    for(int i = 1; i < nums.length; i++){
        if(nums[start] != nums[i])
            nums[++start] = nums[i];
    }
    return start+1;
}
```



[27. Remove Element](https://leetcode.com/problems/remove-element/)

```java
public int removeElement(int[] nums, int val) {
    int start = 0;
    for(int i = 0; i < nums.length; i++){
        if(nums[i] != val)
            nums[start++] = nums[i];
    }
    return start;
}
```



[28. Implement strStr()](https://leetcode.com/problems/implement-strstr/)

**Example 1:**

```
Input: haystack = "hello", needle = "ll"
Output: 2
```

```java
public int strStr(String haystack, String needle) {
    for(int i = 0; i <= haystack.length() - needle.length(); i++){
        int j = 0;
        while(j < needle.length() && needle.charAt(j) == haystack.charAt(i+j)) 
            j++;
        if(j == needle.length()) 
            return i;
    }
    return -1;
}
```



[29. Divide Two Integers](https://leetcode.com/problems/divide-two-integers/)

Suppose `dividend = 15` and `divisor = 3`, `15 - 3 > 0`. We now try to subtract more by *shifting* `3` to the left by `1` bit (`6`). Since `15 - 6 > 0`, shift `6` again to `12`. Now `15 - 12 > 0`, shift `12` again to `24`, which is larger than `15`. So we can at most subtract `12` from `15`. Since `12` is obtained by shifting `3` to left twice, it is `1 << 2 = 4` times of `3`. We add `4` to an answer variable (initialized to be `0`). The above process is like `15 = 3 * 4 + 3`. We now get part of the quotient (`4`), with a remaining dividend `3`.

Then we repeat the above process by subtracting `divisor = 3` from the remaining `dividend = 3` and obtain `0`. We are done. In this case, no shift happens. We simply add `1 << 0 = 1` to the answer variable.

```java
public int divide(int dividend, int divisor) {
    if (divisor == 0 || dividend == Integer.MIN_VALUE && divisor == -1)
        return Integer.MAX_VALUE;
    int sign = ((dividend > 0) ^ (divisor > 0)) ? -1 : 1;
    long num = Math.abs((long) dividend);
    long deno = Math.abs((long) divisor);
    int res = 0;
    while (num >= deno) {
        long temp = deno, mul = 1;
        while (num >= (temp << 1)) {
            temp <<= 1;
            mul <<= 1;
        }
        num -= temp;
        res += mul;
    }
    return sign == 1 ? res : -res;
}
```



[31. Next Permutation](https://leetcode.com/problems/next-permutation/)

1. Find the largest index `k` such that `nums[k] < nums[k + 1]`. If no such index exists, just reverse `nums` and done.
2. Find the largest index `l > k` such that `nums[k] < nums[l]`.
3. Swap `nums[k]` and `nums[l]`.
4. Reverse the sub-array `nums[k + 1:]`.

```java
public void nextPermutation(int[] nums) {
    if (nums == null || nums.length < 2) {
        return;
    }
    int i = nums.length - 2; 
    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }// find first non-increasing number
    if (i >= 0) {
        int j = nums.length - 1;
        while (j > i && nums[j] <= nums[i]) {
            j--;
        }// find the first number that bigger than that non-increasing number,
         // then swap them
        swap(nums, i, j);
    }
    reverse(nums, i + 1); // reverse non-increasing number to the last part
}

private void reverse(int[] nums, int start) {
    int end = nums.length - 1;
    while (start < end) {
        swap(nums, start, end);
        start++;
        end--;
    }
}
private void swap(int[] nums, int i, int j) {
    int tmp = nums[i];
    nums[i] = nums[j];
    nums[j] = tmp;
}
```



[33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

```java
public int search(int[] nums, int target) {
    int start = 0, end = nums.length - 1;
    
    while(start <= end){
        int mid = start + (end - start) / 2;
        if(nums[mid] == target) return mid;
        if(nums[start] <= nums[mid]){
            if(nums[start] <= target && target < nums[mid])
                end = mid - 1;
            else
                start = mid + 1;
        }
        if(nums[mid] <= nums[end]){
            if(nums[mid] < target && target <= nums[end])
                start = mid + 1;
            else
                end = mid - 1;
        }
    }
    return -1;
}
```

[81. Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)

```java
public boolean search(int[] nums, int target) {
    int start = 0, end = nums.length - 1;
    while(start <= end){
        int mid = start + (end - start) / 2;
        if(nums[mid] == target) return true;
        if(nums[start] < nums[mid]){
            if(target < nums[start] || target > nums[mid])
                start = mid + 1;
            else
                end = mid - 1;
        }
        else if(nums[start] > nums[mid]){
            if(target < nums[mid] || target > nums[end])
                end = mid - 1;
            else
                start = mid + 1;
        }
        else
            start++;
    }
    return false;
}
```



[34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

```java
public int[] searchRange(int[] nums, int target) {
    int start = 0, end = nums.length - 1;
    while (start <= end) {
        int mid = (start + end) / 2;
        if (nums[mid] == target) {
            int left = mid, right = mid;
            while (left - 1 >= start && nums[left - 1] == target) {
                left--;
            }
            while (right + 1 <= end && nums[right + 1] == target) {
                right++;
            }
            return new int[]{left, right};
        } else if (nums[mid] < target) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    return new int[]{-1, -1};
}
```



[35. Search Insert Position](https://leetcode.com/problems/search-insert-position/)

```java
public int searchInsert(int[] nums, int target) {
    int start = 0;
    int end = nums.length-1;
    while (start <= end) {
        int mid = start + (end-start)/2;
        if (nums[mid] == target) 
            return mid;
        else if (nums[mid] < target) 
            start = mid+1;
        else 
            end = mid-1;
    }
    return start;
}
```



[36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)

```java
public boolean isValidSudoku(char[][] board) {
    Set seen = new HashSet();
    for (int i=0; i<9; ++i) {
        for (int j=0; j<9; ++j) {
            if (board[i][j] != '.') {
                String b = "(" + board[i][j] + ")";
                if (!seen.add(b + i) || !seen.add(j + b) || !seen.add(i/3 + b + j/3))
                    return false;
            }
        }
    }
    return true;
}
```





[42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

```java
public int trap(int[] height) {
    if(height == null || height.length == 0) return 0;
    int l = 0, r = height.length - 1;
    int leftmax = height[0], rightmax = height[r];
    int water = 0;
    while(l < r){
        leftmax = Math.max(leftmax, height[l]);
        rightmax = Math.max(rightmax, height[r]);
        if(leftmax < rightmax){
            water += leftmax - height[l];
            l++;
        }else{
            water += rightmax - height[r];
            r--;
        }
    }
    return water;
}
```



[43. Multiply Strings](https://leetcode.com/problems/multiply-strings/)



[55. Jump Game](https://leetcode.com/problems/jump-game/)

```java
public boolean canJump(int[] nums) {
    int reach = 0;
    for(int i = 0; i < nums.length && i <= reach; i++){
        reach = Math.max(nums[i] + i, reach);
        if(reach >= nums.length - 1)
            return true;
    }
    return false;
}
```



[45. Jump Game II](https://leetcode.com/problems/jump-game-ii/)

```java
public int jump(int[] A) {
  if(A == null || A.length == 0) {
       return -1;
  }
  int start = 0;
  int end = 0;
  int farthest = 0;
  int step = 0;
  while(end < A.length -1) {
      step++;
      for(int i = start; i <= end; i++) {
          farthest = Math.max(farthest, A[i] + i);
      }
      start = end + 1;
      end = farthest;
  }
    return step;
}
```





[48. Rotate Image](https://leetcode.com/problems/rotate-image/)

```java
/*
 * clockwise rotate
 * first reverse up to down, then swap the symmetry 
 * 1 2 3     7 8 9     7 4 1
 * 4 5 6  => 4 5 6  => 8 5 2
 * 7 8 9     1 2 3     9 6 3
*/
public void rotate(int[][] matrix) {
    int m = matrix.length, n = matrix[0].length;
    
    for(int i = 0; i < m/2; i++){
        for(int j = 0; j < n; j++){
            int temp = matrix[i][j];
            matrix[i][j] = matrix[m-1-i][j];
            matrix[m-1-i][j] = temp;
        }
    }
    for(int i = 0; i < m; i++){
        for(int j = i+1; j < n; j++){
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}

/*
 * anticlockwise rotate
 * first reverse left to right, then swap the symmetry
 * 1 2 3     3 2 1     3 6 9
 * 4 5 6  => 6 5 4  => 2 5 8
 * 7 8 9     9 8 7     1 4 7
*/
void anti_rotate(vector<vector<int> > &matrix) {
    for (auto vi : matrix) reverse(vi.begin(), vi.end());
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = i + 1; j < matrix[i].size(); ++j)
            swap(matrix[i][j], matrix[j][i]);
    }
}
```





[49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)

Use a `map<String, List<String>> ` to store every group. Sort every string then compare.

```java
public List<List<String>> groupAnagrams(String[] strs) {
    if(strs.length == 0 || strs == null)
        return new ArrayList<List<String>>();
    Map<String, List<String>> map = new HashMap<>();
    
    for(String s : strs){
        char[] ch = s.toCharArray();
        Arrays.sort(ch);
        String keyStr = String.valueOf(ch);
        if(!map.containsKey(keyStr))
            map.put(keyStr, new ArrayList<>());
        map.get(keyStr).add(s);
    }
    return new ArrayList<List<String>>(map.values());
}
```



[50. Pow(x, n)](https://leetcode.com/problems/powx-n/)

`x * x` n times. use `long` to convert `n` in order to avoid overflow. 

```java
public double myPow(double x, int n) {
    double res = 1;
    long N = Math.abs((long)n);
    while(N > 0){
        if(N % 2 > 0) res *= x;
        x *= x;
        N /= 2;
    }
    return n > 0 ? res : 1/res;
}
```



[53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

Use `preSum` and `maxSum`.  `        preSum = preSum > 0 ? preSum + nums[i] : nums[i];`.

```java
public int maxSubArray(int[] nums) {
    if(nums.length == 0 || nums == null) return 0;
    int preSum = nums[0];
    int maxSum = preSum;
    for(int i = 1; i < nums.length; i++){
        preSum = preSum > 0 ? preSum + nums[i] : nums[i];
        maxSum = Math.max(preSum, maxSum);
    }
    return maxSum;
}
```



[54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)

Brute force implement. 

```java
public List<Integer> spiralOrder(int[][] matrix) {
    List<Integer> res = new ArrayList<>();
    if(matrix.length == 0 || matrix == null)
        return res;
    
    int rowBegin = 0, rowEnd = matrix.length - 1;
    int colBegin = 0, colEnd = matrix[0].length - 1;
    
    while(rowBegin <= rowEnd && colBegin <= colEnd){
        for(int i = colBegin; i <= colEnd; i++){
            res.add(matrix[rowBegin][i]);
        }
        rowBegin++;
        
        for(int i = rowBegin; i <= rowEnd; i++){
            res.add(matrix[i][colEnd]);
        }
        colEnd--;
        
        if(rowBegin <= rowEnd){
            for(int i = colEnd; i >= colBegin; i--){
                res.add(matrix[rowEnd][i]);
            }
        }
        rowEnd--;
        
        if(colBegin <= colEnd){
            for(int i = rowEnd; i >= rowBegin; i--){
                res.add(matrix[i][colBegin]);
            }
        }
        colBegin++;
    }
    return res;
}
```

[59. Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)

```java
public int[][] generateMatrix(int n) {
    int[][] res = new int[n][n];
    
    int rowBegin = 0, rowEnd = n - 1; //
    int colBegin = 0, colEnd = n - 1; //
    int num = 1;
    
    while(rowBegin <= rowEnd && colBegin <= colEnd){
        for(int i = colBegin; i <= colEnd; i++){
            res[rowBegin][i] = num++;
        }
        rowBegin++;
        
        for(int i = rowBegin; i <= rowEnd; i++){
            res[i][colEnd] = num++;
        }
        colEnd--;
        
        if(colBegin <= colEnd){
            for(int i = colEnd; i >= colBegin; i--){
                res[rowEnd][i] = num++;
            }
        }
        rowEnd--;
        
        if(rowBegin <= rowEnd){
            for(int i = rowEnd; i >= rowBegin; i--){
                res[i][colBegin] = num++;
            }
        }
        colBegin++;
    }
    return res;
}
```

[56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)

Sort all intervals by start in ascending. Then use `last` to record the last interval we traversed.

Compare the current interval with `last`.

```java
public List<Interval> merge(List<Interval> intervals) {
    List<Interval> res = new ArrayList<>();
    if(intervals.size() == 0 || intervals == null) return res;
    
    Collections.sort(intervals, new Comparator<Interval>(){
        public int compare(Interval a, Interval b){
            return a.start - b.start;
        }
    });
    
    Interval last = null;
    for(Interval i : intervals){
        if(last == null || i.start > last.end){
            res.add(i);
            last = i;
        }
        else{
            last.end = Math.max(i.end, last.end);
        }
    }
    
    return res;
}
```



[57. Insert Interval](https://leetcode.com/problems/insert-interval/)

```java
public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
    List<Interval> res = new LinkedList<>();
    int pos = 0;
    
    for(Interval i : intervals){
        if(i.end < newInterval.start){
            res.add(i);
            pos++;
        }else if(newInterval.end < i.start){
            res.add(i);
        }else{
            newInterval.start = Math.min(newInterval.start, i.start);
            newInterval.end = Math.max(newInterval.end, i.end);
        }
    }
    res.add(pos, newInterval);
    
    return res;
}
```



[61. Rotate List](https://leetcode.com/problems/rotate-list/)

```java
public ListNode rotateRight(ListNode head, int k) {
    if(head == null || head.next == null) return head;
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    
    int i;
    ListNode slow = dummy, fast = dummy;
    for(i = 0; fast.next != null; i++)
        fast = fast.next;
    
    for(int j = i - k % i; j > 0; j--)
        slow = slow.next;
    
    fast.next = dummy.next;
    dummy.next = slow.next;
    slow.next = null;
    
    return dummy.next;
}
```



[62. Unique Paths](https://leetcode.com/problems/unique-paths/)

```java
public int uniquePaths(int m, int n) {
    int[][] grid = new int[m][n];
    
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (i==0 || j==0) {
                grid[i][j] = 1;
            } else {
                grid[i][j] = grid[i][j-1] + grid[i-1][j];
            }
        }
    }
    
    return grid[m-1][n-1];
}
```



[63. Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)

```java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int m = obstacleGrid.length;
    int n = obstacleGrid[0].length;
    int[][] grid = new int[m][n];
    
    grid[0][0] = 1;
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            if (obstacleGrid[i][j] == 1) {
                grid[i][j] = 0;
                continue;
            }
            if (i==0 && j > 0) {
                grid[i][j] = grid[i][j-1];
            } else if(j==0 && i > 0) {
                grid[i][j] = grid[i-1][j];
            } else if(i > 0 && j > 0) {
                grid[i][j] = grid[i][j-1] + grid[i-1][j];
            }
        }
    }
    return grid[m-1][n-1];
}
```



[64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

```java
public int minPathSum(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    for(int i = 1; i < m; i++){
        grid[i][0] += grid[i-1][0];
    }
    for(int i = 1; i < n; i++){
        grid[0][i] += grid[0][i-1];
    }
    for(int i = 1; i < m; i++){
        for(int j = 1; j < n; j++){
            grid[i][j] += Math.min(grid[i-1][j], grid[i][j-1]);
        }
    }
    return grid[m-1][n-1];
}   
```



[65. Valid Number](https://leetcode.com/problems/valid-number/)

```java
public boolean isNumber(String s) {
    if(s == null) return true;
    
    s = s.trim();
    boolean pointSeen = false;
    boolean eSeen = false;
    boolean numberSeen = false;
    boolean numberAfterE = true;
    
    for(int i = 0; i < s.length(); i++){
        if(s.charAt(i) >= '0' && s.charAt(i) <= '9'){
            numberSeen = true;
            numberAfterE = true;
        }else if(s.charAt(i) == '.'){
            if(eSeen || pointSeen) 
                return false;
            pointSeen = true;
        }else if(s.charAt(i) == 'e'){
            if(eSeen || !numberSeen) 
                return false;
            eSeen = true;
            numberAfterE = false;
        }else if(s.charAt(i) == '-' || s.charAt(i) == '+'){
            if(i != 0 && s.charAt(i-1) != 'e') 
                return false;
        }else {
            return false;
        }
    }

         
    return numberSeen && numberAfterE;
}
```



[66. Plus One](https://leetcode.com/problems/plus-one/)

```java
public int[] plusOne(int[] digits) {
    int len = digits.length;
    for(int i = len - 1; i >= 0; i--){
        if(digits[i] < 9){
            digits[i]++;
            return digits;
        }
        digits[i] = 0;
    }
    int[] res = new int[len+1];
    res[0] = 1;
    
    return res;
}
```



[69. Sqrt(x)](https://leetcode.com/problems/sqrtx/)

```java
public int mySqrt(int x) {
    if(x == 0) return 0;
    int lo = 1, hi = x/2 + 1;
    while(lo + 1 < hi){
        int mid = lo + (hi - lo) / 2;
        if(mid > x/mid)
            hi = mid;
        else if(mid < x/mid)
            lo = mid;
        else 
            return mid;
    }
    return lo;
}
```



[70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)

Fib

```java
public int climbStairs(int n) {
   int[] a=new int[n+1];
   a[0]=1;
   a[1]=1;
    for(int i=2;i<=n;i++){
       a[i]=a[i-1]+a[i-2]; 
    }
    return a[n];
}
```



[71. Simplify Path](https://leetcode.com/problems/simplify-path/)

```java
public String simplifyPath(String path) {
    StringBuilder sb = new StringBuilder();
    Stack<String> stack = new Stack<>();
    for(String s : path.split("/")){
        if(s.equals("..")){
            if(!stack.isEmpty())
                stack.pop();
        }
        else if(!s.equals(".") && !s.equals(""))
            stack.push(s);
    }
    if(stack.isEmpty())
        return "/";
    for(String s : stack){
        sb.append("/" + s);
    }
    return sb.toString();
}
```



[73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)

```java
public void setZeroes(int[][] matrix) {
    int[] rows = new int[matrix.length];
    int[] cols = new int[matrix[0].length];
    
    for (int i = 0; i < matrix.length; ++i) {
        for (int j = 0; j < matrix[0].length; ++j) {
            if (matrix[i][j] == 0) {
                rows[i] = 1;
                cols[j] = 1;
            }
        }
    }
    for (int i = 0; i < rows.length; ++i) {
        if (rows[i] == 1) {
            makeRowZero(matrix, i);
        }
    }
    for (int i = 0; i < cols.length; ++i) {            
        if (cols[i] == 1) {
            makeColZero(matrix, i);
        }
    }
}
void makeColZero(int[][] matrix, int col) {
    for (int i = 0; i < matrix.length; ++i) {
        matrix[i][col] = 0;
    }
}
void makeRowZero(int[][] matrix, int row) {
    for (int j = 0; j < matrix[0].length; ++j) {
        matrix[row][j] = 0;
    }
}
```



[74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)

```java
public boolean searchMatrix(int[][] matrix, int target) {
    if(matrix.length == 0 || matrix == null) return false;
    int row = matrix.length, col = matrix[0].length; 
    int begin = 0, end = row * col - 1;
    
    while(begin <= end){
        int mid = (begin + end) / 2;
        int val = matrix[mid / col][mid % col];
        if(val == target)
            return true;
        else if(val < target)
            begin = mid + 1;
        else
            end = mid - 1;
    }
    return false;
}
```



[75. Sort Colors](https://leetcode.com/problems/sort-colors/)

```java
public void sortColors(int[] nums) {
    int low = 0, high = nums.length - 1;
    for(int mid = 0; mid <= high; mid++){
        if(nums[mid] == 0){
            swap(nums, mid, low);
            low++;
        }
        else if(nums[mid] == 2){
            swap(nums, mid, high);
            mid--;
            high--;
        }
    }
}

private void swap(int[] nums, int i, int j){
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}
```



[76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

```java
public String minWindow(String s, String t) {
    int[] map = new int[128];
    for(char c : t.toCharArray())
        map[c]++;
    
    int begin = 0, end = 0, head = 0, count = t.length(), d = Integer.MAX_VALUE;
    while(end < s.length()){
        if(map[s.charAt(end++)]-- > 0) count--;
        while(count == 0){
            if(end - begin < d){
                d = end - begin;
                head = begin;
            }
            if(map[s.charAt(begin++)]++ == 0) count++;
        }
    }
    return d == Integer.MAX_VALUE ? "" : s.substring(head, head+d);
}
```



[79. Word Search](https://leetcode.com/problems/word-search/)

```java
public boolean exist(char[][] board, String word) {
    for(int i = 0; i < board.length; i++){
        for(int j = 0; j < board[i].length; j++){
            if(find(board, word.toCharArray(), i, j, 0))
                return true;
        }
    }
    return false;
}

private boolean find(char[][] board, char[] word, int i, int j, int len){
    if(len == word.length)
        return true;
    if(i < 0 || j < 0 || i == board.length || j == board[i].length || board[i][j] != word[len])
        return false;
    
    board[i][j] ^= 256;
    boolean res = find(board, word, i, j + 1, len + 1)
        || find(board, word, i, j - 1, len + 1)
        || find(board, word, i + 1, j, len + 1)
        || find(board, word, i - 1, j, len + 1);
    board[i][j] ^= 256;
    
    return res;
}
```



[80. Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)

```java
public int removeDuplicates(int[] nums) {
    int i = 0;
    for(int n : nums){
        if(i < 2 || n > nums[i - 2])
            nums[i++] = n;
    }
    
    return i;
}
```

[83. Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)

```java
public ListNode deleteDuplicates(ListNode head) {
    if(head == null || head.next == null) return head;
    
    ListNode cur = head;
    
    while(cur != null){
        while(cur != null && cur.next != null && cur.val == cur.next.val)
            cur.next = cur.next.next;
        cur = cur.next;
    }
    
    return head;
}
```



[82. Remove Duplicates from Sorted List II](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)

```java
public ListNode deleteDuplicates(ListNode head) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    
    ListNode pre = dummy, cur = dummy.next;
    while(cur != null){
        while(cur != null && cur.next != null && cur.val == cur.next.val)
            cur = cur.next;
        if(pre.next == cur)
            pre = pre.next;
        else
            pre.next = cur.next;
        cur = cur.next;
    }
    
    return dummy.next;
}
```



[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

```java
public int largestRectangleArea(int[] heights) {
    int len = heights.length;
    Stack<Integer> s = new Stack<>();
    int maxArea = 0;
    for(int i = 0; i <= len; i++){
        int h = (i == len ? 0 : heights[i]);
        if(s.isEmpty() || h >= heights[s.peek()]){
            s.push(i);
        }else{
            int tp = s.pop();
            maxArea = Math.max(maxArea, heights[tp] * (s.isEmpty() ? i : i-1-s.peek()));
            i--;
        }
    }
    return maxArea;
}
```



[86. Partition List](https://leetcode.com/problems/partition-list/)

```java
public ListNode partition(ListNode head, int x) {
    ListNode smallHead = new ListNode(0), bigHead = new ListNode(0);
    ListNode small = smallHead, big = bigHead;
    
    while(head != null){
        if(head.val < x){
            small = small.next = head;
        }else{
            big = big.next = head;
        }
        head = head.next;
    }
    small.next = bigHead.next;
    big.next = null;
    
    return smallHead.next;
}
```



[88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int a = m - 1;
    int b = n - 1;
    int i = m + n - 1;
    while(b >= 0 && a >= 0){
        if(nums1[a] > nums2[b])
            nums1[i--] = nums1[a--];
        else
            nums1[i--] = nums2[b--];
    }
    while(b >= 0){
        nums1[i--] = nums2[b--];
    }
}
```



[91. Decode Ways](https://leetcode.com/problems/decode-ways/)

```java
    public int numDecodings(String s) {
        if(s == null || s.length() == 0)
            return 0;
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for(int i = 2; i <= s.length(); i++){
            int first = Integer.valueOf(s.substring(i-1, i));
            int second = Integer.valueOf(s.substring(i-2, i));
            if(first >= 1 && first <= 9)
                dp[i] += dp[i-1];
            if(second >= 10 && second <= 26)
                dp[i] += dp[i-2];
        }
        return dp[s.length()];
    }
```



[92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

```java
public ListNode reverseBetween(ListNode head, int m, int n) {
    if(head == null) return null;
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode pre = dummy;
    for(int i = 0; i < m-1; i++)
        pre = pre.next;
    
    ListNode start = pre.next;
    ListNode then = start.next;
    
    for(int i=0; i < n - m; i++)
    {
        start.next = then.next;
        then.next = pre.next;
        pre.next = then;
        then = start.next;
    }
    
    return dummy.next;

}
```



[93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)

```java
public List<String> restoreIpAddresses(String s) {
    List<String> res = new ArrayList<>();
    if(s.length() > 12) return res;

    dfs(res, s, "", 0, 0);
    return res;
}
private void dfs(List<String> res, String s, String restored, int pos, int sec){
    if(s.length() - pos > 3 * (4 - sec)) return;
    if(sec == 4 && pos == s.length()){
        res.add(restored);
        return;
    }
    for(int i = 1; i < 4; i++){
        if(pos + i > s.length()) 
            return;
        String part = s.substring(pos, pos+i);
        if(part.length() > 1 && part.startsWith("0") || Integer.parseInt(part) > 255) 
            continue;
        dfs(res, s, sec == 0 ? part : restored + "." + part, pos + i, sec + 1);
    }
}
```



[94. Binary Tree Inorder Traversal]()

```java
    public List<Integer> inorderTraversal(TreeNode root) {
//         List<Integer> list = new ArrayList<Integer>();
//         Stack<TreeNode> stack = new Stack<TreeNode>();
//         if(root ==null){
//             return list ;
//         }
//         TreeNode cur = root ;
// //      注意和先序遍历不一样，这里不能先把根结点先push，要加到while里面判断，
//         while(!stack.isEmpty() || cur != null){
//             while(cur != null){
//                 stack.push(cur) ;
//                 cur = cur.left ;
//             }
//             if(!stack.isEmpty()){
//                 cur = stack.pop() ;
//                 list.add(cur.val) ;
//                 cur = cur.right ;
//             }
//         }
//         return list ;
        
        List<Integer> list = new ArrayList<Integer>();
        recur(list , root) ;
        
        return list ;
    }
    private void recur(List<Integer> list , TreeNode root){
        if(root == null){
            return ;
        }
        recur(list , root.left);
        list.add(root.val);
        recur(list , root.right);
    }
```



[98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

```java
public boolean isValidBST(TreeNode root) {
    return isValidBST(root, Long.MAX_VALUE, Long.MIN_VALUE);
}

private boolean isValidBST(TreeNode root, long max, long min) {
    if (root == null) return true; 
    if (root.val >= max || root.val <= min) return false; 
    
    return isValidBST(root.left, root.val, min) && isValidBST(root.right, max, root.val);
}
```



[100. Same Tree](https://leetcode.com/problems/same-tree/)

```java
public boolean isSameTree(TreeNode p, TreeNode q) {
    if(p == null && q == null)
        return true;
    if(p == null || q == null)
        return false;
    if(p.val == q.val)
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    return false;
}
```



[101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

```java
public boolean isSymmetric(TreeNode root) {
    if(root == null)
        return true;
    return isMirror(root.left, root.right);
}
private boolean isMirror(TreeNode p, TreeNode q){
    if(p == null && q == null) return true;
    if(p == null || q == null) return false;
    return (p.val == q.val) && isMirror(p.left, q.right) && isMirror(p.right, q.left);
}
```



[102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

```java
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        dfs(res, root, 0);
        return res;
    }
    private void dfs(List<List<Integer>> res, TreeNode node, int level){
        if(node == null) return;
        if(level == res.size()) res.add(new ArrayList<Integer>());
        res.get(level).add(node.val);
        dfs(res, node.left, level+1);
        dfs(res, node.right, level+1);
    }
    
//     public List<List<Integer>> levelOrder(TreeNode root) {
//         List<List<Integer>> res = new ArrayList<>();
//         if(root == null) return res;
//         Queue<TreeNode> q = new LinkedList<>();
//         q.add(root);
        
//         while(!q.isEmpty()){
//             int size = q.size();
//             List<Integer> level = new ArrayList<>();
//             for(int i = 0; i < size; i++){
//                 TreeNode node = q.poll();
//                 level.add(node.val);
//                 if(node.left != null){
//                     q.add(node.left);
//                 }
//                 if(node.right != null){
//                     q.add(node.right);
//                 }
//             }
//             res.add(level);
//         }
//         return res;
//     }
```



[103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

```java
public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    if(root == null) return res;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    boolean zigzag = false;
    while(!q.isEmpty()){
        int size = q.size();
        List<Integer> list = new ArrayList<>();

        for(int i = 0; i < size; i++){
            TreeNode node = q.poll();
            if(zigzag)
                list.add(0, node.val);
            else
                list.add(node.val);
            if(node.left != null) q.offer(node.left);
            if(node.right != null) q.offer(node.right);
        }
        res.add(list);
        zigzag = !zigzag;
    }
    return res;
}
```



[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

```java
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
```



[111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

```java
public int minDepth(TreeNode root) {
    if (root == null) return 0;
    int left = minDepth(root.left);
    int right = minDepth(root.right);
    if (left == 0) {
        return right + 1;
    }
    if (right == 0) {
        return left + 1;
    }
    return Math.min(left, right) + 1;
}
```





[108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

```java
public TreeNode sortedArrayToBST(int[] nums) {
    return toBST(nums, 0, nums.length - 1);
}
private TreeNode toBST(int[] nums, int s, int e){
    if(s > e) return null;
    int m = s + (e - s) / 2;
    TreeNode root = new TreeNode(nums[m]);
    root.left = toBST(nums, s, m - 1);
    root.right = toBST(nums, m + 1, e);
    return root;
}
```



[109. Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/)

```java
public TreeNode sortedListToBST(ListNode head) {
    if(head == null) return null;
    return helper(head, null);
}
private TreeNode helper(ListNode head, ListNode tail){
    if(head == tail) return null;
    ListNode slow = head, fast = head;
    while(fast != tail && fast.next != tail){
        fast = fast.next.next;
        slow = slow.next;
    }
    TreeNode treeHead = new TreeNode(slow.val);
    treeHead.left = helper(head, slow);
    treeHead.right = helper(slow.next, tail);
    return treeHead;
}
```



[110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

```java
public boolean isBalanced(TreeNode root) {
    return dfs(root) != -1;
}
private int dfs(TreeNode node){
    if(node == null) return 0;
    
    int left = dfs(node.left);
    if(left == -1) return -1;
    
    int right = dfs(node.right);
    if(right == -1) return -1;
    
    if(Math.abs(left - right) >= 2) return -1;
    
    return Math.max(left, right) + 1;
}
```



[112. Path Sum](https://leetcode.com/problems/path-sum/)

```java
public boolean hasPathSum(TreeNode root, int sum) {
    if(root == null) return false;
    if(root.left == null && root.right == null && sum - root.val == 0) return true;
    
    return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
}
```



[113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)

```java
public List<List<Integer>> pathSum(TreeNode root, int sum) {
    List<List<Integer>> res = new ArrayList<>();
    dfs(root, sum, new ArrayList<>(), res);
    return res;
}
private void dfs(TreeNode root, int sum, List<Integer> list, List<List<Integer>> res){
    if(root == null) return;
    list.add(root.val);
    if(root.left == null && root.right == null && sum == root.val){
        res.add(new ArrayList<>(list));
    }
    dfs(root.left, sum - root.val, list, res);
    dfs(root.right, sum - root.val, list, res);
    list.remove(list.size() - 1);
}
```



[118. Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/)

```java
public List<List<Integer>> generate(int numRows) {
    List<List<Integer>> allrows = new ArrayList<>();
    List<Integer> row = new ArrayList<>();
    for(int i = 0; i < numRows; i++){
        row.add(0, 1);
        for(int j = 1; j < row.size() - 1; j++){
            row.set(j, row.get(j) + row.get(j+1));
        }
        allrows.add(new ArrayList<Integer>(row));
    }
    return allrows;
}
```

[119. Pascal's Triangle II](https://leetcode.com/problems/pascals-triangle-ii/)

```java
public List<Integer> getRow(int rowIndex) {
    List<Integer> res = new ArrayList<>();
    for(int i = 0; i <= rowIndex; i++){
        res.add(0, 1);
        for(int j = 1; j < res.size() - 1; j++)
            res.set(j, res.get(j) + res.get(j+1));
    }
    return res;
}
```



[120. Triangle](https://leetcode.com/problems/triangle/)

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int[] dp = new int[triangle.size() + 1];
    for(int i = triangle.size() - 1; i >= 0; i--){
        for(int j = 0; j < triangle.get(i).size(); j++){
            dp[j] = Math.min(dp[j], dp[j+1]) + triangle.get(i).get(j);
        }
    }
    return dp[0];
}
```



[114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

```java
    public void flatten(TreeNode root) {
        flatten(root, null);
    }
    private TreeNode flatten(TreeNode root, TreeNode pre){
        if(root == null) return pre;
        pre = flatten(root.right, pre);
        pre = flatten(root.left, pre);
        root.right = pre;
        root.left = null;
        return root;
    }
```



[116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

```java
    public void connect(TreeLinkNode root) {
        if(root == null)
            return;
        TreeLinkNode pre = root;
        TreeLinkNode cur = null;
        while(pre.left != null){
            cur = pre;
            while(cur != null){
                cur.left.next = cur.right;
                if(cur.next != null)
                    cur.right.next = cur.next.left;
                cur = cur.next;
            }
            pre = pre.left;
        }
    }
```

[117. Populating Next Right Pointers in Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)

```java
    public void connect(TreeLinkNode root) {
        
        while(root != null){
            TreeLinkNode tempChild = new TreeLinkNode(0);
            TreeLinkNode currentChild = tempChild;
            while(root!=null){
                if(root.left != null) { 
                    currentChild.next = root.left; 
                    currentChild = currentChild.next;
                }
                if(root.right != null) { 
                    currentChild.next = root.right; 
                    currentChild = currentChild.next;
                }
                root = root.next;
            }
            root = tempChild.next;
        }
    }
```



[105. Construct Binary Tree from Preorder and Inorder Traversal]()

```java
    Map<Integer, Integer> map = new HashMap<>();
    
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return helper(preorder, inorder, 0, preorder.length - 1, 0, inorder.length);
    }
    
    private TreeNode helper(int[] preorder, int[] inorder, int pStart, int pEnd, int iStart, int iEnd) {
        if (pStart > pEnd || iStart > iEnd) return null;
        int pivot = map.get(preorder[pStart]);
        int leftLen = pivot - iStart;
        TreeNode node = new TreeNode(preorder[pStart]);
        node.left = helper(preorder, inorder, pStart + 1, pStart + leftLen, iStart, pivot - 1);
        node.right = helper(preorder, inorder, pStart + leftLen + 1, pEnd, pivot + 1, iEnd);
        return node;
    }
```



[121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

```java
    public int maxProfit(int[] prices) {
        int curSum = 0, maxPro = 0;
        for(int i = 1; i < prices.length; i++){
            curSum += prices[i] - prices[i-1];
            curSum = Math.max(0, curSum);
            maxPro = Math.max(curSum, maxPro);
        }
        return maxPro;
    }
```



[122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

```java
    public int maxProfit(int[] prices) {
        int profit = 0;
        for(int i = 1; i < prices.length; i++){
            if(prices[i] > prices[i-1])
                profit += prices[i] - prices[i-1];
        }
        return profit;
    }
```



[124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

```java
    private int maxSum = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        tranverse(root);
        return maxSum;
    }
    
    private int tranverse(TreeNode node){
        if(node == null) return 0;
        
        int left = Math.max(0, tranverse(node.left));
        int right = Math.max(0, tranverse(node.right));
        
        // if doesn't include negative number
        // int left = tranverse(node.left);
        // int right = tranverse(node.right);
        
        maxSum = Math.max(left + right + node.val, maxSum);
        
        return Math.max(left, right) + node.val;
    }
```

[125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

```java
    public boolean isPalindrome(String s) {
        if(s.length() == 0 || s == null) return true;
        char[] c = s.toCharArray();
        int i = 0, j = s.length() - 1;
        while(i < j){
            if(!Character.isLetterOrDigit(c[i])) i++;
            else if(!Character.isLetterOrDigit(c[j])) j--;
            else if(Character.toLowerCase(c[i++]) != Character.toLowerCase(c[j--]))
                return false;
        }
        return true;
    }
```

[128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

```java
    public int longestConsecutive(int[] nums){
        Set<Integer> set = new HashSet<>();
        int res = 0;
        for(int n : nums) set.add(n);
        
        for(int n : nums){
            int localLen = 1;
            if(!set.contains(n-1)){
                int m = n+1;
                while(set.contains(m)){
                    localLen++;
                    m++;
                }
                if(res < localLen) res = localLen;
            }
        }
        return res;
    }
//     public int longestConsecutive(int[] nums) {
//         if(nums == null || nums.length == 0) return 0;
//         int res = 0;
//         Map<Integer, Integer> map = new HashMap<>();
//         for(int n : nums){
//             if(map.containsKey(n)) continue;
            
//             int left = map.getOrDefault(n-1, 0);
//             int right = map.getOrDefault(n+1, 0);
//             int sum = left + right + 1;
//             res = Math.max(res, sum);
            
//             if(left > 0) map.put(n-left, sum);
//             if(right > 0) map.put(n+right, sum);
//             map.put(n, sum);
//         }
//         return res;
//     }
```

[133. Clone Graph](https://leetcode.com/problems/clone-graph/)

```java
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node){
        Map<Integer, UndirectedGraphNode> map = new HashMap<>();
        return dfs(node, map);
    }

    private UndirectedGraphNode dfs(UndirectedGraphNode node, Map<Integer, UndirectedGraphNode> map){
        if(node == null) 
            return null;
        if(map.containsKey(node.label)) 
            return map.get(node.label);
        UndirectedGraphNode cloneNode = new UndirectedGraphNode(node.label);
        map.put(cloneNode.label, cloneNode);
        for(UndirectedGraphNode n : node.neighbors){
            cloneNode.neighbors.add(dfs(n, map));
        }
        return cloneNode;
    }
```

[134. Gas Station](https://leetcode.com/problems/gas-station/)

```java
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int sumGas = 0, sumCost = 0;
        int tank = 0;
        int start = 0;
        for(int i = 0; i < gas.length; i++){
            sumGas += gas[i];
            sumCost += cost[i];
            tank += gas[i] - cost[i];
            if(tank < 0){
                start = i + 1;
                tank = 0;
            }
        }
        if(sumGas < sumCost)
            return -1;
        else 
            return start;
    }
```

[138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

```java
    public RandomListNode copyRandomList(RandomListNode head) {
        if(head == null) return null;
        
        RandomListNode cur = head;
        while(cur != null){
            RandomListNode newNode = new RandomListNode(cur.label);
            newNode.next = cur.next;
            cur.next = newNode;
            cur = cur.next.next;
        }
        
        cur = head;
        while(cur != null){
            if(cur.random != null)
                cur.next.random = cur.random.next;
            cur = cur.next.next;
        }
        
        cur = head;
        RandomListNode copyHead = head.next;
        RandomListNode copyIter = copyHead;
        
        while(copyIter.next != null){
            cur.next = cur.next.next;
            cur = cur.next;
            
            copyIter.next = copyIter.next.next;
            copyIter = copyIter.next;
        }
        cur.next = cur.next.next;
        
        return copyHead;
    }
```

[139. Word Break](https://leetcode.com/problems/word-break/)

```java
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] f = new boolean[s.length() + 1];
        f[0] = true;
        int len = 0;
        for(String word : wordDict)
            len = Math.max(len, word.length());
        
        for(int i = 1; i <= s.length(); i++){
            for(int j = 1; j <= len; j++){
                if(i- j >= 0 && f[i-j] && wordDict.contains(s.substring(i-j, i))){
                    f[i] = true;
                    break;
                }
            }
        }
        
        return f[s.length()];
    }
```

[140. Word Break II](https://leetcode.com/problems/word-break-ii/)

```java
HashMap<String,List<String>> map = new HashMap<String,List<String>>();

public List<String> wordBreak(String s, Set<String> wordDict) {
    List<String> res = new ArrayList<String>();
    if(s == null || s.length() == 0) {
        return res;
    }
    if(map.containsKey(s)) {
        return map.get(s);
    }
    if(wordDict.contains(s)) {
        res.add(s);
    }
    for(int i = 1 ; i < s.length() ; i++) {
        String t = s.substring(i);
        if(wordDict.contains(t)) {
            List<String> temp = wordBreak(s.substring(0 , i) , wordDict);
            if(temp.size() != 0) {
                for(int j = 0 ; j < temp.size() ; j++) {
                    res.add(temp.get(j) + " " + t);
                }
            }
        }
    }
    map.put(s , res);
    return res;
}
```



[141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

```java
    public boolean hasCycle(ListNode head) {
        if(head == null)
            return false;
        ListNode slow = head;
        ListNode fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast)
                return true;
        }
        return false;
    }
```

[142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

```java
    public ListNode detectCycle(ListNode head) {
        if(head == null || head.next == null)
             return null;
        ListNode slow = head, fast = head, res = head;
        
        while(fast.next != null && fast.next.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast){
                while(slow != res){
                    slow = slow.next;
                    res = res.next;
                }
                return res;
            }
        }
        return null;
    }
```

[143. Reorder List](https://leetcode.com/problems/reorder-list/)

```java
    public void reorderList(ListNode head) {
        if(head == null || head.next == null) return;
        
        ListNode prev = null, slow = head, fast = head, l1 = head;
        
        while(fast != null && fast.next != null){
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        prev.next = null;
        
        ListNode l2 = reverse(slow);
        
        merge(l1, l2);
    }
    
    private ListNode reverse(ListNode root){
        ListNode prev = null, cur = root, next = null;
        while(cur != null){
            next = cur.next;
            cur.next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }
    
    private void merge(ListNode l1, ListNode l2){
        while(l1 != null){
            ListNode n1 = l1.next, n2 = l2.next;
            
            l1.next = l2;
            if(n1 == null) break;
            l2.next = n1;
            
            l1 = n1;
            l2 = n2;
        }
    }
```

[146. LRU Cache](https://leetcode.com/problems/lru-cache/)

```java
    class Node {
        int key;
        int val;
        Node next;
        Node prev;
        public Node(){
            this.key = 0;
            this.val = 0;
        }
        public Node(int key, int val){
            this.key = key;
            this.val = val;
        }
    }
    private int capacity;
    private Map<Integer, Node> map;
    Node head = null;
    Node tail = null;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>();
        head = new Node();
        tail = new Node();
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        if(!map.containsKey(key))
            return -1;
        moveToFront(map.get(key));
        return map.get(key).val;
    }
    
    public void put(int key, int value) {
        if(capacity == 0){
            return;
        }
        if(map.containsKey(key)){
            map.get(key).val = value;
            moveToFront(map.get(key));
        }
        else{
            freeSpace();
            Node node = new Node(key, value);
            map.put(key, node);
            addToFront(node);
        }
    }
    
    private void freeSpace(){
        if(map.size() == capacity){
            Node toRemove = head.next;
            map.remove(toRemove.key);
            Node next = toRemove.next;
            head.next = next;
            next.prev = head;
        }
    }
    
    private void moveToFront(Node newNode){
        Node next = newNode.next;
        Node prev = newNode.prev;
        prev.next = next;
        next.prev = prev;
        
        addToFront(newNode);
    }
    
    private void addToFront(Node newNode){
        Node prev = tail.prev;
        prev.next = newNode;
        newNode.prev = prev;
        newNode.next = tail;
        tail.prev = newNode;
    }
```

[148. Sort List](https://leetcode.com/problems/sort-list/)

```java
    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null)
            return head;
        
        ListNode prev = null, slow = head, fast = head;
        
        while(fast != null && fast.next != null){
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        
        prev.next = null; //
        
        ListNode l1 = sortList(head);
        ListNode l2 = sortList(slow);
        
        return merge(l1, l2);
    }
    
    private ListNode merge(ListNode l1, ListNode l2){
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;
        
        while(l1 != null && l2 != null){
            if(l1.val < l2.val){
                tail.next = l1;
                l1 = l1.next;
            }else{
                tail.next = l2;
                l2 = l2.next;
            }
            tail = tail.next;
        }
        
        if(l1 != null)
            tail.next = l1;
        if(l2 != null)
            tail.next = l2;
        
        return dummy.next;
    }
```

[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

```java
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        int a, b;
        for(String s : tokens){
            if(s.equals("+")){
                stack.add(stack.pop() + stack.pop());
            }else if(s.equals("-")){
                b = stack.pop();
                a = stack.pop();
                stack.add(a-b);
            }else if(s.equals("*")){
                stack.add(stack.pop() * stack.pop());
            }else if(s.equals("/")){
                b = stack.pop();
                a = stack.pop();
                stack.add(a/b);                  
            }else
                stack.add(Integer.parseInt(s));
        }
        return stack.pop();
    }
```

[151. Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/)

```java
    public String reverseWords(String s) {
        String[] strs = s.split(" ");
        StringBuilder sb = new StringBuilder();
        for(int i = strs.length - 1; i >= 0 ; i--){
            if(!strs[i].equals(""))
                sb.append(strs[i]).append(" ");
        }
        return sb.length() == 0 ? "" : sb.substring(0, sb.length() - 1);
    }
```

[152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

```java
    public int maxProduct(int[] nums) {
        int res = nums[0];
        int curMax = res, curMin = res;
        for(int i = 1; i < nums.length; i++){
            if(nums[i] < 0){
                int temp = curMax;
                curMax = curMin;
                curMin = temp;
            }
            curMax = Math.max(nums[i], nums[i] * curMax);
            curMin = Math.min(nums[i], nums[i] * curMin);
            
            res = Math.max(curMax, res);
        }
        
        return res;
    }
```

[153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

```java
    public int findMin(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        while(lo < hi){
            int m = lo + (hi - lo) / 2;
            if(nums[m] <= nums[hi])
                hi = m;
            else
                lo = m + 1;
        }
        return nums[lo];
    }
```

[154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

```java
    public int findMin(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        while(lo < hi){
            int m = lo + (hi - lo) / 2;
            if(nums[m] > nums[hi])
                lo = m + 1;
            else if(nums[m] > nums[lo])
                hi = m;
            else
                hi--;
        }
        return nums[lo];
    }
```



[155. Min Stack](https://leetcode.com/problems/min-stack/)

```java
    private Stack<Integer> st;
    private int min;

    /** initialize your data structure here. */
    public MinStack() {
        st = new Stack<Integer>();
        min = Integer.MAX_VALUE;
    }
    
    public void push(int x) {
        if(min >= x){
            st.push(min);
            min = x;
        }
        st.push(x);
    }
    
    public void pop() {
        if(st.size() == 0)
             return;
        if(st.pop() == min)
            min = st.pop();
    }
    
    public int top() {
        return st.peek();
    }
    
    public int getMin() {
        return min;
    }
```

[160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

```java
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null)
            return null;
        
        ListNode a = headA;
        ListNode b = headB;
        
        while(a != b){
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return a;
    }
```

[162. Find Peak Element](https://leetcode.com/problems/find-peak-element/)

```java
    public int findPeakElement(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        while(lo < hi){
            int mid1 = lo + (hi - lo) / 2;
            int mid2 = mid1 + 1;
            if(nums[mid1] < nums[mid2])
                lo = mid2;
            else
                hi = mid1;
        }
        return lo;
    }
```

[165. Compare Version Numbers](https://leetcode.com/problems/compare-version-numbers/)

```java
//     public int compareVersion(String version1, String version2) {
//         String[] v1 = version1.split("\\.");
//         String[] v2 = version2.split("\\.");
        
//         int i = 0;
//         int len = Math.max(v1.length, v2.length);
//         while(i < len){
//             Integer x = i < v1.length ? Integer.parseInt(v1[i]) : 0;
//             Integer y = i < v2.length ? Integer.parseInt(v2[i]) : 0;
//             int res = x.compareTo(y);
//             if(res != 0) return res;
//             i++;
//         }
//         return 0;
//     }
    public int compareVersion(String version1, String version2) {
        int temp1 = 0,temp2 = 0;
        int len1 = version1.length(),len2 = version2.length();
        int i = 0,j = 0;
        while(i<len1 || j<len2) {
            temp1 = 0;
            temp2 = 0;
            while(i<len1 && version1.charAt(i) != '.') {
                temp1 = temp1*10 + version1.charAt(i++)-'0';

            }
            while(j<len2 && version2.charAt(j) != '.') {
                temp2 = temp2*10 + version2.charAt(j++)-'0';
            }
            if(temp1>temp2) return 1;
            else if(temp1<temp2) return -1;
            else {
                i++;
                j++;
            }
        }
        return 0;
    }
```
[166. Fraction to Recurring Decimal](https://leetcode.com/problems/fraction-to-recurring-decimal/)
```java
    public String fractionToDecimal(int numerator, int denominator) {
        if(numerator == 0) return "0";
        
        long num = Math.abs((long)numerator);
        long deno = Math.abs((long)denominator);
        StringBuilder res = new StringBuilder();
        
        res.append((numerator > 0) ^ (denominator > 0) ? "-" : "");
        
        res.append(num / deno);
        num %= deno;
        
        if(num != 0)
            res.append(".");
        
        Map<Long, Integer> map = new HashMap<>();
        map.put(num, res.length());
        while(num > 0){
            num *= 10;
            res.append(num / deno);
            num %= deno;
            
            Integer index = map.get(num);
            if(index != null){
                res.insert(index, "(");
                res.append(")");
                break;
            }else{
                map.put(num, res.length());
            }
        }
        return res.toString();
    }
```
[167. Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

```java
    public int[] twoSum(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while(l < r){
            int m = l + (r - l) / 2;
            if(nums[l] + nums[r] > target)
                r--;
            else if(nums[l] + nums[r] < target)
                l++;
            else
                return new int[]{l+1, r+1};
        }
        return new int[]{0, 0};
    }
```

[168. Excel Sheet Column Title](https://leetcode.com/problems/excel-sheet-column-title/)

```java
    public String convertToTitle(int n) {
        StringBuilder sb = new StringBuilder();
        while(n != 0){
            int i = (n-1) % 26;
            sb.append((char)(i+'A'));
            n = (n-1) / 26;
        }
        return sb.reverse().toString();
    }
```

[169. Majority Element](https://leetcode.com/problems/majority-element/)

```java
    public int majorityElement(int[] nums) {
        int major = nums[0];
        int count = 1;
        for(int i = 1; i < nums.length; i++){
            if(count == 0){
                major = nums[i];
                count++;
            }else if(major == nums[i])
                count++;
            else
                count--;
        }
        return major;
    }
```

[171. Excel Sheet Column Number](https://leetcode.com/problems/excel-sheet-column-number/)

```java
    public int titleToNumber(String s) {
        int res = 0;
        for(int i = 0; i < s.length(); i++){
            res = res * 26 + (s.charAt(i) - 'A' + 1);
        }
        return res;
    }
```

[172. Factorial Trailing Zeroes](https://leetcode.com/problems/factorial-trailing-zeroes/)

```java
    public int trailingZeroes(int n) {
        return n == 0 ? 0 : n/5 + trailingZeroes(n / 5);
    }
```

[173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)

```java
    private Stack<TreeNode> stack = new Stack<>();
    
    public BSTIterator(TreeNode root) {
        pushAllLeft(root);
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stack.isEmpty();
    }

    /** @return the next smallest number */
    public int next() {
        TreeNode node = stack.pop();
        pushAllLeft(node.right);
        return node.val;
    }
    
    private void pushAllLeft(TreeNode node){
        while(node != null){
            stack.push(node);
            node = node.left;
        }
    }
```

[186. Reverse Words in a String II](https://leetcode.com/problems/reverse-words-in-a-string-ii/)

```java
    public void reverseWords(char[] str) {
        reverse(str, 0, str.length - 1);
        int start = 0;
        for(int i = 0; i < str.length; i++){
            if(str[i] == ' '){
                reverse(str, start, i-1);
                start = i+1;
            }
        }
        reverse(str, start, str.length - 1);
    }
    private void reverse(char[] str, int i , int j){
        while(i < j){
            char temp = str[i];
            str[i++] = str[j];
            str[j--] = temp;
        }
    }
```

[189. Rotate Array](https://leetcode.com/problems/rotate-array/)

```java
    public void rotate(int[] nums, int k) {
    k %= nums.length;
    reverse(nums, 0, nums.length - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, nums.length - 1);
}

public void reverse(int[] nums, int i, int j) {
    while (i < j) {
        int temp = nums[i];
        nums[i++] = nums[j];
        nums[j--] = temp;
    }
}
```

[191. Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)

```java
    public int hammingWeight(int n) {
        int count = 0;
        while(n != 0){
            n = n & (n - 1);
            count++;
        }
        return count;
    }
```

[199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)

```java
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        dfs(root, res, 0);
        return res;
    }
    private void dfs(TreeNode node, List<Integer> res, int depth){
        if(node == null) return;
        if(depth == res.size()) res.add(node.val);
        dfs(node.right, res, depth+1);
        dfs(node.left, res, depth+1);
    }
```

[200. Number of Islands](https://leetcode.com/problems/number-of-islands/)

```java
public int numIslands(char[][] grid){
    if(grid.length == 0 || grid[0].length == 0) return 0;
    int m = grid.length, n = grid[0].length;
    int res = 0;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(grid[i][j] == '1'){
                dfs(grid, i, j);
                res++;
            }
        }
    }
    return res;
}
private void dfs(char[][] grid, int i, int j){
    if(i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == '0') return;
    grid[i][j] = '0';
    dfs(grid, i+1, j);
    dfs(grid, i-1, j);
    dfs(grid, i, j+1);
    dfs(grid, i, j-1);
}
```

[204. Count Primes](https://leetcode.com/problems/count-primes/)

```java
    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n];
        int count = 0;
        for(int i = 2; i < n; i++){
            if(notPrime[i] == false){
                count++;
                for(int j = 2; j*i < n; j++){
                    notPrime[j*i] = true;
                }
            }
        }
        return count;
    }
```

[206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

```java
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        while(head != null){
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }
```

[208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

```java
class TrieNode {
    boolean isWord;
    TrieNode[] children;
    public TrieNode(){
        children = new TrieNode[26];
        isWord = false;
    }
}
private TrieNode root;

/** Initialize your data structure here. */
public Trie() {
    root = new TrieNode();
}

/** Inserts a word into the trie. */
public void insert(String word) {
    if(word == null) return;
    TrieNode node = root;
    for(char c : word.toCharArray()){
        if(node.children[c-'a'] == null)
            node.children[c-'a'] = new TrieNode();
        node = node.children[c-'a'];
    }
    node.isWord = true;
}

/** Returns if the word is in the trie. */
public boolean search(String word) {
    if(word == null) return false;
    TrieNode node = root;
    for(char c : word.toCharArray()){
        if(node.children[c-'a'] == null)
            return false;
        node = node.children[c-'a'];
    }
    return node.isWord;
}

/** Returns if there is any word in the trie that starts with the given prefix. */
public boolean startsWith(String prefix) {
    if(prefix == null) return false;
    TrieNode node = root;
    for(char c : prefix.toCharArray()){
        if(node.children[c-'a'] == null)
            return false;
        node = node.children[c-'a'];
    }
    return true;
}
```

[209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)

```java
    public int minSubArrayLen(int s, int[] nums) {
        if(nums== null || nums.length == 0)
            return 0;
        
        int i = 0, j = 0, sum = 0, min = Integer.MAX_VALUE;
        
        while(j < nums.length){
            sum += nums[j++];
            
            while(sum >= s){
                min = Math.min(min, j - i);
                sum -= nums[i++];
            }
        }
        
        return min == Integer.MAX_VALUE ? 0 : min;
    }
```

[215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

```java
    public int findKthLargest(int[] nums, int k) {
        int left = 0, right = nums.length - 1, target = nums.length - k;
        while(left <= right){
            int pivot = partition(nums, left, right);
            if(target < pivot)
                right = pivot - 1;
            else if(target > pivot)
                left = pivot + 1;
            else 
                return nums[target];
        }
        return nums[target];
    }
    private int partition(int[] nums, int left, int right){
        int pivot = left;
        while(left <= right){
            while(left <= right && nums[left] <= nums[pivot])
                left++;
            while(left <= right && nums[right] > nums[pivot])
                right--;
            if(left > right)
                break;
            swap(nums, left, right);
        }
        swap(nums, right, pivot);
        
        return right;
    }
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
```

[224. Basic Calculator](https://leetcode.com/problems/basic-calculator/)

```java
    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        int num = 0, sum = 0, sign = 1;
        for(int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            if(Character.isDigit(c)){
                num = num * 10 + c - '0';
            }
            if(c == '+' || c == '-' || i == s.length() - 1){
                sum += num * sign;
                sign = c == '+' ? 1 : -1;
                num = 0;
            }   
            if(c == '('){
                stack.push(sum);
                stack.push(sign);
                sum = 0;
                sign = 1;
            }
            if(c == ')'){
                sum += num * sign;
                num = 0;
                sum = sum * stack.pop() + stack.pop();
                //first stack.pop() is the sign before the parenthesis
                //second stack.pop() is the result calculated before the parenthesis
            }
        }
        return sum;
    }
```

[227. Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/)

```java
    public int calculate(String s) {
        if(s == null || s.length() == 0) return 0;
        Stack<Integer> stack = new Stack<>();
        char sign = '+';
        int num = 0, res = 0;
        for(int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            if(c >= '0' && c <= '9')
                num = num * 10 + c - '0';
            if("+-*/".indexOf(c) >= 0 || i == s.length() - 1){
                if("*/".indexOf(sign) >= 0){
                    res -= stack.peek();
                }
                switch(sign){
                    case '+': stack.push(num); break;
                    case '-': stack.push(-num); break;
                    case '*': stack.push(stack.pop() * num); break;
                    case '/': stack.push(stack.pop() / num); break;
                }
                num = 0;
                sign = c;
                res += stack.peek();
            }
        }
        return res;
    }
```

[225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)

```java
class MyStack {
    Queue<Integer> queue;
    
    public MyStack(){
        this.queue=new LinkedList<Integer>();
    }
    
    public void push(int x) {
       queue.add(x);
       for(int i=0;i<queue.size()-1;i++){
           queue.add(queue.poll());
       }
    }

    public void pop() {
        queue.poll();
    }

    public int top(){
        return queue.peek();
    }

    public boolean empty() {
        return queue.isEmpty();
    }
}
```



[229. Majority Element II](https://leetcode.com/problems/majority-element-ii/)

```java
    public List<Integer> majorityElement(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if(nums.length == 0 || nums == null)
            return res;
        int counter1 = 0, counter2 = 0;
        int num1 = nums[0]; 
        int num2 = nums[0];
        for(int num : nums){
            if(num == num1)
                counter1++;
            else if(num == num2)
                counter2++;
            else if(counter1 == 0){
                num1 = num;
                counter1++;
            }else if(counter2 == 0){
                num2 = num;
                counter2++;
            }else{
                counter1--;
                counter2--;
            }
        }
        counter1 = 0;
        counter2 = 0;
        for(int num : nums){
            if(num == num1)
                counter1++;
            else if(num == num2)
                counter2++;
        }
        if(counter1 > nums.length/3)
            res.add(num1);
        if(counter2 > nums.length/3)
            res.add(num2);
        
        return res;
    }
```

[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

```java
    private int count = 0;
    private int val = 0;
    
    public int kthSmallest(TreeNode root, int k) {
        inorder(root, k);
        return val;
    }
    
    private void inorder(TreeNode root, int k){
        if(root == null) return;
        inorder(root.left, k);
        if(k == ++count){
            val = root.val;
            return;
        }
        inorder(root.right, k);
    }
```

[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)

```java
    Stack<Integer> input = new Stack();
    Stack<Integer> output = new Stack();

    /** Initialize your data structure here. */
    public MyQueue() {
        output = new Stack<>();
        input = new Stack<>();
    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
        input.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        peek();
        return output.pop();
    }
    
    /** Get the front element. */
    public int peek() {
        if(output.isEmpty()){
            while(!input.isEmpty())
                output.push(input.pop());
        }
        return output.peek();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
        return input.isEmpty() && output.isEmpty();
    }
```

[234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

```java
    public boolean isPalindrome(ListNode head) {
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        if(fast != null)
            slow = slow.next;
        
        slow = reverse(slow);
        fast = head;
        
        while(slow != null){
            if(slow.val != fast.val)
                return false;
            fast = fast.next;
            slow = slow.next;
        }
        return true;
    }
    
    private ListNode reverse(ListNode head){
        ListNode prev = null;
        while(head != null){
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }
```

[235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```java
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        while((root.val - p.val) * (root.val - q.val) > 0)
            root = p.val < root.val ? root.left : root.right; 
        return root;
    }
```

[236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```java
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || p == root || q == root) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left != null && right != null) return root;
        return left == null ? right : left;
    }
```

[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)

```java
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
```

[238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

```java
    public int[] productExceptSelf(int[] nums) {
        if(nums == null || nums.length == 0) return null;
        int[] res = new int[nums.length];
        int n = nums.length;
        res[0] = 1;
        for(int i = 1; i < n; i++){
            res[i] = res[i-1] * nums[i-1];
        }
        
        int right = 1;
        for(int i = n-1; i >= 0; i--){
            res[i] *= right;
            right *= nums[i];
        }
        
        return res;
    }
```

[240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)

```java
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return false;
        int col = matrix[0].length - 1, row = 0;
        
        while(col >= 0 && row < matrix.length){
            if(target == matrix[row][col])
                return true;
            else if(target < matrix[row][col])
                col--;
            else if(target > matrix[row][col])
                row++;
        }
        return false;
    }
```

[242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)

```java
    public boolean isAnagram(String s, String t) {
        if(s.length() != t.length())
            return false;
        
        int[] count = new int[26];
        for(int i = 0; i < s.length(); i++){
            count[s.charAt(i) - 'a']++;
            count[t.charAt(i) - 'a']--;
        }
        
        for(int i : count){
            if(i != 0)
                return false;
        }
        return true;
    }
```

[253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

```java
    public int minMeetingRooms(Interval[] intervals) {
        int[] starts = new int[intervals.length];
        int[] ends = new int[intervals.length];
        
        for(int i = 0; i < intervals.length; i++){
            starts[i] = intervals[i].start;
            ends[i] = intervals[i].end;
        }
        
        Arrays.sort(starts);
        Arrays.sort(ends);
        
        int room = 0, endIdx = 0;
        for(int i = 0; i < intervals.length; i++){
            if(starts[i] < ends[endIdx])
                room++;
            else 
                endIdx++;
        }
        return room;
    }
```

[266. Palindrome Permutation](https://leetcode.com/problems/palindrome-permutation/)

```java
    public boolean canPermutePalindrome(String s) {
        Set<Character> set = new HashSet<>();
        for(char c : s.toCharArray()){
            if(!set.add(c))
                set.remove(c);
        }
        return set.size() <= 1;
    }
```

[273. Integer to English Words](https://leetcode.com/problems/integer-to-english-words/)

```java
    private final String[] LESS_THAN_20 = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    private final String[] TENS = {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    private final String[] THOUSANDS = {"", "Thousand", "Million", "Billion"};

    public String numberToWords(int num){
        if(num == 0) return "Zero";
        
        int i = 0;
        String words = "";

        while(num > 0){
            if(num % 1000 != 0){
                words = helper(num % 1000) + THOUSANDS[i] + " " + words;
            }
            num /= 1000;
            i++;
        }
        return words.trim();
    }

    private String helper(int num){
        if(num == 0) return "";
        else if(num < 20) return LESS_THAN_20[num] + " ";
        else if(num < 100) return TENS[num / 10] + " " + helper(num % 10);
        else return LESS_THAN_20[num / 100] + " Hundred " + helper(num % 100);
    }
```

[283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)

```java
    public void moveZeroes(int[] nums) {
        if(nums.length == 0 || nums == null)
            return;
        int i = 0;
        for(int num : nums){
            if(num != 0)
                nums[i++] = num;
        }
        while(i < nums.length){
            nums[i++] = 0;
        }
    }
```

[285. Inorder Successor in BST](https://leetcode.com/problems/inorder-successor-in-bst/)

```java
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        TreeNode succ = null;
        while(root != null){
            if(p.val < root.val){
                succ = root;
                root = root.left;
            }
            else{
                root = root.right;
            }
        }
        return succ;
    }
```

[286. Walls and Gates](https://leetcode.com/problems/walls-and-gates/)

```java
private int m = -1;
private int n = -1;
public void wallsAndGates(int[][] rooms) {
    if (rooms == null || rooms.length == 0 || rooms[0].length == 0) return;
    m = rooms.length;
    n = rooms[0].length;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            if (rooms[i][j] == 0){
                fillRoom (rooms, i, j, 0);
            }
        }
    }
}
public void fillRoom (int[][] rooms, int i, int j , int distance){
    if (i < 0 || j < 0 || i >= m || j >= n || rooms[i][j] < distance) return;
    rooms[i][j] = distance;
    fillRoom(rooms, i + 1, j, distance + 1);
    fillRoom(rooms, i - 1, j, distance + 1);
    fillRoom(rooms, i, j + 1, distance + 1);
    fillRoom(rooms, i, j - 1, distance + 1);
}
```

[287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

```java
    public int findDuplicate(int[] nums) {
        if(nums.length == 0 || nums == null) return -1;
        int slow = nums[0], fast = nums[nums[0]];
        while(fast != slow){
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        fast = 0;
        while(fast != slow){
            fast = nums[fast];
            slow = nums[slow];
        }
        return slow;
    }
```

[289. Game of Life](https://leetcode.com/problems/game-of-life/)

[2nd bit, 1st bit] = [next state, current state]

- 00  dead (next) <- dead (current) 
- 01  dead (next) <- live (current) 
- 10  live (next) <- dead (current) 
- 11  live (next) <- live (current) 
```java
    public void gameOfLife(int[][] board) {
        if(board == null || board.length == 0)
            return;
        
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                int lives = liveNeighbor(board, i, j);
                
                if(board[i][j] == 1 && lives >= 2 && lives <= 3)
                    board[i][j] = 3;
                if(board[i][j] == 0 && lives == 3)
                    board[i][j] = 2;
            }
        }
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                board[i][j] >>= 1;
            }
        }
    }
    
    private int liveNeighbor(int[][] board, int i, int j){
        int lives = 0;
        for(int x = Math.max(0, i - 1); x <= Math.min(i+1, board.length-1); x++){
            for(int y = Math.max(0, j-1); y <= Math.min(j+1, board[0].length-1); y++){
                lives += board[x][y] & 1;
            }
        }
        lives -= board[i][j] & 1;
        return lives;
    }
```

[295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

```java
    PriorityQueue<Integer> minHeap;
    PriorityQueue<Integer> maxHeap;

    public MedianFinder() {
        minHeap = new PriorityQueue<>();
        maxHeap = new PriorityQueue<>((a,b) -> b - a);
    }
    
    public void addNum(int num) {
        minHeap.offer(num);
        maxHeap.offer(minHeap.poll());
        if (minHeap.size() < maxHeap.size())
            minHeap.offer(maxHeap.poll());
    }
    
    public double findMedian() {
        return minHeap.size() > maxHeap.size() ? minHeap.peek() : (maxHeap.peek() + minHeap.peek()) / 2.0;
    }
```

[297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

```java
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder res = new StringBuilder();
        buildString(root, res);
        return res.toString();
    }
    private void buildString(TreeNode root, StringBuilder res){
        if(root == null) res.append("#").append(" ");
        else{
            res.append(root.val).append(" ");
            buildString(root.left, res);
            buildString(root.right, res);
        }
        
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> q = new LinkedList<>();
        q.addAll(Arrays.asList(data.split(" ")));
        return buildTree(q);
    }
    private TreeNode buildTree(Queue<String> q){
        String cur = q.remove();
        if(cur.equals("#")) return null;
        TreeNode root = new TreeNode(Integer.valueOf(cur));
        root.left = buildTree(q);
        root.right = buildTree(q);
        
        return root;
    }
```

[300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

```java
    public int lengthOfLIS(int[] nums) {
        int[] tails = new int[nums.length];
        int size = 0;
        for(int x : nums){
            int i = 0, j = size;
            while(i < j){
                int mid = i+(j-i)/2;
                if(tails[mid] < x)
                    i = mid + 1;
                else
                    j = mid;
            }
            tails[i] = x;
            if(i == size) ++size;
        }
        return size;
    }
```

[314. Binary Tree Vertical Order Traversal](https://leetcode.com/problems/binary-tree-vertical-order-traversal/)

```java
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if(root == null) return res;
        
        Map<Integer, List<Integer>> map = new HashMap<>();
        Queue<Integer> cols = new LinkedList<>();
        Queue<TreeNode> q = new LinkedList<>();
        int min = 0, max = 0;
        
        cols.offer(0);
        q.offer(root);
        
        while(!q.isEmpty()){
            int col = cols.poll();
            TreeNode node = q.poll();
            
            if(!map.containsKey(col))
                map.put(col, new ArrayList<>());
            map.get(col).add(node.val);
            
            if(node.left != null){
                q.offer(node.left);
                cols.add(col - 1);
                min = Math.min(min, col-1);
            }
            if(node.right != null){
                q.offer(node.right);
                cols.add(col+1);
                max = Math.max(max, col+1);
            }
        }
        for(int i = min; i <= max; i++)
            res.add(map.get(i));
        
        return res;
    }
```

[328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/)

```java
    public ListNode oddEvenList(ListNode head) {
        if(head == null)
            return null;
        ListNode odd = head, even = head.next, evenHead = even;
        
        while(even != null && even.next != null){
            odd.next = odd.next.next;
            even.next = even.next.next;
            odd = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        
        return head;
    }
```

[333. Largest BST Subtree](https://leetcode.com/problems/largest-bst-subtree/)

```java
    // return array for each node: 
    //     [0] --> min
    //     [1] --> max
    //     [2] --> largest BST in its subtree(inclusive)
    public int largestBSTSubtree(TreeNode root) {
        int[] res = helper(root);
        return res[2];
    }
    private int[] helper(TreeNode root){
        if(root == null)
            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
        int[] left = helper(root.left);
        int[] right = helper(root.right);
        if(root.val > left[1] && root.val < right[0]){
            return new int[]{Math.min(root.val, left[0]), 
                             Math.max(root.val, right[1]), 
                             left[2] + right[2] + 1};
        }else{
            return new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE, Math.max(left[2], right[2])};
        }
    }
```

[341. Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/)

```java
    Stack<NestedInteger> stack = new Stack<>();
    public NestedIterator(List<NestedInteger> nestedList) {
        for(int i = nestedList.size() - 1; i >= 0; i--)
            stack.push(nestedList.get(i));
    }

    @Override
    public Integer next() {
        return stack.pop().getInteger(); // if hasNext() return true, then next() can pop this value directly.
    }

    @Override
    public boolean hasNext() {
        while(!stack.isEmpty()){
            NestedInteger cur = stack.peek();    
            if(cur.isInteger())
                return true;
            stack.pop();
            for(int i = cur.getList().size() - 1; i >= 0; i--)
                stack.push(cur.getList().get(i));
        }
        return false;
    }
```

[344. Reverse String](https://leetcode.com/problems/reverse-string/)

```java
    public String reverseString(String s) {
        char[] word = s.toCharArray();
        int i = 0;
        int j = s.length() - 1;
        while (i < j) {
            char temp = word[i];
            word[i] = word[j];
            word[j] = temp;
            i++;
            j--;
        }
        return new String(word);
    }
```

[346. Moving Average from Data Stream](https://leetcode.com/problems/moving-average-from-data-stream/)

```java
    private int size;
    private int runningSum;
    private List<Integer> values = new ArrayList<>();

    /** Initialize your data structure here. */
    public MovingAverage(int size) {
        this.size = size;
    }
    
    public double next(int val) {
        runningSum += val;
        values.add(val);
        
        if (values.size() > size) {
            runningSum -= values.get(0);
            values.remove(0);
        }
        
        return runningSum / ((double) values.size());
    }
```

[347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
Bucket sort. Use a list of `buckets` and a `map` to record the frequency of each number. `bucket[]` : index is the frequency of element, values are numbers. 


```java
    public List<Integer> topKFrequent1(int[] nums, int k) {
        List<Integer>[] bucket = new List[nums.length + 1];
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> res = new ArrayList<>();

        for(int n : nums)
            map.put(n, map.getOrDefault(n, 0) + 1);
        
        for(int num : map.keySet()){
            int freq = map.get(num);
            if(bucket[freq] == null)
                bucket[freq] = new ArrayList<>();
            bucket[freq].add(num);
        }
        
        for(int i = bucket.length - 1; i >= 0 && res.size() < k; i--){
            if(bucket[i] != null)
                res.addAll(bucket[i]);
        }
        return res.subList(0, k);
    }
    public List<Integer> topKFrequent2(int[] nums, int k) {
        List<Integer> res=new ArrayList<>();
        if(nums==null||nums.length==0) return res;
        Map<Integer,Integer> map=new HashMap<>();
        for(int num:nums){                  //count frequency
            map.put(num,map.getOrDefault(num,0)+1);
        }
        //maxHeap, Comparator:map.value 
        PriorityQueue<Integer> pq=new PriorityQueue<>((a,b)->map.get(b)-map.get(a));
        for(int key:map.keySet()){
            pq.offer(key);
            if(pq.size()>map.size()-k)
                res.add(pq.poll()); //k in res,(n-k)in pq
        } 
        return res;
    }
```

[348. Design Tic-Tac-Toe](https://leetcode.com/problems/design-tic-tac-toe/)

```java
    private int[] rows, cols;
    private int diagonal, antiDiagonalonal, n;
    
    /** Initialize your data structure here. */
    public TicTacToe(int n) {
        rows = new int[n];
        cols = new int[n];
        diagonal = 0;
        antiDiagonalonal = 0;
        this.n = n;
    }
    
    /** Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins. */
    public int move(int row, int col, int player) {
        int val = player == 1 ? 1 : -1;
        int target = player == 1 ? n : -n;
        
        if(row == col){
            diagonal += val;
            if(diagonal == target)
                return player;
        }
        if(row + col + 1 == n){
            antiDiagonalonal += val;
            if(antiDiagonalonal == target)
                return player;
        }
        rows[row] += val;
        cols[col] += val;
        if(rows[row] == target || cols[col] == target)
            return player;
        
        return 0;
    }
```

[349. Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)

```java
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet<>();
        Set<Integer> intersect = new HashSet<>();
        
        for(int num : nums1)
            set.add(num);
        for(int num : nums2){
            if(set.contains(num))
                intersect.add(num);
        }
        
        int[] res = new int[intersect.size()];
        int i = 0;
        for(Integer n : intersect)
            res[i++] = n;
        
        return res;
    }
```

[362. Design Hit Counter](https://leetcode.com/problems/design-hit-counter/)

```java
    private int[] timestamps;
    private int[] hits;

    /** Initialize your data structure here. */
    public HitCounter() {
        timestamps = new int[300];
        hits = new int[300];
    }
    
    /** Record a hit.
        @param timestamp - The current timestamp (in seconds granularity). */
    public void hit(int timestamp) {
        int i = timestamp % 300;
        if(timestamps[i] == timestamp){
            hits[i]++;
        }else{
            timestamps[i] = timestamp;
            hits[i] = 1;
        }
    }
    
    /** Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity). */
    public int getHits(int timestamp) {
        int totalHits = 0;
        for(int i = 0; i < 300; i++){
            if(timestamp - timestamps[i] < 300){
                totalHits += hits[i];
            }
        }
        return totalHits;
    }
```

[373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

```java
class Solution {
    public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<Tuple> pq = new PriorityQueue<>();
        List<int[]> res = new ArrayList<>();
        int m = nums1.length, n = nums2.length;
        if(m == 0 || n == 0 || nums1 == null || nums2 == null || k <= 0) 
            return res;
        for(int j = 0; j < n; j++) 
            pq.offer(new Tuple(0, j, nums1[0] + nums2[j]));
        for(int i = 0; i < Math.min(k, m * n); i++){
            Tuple t = pq.poll();
            res.add(new int[]{nums1[t.x], nums2[t.y]});
            if(t.x < m - 1){
                pq.offer(new Tuple(t.x + 1, t.y, nums1[t.x+1] + nums2[t.y]));
            }
        }
        return res;
    }
}

class Tuple implements Comparable<Tuple> {
    int x, y, val;
    public Tuple(int x, int y, int val){
        this.x = x;
        this.y = y;
        this.val = val;
    }
    
    public int compareTo(Tuple that){
        return this.val - that.val;
    }
}
```

[378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

```java
    public int kthSmallest(int[][] matrix, int k) {
        int lo = matrix[0][0], hi = matrix[matrix.length - 1][matrix[0].length - 1];
        while(lo < hi){
            int mid = lo + (hi - lo) / 2;
            int count = 0, j = matrix[0].length - 1;
            for(int i = 0; i < matrix.length; i++){
                while(j >= 0 && matrix[i][j] > mid) // find nums that smaller than mid
                    j--;
                count += j + 1;
            }
            if(count < k) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
    public int kthSmallest1(int[][] matrix, int k) {
        int n = matrix.length;
        PriorityQueue<Tuple> pq = new PriorityQueue<>();
        for(int j = 0; j < n; j++) pq.offer(new Tuple(0, j, matrix[0][j]));
        for(int i = 0; i < k-1; i++){
            Tuple t = pq.poll();
            if(t.x < n - 1){
                pq.offer(new Tuple(t.x + 1, t.y, matrix[t.x + 1][t.y]));
            }
        }
        return pq.poll().val;
    }
    
    class Tuple implements Comparable<Tuple>{
        int x, y, val;

        public Tuple(int x, int y, int val){
            this.x = x;
            this.y = y;
            this.val = val;
        }
        
        public int compareTo(Tuple that){
            return this.val - that.val;
        }
    }
```

[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)

```java
	ArrayList<Integer> nums;
    HashMap<Integer, Integer> locs;
    java.util.Random rand = new java.util.Random();
    
    public RandomizedSet() {
        nums = new ArrayList<Integer>();
        locs = new HashMap<Integer, Integer>();
    }
    
    public boolean insert(int val) {
        //boolean contain = locs.containsKey(val);
        if(locs.containsKey(val)) 
            return false;
        locs.put(val, nums.size());
        nums.add(val);
        return true;
    }
    
    public boolean remove(int val) {
        //boolean contain = locs.containsKey(val);
        if (!locs.containsKey(val)) return false;
        int loc = locs.get(val);
        if (loc < nums.size() - 1) { // not the last one than swap the last one with this val
            int lastOne = nums.get(nums.size() - 1);
            nums.set(loc, lastOne);
            locs.put(lastOne, loc);
        }
        locs.remove(val);
        nums.remove(nums.size() - 1);
        return true;
    }
    
    public int getRandom() {
        return nums.get(rand.nextInt(nums.size()));
    }
```

[384. Shuffle an Array](https://leetcode.com/problems/shuffle-an-array/)

```java
    private int[] nums;
    
    public Solution(int[] nums) {
        this.nums = nums;
    }
    
    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
        return nums;
    }
    
    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
        int[] rand = new int[nums.length];
        for (int i = 0; i < nums.length; i++){
            int r = (int) (Math.random() * (i+1));
            rand[i] = rand[r];
            rand[r] = nums[i];
        }
        return rand;
    }
```

[394. Decode String](https://leetcode.com/problems/decode-string/)

```java
    public String decodeString(String s) {
        Stack<Integer> intStack = new Stack<>();
        Stack<StringBuilder> strStack = new Stack<>();
        StringBuilder cur = new StringBuilder();
        int k = 0;
        
        for(char ch : s.toCharArray()){
            if(Character.isDigit(ch)){
                k = k * 10 + ch - '0';
            }
            else if(ch == '['){
                intStack.push(k);
                strStack.push(cur);
                cur = new StringBuilder();
                k = 0;
            }
            else if(ch == ']'){
                StringBuilder temp = cur;
                cur = strStack.pop();
                for(k = intStack.pop(); k > 0; k--)
                    cur.append(temp);
            }
            else
                cur.append(ch);
        }
        return cur.toString();
    }
```

[402. Remove K Digits](https://leetcode.com/problems/remove-k-digits/)

```java
    public String removeKdigits1(String num, int k) {
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < num.length(); i++){
            char cur = num.charAt(i);
            while(k > 0 && sb.length() > 0 && sb.charAt(sb.length() - 1) > cur){
                sb.deleteCharAt(sb.length()-1);
                k--;
            }
            if(sb.length() == 0 && cur == '0') continue;
            sb.append(cur);
        }
        while(k-- > 0 && sb.length() > 0){
            sb.deleteCharAt(sb.length() - 1);
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    public String removeKdigits2(String num, int k) {
        int digits=num.length()-k;
        char[] st=new char[num.length()];
        int top=0;
        for(int i=0;i<num.length();i++){
            char c=num.charAt(i);
            while(top>0 && k>0 && st[top-1]>c){
                top--;
                k--;
            }
            st[top++]=c;
        }
        int idx=0;
        while(idx<digits && st[idx]=='0')idx++;
        return idx==digits?"0":new String(st,idx,digits-idx);
    }
```

[412. Fizz Buzz](https://leetcode.com/problems/fizz-buzz/)

```java
    public List<String> fizzBuzz(int n) {
        List<String> res = new LinkedList<>();
        for(int i = 1; i <= n; i++){
            if(i % 3 == 0 && i % 5 == 0)
                res.add("FizzBuzz");
            else if(i % 5 == 0)
                res.add("Buzz");
            else if(i % 3 == 0)
                res.add("Fizz");
            else
                res.add(String.valueOf(i));
        }
        return res;
    }
```



[415. Add Strings](https://leetcode.com/problems/add-strings/)

```java
    public String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        for(int i = num1.length() - 1, j = num2.length() - 1; i >= 0 || j >= 0 || carry > 0; i--, j--){
            int x = i < 0 ? 0 : num1.charAt(i) - '0';
            int y = j < 0 ? 0 : num2.charAt(j) - '0';
            sb.append((x + y + carry) % 10);
            carry = (x + y + carry) / 10;
        }
        return sb.reverse().toString();
    }
```

[417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

```java
    private int m, n;
    private int[][] matrix;
    private int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public List<int[]> pacificAtlantic(int[][] matrix) {
        List<int[]> ret = new ArrayList<>();
        if (matrix == null || matrix.length == 0) {
            return ret;
        }

        m = matrix.length;
        n = matrix[0].length;
        this.matrix = matrix;
        boolean[][] canReachP = new boolean[m][n];
        boolean[][] canReachA = new boolean[m][n];

        for (int i = 0; i < m; i++) {
            dfs(i, 0, canReachP);
            dfs(i, n - 1, canReachA);
        }
        for (int i = 0; i < n; i++) {
            dfs(0, i, canReachP);
            dfs(m - 1, i, canReachA);
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (canReachP[i][j] && canReachA[i][j]) {
                    ret.add(new int[]{i, j});
                }
            }
        }

        return ret;
    }

    private void dfs(int r, int c, boolean[][] canReach) {
        if (canReach[r][c]) {
            return;
        }
        canReach[r][c] = true;
        for (int[] d : direction) {
            int nextR = d[0] + r;
            int nextC = d[1] + c;
            if (nextR < 0 || nextR >= m || nextC < 0 || nextC >= n
                    || matrix[r][c] > matrix[nextR][nextC]) {
                continue;
            }
            dfs(nextR, nextC, canReach);
        }
    }
```

[419. Battleships in a Board](https://leetcode.com/problems/battleships-in-a-board/)

```java
    public int countBattleships(char[][] board) {
        if(board.length == 0)
            return 0;
        int count = 0;
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(board[i][j] == '.') continue;
                if(i > 0 && board[i-1][j] == 'X') continue;
                if(j > 0 && board[i][j-1] == 'X') continue;
                count++;
            }
        }
        return count;
    }
```

[442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)

```java
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < nums.length; i++){
            int index = Math.abs(nums[i]) - 1;
            if(nums[index] < 0)
                res.add(index + 1);
            nums[index] = -nums[index];
        }
        return res;
    }
```

[443. String Compression](https://leetcode.com/problems/string-compression/)

```java
    public int compress(char[] chars) {
        int i = 0, res = 0;
        while(i < chars.length){
            char cur = chars[i];
            int count = 0;
            while(i < chars.length && cur == chars[i]){
                i++;
                count++;
            }
            chars[res++] = cur;
            if(count != 1){
                for(char num : Integer.toString(count).toCharArray()){
                    chars[res++] = num;
                }
            }
        }
        return res;
    }
```

[445. Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/)

```java
       public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        //新建两个stack，分别把l1,l2放入
        Stack <Integer> stack1 = new Stack<Integer>();
        Stack <Integer> stack2 = new Stack<Integer>();
        while(l1 != null){
            stack1.push(l1.val); 
            l1 = l1.next;
        }
        while(l2 != null){
            stack2.push(l2.val);
            l2 = l2.next;
        }
        // 新建一个list用来存放相加后多的位数，并把它放在head位置
        int sum = 0, carry = 0;
        ListNode cur = new ListNode(0);
        while(!stack1.empty() || !stack2.empty()){
            //sum = carry;
            if(!stack1.empty()){
                carry += stack1.pop();
            }
            if(!stack2.empty()){
                carry += stack2.pop();
            }
            cur.val = carry % 10;
            carry = carry/10;
            ListNode carryNode = new ListNode(carry);
            carryNode.next = cur;
            cur = carryNode;
        }
        return (cur.val == 0)? cur.next : cur ;
    
    }
```

[448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)

```java
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < nums.length; i++){
            int val = Math.abs(nums[i]) - 1;
            if(nums[val] > 0)
                nums[val] = -nums[val];
        }
        for(int i = 0; i < nums.length; i++){
            if(nums[i] > 0)
                res.add(i+1);
        }
        return res;
    }
```

[449. Serialize and Deserialize BST](https://leetcode.com/problems/serialize-and-deserialize-bst/)

```java
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        buildString(root, sb);
        return sb.toString();
    }
    private void buildString(TreeNode root, StringBuilder path){
        if(root == null)  path.append("#").append(" ");
        else{
            path.append(root.val).append(" ");
            buildString(root.left, path);
            buildString(root.right, path);
        }
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> q = new LinkedList<>();
        q.addAll(Arrays.asList(data.split(" ")));
        return buildTree(q);
    }
    private TreeNode buildTree(Queue<String> q){
        String node = q.poll();
        if(node.equals("#")) return null;
        TreeNode root = new TreeNode(Integer.valueOf(node));
        root.left = buildTree(q);
        root.right = buildTree(q);
        
        return root;
    }
```

[450. Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/)

```java
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null) return null;
        
        if(root.val < key){
            root.right = deleteNode(root.right, key);
        }else if(root.val > key){
            root.left = deleteNode(root.left, key);
        }else{
            if(root.left == null) return root.right;
            if(root.right == null) return root.left;
            
            TreeNode rightSmallest = root.right;
            while(rightSmallest.left != null)
                rightSmallest = rightSmallest.left;
            rightSmallest.left = root.left;
            return root.right;
        }
        return root;
    }
```

[451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/)

```java
    public String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for(char c : s.toCharArray())
            map.put(c, map.getOrDefault(c, 0) + 1);
        
        PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>((a, b) -> b.getValue() - a.getValue());

        pq.addAll(map.entrySet());
        
        StringBuilder sb = new StringBuilder();
        while(!pq.isEmpty()){
            Map.Entry e = pq.poll();
            for(int i = 0; i < (int)e.getValue(); i++){
                sb.append(e.getKey());
            }
        }
        return sb.toString();
    }
```
[490. The Maze](https://leetcode.com/problems/the-maze/)
```java
    private int[][] directions = new int[][] {{1,0},{-1,0},{0,1},{0,-1}};
    
    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        if (maze.length == 0) {
            return false;
        }
        boolean[][] visited = new boolean[maze.length][maze[0].length];
        return helper(maze, start, destination, visited);
    }
    
    private boolean helper(int[][] maze, int[] start, int[] destination, boolean[][] visited) {
        if (maze[start[0]][start[1]] == 1) {
            return false;
        }
        if(visited[start[0]][start[1]]) return false;
    
        if(start[0] == destination[0] && start[1] == destination[1]) return true;
        
        visited[start[0]][start[1]] = true;
        
        for(int[] dir : directions){
            int x = start[0];
            int y = start[1];
            if(dir[0] == 0){
                while(y+dir[1]>=0 && y+dir[1]<maze[0].length && maze[x][y+dir[1]]!=1){
                    y+=dir[1];
                }
            }
            else{
                while(x+dir[0]>=0 && x+dir[0]<maze.length && maze[x+dir[0]][y]!=1){
                    x+=dir[0];
                }
            }
            if(helper(maze, new int[]{x,y}, destination, visited)) 
                return true;
        }
        return false;
        
    }
```

[503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)

```java
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] next = new int[n];
        Arrays.fill(next, -1);
        Stack<Integer> stack = new Stack<>();
        
        for(int i = 0; i < n * 2; i++){
            int num = nums[i % n];
            while(!stack.isEmpty() && nums[stack.peek()] < num)
                next[stack.pop()] = num;
            if(i < n)
                stack.push(i);
        }
        return next;
    }
```

[523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)

```java
    public boolean checkSubarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>(){{put(0,-1);}};;
        int sum = 0;
        for(int i = 0; i < nums.length; i++){
            sum += nums[i];
            if(k != 0)
                sum %= k;
            Integer prev = map.get(sum);
            if(prev != null){
                if(i - prev > 1)
                    return true;
            }
            else
                map.put(sum, i);
        }
        return false;
    }
```

[545. Boundary of Binary Tree](https://leetcode.com/problems/boundary-of-binary-tree/)

```java
    List<Integer> res = new ArrayList<>(1000);
    public List<Integer> boundaryOfBinaryTree(TreeNode root) {
        if(root == null) return res;
        
        res.add(root.val);
        
        leftSubTree(root.left);
        leaves(root.left);
        leaves(root.right);
        rightSubTree(root.right);
        
        return res;
    }
    
    private void leftSubTree(TreeNode root){
        if(root == null || (root.left == null && root.right == null)) // skip leaves
            return;
        res.add(root.val);
        if(root.left == null)
            leftSubTree(root.right);
        else
            leftSubTree(root.left);
    }
    
    private void rightSubTree(TreeNode root){
        if(root == null || (root.left == null && root.right == null)) 
            return;
        if(root.right == null)
            rightSubTree(root.left);
        else
            rightSubTree(root.right);
        res.add(root.val);
    }
    
    private void leaves(TreeNode root){
        if(root == null) 
            return;
        if(root.left == null && root.right == null) {
            res.add(root.val);
            return;
        }
        leaves(root.left);
        leaves(root.right);
    }
```

[547. Friend Circles](https://leetcode.com/problems/friend-circles/)

```java
    public int findCircleNum(int[][] M) {
        int n = M.length, count = 0;;
        boolean[] visited = new boolean[n];
        for(int i = 0; i < n; i++){
            if(!visited[i]){
                dfs(M, visited, i);
                count++;
            }
        }
        return count;
    }
    private void dfs(int[][] M, boolean[] visited, int cur){
        for(int other = 0; other < M.length; other++){
            if(M[cur][other] == 1 && !visited[other]){
                visited[other] = true;
                dfs(M, visited, other);
            }
        }
    }
```

[557. Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/)

```java
    public String reverseWords(String s) {
        String[] str = s.split(" ");
        for(int i = 0; i < str.length; i++)
            str[i] = new StringBuilder(str[i]).reverse().toString();
        StringBuilder res = new StringBuilder();
        for(String st : str)
            res.append(st + " ");
        return res.toString().trim();
    }
```

[572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)

```java
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if(s == null) return false;
        if(isSame(s, t)) return true;
        
        return isSubtree(s.left, t) || isSubtree(s.right, t);
    }
    private boolean isSame(TreeNode s, TreeNode t){
        if(s == null && t == null) return true;
        if(s == null || t == null) return false;
        
        if(s.val != t.val) return false;
        
        return isSame(s.left, t.left) && isSame(s.right, t.right);
    }
```

[621. Task Scheduler](https://leetcode.com/problems/task-scheduler/)

```java
    public int leastInterval(char[] tasks, int n) {
        int[] c = new int[26];
        for(char t : tasks){
            c[t - 'A']++;
        }
        Arrays.sort(c);
        int i = 25;
        while(i >= 0 && c[i] == c[25]) i--;

        return Math.max(tasks.length, (c[25] - 1) * (n + 1) + 25 - i);
    }
```

[636. Exclusive Time of Functions](https://leetcode.com/problems/exclusive-time-of-functions/)

```java
    public int[] exclusiveTime(int n, List<String> logs) {
        Stack<Integer> st = new Stack<>();
        int pre = 0;
        int[] res = new int[n];
        
        for(String log : logs){
            String[] arr = log.split(":");
            if(arr[1].equals("start")){
                if(!st.isEmpty())
                    res[st.peek()] += Integer.parseInt(arr[2]) - pre;
                st.push(Integer.parseInt(arr[0]));
                pre = Integer.parseInt(arr[2]);
            }
            else{
                res[st.pop()] += Integer.parseInt(arr[2]) - pre + 1;
                pre = Integer.parseInt(arr[2]) + 1;
            }
        }
        return res;
    }
```

[647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

```java
    int count = 0;
    public int countSubstrings(String s) {
        if(s.length() == 0 || s == null)
            return 0;
        for(int i = 0; i < s.length(); i++){
            findPalindromic(s, i, i);
            findPalindromic(s, i, i+1);
        }
        return count;
    }
    private void findPalindromic(String s, int left, int right){
        while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)){
            left--;  //
            right++; //
            count++;
        }
    }
```

[652. Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees/)

```java
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        List<TreeNode> res = new ArrayList<>();
        preOrder(root, new HashMap<>(), res);
        return res;
    }
    private String preOrder(TreeNode cur, Map<String, Integer> map, List<TreeNode> res){
        if(cur == null) return "#";
        String path = cur.val + "," + preOrder(cur.left, map, res) + "," + preOrder(cur.right, map, res);
        if(map.containsKey(path) && map.get(path) == 1) res.add(cur);
        map.put(path, map.getOrDefault(path, 0) + 1);
        return path;
    }
```



[662. Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree/)

```java
    public int widthOfBinaryTree(TreeNode root) {
        return dfs(root, 0, 1, new ArrayList<>());
    }

    private int dfs(TreeNode node, int level, int index, List<Integer> starts) {
        if (node == null) return 0;
        if (starts.size() == level) 
            starts.add(index);

        int cur = index - starts.get(level) + 1;
        int leftResult = dfs(node.left, level + 1, index * 2, starts);
        int rightResult = dfs(node.right, level + 1, index * 2 + 1, starts);
        return Math.max(cur, Math.max(leftResult, rightResult));
    }
```

[669. Trim a Binary Search Tree](https://leetcode.com/problems/trim-a-binary-search-tree/)

```java
    public TreeNode trimBST(TreeNode root, int L, int R) {
        if(root == null) return null;
        if(root.val < L) return trimBST(root.right, L, R);
        if(root.val > R) return trimBST(root.left, L, R);
        root.left = trimBST(root.left, L, R);
        root.right = trimBST(root.right, L, R);
        return root;
    }
```

[671. Second Minimum Node In a Binary Tree](https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/)

```java
    public int findSecondMinimumValue(TreeNode root) {
        if(root.left == null) return -1;
        
        int l = root.left.val == root.val ? findSecondMinimumValue(root.left) : root.left.val;
        int r = root.right.val == root.val ? findSecondMinimumValue(root.right) : root.right.val;
        
        return (l == -1 || r == -1) ? Math.max(l, r) : Math.min(l, r);
    }
```

[692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)

```java
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> map = new HashMap<>();
        int maxFre = 0;
        for(String word : words){
            map.put(word, map.getOrDefault(word, 0) + 1);
            maxFre = Math.max(maxFre, map.get(word));
        }
        List<String>[] bucket = new ArrayList[maxFre + 1];
        for(Map.Entry<String, Integer> freSet: map.entrySet()){
            int fre = freSet.getValue();
            if(bucket[fre] == null)
                bucket[fre] = new ArrayList<>();
            bucket[fre].add(freSet.getKey());
        }
        List<String> res = new ArrayList<>();
        for(int i = maxFre; i >= 0 && res.size() < k; i--){
            if(bucket[i] != null){
                Collections.sort(bucket[i]);
                res.addAll(bucket[i]);
            }
        }
        return res.subList(0, k);
    }
```

[694. Number of Distinct Islands](https://leetcode.com/problems/number-of-distinct-islands/)

```java
    public int numDistinctIslands(int[][] grid) {
        Set<String> set = new HashSet<>();
        
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == 1){
                    StringBuilder sb = new StringBuilder();
                    dfs(grid, i, j, sb, 0);
                    grid[i][j] = 0;
                    set.add(sb.toString());
                }
            }
        }
        return set.size();
    }
    private void dfs(int[][] grid, int i, int j, StringBuilder path, int dir){
        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] != 1)
            return;
        path.append(dir);
        grid[i][j] = 0;
        dfs(grid, i+1, j, path, 1);
        dfs(grid, i-1, j, path, 2);
        dfs(grid, i, j+1, path, 3);
        dfs(grid, i, j-1, path, 4);
        path.append("#");
    }
```

[695. Max Area of Island](https://leetcode.com/problems/max-area-of-island/)

```java
    public int maxAreaOfIsland(int[][] grid) {
        int maxArea = 0;
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                maxArea = Math.max(maxArea, dfs(grid, i, j));
            }
        }
        return maxArea;
    }
    
    private int dfs(int[][] grid, int i , int j){
        if(i >= 0 && j >= 0 && i < grid.length && j < grid[0].length && grid[i][j] == 1){
            grid[i][j] = 0;
            return 1 + dfs(grid, i+1, j) + dfs(grid, i-1, j) + dfs(grid, i, j+1) + dfs(grid, i, j-1);
        }
        return 0;
    }
```

[722. Remove Comments](https://leetcode.com/problems/remove-comments/)

```java
    public List<String> removeComments(String[] source) {
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean mode = false;
        for(String s : source){
            for(int i = 0; i < s.length(); i++){
                if(mode){
                    if(s.charAt(i) == '*' && i < s.length() - 1 && s.charAt(i+1) == '/'){
                        mode = false;
                        i++;
                    }
                }else{
                    if(s.charAt(i) == '/' && i < s.length() - 1 && s.charAt(i+1) == '/'){
                        break;
                    }
                    else if(s.charAt(i) == '/' && i < s.length() - 1 && s.charAt(i+1) == '*'){
                        mode = true;
                        i++;
                    }
                    else
                        sb.append(s.charAt(i));
                }
            }
            if(!mode && sb.length() > 0){
                res.add(sb.toString());
                sb = new StringBuilder();
            }
        }
        return res;
    }
```

[784. Letter Case Permutation](https://leetcode.com/problems/letter-case-permutation/)

```java
    public List<String> letterCasePermutation(String S) {
        if(S == null)
             return new LinkedList<>();
        Queue<String> q = new LinkedList<>();
        q.offer(S);
        
        for(int i = 0; i < S.length(); i++){
            if(Character.isDigit(S.charAt(i)))
                continue;
            int size = q.size();
            
            for(int j = 0; j < size; j++){
                char[] chs = q.poll().toCharArray();
                
                chs[i] = Character.toUpperCase(chs[i]);
                q.offer(String.valueOf(chs));
                
                chs[i] = Character.toLowerCase(chs[i]);
                q.offer(String.valueOf(chs));
            }
        }
        return new LinkedList<>(q);
    }

    public List<String> letterCasePermutation(String S) {
        if (S == null) {
            return new LinkedList<>();
        }
        
        List<String> res = new LinkedList<>();
        helper(S.toCharArray(), res, 0);
        return res;
    }
    
    public void helper(char[] chs, List<String> res, int pos) {
        if (pos == chs.length) {
            res.add(new String(chs));
            return;
        }
        if (chs[pos] >= '0' && chs[pos] <= '9') {
            helper(chs, res, pos + 1);
            return;
        }
        
        chs[pos] = Character.toLowerCase(chs[pos]);
        helper(chs, res, pos + 1);
        
        chs[pos] = Character.toUpperCase(chs[pos]);
        helper(chs, res, pos + 1);
    }
```

[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

```java
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        int INF = 0x3F3F3F3F;
        
        int[] cost = new int[n];
        Arrays.fill(cost, INF);
        cost[src] = 0; //
        
        int res = cost[dst];
        
        for(int i = K; i >= 0; i--){
            int[] cur = new int[n];
            Arrays.fill(cur, INF);
            for(int[] flight : flights)
                cur[flight[1]] = Math.min(cur[flight[1]], cost[flight[0]] + flight[2]);
            cost = cur;
            res = Math.min(res, cost[dst]);
        }
        return res == INF ? -1 : res;
    }
```

[794. Valid Tic-Tac-Toe State](https://leetcode.com/problems/valid-tic-tac-toe-state/)

```java
    public boolean validTicTacToe(String[] board) {
        boolean xWin = false, oWin = false;
        int diagonal = 0, antiDiagonal = 0;
        int turns = 0;
        int[] rows = new int[3];
        int[] cols = new int[3];
        
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                if(board[i].charAt(j) == 'X'){
                    turns++; rows[i]++; cols[j]++;
                    if(i == j) diagonal++;
                    if(i + j == 2) antiDiagonal++;
                }
                else if(board[i].charAt(j) == 'O'){
                    turns--; rows[i]--; cols[j]--;
                    if(i == j) diagonal--;
                    if(i + j == 2) antiDiagonal--;
                }
            }
        }
        xWin = rows[0] == 3 || rows[1] == 3 || rows[2] == 3 || 
               cols[0] == 3 || cols[1] == 3 || cols[2] == 3 || 
               diagonal == 3 || antiDiagonal == 3;
        oWin = rows[0] == -3 || rows[1] == -3 || rows[2] == -3 || 
               cols[0] == -3 || cols[1] == -3 || cols[2] == -3 || 
               diagonal == -3 || antiDiagonal == -3;
        
        if(turns == 0 && xWin || turns == 1 && oWin)
            return false;
        return (turns == 0 || turns == 1) && (!xWin || !oWin);
    }
```

[796. Rotate String](https://leetcode.com/problems/rotate-string/)

```java
    public boolean rotateString(String A, String B) {
        return (A.length() == B.length()) && (A+A).contains(B);
    }·
```

[706. Design HashMap](https://leetcode.com/problems/design-hashmap/)

```java
    final ListNode[] nodes = new ListNode[10000];

    /** Initialize your data structure here. */
    public MyHashMap() {
        
    }
    
    /** value will always be non-negative. */
    public void put(int key, int value) {
        int i = getIndex(key);
        if(nodes[i] == null)
            nodes[i] = new ListNode(-1, -1);
        ListNode prev = find(nodes[i], key);
        if(prev.next == null)
            prev.next = new ListNode(key, value);
        else
            prev.next.val = value;
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    public int get(int key) {
        int i = getIndex(key);
        if(nodes[i] == null)
            return -1;
        ListNode prev = find(nodes[i], key);
        return prev.next == null ? -1 : prev.next.val;
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    public void remove(int key) {
        int i = getIndex(key);
        if(nodes[i] == null)
            return;
        ListNode prev = find(nodes[i], key);
        if(prev.next == null)
            return;
        prev.next = prev.next.next;
    }
    
    private int getIndex(int key){
        return Integer.hashCode(key) % nodes.length;
    } 
    
    private ListNode find(ListNode bucket, int key){
        ListNode p = bucket, prev = null;
        while(p != null && p.key != key){
            prev = p;
            p = p.next;
        }
        return prev;
    }
    
    class ListNode{
        int key;
        int val;
        ListNode next;
        
        public ListNode(int key, int val){
            this.key = key;
            this.val = val;
        }
    }
```

[819. Most Common Word](https://leetcode.com/problems/most-common-word/)

```java
    public String mostCommonWord(String paragraph, String[] banned) {
        Set<String> ban = new HashSet<>(Arrays.asList(banned));
        Map<String, Integer> counter = new HashMap<>();
        String[] words = paragraph.replaceAll("\\pP", " ").toLowerCase().split("\\s+");
        String res = "";
        int max = 0;
        
        for(String word : words){
            if(!ban.contains(word)){
                counter.put(word, counter.getOrDefault(word, 0) + 1);
                if(max < counter.get(word)){
                    res = word;
                    max = counter.get(word);
                }
            }
        }
        
        return res;
    }
```

[708. Insert into a Cyclic Sorted List](https://leetcode.com/problems/insert-into-a-cyclic-sorted-list/)

```java
class Solution {
    public Node insert(Node head, int x) {
        if(head == null){
            Node node = new Node(x, null);
            node.next = node;
            return node;
        }
        Node cur = head;
        while(true){
            if(cur.val > cur.next.val){
                if(cur.val <= x || x <= cur.next.val){
                    insertAfter(cur, x);
                    break;
                }
            }else if(cur.val < cur.next.val){
                if(cur.val <= x && x <= cur.next.val){
                    insertAfter(cur, x);
                    break;
                }
            }else{
                if(cur.next == head){
                    insertAfter(cur, x);
                    break;
                }
            }
            cur = cur.next;
        }
        return head;
    }
    private void insertAfter(Node cur, int x){
        cur.next = new Node(x, cur.next);
    }
```

[836. Rectangle Overlap](https://leetcode.com/problems/rectangle-overlap/)

```java
    public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
        return rec1[0] < rec2[2] && rec2[0] < rec1[2] && rec1[1] < rec2[3] && rec2[1] < rec1[3];
    }
```

[852. Peak Index in a Mountain Array](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

```java
    public int peakIndexInMountainArray(int[] A) {
        int lo = 0, hi = A.length - 1;
        while(lo < hi){
            int mid1 = lo + (hi - lo) / 2;
            int mid2 = mid1 + 1;
            if(A[mid1] < A[mid2])
                lo = mid1 + 1;
            else if(A[mid1] > A[mid2])
                hi = mid2 - 1;
            else
                return mid1;
        }
        return lo;
    }
```

[863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)

```java
    private Map<TreeNode, Integer> map = new HashMap<>();
    
    public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
        List<Integer> res = new ArrayList<>();
        find(root, target);
        dfs(root, target, K, map.get(root), res);
        return res;
    }
    private int find(TreeNode root, TreeNode target){
        if(root == null) return -1;
        if(root == target){
            map.put(root, 0);
            return 0;
        }
        int left = find(root.left, target);
        if(left >= 0){
            map.put(root, left + 1);
            return left + 1;
        }        
        int right = find(root.right, target);
        if(right >= 0){
            map.put(root, right + 1);
            return right + 1;
        }
        return -1;
    }
    private void dfs(TreeNode root, TreeNode target, int K, int len, List<Integer> res){
        if(root == null) return;
        if(map.containsKey(root)) len = map.get(root);
        if(len == K) res.add(root.val);
        dfs(root.left, target, K, len+1, res);
        dfs(root.right, target, K, len+1, res);
    }
```

[905. Sort Array By Parity](https://leetcode.com/problems/sort-array-by-parity/)

```java
    public int[] sortArrayByParity(int[] A) {
        if(A.length <= 1 || A == null) return A;
        int left = 0, right = A.length - 1;
        while(left < right){
            if(A[left] % 2 != 0 && A[right] % 2 != 1)
                swap(A, left++, right--);
            if(A[left] % 2 == 0)
                left++;
            if(A[right] % 2 != 0)
                right--;
        }
        return A;
    }
    private void swap(int[] A, int i , int j){
        int temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
```

[917. Reverse Only Letters](https://leetcode.com/problems/reverse-only-letters/)

```java
    public String reverseOnlyLetters(String S) {
        char[] chs = S.toCharArray();
        int i = 0, j = chs.length - 1;
        while(i <= j){
            if(Character.isLetter(chs[i]) && Character.isLetter(chs[j]))
                swap(chs, i++, j--);
            else if(!Character.isLetter(chs[i]) && i <= j)
                i++;
            else if(!Character.isLetter(chs[j]) && i <= j)
                j--;
                
        }
        return new String(String.valueOf(chs));
    }
    private void swap(char[] chs, int i, int j){
        char temp = chs[i];
        chs[i] = chs[j];
        chs[j] = temp;
    }
```

[958. Check Completeness of a Binary Tree](https://leetcode.com/problems/check-completeness-of-a-binary-tree/)

```java
    public boolean isCompleteTree(TreeNode root) {
        if(root == null) return false;
        
        boolean end = false;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        
        
        while(!q.isEmpty()){
            TreeNode cur = q.poll();
            if(cur == null) end = true;
            else{
                if(end) return false;
                q.add(cur.left);
                q.add(cur.right);
            }
        }
        return true;
    }
```

[981. Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)

```java
    private Map<String, TreeMap<Integer, String>> map;
    
    /** Initialize your data structure here. */
    public TimeMap() {
        map = new HashMap<>();
    }
    
    public void set(String key, String value, int timestamp) {
        map.putIfAbsent(key, new TreeMap<>());
        map.get(key).put(timestamp, value);
    }
    
    public String get(String key, int timestamp) {
        TreeMap<Integer, String> treeMap = map.get(key);
        if(treeMap == null) return "";
        Integer time = treeMap.floorKey(timestamp);
        if(time == null) return "";
        return treeMap.get(time);
    }
```

[983. Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/)

```java
    public int mincostTickets(int[] days, int[] costs) {
        int[] total = new int[366];
        int cur = 0;
        
        for(int day = 1; day <= 365; day++){
            if(cur == days.length)
                break;
            if(day != days[cur]){
                total[day] = total[day-1];
                continue;
            }
            int min = total[day - 1] + costs[0];
            min = Math.min(min, total[Math.max(0, day - 7)] + costs[1]);
            min = Math.min(min, total[Math.max(0, day - 30)] + costs[2]);
            
            total[day] = min;
            cur++;
        }
        return total[days[days.length - 1]];
    }
```
[984. String Without AAA or BBB](https://leetcode.com/problems/string-without-aaa-or-bbb/)
```java
    public String strWithout3a3b(int A, int B) {
        StringBuilder sb = new StringBuilder();
        while(A + B > 0){
            int len = sb.length();
            if(len > 1 && sb.charAt(len - 1) == sb.charAt(len - 2)){
                if(sb.charAt(len - 1) == 'a'){
                    sb.append('b');
                    B--;
                }else{
                    sb.append('a');
                    A--;
                }
            }else{
                if(A > B){
                    sb.append('a');
                    A--;
                }else{
                    sb.append('b');
                    B--;
                }
            }
        }
        
        return sb.toString();
    }
```
[985. Sum of Even Numbers After Queries](https://leetcode.com/problems/sum-of-even-numbers-after-queries/)
```java
    public int[] sumEvenAfterQueries(int[] A, int[][] queries) {
        int[] res = new int[queries.length];
        int sum = 0;
        int pos = 0;
        for(int a : A){
            if(a % 2 == 0)
                sum += a;
        }
        for(int[] q : queries){
            int val = q[0];
            int i = q[1];
            if(A[i] % 2 == 0)
                sum -= A[i];
            if((val + A[i]) % 2 == 0){
                sum += val + A[i];
            }
            A[i] += val;
            
            res[pos++] = sum;
        }
        
        return res;
    }
```

[986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)
```java
    public Interval[] intervalIntersection(Interval[] A, Interval[] B) {
        if(A == null || A.length == 0 || B == null || B.length == 0)
            return new Interval[]{};
        int m = A.length, n = B.length;
        int i = 0, j = 0;
        List<Interval> res = new ArrayList<>();
        
        while(i < m && j < n){
            Interval a = A[i];
            Interval b = B[j];
            
            int start = Math.max(a.start, b.start);
            int end = Math.min(a.end, b.end);
            
            if(start <= end){
                res.add(new Interval(start, end));
            }
            
            if(a.end == end) i++;
            if(b.end == end) j++;
        }
        return res.toArray(new Interval[0]);
    }
```
[988. Smallest String Starting From Leaf](https://leetcode.com/problems/smallest-string-starting-from-leaf/)
```java
    String ret = "~";
    public String smallestFromLeaf(TreeNode root) {
        dfs(root, "");
        return ret;
    }

    void dfs(TreeNode cur, String s){
        if(cur == null) return;
        s = (char)('a'+cur.val) + s;
        if(cur.left == null && cur.right == null){
            if(s.compareTo(ret) < 0){
                ret = s;
            }
        }
        dfs(cur.left, s);
        dfs(cur.right, s);
    }
```
[989. Add to Array-Form of Integer](https://leetcode.com/problems/add-to-array-form-of-integer/)
```java
    public List<Integer> addToArrayForm(int[] A, int K) {
        List<Integer> res = new LinkedList<>();
        int carry = 0;
        int i = A.length - 1;
        while(K > 0 || i >= 0){
            if(i >= 0)
                carry += A[i--] + K % 10;
            else
                carry += K % 10;
            K /= 10;
            res.add(0, carry % 10);
            carry /= 10;
        }
        if(carry != 0){
            res.add(0, carry);
        }
        
        return res;
    }
```
[990. Satisfiability of Equality Equations](https://leetcode.com/problems/satisfiability-of-equality-equations/)
```java
    public boolean equationsPossible(String[] equations) {
        int[] var = new int[26];
        for(int i = 0; i < 26; i++)
            var[i] = i;
        for(String e : equations){
            if(e.charAt(1) == '=')
                var[find(var, e.charAt(0) - 'a')] = find(var, e.charAt(3) - 'a');
        }
        for(String e : equations){
            if(e.charAt(1) == '!' && find(var, e.charAt(0) - 'a') == find(var, e.charAt(3) - 'a'))
                return false;
        }
        return true;
    }
    private int find(int[] var, int k){
        if(var[k] != k)
            return find(var, var[k]);
        return var[k];
    }
```
[991. Broken Calculator](https://leetcode.com/problems/broken-calculator/)
```java
    public int brokenCalc(int X, int Y) {
        int res = 0;
        while(Y > X){
            if(Y % 2 == 1)
                Y += 1;
            else
                Y /= 2;
            res++;
        }
        return res + X - Y;
    }
```
[994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)
```java
    public int orangesRotting(int[][] grid) {
        int total = 0, rotted = 0, fresh = 0;
        int m = grid.length, n = grid[0].length;
        Queue<int[]> q = new LinkedList<>();
        int res = 0;
        int[][] dirs = {{0,1}, {1,0}, {0,-1}, {-1,0}};
        
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(grid[i][j] == 1){
                    total++;
                    fresh++;
                }
                if(grid[i][j] == 2){
                    total++;
                    rotted++;
                    q.add(new int[]{i, j});
                }
            }
        }
        if(fresh == 0) return 0;
        
        while(!q.isEmpty()){
            res++;
            int size = q.size();
            for(int i = 0; i < size; i++){
                int[] cur = q.poll();
                for(int[] dir : dirs){
                    int x = cur[0] + dir[0];
                    int y = cur[1] + dir[1];
                    if(x < 0 || x >= m || y < 0 || y >= n || grid[x][y] != 1) continue;
                    grid[x][y] = 2;
                    q.add(new int[]{x, y});
                    rotted++;
                    fresh--;
                }
            }
        }
        if(fresh != 0) return -1;
        return res - 1;
    }
```
[993. Cousins in Binary Tree](https://leetcode.com/problems/cousins-in-binary-tree/)
```java
    public boolean isCousins(TreeNode root, int x, int y) {
        if(root == null) return false;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        
        while(!q.isEmpty()){
            boolean xFound = false, yFound = false;
            int size = q.size();
            for(int i = 0; i < size; i++){
                TreeNode cur = q.poll();
                if(cur.val == x) xFound = true;
                if(cur.val == y) yFound = true;
                if(cur.left != null && cur.right != null){
                    if(cur.left.val == x && cur.right.val == y)
                        return false;
                    if(cur.left.val == y && cur.right.val == x)
                        return false;
                }
                if(cur.left != null){
                    q.add(cur.left);
                }
                if(cur.right != null){
                    q.add(cur.right);
                }
            }
            if(xFound && yFound) return true;
        }
        return false;
    }
```
[997. Find the Town Judge](https://leetcode.com/problems/find-the-town-judge/)
```java
    public int findJudge(int N, int[][] trust) {
        int[] trusted = new int[N+1];
        
        for(int[] cur : trust){
            trusted[cur[0]]--;
            trusted[cur[1]]++;
        }
        for(int i = 1; i <= N; i++){
            if(trusted[i] == N - 1)
                return i;
        }
        return -1;
    }
```
[992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/)
```java
    public int subarraysWithKDistinct(int[] A, int K) {
        return atMostK(A, K) - atMostK(A, K - 1);
    }
    int atMostK(int[] A, int K) {
        int i = 0, res = 0;
        int[] count = new int[A.length + 1];
        for (int j = 0; j < A.length; ++j) {
            if (count[A[j]]++ == 0) K--;
            while (K < 0) {
                count[A[i]]--;
                if (count[A[i]] == 0) K++;
                i++;
            }
            res += j - i + 1;
        }
        return res;
    }
```
[987. Vertical Order Traversal of a Binary Tree](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)
```java
public List<List<Integer>> verticalTraversal(TreeNode root) {
    TreeMap<Integer, TreeMap<Integer, PriorityQueue<Integer>>> map = new TreeMap<>();
    dfs(root, 0, 0, map);
    List<List<Integer>> res = new ArrayList<>();
    for(TreeMap<Integer, PriorityQueue<Integer>> ys : map.values()){
        res.add(new ArrayList<>());
        for(PriorityQueue<Integer> q : ys.values()){
            while(!q.isEmpty()){
                res.get(res.size() - 1).add(q.poll());
            }
        }
    }
    return res;
}
private void dfs(TreeNode root, int x, int y, TreeMap<Integer, TreeMap<Integer,PriorityQueue<Integer>>> map){
    if(root == null) return;
    if(!map.containsKey(x))
        map.put(x, new TreeMap<>());
    if(!map.get(x).containsKey(y))
        map.get(x).put(y, new PriorityQueue<>());
    map.get(x).get(y).add(root.val);
    dfs(root.left, x - 1, y + 1, map);
    dfs(root.right, x + 1, y + 1, map);
}
```
[999. Available Captures for Rook](https://leetcode.com/problems/available-captures-for-rook/)
```java
    public int numRookCaptures(char[][] board) {
        int row = -1;
        int column = -1;
        int count = 0;
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                if(board[i][j] == 'R'){
                    row = i;
                    column = j;
                }
            }
        }
        int left = column-1;
        while(left >=0 && board[row][left] != 'B'){
            if(board[row][left] == 'p'){
                count++;
                break;
            }
            left--;
        }
        int right = column+1;
        while(right <8 && board[row][right] != 'B'){
            if(board[row][right] == 'p'){
                count++;
                break;
            }
            right++;
        }
        int up = row-1;
        while(up >=0 && board[up][column] != 'B'){
            if(board[up][column] == 'p'){
                count++;
                break;
            }
            up--;
        }       
        int down = row+1;
        while(down < 8 && board[down][column] != 'B'){
            if(board[down][column] == 'p'){
                count++;
                break;
            }
            down++;
        }  
        return count;
    }
```
[654. Maximum Binary Tree](https://leetcode.com/problems/maximum-binary-tree/)
```java
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return buildTree(nums, 0, nums.length - 1);
    }
    private TreeNode buildTree(int[] nums, int lo, int hi){
        if(lo > hi) return null;
        int max = 0, maxId = lo;
        for(int i = lo; i <= hi; i++){
            if(max < nums[i]){
                max = nums[i];
                maxId = i;
            }
        }
        TreeNode root = new TreeNode(max);
        root.left = buildTree(nums, lo, maxId-1);
        root.right = buildTree(nums, maxId+1, hi);
        return root;
    }
```
[998. Maximum Binary Tree II](https://leetcode.com/problems/maximum-binary-tree-ii/)
```java
    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        TreeNode node = new TreeNode(val), cur = root;
        if(root.val < val){
            node.left = root;
            return node;
        }
        while(cur.right != null && cur.right.val > val){
            cur = cur.right;
        }
        node.left = cur.right;
        cur.right = node;
        
        return root;
    }
```
[979. Distribute Coins in Binary Tree](https://leetcode.com/problems/distribute-coins-in-binary-tree/)
```java
    int res = 0;
    public int distributeCoins(TreeNode root) {
        dfs(root);
        return res;
    }

    public int dfs(TreeNode root) {
        if (root == null) return 0;
        int left = dfs(root.left), right = dfs(root.right);
        res += Math.abs(left) + Math.abs(right);
        return root.val + left + right - 1;
    }
```

[968. Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/)
Apply a recursion function dfs.
Return `0` if it's a leaf.
Return `1` if it's a parent of a leaf, with a camera on this node.
Return `2` if it's covered, without a camera on this node.

For each node,
if it has a child, which is leaf (node 0), then it needs camera.
if it has a child, which is the parent of a leaf (node 1), then it's covered.

If it needs camera, then `res++` and we return `1`.
If it's covered, we return `2`.
Otherwise, we return `0`.

```java
    int res = 0;
    public int minCameraCover(TreeNode root) {
        return (dfs(root) < 1 ? 1 : 0) + res;
    }

    public int dfs(TreeNode root) {
        if (root == null) return 2;
        int left = dfs(root.left), right = dfs(root.right);
        if (left == 0 || right == 0) {
            res++;
            return 1;
        }
        return left == 1 || right == 1 ? 2 : 0;
    }
```
[976. Largest Perimeter Triangle](https://leetcode.com/problems/largest-perimeter-triangle/)
```java
    public int largestPerimeter(int[] A) {
        Arrays.sort(A);
        int i = A.length - 1;
        while(i >= 2){
            if(A[i] < A[i-1] + A[i-2]){
                return A[i] + A[i-1] + A[i-2];
            }else
                i--;
        }
        return 0;
    }
```
[977. Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)
```java
    public int[] sortedSquares(int[] A) {
        int[] res = new int[A.length];
        int lo = 0, hi = A.length - 1;
        for (int i = A.length - 1; i >= 0; i--) {
            if (Math.abs(A[lo]) > Math.abs(A[hi])) {
                res[i] = A[lo] * A[lo];
                lo++;
            } else {
                res[i] = A[hi] * A[hi];
                hi--;
            }
        }
        return res;
    }
```
[978. Longest Turbulent Subarray](https://leetcode.com/problems/longest-turbulent-subarray/)
```java
    public int maxTurbulenceSize(int[] A) {
        int inc = 1, dec = 1, res = 1;
        for(int i = 1; i < A.length; i++){
            if(A[i] < A[i-1]){
                dec = inc + 1;
                inc = 1;
            }else if(A[i] > A[i-1]){
                inc = dec + 1;
                dec = 1;
            }else{
                inc = 1;
                dec = 1;
            }
            res = Math.max(res, Math.max(inc, dec));
        }
        return res;
    }
```
[974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)
If a subarray is divisible by K, it has to be a multiple of K

`a-b=n*k`, `a = running total`, `b = any previous subarray sum`, same as original prefix sum problems.

We want to solve for b, so using basic algebra, `b=a-n*k`.

We don't know what n is, so we can get rid of n by modding every element by k, `(b%k) = (a%k) - (n*k)%k`

since `n*k` is a multiple of k and k goes into it evenly, the result of the `(n *k)%k` will be 0

therefore
`b%k = a%k`

is the same as the formula we defined earlier, `a-b=n*k`

where `b = running total`, `a = any previous subarray sum`

So we just have to see if running total mod k is equal to any previous running total mod k
```java
    public int subarraysDivByK(int[] A, int K) {
        int[] count = new int[K];
        count[0] = 1;
        int prefix = 0, res = 0;
        for (int a : A) {
            prefix = (prefix + a % K + K) % K;
            res += count[prefix]++;
        }
        return res;
    }
```
[973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)
```java
public int[][] kClosest(int[][] points, int K) {
    int len =  points.length, l = 0, r = len - 1;
    while (l <= r) {
        int mid = helper(points, l, r);
        if (mid == K) break;
        if (mid < K) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return Arrays.copyOfRange(points, 0, K);
}

private int helper(int[][] A, int l, int r) {
    int[] pivot = A[l];
    while (l < r) {
        while (l < r && compare(A[r], pivot) >= 0) r--;
        A[l] = A[r];
        while (l < r && compare(A[l], pivot) <= 0) l++;
        A[r] = A[l];
    }
    A[l] = pivot;
    return l;
}

private int compare(int[] p1, int[] p2) {
    return p1[0] * p1[0] + p1[1] * p1[1] - p2[0] * p2[0] - p2[1] * p2[1];
}
```
[560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
```java
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> preSum = new HashMap<>();
        preSum.put(0, 1);
        int sum = 0, res = 0;
        for(int num : nums){
            sum += num;
            if(preSum.containsKey(sum - k)){
                res += preSum.get(sum - k);
            }
            preSum.put(sum, preSum.getOrDefault(sum, 0) + 1);
        }
        return res;
    }
```
[1008. Construct Binary Search Tree from Preorder Traversal](https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/)
```java
    private int i = 0;
    public TreeNode bstFromPreorder(int[] preorder) {
        return buildTree(preorder, Integer.MAX_VALUE);
    }
    private TreeNode buildTree(int[] preorder, int bound){
        if(i >= preorder.length || preorder[i] > bound) return null;
        TreeNode node = new TreeNode(preorder[i++]);
        node.left = buildTree(preorder, node.val);
        node.right = buildTree(preorder, bound);
        return node;
    }
```
[961. N-Repeated Element in Size 2N Array](https://leetcode.com/problems/n-repeated-element-in-size-2n-array/)
```java
    public int repeatedNTimes(int[] A) {
        int i = 0, j = 0, n = A.length;
        while (i == j || A[i] != A[j]) {
            i = (int)(Math.random() * n);
            j = (int)(Math.random() * n);
        }
        return A[i];
    }
```
[962. Maximum Width Ramp](https://leetcode.com/problems/maximum-width-ramp/)
maintain a decreasing `stack` to record all possible candidates. Then traverse from the end of array to find the number which is bigger than `st.peek()`, `end - st.peek()` will be the result, compare `res` to `end - st.peek()` in each loop.

```java
    public int maxWidthRamp(int[] A) {
        Deque<Integer> st = new LinkedList<>();
        for(int i = 0; i < A.length; i++){
            if(st.isEmpty() || A[st.peek()] > A[i])
                st.push(i);
        }
        int res = 0;
        for(int end = A.length - 1; end >= 0; end--){
            while(!st.isEmpty() && A[st.peek()] <= A[end]){
                res = Math.max(res, end - st.pop());
            }
        }
        return res;
    }
```
[959. Regions Cut By Slashes](https://leetcode.com/problems/regions-cut-by-slashes/)
Convert into count islands problem. Construct a `m*3 x n*3` array, mark slashes as `1`, then count islands.
```java
    int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    public int regionsBySlashes(String[] grid) {
        int m = grid.length, n = grid[0].length();
        int[][] g = new int[m*3][n*3];
        int res = 0;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(grid[i].charAt(j) == '\\'){
                    g[i*3][j*3] = 1;
                    g[i*3 + 1][j*3 + 1] = 1;
                    g[i*3 + 2][j*3 + 2] = 1;
                }
                else if(grid[i].charAt(j) == '/'){
                    g[i*3 + 2][j*3] = 1;
                    g[i*3 + 1][j*3 + 1] = 1;
                    g[i*3][j*3 + 2] = 1;
                }
            }
        }
        for(int i = 0; i < g.length; i++){
            for(int j = 0; j < g[0].length; j++){
                if(g[i][j] == 0){
                    dfs(g, i, j);
                    res++;
                }
            }
        }
        
        return res;
    }
    private void dfs(int[][] g, int i, int j){
        if(i < 0 || j < 0 || i >= g.length || j >= g[0].length || g[i][j] != 0)
            return;
        g[i][j] = 1;
        for(int[] dir : dirs){
            dfs(g, i + dir[0], j + dir[1]);
        }
    }
```
[969. Pancake Sorting](https://leetcode.com/problems/pancake-sorting/)
find the largest, then flip all elements before largest one(included), then flip before the index where the largest one should be.
```java
    public List<Integer> pancakeSort(int[] A) {
        List<Integer> res = new ArrayList<>();
        int largest = A.length;
        for(int i = 0; i < A.length; i++){
            int index = find(A, largest);
            flip(A, index);
            flip(A, largest - 1);
            res.add(index + 1);
            res.add(largest);
            largest--;
        }
        return res;
    }
    private int find(int[] A, int target) {
        for (int i = 0; i < A.length; i++) {
            if (A[i] == target) {
                return i;
            }
        }
        return -1;
    }
    private void flip(int[] A, int index) {
        int i = 0, j = index;
        while (i < j) {
            int temp = A[i];
            A[i++] = A[j];
            A[j--] = temp;
        }
    }
```
[967. Numbers With Same Consecutive Differences](https://leetcode.com/problems/numbers-with-same-consecutive-differences/)
```java
    public int[] numsSameConsecDiff(int N, int K) {
        List<Integer> res = Arrays.asList(0,1,2,3,4,5,6,7,8,9);
        for(int i = 2; i <= N; i++){
            List<Integer> list = new ArrayList<>();
            for(int x : res){
                int y = x % 10;
                if(x > 0 && y + K < 10)
                    list.add(x * 10 + y + K);
                if(x > 0 && K > 0 && y - K >= 0)
                    list.add(x * 10 + y - K);
            }
            res = list;
        }
        int[] result = new int[res.size()];
        for(int i = 0; i < res.size(); ++i) {
            result[i] = res.get(i);
        }
        return result;
    }
```


https://leetcode.com/problems/k-closest-points-to-origin/discuss/220235/Java-Three-solutions-to-this-classical-K-th-problem.

[701. Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
```java
    public TreeNode insertIntoBST1(TreeNode root, int val) {
        if(root == null) return new TreeNode(val);
        if(root.val < val)
            root.right = insertIntoBST(root.right, val);
        else if(root.val > val)
            root.left = insertIntoBST(root.left, val);
        return root;
    }

    public TreeNode insertIntoBST2(TreeNode root, int val) {
        if(root == null) return new TreeNode(val);
        TreeNode cur = root;
        
        while(cur != null){
            if(cur.val < val){
                if(cur.right != null) cur = cur.right;
                else{
                    cur.right = new TreeNode(val);
                    break;
                } 
            }else{
                if(cur.left != null) cur = cur.left;
                else{
                    cur.left = new TreeNode(val);
                    break;
                }
            }
        }
        return root;
    }
```
[338. Counting Bits](https://leetcode.com/problems/counting-bits/)
`f[i] = f[i / 2] + i % 2`
Take number X for example, `10011001`.
Divide it in 2 parts:
<1>the last digit ( 1 or 0, which is `i&1`, equivalent to `i%2` )
<2>the other digits ( the number of 1, which is `f[i >> 1]`, equivalent to `f[i/2]` )
```java
    public int[] countBits(int num) {
        int[] res = new int[num + 1];
        for(int i = 1; i <= num; i++){
            res[i] = res[i >> 1] + (i & 1);
        }
        return res;
    }
```
[280. Wiggle Sort](https://leetcode.com/problems/wiggle-sort/)
If `i` is even, `nums[i] <= nums[i+1]`
If `i` is odd, `nums[i] >= nums[i+1]`
```java
    public void wiggleSort(int[] nums) {
        if(nums.length <= 1) return;
        for(int i = 1; i < nums.length; i++){
            if(i % 2 == 0 && nums[i] > nums[i-1]) // even position should be smaller than previous
                swap(nums, i, i - 1);
            else if(i % 2 == 1 && nums[i] < nums[i-1])// odd position should be bigger than previous
                swap(nums, i, i - 1);
        }
        return;
    }
    private void swap(int[] nums, int i, int j){
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
```
[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
```java
    public int[] dailyTemperatures(int[] T) {
        Deque<Integer> st = new LinkedList<>();
        int[] res = new int[T.length];
        for(int i = 0; i < T.length; i++){
            while(!st.isEmpty() && T[i] > T[st.peek()]){
                res[st.peek()] = i - st.pop();
            }
            st.push(i);
        }
        return res;
    }
```
[582. Kill Process](https://leetcode.com/problems/kill-process/)
```java
    public List<Integer> killProcess(List<Integer> pid, List<Integer> ppid, int kill) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for(int i = 0; i < ppid.size(); i++){
            if(ppid.get(i) != 0){
                if(map.get(ppid.get(i)) == null)
                    map.put(ppid.get(i), new ArrayList<>());
                map.get(ppid.get(i)).add(pid.get(i));
            }
        }
        List<Integer> res = new ArrayList<>();
        Queue<Integer> q = new LinkedList<>();
        q.add(kill);
        
        while(!q.isEmpty()){
            int cur = q.poll();
            List<Integer> children = map.get(cur);
            if(children != null){
                q.addAll(children);
            }
            res.add(cur);
        }
        return res;
    }
```

[Class UF]()
```java
private class UF {

    private int[] id;

    UF(int N) {
        id = new int[N + 1];
        for (int i = 0; i < id.length; i++) {
            id[i] = i;
        }
    }

    void union(int u, int v) {
        int uID = find(u);
        int vID = find(v);
        if (uID == vID) {
            return;
        }
        for (int i = 0; i < id.length; i++) {
            if (id[i] == uID) {
                id[i] = vID;
            }
        }
    }

    int find(int p) {
        return id[p];
    }

    boolean connect(int u, int v) {
        return find(u) == find(v);
    }
}
```