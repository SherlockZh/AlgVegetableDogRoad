[1319. Number of Operations to Make Network Connected](https://leetcode.com/problems/number-of-operations-to-make-network-connected/)

```java
public int makeConnected(int n, int[][] connections) {
    if(n - 1 > connections.length) return -1;
    int[] parent = new int[n];
    for(int i = 0; i < n; i++) parent[i] = i;
    
    for(int[] con : connections){
        int x = find(con[0], parent), y = find(con[1], parent);
        if(x != y) {
            n--;
            parent[x] = y;
        }
    }
    return n - 1;
}

private int find(int x, int[] parent){
    if(parent[x] != x){
        parent[x] = find(parent[x], parent);
    }
    return parent[x];
}
```

[1362. Closest Divisors](https://leetcode.com/problems/closest-divisors/)

```java
    public int[] closestDivisors(int num) {
        for(int i = (int)Math.sqrt(num + 2); i >= 1; i--){
            if((num + 1) % i == 0) return new int[]{i, (num + 1) / i};
            if((num + 2) % i == 0) return new int[]{i, (num + 2) / i};
        }
        return null;
    }
```

[95. Unique Binary Search Trees II](https://leetcode.com/problems/unique-binary-search-trees-ii/)

```java
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> res = genTree(1, n);
        if(res.get(0) == null) return new ArrayList<>();
        else return res;
    }
    
    private List<TreeNode> genTree(int start, int end){
        List<TreeNode> res = new ArrayList<>();
        if(start > end){
            res.add(null);
            return res;
        }
        for(int cur = start; cur <= end; cur++){
            List<TreeNode> leftSub = genTree(start, cur - 1);
            List<TreeNode> rightSub = genTree(cur + 1, end);
            for(TreeNode l : leftSub){
                for(TreeNode r : rightSub){
                    TreeNode node = new TreeNode(cur);
                    node.left = l;
                    node.right = r;
                    res.add(node);
                }
            }
        }
        return res;
    }
```

[131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/)

```java
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        List<String> list = new ArrayList<>();
        backtrack(s, 0, list, res);
        
        return res;
    }
    
    private void backtrack(String s, int start, List<String> list, List<List<String>> res){
        if(start == s.length()){
            res.add(new ArrayList<>(list));
        }
        else{
            for(int i = start; i < s.length(); i++){
                String str = s.substring(start, i+1);
                if(valid(str)){
                    list.add(str);
                    backtrack(s, i+1, list, res);
                    list.remove(list.size() - 1);
                }
            }
        }
    }
    
    private boolean valid(String str){
        int i = 0, j = str.length() - 1;
        while(i < j){
            if(str.charAt(i++) != str.charAt(j--)) 
                return false;
        }
        return true;
    }
```

[880. Decoded String at Index](https://leetcode.com/problems/decoded-string-at-index/)
```java
    public String decodeAtIndex(String S, int K) {
        long size = 0;
        int N = S.length();
        
        for (int i = 0; i < N; ++i) {
            char c = S.charAt(i);
            if (Character.isDigit(c))
                size *= c - '0';
            else
                size++;
        }

        for (int i = N-1; i >= 0; --i) {
            char c = S.charAt(i);
            K %= size;
            if (K == 0 && Character.isLetter(c))
                return Character.toString(c);

            if (Character.isDigit(c))
                size /= c - '0';
            else
                size--;
        }

        throw null;
    }
```