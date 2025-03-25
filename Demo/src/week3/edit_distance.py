def edit_distance(str1, str2):
    # create a table to store results of subproblems
    dp = [[0 for x in range(len(str2) + 1)] for x in range(len(str1) + 1)]

    # fill dp[][] in bottom up manner
    for i in range(len(str1) + 1):
        for j in range(len(str2) + 1):
            # if first strinng is empty, insert all characters of second string
            if i == 0:
                dp[i][j] = j

            # if second string is empty, remove all characters of the first string
            elif j == 0:
                dp[i][j] = i

            # if last characters are the same, ignore the last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            # if las character is different, consider all possibilities
            # and find minium
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])

    return dp[len(str1)][len(str2)]

# test
str1 = "hello"
str2 = "elhlo"
print(f"Edit Distance between '{str1}' and '{str2}': ", edit_distance(str1, str2))