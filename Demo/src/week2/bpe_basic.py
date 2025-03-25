from collections import Counter
from hmac import new

words = [list("cam_")] * 5 + [list("nham_")] * 4 + [list("tam_")] * 3 + [list("can_")] * 6 + [list("ham_")] * 4
print(words)

aSet = set()
for word in words:
    for cha in word:
        aSet.add(cha)
print(aSet)

# tìm cặp token xuất hiện nhiều nhất
# Return most occurence
def most_common(words_):
    counter = Counter()
    # {"c":0, "am":2}
    for word in words_:
        for i in range(len(word) - 1):
            # print(word[i], word[i+1])
            counter["{}{}".format(word[i], word[i+1])] += 1
    print(counter)
    return counter.most_common(1)

# Vòng lặp thứ nhất
common_tokens_and_cnt = most_common(words)
print(common_tokens_and_cnt)

# Hợp token
token = common_tokens_and_cnt[0][0]
def merge_tokens(words, token):
    token_to_merge = token
    for i in range(len(words)):
        word = words[i]
        for j in range(len(word) - 1):
            if "{}{}".format(word[j], word[j+1]) == token_to_merge:
                # print('Go')
                new_word = word[:j] + [token_to_merge] + word[j+2:]
                words[i] = new_word
    return words

# ngữ liệu mới
new_words = merge_tokens(words, common_tokens_and_cnt[0][0])
print(new_words)

# vòng lặp thứ 2
common_tokens_and_cnt = most_common(new_words)
print(common_tokens_and_cnt)
new_words = merge_tokens(words, common_tokens_and_cnt[0][0])
print(new_words)

# vòng lặp thứ 3
common_tokens_and_cnt = most_common(new_words)
print(common_tokens_and_cnt)
new_words = merge_tokens(words, common_tokens_and_cnt[0][0])
print(new_words)

