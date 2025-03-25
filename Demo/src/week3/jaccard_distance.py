from blis.py import gemm


def jaccard_distance_ngrams(str1, str2, n = 2):
    # function generate n-grams from a given string
    def generate_ngrams(s, n):
        ngrams = set()
        for i in range(len(s) - n + 1):
            ngrams.add(s[i:i + n])
        return ngrams

    # generate n-grams for both strings
    ngrams1 = generate_ngrams(str1, n)
    ngrams2 = generate_ngrams(str2, n)

    # calculate intersection and union of n-grams sets
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    # calculate Jaccard distance
    distance = 1 - (len(intersection)/len(union))

    return distance

# test
str1 = "hello"
str2 = "hlleo"
distance = jaccard_distance_ngrams(str1, str2, 2)
print(distance)