import re
parent_str = 'Con chó đang đi chơi trên đồng. Có con trâu trên đồng. Đồng nay đẹp quá'
pattern = 'đồng'

print("Example string 1: ", parent_str)
# function to search for a substring matching a pattern
print("use re.search: ", re.search(pattern, parent_str))

# function to find all substrings matching the pattern
print("use re.findall: ", re.findall(pattern, parent_str))

# However, this function does not return span - the terminal index of the substrings,
# so you can use the findter function to process it
print("use re.finditer: ")
matches = re.finditer(pattern, parent_str)
for match in matches:
    span = match.span()
    print(f"Math-at: {span}, string: {parent_str[span[0]:span[1]]}")

print("regex selection:")
pattern_2 = '[Đđ]ồng'
print(re.findall(pattern_2, parent_str))

parent_str = 'Mật*********khẩu********bạn***không**được**trống'
print("Example string 2: ", parent_str)
pattern = '\*+'
print(re.split(pattern, parent_str))

parent_str = 'mật       khẩu    rỗng'
print("Example string 3: ", parent_str)
pattern = '\s+'
print(re.sub(pattern,' ', parent_str))