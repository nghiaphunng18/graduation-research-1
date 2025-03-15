import re

with open("../../datasets/week2/emails_and_names.txt", "r", encoding="utf-8") as file:
    content = file.read()
    pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    emails = re.findall(pattern, content)
print(emails)
