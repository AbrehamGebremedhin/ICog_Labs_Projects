# import re
# from langchain_community.document_loaders import PyPDFLoader


# def convert_pdf_to_text(path: str):
#     loader = PyPDFLoader(path, extract_images=False)
#     pages = str(loader.load_and_split())

#     filename = "output.txt"

#     with open(filename, "w", encoding="utf-8") as file:
#         file.write(pages + "\n")


# convert_pdf_to_text("D:\Projects\\amharic_nanogpt\Driving.pdf")

# def clean_text(path: str):
#     # Read the contents of the file
#     with open(path, "r", encoding="utf-8") as file:
#         content = file.read()

#     no = 0
#     # Loop through numbers from 1 to 468
#     for num in range(1, 469):
#         string_to_remove = f"page: {num}, page_content={no}"

#         # Remove all instances of the string
#         content = content.replace(string_to_remove, "")

#         no += 1

#     # Write the modified content back to the file
#     with open("output_cleaned.txt", "w", encoding="utf-8") as file:
#         file.write(content)

#     print("The specified string has been removed from the file.")


# def replace_letter_in_file(input_file_path, output_file_path):
#     # Read the input file content
#     with open(input_file_path, 'r', encoding='utf-8') as file:
#         content = file.read()

#     # Define the pattern to search for "ዯ" followed by "መቀች"
#     pattern = r'ዯ(?=መቀች)'

#     # Replace "ዯ" with "ደ" where the pattern matches
#     updated_content = re.sub(pattern, 'ደ', content)

#     # Save the updated content to the output file
#     with open(output_file_path, 'w', encoding='utf-8') as file:
#         file.write(updated_content)


# # File paths
# input_file_path = 'D:\Projects\\amharic_nanogpt\\train.txt'
# output_file_path = 'D:\Projects\\amharic_nanogpt\\new_train.txt'

# # Replace the letter in the file
# replace_letter_in_file(input_file_path, output_file_path)

# print("Replacement completed.")
with open("D:\Projects\\amharic_nanogpt\\new_train.txt", "r", encoding="utf-8") as file:
    content = file.read()

print(len(content))

print(sorted(list(set(content))))
# Read the content of the uploaded file

# file_path = 'D:\Projects\\amharic_nanogpt\mawek.txt'


# def clean_text(path: str):
#     # Read the contents of the file
#     with open(path, "r", encoding="utf-8") as file:
#         content = file.read()

#     # Loop through numbers from 1 to 468
#     for num in range(0, 9):
#         string_to_remove = f"እራስን የማወቅ ጉዞ"

#         # Remove all instances of the string
#         content = content.replace(string_to_remove, "")

#     # Write the modified content back to the file
#     with open("output_cleaned.txt", "w", encoding="utf-8") as file:
#         file.write(content)

#     print("The specified string has been removed from the file.")


# clean_text("D:\Projects\\amharic_nanogpt\\train.txt")
