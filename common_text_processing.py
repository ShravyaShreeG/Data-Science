import pandas as pd
import re
from nltk.corpus import stopwords

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)

#---------------------------------------------------------#
# Full products cleaned
df = pd.read_csv('/cleaned_csv/full_products_merged.csv')
df.rename(columns={"Ingredients_cleaned": "Ingredients"}, inplace=True)
#----------------------------------------------------------#

# Remove NA
df = df.dropna(subset=['Ingredients'])
# Remove if empty string
df = df[df['Ingredients'].str.strip() != ""]
# # print(df)

# Assign to a list of strings
ingredients_data = df['Ingredients']

# Lower case all text
ingredients_data = [re.sub(r"[A-Z]", lambda x: x.group(0).lower(), text) for text in ingredients_data]

# # new line to commas as it could indicate new product
ingredients_data = [re.sub(r"\n", ",", text) for text in ingredients_data]

# Replace "and" and "also" by ","
ingredients_data = [re.sub(r'\b(?:and|also)\b', ', ', text) for text in ingredients_data]

# Remove texts within parenthesis
ingredients_data = [re.sub(r"\([^()]*\)", "", text) for text in ingredients_data]

# Remove all special characters and numbers
ingredients_data = [re.sub(r"[^a-zA-Z,\s]", " ", text) for text in ingredients_data]

# Remove single letters
ingredients_data = [re.sub(r"\b\w\b", "", text) for text in ingredients_data]

# Remove stop words
stop_words = set(stopwords.words("english"))
ingredients_data = [" ".join(filter(lambda word: word.lower() not in stop_words, text.split())) for text in ingredients_data]

# Remove common words
common_words = pd.read_excel("/text_processing/words_to_remove.xlsx")
words_to_remove = common_words['words_to_remove']
pattern = r'\b(?:' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
ingredients_data = [re.sub(pattern, '', text) for text in ingredients_data]

def remove_numeric_words(strings):
    pattern = r'\b(?:one|two|three|four|five|six|seven|eight|nine)\b'
    return [re.sub(pattern, '', string, flags=re.IGNORECASE) for string in strings]


ingredients_data = remove_numeric_words(ingredients_data)

# if end result is of len 1, remove it
ingredients_data = [text if len(text) >= 2 else "" for text in ingredients_data]

# remove if there consecutive spaces
ingredients_data = [re.sub(r'\s+', " ", text) for text in ingredients_data]

# remove spaces between commas
ingredients_data = [re.sub(r',\s*,', " ", text) for text in ingredients_data]

# remove consecutive commas and replace with single comma
ingredients_data = [re.sub(r',+', ",", text) for text in ingredients_data]

# remove if there consecutive spaces
ingredients_data = [re.sub(r'\s+', " ", text) for text in ingredients_data]

#------------------------------------------------------------------------#

ingredients_data = [re.sub(r'\s*,\s*', ",", text) for text in ingredients_data]

# remove leading spaces followed by comma or trailing spaces after commas at the end of the string
ingredients_data = [re.sub(r'^\s*,|,\s*$', "", text) for text in ingredients_data]

for text in ingredients_data:
    a = 1
    print(text)

# df['Ingredients_cleaned'] = ingredients_data
df['Ingredients_cleaned'] = ingredients_data
# print(df)

df.to_csv('full_products_cleaned.csv', index=False)
#--------------------------------------------------------------------------#


