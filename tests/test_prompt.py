import json
from handyllm import PromptConverter
converter = PromptConverter()

converter.read_substitute_content('substitute.txt')  # read substitute map

# chat can be used as the message parameter for OpenAI API
chat = converter.rawfile2chat('prompt.txt')  # variables are substituted according to map
print(json.dumps(chat, indent=2))
print()

# variables wrapped in %s can be replaced at runtime
new_chat = converter.chat_replace_variables(chat, {r'%misc%': 'Note: do not use any bad word.'})
print(json.dumps(new_chat, indent=2))
