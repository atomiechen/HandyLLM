import json
from handyllm import PromptConverter
converter = PromptConverter()

converter.read_substitute_content('substitute.txt')  # read substitute map

# chat can be used as the message parameter for OpenAI API
chat = converter.rawfile2chat('prompt.txt')  # variables are substituted according to map
# print(json.dumps(chat, indent=2))
print(converter.chat2raw(chat))
print('-----')

# variables wrapped in %s can be replaced at runtime
new_chat = converter.chat_replace_variables(
    chat, 
    {
        r'%misc1%': 'Note1: do not use any bad word.',
        r'%misc2%': 'Note2: be optimistic.',
    }
)
# print(json.dumps(new_chat, indent=2))
print(converter.chat2raw(new_chat))
