from handyllm import PromptConverter
converter = PromptConverter()

converter.read_substitute_content('./assets/substitute.txt')  # read substitute map

# msgs can be used as the message parameter for OpenAI API
msgs = converter.rawfile2msgs('./assets/prompt.txt')  # variables are substituted according to map
# print(json.dumps(msgs, indent=2))
print(converter.msgs2raw(msgs))
print('-----')

# variables wrapped in %s can be replaced at runtime
new_msgs = converter.msgs_replace_variables(
    msgs, 
    {
        r'%misc1%': 'Note1: do not use any bad word.',
        r'%misc2%': 'Note2: be optimistic.',
    }
)
# print(json.dumps(new_msgs, indent=2))
print(converter.msgs2raw(new_msgs))
