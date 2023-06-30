import re

class PromptConverter:
    
    def __init__(self):
        self.substitute_map = {}

    def read_substitute_content(self, path: str):
        # 从文本文件读取所有prompt中需要替换的内容
        with open(path, 'r', encoding='utf-8') as fin:
            content = fin.read()
    
        self.substitute_map = {}
        blocks = re.split(r'(%\w+%)', content)
        for idx in range(1, len(blocks), 2):
            key = blocks[idx]
            value = blocks[idx+1]
            self.substitute_map[key] = value.strip()

    def raw2chat(self, raw_prompt: str):
        # substitute pre-defined variables
        for key, value in self.substitute_map.items():
            raw_prompt = raw_prompt.replace(key, value)

        # convert plain text to chat format
        chat = []
        blocks = re.split(r'(\$\w+\$)', raw_prompt)
        for idx in range(1, len(blocks), 2):
            key = blocks[idx]
            value = blocks[idx+1]
            chat.append({"role": key[1:-1], "content": value.strip()})
        
        return chat
    
    def rawfile2chat(self, raw_prompt_path: str):
        with open(raw_prompt_path, 'r', encoding='utf-8') as fin:
            raw_prompt = fin.read()
        
        return self.raw2chat(raw_prompt)
    
    def chat2raw(self, chat):
        # convert chat format to plain text
        messages = []
        for message in chat:
            messages.append(f"${message['role']}$\n{message['content']}")
        raw_prompt = "\n\n".join(messages)
        return raw_prompt
    
    def chat2rawfile(self, chat, raw_prompt_path: str):
        raw_prompt = self.chat2raw(chat)
        with open(raw_prompt_path, 'w', encoding='utf-8') as fout:
            fout.write(raw_prompt)
    
    def chat_replace_variables(self, chat, variable_map: dict, inplace=False):
        # replace every variable in chat content
        if inplace:
            for message in chat:
                for var, value in variable_map.items():
                    if var in message['content']:
                        message['content'] = message['content'].replace(var, value)
            return chat
        else:
            new_chat = []
            for message in chat:
                for var, value in variable_map.items():
                    if var in message['content']:
                        new_message = {"role": message['role'], "content": message['content'].replace(var, value)}
                    else:
                        new_message = {"role": message['role'], "content": message['content']}
                new_chat.append(new_message)
            return new_chat

