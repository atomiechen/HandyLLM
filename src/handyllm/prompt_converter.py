import io
import re
from typing import Optional, Tuple
import yaml


class PromptConverter:
    
    role_keys = ['system', 'user', 'assistant', 'tool']

    def __init__(self):
        self.substitute_map = {}

    @property
    def split_pattern(self):
        # build a regex pattern to split the prompt by role keys
        # return r'^\$(' + '|'.join(self.role_keys) + r')\$$'
        return r'^\$(' + '|'.join(self.role_keys) + r')\$[^\S\r\n]*({[^}]*?})?[^\S\r\n]*$'

    def detect(self, raw_prompt: str):
        # detect the role keys in the prompt
        if re.search(self.split_pattern, raw_prompt, flags=re.MULTILINE):
            return True
        return False

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

    def raw2msgs(self, raw_prompt: str):
        # substitute pre-defined variables
        for key, value in self.substitute_map.items():
            raw_prompt = raw_prompt.replace(key, value)

        # convert plain text to chat format
        chat = []
        blocks = re.split(self.split_pattern, raw_prompt, flags=re.MULTILINE)
        for idx in range(1, len(blocks), 3):
            role = blocks[idx]
            extra = blocks[idx+1]
            content = blocks[idx+2]
            if content:
                content = content.strip()
            msg = {"role": role, "content": content}
            if extra:
                # remove curly braces
                key_values_pairs = re.findall(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\')', extra[1:-1])
                # parse extra properties
                extra_properties = {}
                for key, value in key_values_pairs:
                    # remove quotes of the value
                    extra_properties[key] = value[1:-1]
                if 'type' in extra_properties:
                    type_of_msg = extra_properties.pop('type')
                    if type_of_msg == 'tool_calls':
                        msg['tool_calls'] = yaml.safe_load(content)
                        msg['content'] = None
                    elif type_of_msg == 'content_array':
                        # parse content array
                        msg['content'] = yaml.safe_load(content)
                for key in extra_properties:
                    msg[key] = extra_properties[key]
            chat.append(msg)
        
        return chat
    
    def rawfile2msgs(self, raw_prompt_path: str):
        with open(raw_prompt_path, 'r', encoding='utf-8') as fin:
            raw_prompt = fin.read()
        
        return self.raw2msgs(raw_prompt)
    
    @staticmethod
    def msgs2raw(chat):
        # convert chat format to plain text
        messages = []
        for message in chat:
            role = message.get('role')
            content = message.get('content')
            tool_calls = message.get('tool_calls')
            extra_properties = {key: message[key] for key in message if key not in ['role', 'content', 'tool_calls']}
            if tool_calls:
                extra_properties['type'] = 'tool_calls'
                content = yaml.dump(tool_calls)
            elif isinstance(content, list):
                extra_properties['type'] = 'content_array'
                content = yaml.dump(content)
            if extra_properties:
                extra = " {" + " ".join([f'{key}="{extra_properties[key]}"' for key in extra_properties]) + "}"
            else:
                extra = ""
            messages.append(f"${role}${extra}\n{content}")
        raw_prompt = "\n\n".join(messages)
        return raw_prompt

    @staticmethod
    def stream_msgs2raw(gen_sync, fd: Optional[io.IOBase] = None) -> Tuple[str, str]:
        # stream response to fd
        role = ""
        content = ""
        tool_calls = []
        role_completed = False
        for r, text, tool_call in gen_sync:
            if r != role:
                role = r
                if fd:
                    fd.write(f"${role}$")  # do not add newline
            if tool_call:
                if not role_completed:
                    if fd:
                        fd.write(' {type="tool_calls"}\n')
                    role_completed = True
                tool_calls.append(tool_call)  # do not stream, wait for the end
            elif text:
                if not role_completed:
                    if fd:
                        fd.write('\n')
                    role_completed = True
                if fd:
                    fd.write(text)
                content += text
        if tool_calls and fd:
            # dump tool calls
            fd.write(yaml.dump(tool_calls))
        return role, content, tool_calls

    @staticmethod
    async def astream_msgs2raw(gen_async, fd: Optional[io.IOBase] = None) -> Tuple[str, str]:
        # stream response to fd
        role = ""
        content = ""
        tool_calls = []
        role_completed = False
        async for r, text, tool_call in gen_async:
            if r != role:
                role = r
                if fd:
                    fd.write(f"${role}$")  # do not add newline
            if tool_call:
                if not role_completed:
                    if fd:
                        fd.write(' {type="tool_calls"}\n')
                    role_completed = True
                tool_calls.append(tool_call)  # do not stream, wait for the end
            elif text:
                if not role_completed:
                    if fd:
                        fd.write('\n')
                    role_completed = True
                if fd:
                    fd.write(text)
                content += text
        if tool_calls and fd:
            # dump tool calls
            fd.write(yaml.dump(tool_calls))
        return role, content, tool_calls
    
    @classmethod
    def msgs2rawfile(cls, chat, raw_prompt_path: str):
        raw_prompt = cls.msgs2raw(chat)
        with open(raw_prompt_path, 'w', encoding='utf-8') as fout:
            fout.write(raw_prompt)
    
    @staticmethod
    def msgs_replace_variables(chat, variable_map: dict, inplace=False):
        # replace every variable in chat content
        if inplace:
            for message in chat:
                for var, value in variable_map.items():
                    if message.get('content') and var in message['content']:
                        message['content'] = message['content'].replace(var, value)
            return chat
        else:
            new_chat = []
            for message in chat:
                new_message = message.copy()
                for var, value in variable_map.items():
                    if new_message.get('content') and var in new_message['content']:
                        new_message['content'] = new_message['content'].replace(var, value)
                new_chat.append(new_message)
            return new_chat
    
    @staticmethod
    def chat_append_msg(chat, content: str, role: str = 'user', inplace=False):
        if inplace:
            chat.append({"role": role, "content": content})
            return chat
        else:
            new_chat = chat.copy()
            new_chat.append({"role": role, "content": content})
            return new_chat
    
    raw2chat = raw2msgs
    rawfile2chat = rawfile2msgs
    chat2raw = msgs2raw
    chat2rawfile = msgs2rawfile
    chat_replace_variables = msgs_replace_variables
