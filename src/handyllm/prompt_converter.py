__all__ = [
    'PromptConverter'
]

import re
from typing import IO, Generator, MutableMapping, MutableSequence, Optional
import yaml

from .types import PathType, ShortChatChunk


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

    def read_substitute_content(self, path: PathType):
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

        # convert plain text to messages format
        msgs = []
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
            msgs.append(msg)
        
        return msgs
    
    def rawfile2msgs(self, raw_prompt_path: PathType):
        with open(raw_prompt_path, 'r', encoding='utf-8') as fin:
            raw_prompt = fin.read()
        
        return self.raw2msgs(raw_prompt)
    
    @staticmethod
    def msgs2raw(msgs):
        # convert messages format to plain text
        messages = []
        for message in msgs:
            role = message.get('role')
            content = message.get('content')
            tool_calls = message.get('tool_calls')
            extra_properties = {key: message[key] for key in message if key not in ['role', 'content', 'tool_calls']}
            if tool_calls:
                extra_properties['type'] = 'tool_calls'
                content = yaml.dump(tool_calls, allow_unicode=True)
            elif isinstance(content, MutableSequence):
                extra_properties['type'] = 'content_array'
                content = yaml.dump(content, allow_unicode=True)
            if extra_properties:
                extra = " {" + " ".join([f'{key}="{extra_properties[key]}"' for key in extra_properties]) + "}"
            else:
                extra = ""
            messages.append(f"${role}${extra}\n{content}")
        raw_prompt = "\n\n".join(messages)
        return raw_prompt

    @staticmethod
    def consume_stream2fd(fd: IO[str]) -> Generator[Optional[ShortChatChunk], ShortChatChunk, None]:
        # stream response to fd
        role = ""
        role_completed = False
        data = None
        while True:
            data = yield data
            r, text, tool_call = data
            if r != role:
                role = r
                fd.write(f"${role}$")  # do not add newline
            if tool_call:
                if not role_completed:
                    fd.write(' {type="tool_calls"}\n')
                    role_completed = True
                # dump tool calls
                fd.write(yaml.dump([tool_call], allow_unicode=True))
            elif text:
                if not role_completed:
                    fd.write('\n')
                    role_completed = True
                fd.write(text)
    
    @classmethod
    def msgs2rawfile(cls, msgs, raw_prompt_path: PathType):
        raw_prompt = cls.msgs2raw(msgs)
        with open(raw_prompt_path, 'w', encoding='utf-8') as fout:
            fout.write(raw_prompt)
    
    @classmethod
    def msgs_replace_variables(cls, msgs, variable_map: MutableMapping, inplace=False):
        # replace every variable in messages content
        if inplace:
            for message in msgs:
                content = message.get('content')
                if content:
                    message['content'] = cls._replace_deep(content, variable_map)
            return msgs
        else:
            new_msgs = []
            for message in msgs:
                new_message = message.copy()
                new_msgs.append(new_message)
                content = new_message.get('content')
                if content:
                    new_message['content'] = cls._replace_deep(content, variable_map)
            return new_msgs
    
    @classmethod
    def _replace_deep(cls, content, variable_map: MutableMapping):
        if isinstance(content, str):
            for var, value in variable_map.items():
                if var in content:
                    content = content.replace(var, value)
        elif isinstance(content, MutableMapping):
            for key, value in content.items():
                content[key] = cls._replace_deep(value, variable_map)
        elif isinstance(content, MutableSequence):
            for idx, value in enumerate(content):
                content[idx] = cls._replace_deep(value, variable_map)
        return content
    
    raw2chat = raw2msgs
    rawfile2chat = rawfile2msgs
    chat2raw = msgs2raw
    chat2rawfile = msgs2rawfile
    chat_replace_variables = msgs_replace_variables
