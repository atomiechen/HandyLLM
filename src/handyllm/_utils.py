import collections.abc
import copy
import json
from urllib.parse import quote_plus
import time
import inspect

from ._constants import _API_TYPES_AZURE
from .prompt_converter import PromptConverter


def get_request_url(request_url, api_type, api_version, engine):
    if api_type and api_type.lower() in _API_TYPES_AZURE:
        if api_version is None:
            raise Exception("api_version is required for Azure OpenAI API")
        if engine is None:
            return f'/openai/deployments?api-version={api_version}'
        else:
            return f'/openai/deployments/{quote_plus(engine)}{request_url}?api-version={api_version}'
    else:
        if engine is not None:
            return f'/engines/{quote_plus(engine)}{request_url}'
    return request_url

def join_url(base_url, *args):
    url = base_url.rstrip('/')
    for arg in args:
        url += '/' + arg.lstrip('/')
    return url

def isiterable(arg):
    return (
        isinstance(arg, collections.abc.Iterable) 
        and not isinstance(arg, str)
    )

def wrap_log_input(input_content: str, log_marks, kwargs):
    arguments = copy.deepcopy(kwargs)
    # check if log_marks is iterable
    if isiterable(log_marks):
        input_lines = [str(item) for item in log_marks]
    else:
        input_lines = [str(log_marks)]
    input_lines.append(json.dumps(arguments, indent=2, ensure_ascii=False))
    input_lines.append(" INPUT START ".center(50, '-'))
    input_lines.append(input_content)
    input_lines.append(" INPUT END ".center(50, '-')+"\n")
    input_str = "\n".join(input_lines)
    return input_str

def log_result(logger, tag: str, duration: float, log_marks, kwargs, input_content: str, output_content: str):
    input_str = wrap_log_input(input_content, log_marks, kwargs)
    ## log this on result
    log_strs = []
    log_strs.append(f"{tag} result ({duration:.2f}s)")
    log_strs.append(input_str)
    log_strs.append(" OUTPUT START ".center(50, '-'))
    log_strs.append(output_content)
    log_strs.append(" OUTPUT END ".center(50, '-')+"\n")
    logger.info('\n'.join(log_strs))

def log_exception(logger, tag: str, duration: float, log_marks, kwargs, input_content: str, err_msg: str):
    input_str = wrap_log_input(input_content, log_marks, kwargs)
    ## log this on exception
    log_strs = []
    log_strs.append(f"{tag} error ({duration:.2f}s)")
    log_strs.append(input_str)
    log_strs.append(" EXCEPTION START ".center(50, '-'))
    log_strs.append(err_msg)
    log_strs.append(" EXCEPTION END ".center(50, '-')+"\n")
    logger.error('\n'.join(log_strs))

def exception2err_msg(exception: Exception):
    err_msg = f"Exception: {type(exception).__module__}.{type(exception).__name__}"
    err_msg += f"\nDetailed info: {repr(exception)}"
    if exception.args:
        print(f"\nException arguments: {exception.args}")
    return err_msg

def _chat_log_response_final(logger, log_marks, kwargs, messages, start_time, role, content, err_msg=None):
    end_time = time.perf_counter()
    duration = end_time - start_time
    input_content = PromptConverter.chat2raw(messages)
    if not err_msg:
        output_content = PromptConverter.chat2raw([{'role': role, 'content': content}])
        log_result(logger, "Chat request", duration, log_marks, kwargs, input_content, output_content)
    else:
        log_exception(logger, "Chat request", duration, log_marks, kwargs, input_content, err_msg)

def _chat_log_response(logger, log_marks, kwargs, messages, start_time, response, stream):
    if logger is not None:
        if stream:
            if inspect.isasyncgen(response):
                async def wrapper(response):
                    content = ''
                    role = ''
                    async for data in response:
                        try:
                            message = data['choices'][0]['delta']
                            if 'role' in message:
                                role = message['role']
                            if 'content' in message:
                                content += message['content']
                        except (KeyError, IndexError):
                            pass
                        yield data
                    _chat_log_response_final(logger, log_marks, kwargs, messages, start_time, role, content)
            elif inspect.isgenerator(response):
                def wrapper(response):
                    content = ''
                    role = ''
                    for data in response:
                        try:
                            message = data['choices'][0]['delta']
                            if 'role' in message:
                                role = message['role']
                            if 'content' in message:
                                content += message['content']
                        except (KeyError, IndexError):
                            pass
                        yield data
                    _chat_log_response_final(logger, log_marks, kwargs, messages, start_time, role, content)
            else:
                raise Exception("response is not a generator or async generator in stream mode")
            response = wrapper(response)
        else:
            role = content = err_msg = None
            try:
                role = response['choices'][0]['message']['role']
                content = response['choices'][0]['message']['content']
            except (KeyError, IndexError):
                err_msg = "Wrong response format, no message found"
            _chat_log_response_final(logger, log_marks, kwargs, messages, start_time, role, content, err_msg)
    return response

def _chat_log_exception(logger, log_marks, kwargs, messages, start_time, exception: Exception):
    if logger is not None:
        end_time = time.perf_counter()
        duration = end_time - start_time
        input_content = PromptConverter.chat2raw(messages)
        err_msg = exception2err_msg(exception)
        log_exception(logger, "Chat request", duration, log_marks, kwargs, input_content, err_msg)

def _completions_log_response_final(logger, log_marks, kwargs, prompt, start_time, text, err_msg=None):
    end_time = time.perf_counter()
    duration = end_time - start_time
    input_content = prompt
    if not err_msg:
        output_content = text
        log_result(logger, "Completions request", duration, log_marks, kwargs, input_content, output_content)
    else:
        log_exception(logger, "Completions request", duration, log_marks, kwargs, input_content, err_msg)

def _completions_log_response(logger, log_marks, kwargs, prompt, start_time, response, stream):
    if logger is not None:
        if stream:
            if inspect.isasyncgen(response):
                async def wrapper(response):
                    text = ''
                    async for data in response:
                        try:
                            text += data['choices'][0]['text']
                        except (KeyError, IndexError):
                            pass
                        yield data
                    _completions_log_response_final(logger, log_marks, kwargs, prompt, start_time, text)
            elif inspect.isgenerator(response):
                def wrapper(response):
                    text = ''
                    for data in response:
                        try:
                            text += data['choices'][0]['text']
                        except (KeyError, IndexError):
                            pass
                        yield data
                    _completions_log_response_final(logger, log_marks, kwargs, prompt, start_time, text)
            else:
                raise Exception("response is not a generator or async generator in stream mode")
            response = wrapper(response)
        else:
            text = err_msg = None
            try:
                text = response['choices'][0]['text']
            except (KeyError, IndexError):
                err_msg = "Wrong response format, no text found"
            _completions_log_response_final(logger, log_marks, kwargs, prompt, start_time, text, err_msg)
    return response

def _completions_log_exception(logger, log_marks, kwargs, prompt, start_time, exception: Exception):
    if logger is not None:
        end_time = time.perf_counter()
        duration = end_time - start_time
        input_content = prompt
        err_msg = exception2err_msg(exception)
        log_exception(logger, "Completions request", duration, log_marks, kwargs, input_content, err_msg)

