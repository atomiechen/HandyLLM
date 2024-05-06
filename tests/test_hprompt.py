from pathlib import Path

from handyllm import hprompt


cur_dir = Path(__file__).parent

# load hprompt
prompt: hprompt.ChatPrompt = hprompt.load_from(cur_dir / './assets/magic.hprompt')
print(prompt.data)
print(prompt)
print(repr(prompt))

# run hprompt
result_prompt = prompt.run()
print(result_prompt.result_str)

# dump result hprompt
result_prompt.dump_to(cur_dir / './assets/tmp_out.hprompt')

# chain result hprompt
prompt += result_prompt
# chain another hprompt
prompt += hprompt.load_from(cur_dir / './assets/magic.hprompt')
# run again
result2 = prompt.run()


