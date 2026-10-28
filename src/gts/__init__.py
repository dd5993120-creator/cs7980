"""Generate-then-select track of the CS 7980 project "Enhancing Memory of LLMs".

Modules:
    paths     where data and outputs live
    data      the 9,108-paper sample and the shipped Qwen3-32B greedy run
    prompts   the open-question and multiple-choice prompts
    matching  name matching on the scale of the shipped run
    coverage  coverage@k and the chance baseline
    mc        multiple-choice items: distractors, option orders, controls
    generate  vLLM runners for Colab (imports vllm lazily)
"""
