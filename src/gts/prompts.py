"""Prompts, verbatim from the Dreaming Up Authors release.

Open question: the Section 7 Qwen3-32B prompt (dua.probing.stages.stage1_generate,
SYSTEM_MSG and _user_text), with Qwen3's chat template and thinking off. Using
it unchanged makes the shipped greedy run our default-answer baseline.

Multiple choice: config/prompts/mc_recognition.txt, the Section 6 template.
"""

from __future__ import annotations

OPEN_SYSTEM = (
    "You are a knowledgeable assistant. Answer factually and concisely. "
    "When asked for authors, return a semicolon separated list of full names only, "
    "with no additional text."
)


def open_user_text(title: str, year) -> str:
    return (
        f"Who are the authors of the paper titled '{title}' "
        f"which was published in {year}? "
        "Return a semicolon separated list of author names only."
    )


MC_TEMPLATE = """You are an academic assistant. Your task is to identify the correct author list for an academic paper. You will be given a paper title, its publication year, and four candidate author lists labelled A, B, C, and D. Exactly one is correct.

- Respond with a single letter: A, B, C, or D.
- Do not add any explanation, notes, or extra content.

Here are some examples:

Question: Which of the following is the author list of the paper titled 'Why men are at higher risk for hepatocellular carcinoma?' which was published in 2012?
A: V. Keng, D. Largaespada, A. Villanueva
B: M. Sherman, J. Bruix, A. Forner
C: J. Llovet, S. Ricci, V. Mazzaferro
D: H. El-Serag, K. Rudolph, A. Marrero
Answer: A

Question: Which of the following is the author list of the paper titled 'Research as an influence on teaching' which was published in 1984?
A: J. Centra
B: K. R. Jolls
C: P. Ramsden
D: M. Prosser
Answer: B

Now, answer the following question:
Question: Which of the following is the author list of the paper titled '{title}' which was published in {year}?
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
Answer:
"""

OPTION_LABELS = ("A", "B", "C", "D")

# Shown in place of the title in the title-hidden control.
HIDDEN_TITLE = "[title not shown]"


def mc_user_text(title: str, year, options: list[list[str]]) -> str:
    """The Section 6 question with four author lists, each joined with ", "."""
    if len(options) != 4:
        raise ValueError(f"expected 4 options, got {len(options)}")
    return MC_TEMPLATE.format(
        title=title, year=year,
        **{f"option_{label.lower()}": ", ".join(opt) for label, opt in zip(OPTION_LABELS, options)},
    )


def chat_prompt(tokenizer, user: str, system: str | None = None, thinking: bool = False) -> str:
    """Qwen3 chat template; thinking off by default, as stage1_generate built it."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )


def open_prompt(tokenizer, title: str, year, thinking: bool = False) -> str:
    return chat_prompt(tokenizer, open_user_text(title, year), OPEN_SYSTEM, thinking=thinking)


def split_thinking(text: str) -> tuple[str, str, bool]:
    """(reasoning, answer, finished) for a thinking-mode output.

    "</think>" is an ordinary token in Qwen3's vocabulary, so it survives
    detokenisation. Without it the model ran out of tokens mid-reasoning:
    there is no answer, and the reasoning must not be scored as one.
    """
    if "</think>" not in text:
        return text, "", False
    reasoning, answer = text.split("</think>", 1)
    return reasoning.replace("<think>", "").strip(), answer.strip(), True


# Joint multiple choice over the model's own top candidates (selection, not measurement).
CAND_MC_TEMPLATE = """You are an academic assistant. Your task is to identify an author of an academic paper. You will be given a paper title, its publication year, and four candidate names labelled A, B, C, and D, followed by option E. Choose a candidate who is an author of the paper. If none of the four is an author, choose E.

- Respond with a single letter: A, B, C, D, or E.
- Do not add any explanation, notes, or extra content.

Here are some examples:

Question: Who among the following is an author of the paper titled 'Why men are at higher risk for hepatocellular carcinoma?' which was published in 2012?
A: M. Sherman
B: D. Largaespada
C: J. Llovet
D: H. El-Serag
E: None of these
Answer: B

Question: Who among the following is an author of the paper titled 'Research as an influence on teaching' which was published in 1984?
A: J. Centra
B: P. Ramsden
C: M. Prosser
D: R. Barnett
E: None of these
Answer: E

Now, answer the following question:
Question: Who among the following is an author of the paper titled '{title}' which was published in {year}?
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
E: None of these
Answer:
"""

CAND_LABELS = ("A", "B", "C", "D", "E")


def cand_mc_user_text(title: str, year, names: list[str]) -> str:
    if len(names) != 4:
        raise ValueError(f"expected 4 candidate names, got {len(names)}")
    return CAND_MC_TEMPLATE.format(
        title=title, year=year,
        **{f"option_{label.lower()}": name for label, name in zip(OPTION_LABELS, names)},
    )


# Catch trials: the Section 6 question with "None of the above" as one of the four
# options. One example has the true list among the options, the other does not
# (the second example's true author, K. R. Jolls, is left out), so the examples
# favour neither answer.
NONE_OPTION = "None of the above"

MC_NONE_TEMPLATE = """You are an academic assistant. Your task is to identify the correct author list for an academic paper. You will be given a paper title, its publication year, and four options labelled A, B, C, and D. One option is "None of the above": choose it if none of the listed author lists is correct.

- Respond with a single letter: A, B, C, or D.
- Do not add any explanation, notes, or extra content.

Here are some examples:

Question: Which of the following is the author list of the paper titled 'Why men are at higher risk for hepatocellular carcinoma?' which was published in 2012?
A: V. Keng, D. Largaespada, A. Villanueva
B: M. Sherman, J. Bruix, A. Forner
C: None of the above
D: H. El-Serag, K. Rudolph, A. Marrero
Answer: A

Question: Which of the following is the author list of the paper titled 'Research as an influence on teaching' which was published in 1984?
A: J. Centra
B: None of the above
C: P. Ramsden
D: M. Prosser
Answer: B

Now, answer the following question:
Question: Which of the following is the author list of the paper titled '{title}' which was published in {year}?
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
Answer:
"""


def mc_none_user_text(title: str, year, options: list) -> str:
    """Four options, each an author list or NONE_OPTION."""
    if len(options) != 4:
        raise ValueError(f"expected 4 options, got {len(options)}")
    return MC_NONE_TEMPLATE.format(
        title=title, year=year,
        **{f"option_{label.lower()}": opt if isinstance(opt, str) else ", ".join(opt)
           for label, opt in zip(OPTION_LABELS, options)},
    )


# Yes/no recognition: one author list at a time (signal detection, d').
YESNO_SYSTEM = (
    "You are a knowledgeable assistant. Answer factually and concisely. "
    "Answer with a single word: Yes or No."
)


def yesno_list_user_text(names: list[str], title: str, year) -> str:
    return (f"Is {', '.join(names)} the author list of the paper titled '{title}' "
            f"which was published in {year}? Answer Yes or No.")


# Likelihood scoring: the log-probability of a continuation after a prefix.
#   list    an author list as the answer to the open question; with the title
#           hidden, the same list's prior, so their difference is a PMI
#   author  one name as the answer to "name one author", with or without the title
#   title   the title after a reference-list author and year (authors -> title)
ONE_AUTHOR_SYSTEM = (
    "You are a knowledgeable assistant. Answer factually and concisely. "
    "When asked for an author, return one full name only, with no additional text."
)
END_OF_TURN = "<|im_end|>"
REFERENCE_HEADER = "References\n\n"


def one_author_user_text(title: str | None, year) -> str:
    if title is None:
        return f"Name one author of a paper which was published in {year}. Return the name only."
    return (f"Name one author of the paper titled '{title}' which was published in {year}. "
            "Return the name only.")


def reference_prefix(author: str | None, year) -> str:
    """A reference-list entry up to the title: "Shi, P. et al. (2014)."; "Anonymous (2014)." without an author."""
    return f"{REFERENCE_HEADER}{author} et al. ({year})." if author else f"{REFERENCE_HEADER}Anonymous ({year})."


def score_texts(rec: dict, tokenizer) -> tuple[str, str]:
    """(prefix, continuation) for one scoring record; only the continuation is scored.

    Chat prefixes end with the empty thinking block, so the continuation sits where
    the model's answer would start; reference prefixes end before the space that
    BPE attaches to the title's first word.
    """
    kind = rec["kind"]
    if kind == "list":
        return open_prompt(tokenizer, rec["title"], rec["year"]), "; ".join(rec["names"]) + END_OF_TURN
    if kind == "author":
        return (chat_prompt(tokenizer, one_author_user_text(rec.get("title"), rec["year"]), ONE_AUTHOR_SYSTEM),
                rec["name"] + END_OF_TURN)
    if kind == "title":
        return reference_prefix(rec.get("author"), rec["year"]), f" {rec['title']}."
    raise ValueError(f"unknown scoring record kind {kind!r}")
