# Talk script: Generate-then-Select results (Minyan Xu)

Six slides, 931 words: about 7 minutes at 140 words a minute. The same text is in each slide’s speaker notes.

## 01 Setup: Sampling 50 Author Lists for Every Paper

*About 1 min*

I’ll start with what I ran. For all 9,108 papers I gave Qwen3-32B the title and year, with no search and no retrieval, and asked for the author list fifty times, at temperature one with no truncation. That is 455,400 author lists in total. Each name is matched with the paper’s own matcher, last name plus first initial, and pooled with a count per paper.

Before trusting any of it, I checked that my setup is the paper’s setup. My default answers reproduce the paper’s own Qwen3-32B run: 1,211 versus 1,202 true authors named, the same 222 papers mostly named, and the same mean HR2.

One trap: with fifty lists, common surnames match by luck. So every count I show is net of a chance baseline, either the authors of random papers from the same field and citation bin, or, stricter, the authors of the three most similar titles. Only matches above the baseline count.

## 02 Recognition: The Recognition Test Leaks; Own Wrong Lists Are Harder

*About 1.25 min*

Multiple choice is our shared measurement: it decides which papers count as known. So before using it, I tested how much it can be passed without knowing the paper.

The blue bars are the normal test, the orange bars hide the title, and the grey bars show another paper’s title. With the team’s design, real teams of other papers, 70% of papers pass. But with the title hidden, 24% still pass, and even with a wrong title 16% do. The model keeps picking the same list in every order, so the real floor is about 16%, not the 5% that pure guessing would give.

Most of the hidden-title advantage is a year cue: when I draw the wrong teams from papers of the same years, it falls from 24% to 18%, close to the floor.

Finally, my own harder test: the wrong options are the model’s own most frequent wrong answers. Known drops from 70% to 41%. Of the 6,179 known papers, 2,009 become unknown. The model often trusts its own invented lists more than the truth.

## 03 Failure split: Known Papers: The Right Author Is Often in the Samples

*About 1 min*

This is the failure split: when the default answer misses an author, was it never generated, or generated but not chosen?

On the known papers, the default answer names 11 true authors per 100 papers. But across the fifty samples, another 64 per 100 show up, after subtracting the topic baseline. 42% of known papers have at least one such author. For unknown papers it is only 13 per 100, five times less, which also tells us the recognition label does carry signal.

On the right, by citation bin: the grey bars are the default answer, the blue bars what the samples add. It grows with citations, but even the least-cited papers are above the baseline.

And this is not saturated. On a 1,023-paper subset, 50, 100 and 200 samples reach 24, 30 and 36% of true authors. So 'never generated' always has to say at how many samples.

## 04 Selection: Counting Beats Asking: Frequency Is the Selector to Beat

*About 1.25 min*

If the right author is in the samples, can we pick it out? I compared three ways to select.

First, simply counting: how many of the fifty samples name a candidate. True candidates beat false ones of the same paper 74% of the time.

Second, asking the model directly, 'is X an author?', which is P(True). That is worse than counting, 61%: the model’s yes or no adds nothing beyond how often it writes the name.

Third, a joint choice among the four most frequent candidates plus 'none of these'. It is a little better than taking the most frequent one, by three points, but it rarely says 'none' when none is true.

On the right is the held-out result. I chose the rule on the tune split and scored the test split once. Keeping names that appear in at least seven of the fifty samples names 29% more true authors than the default answer and invents 80% fewer names.

## 05 Equal compute: At Equal Compute, Counting Matches Thinking

*About 1 min*

The natural rival is letting the model think first. Thinking costs tokens, so I compare at equal compute. One thinking answer costs about 650 tokens, the same as twenty ordinary samples.

On the left, test split: thinking mode names 15.4 true authors per 100 papers; counting over twenty samples names 18.6, and invents 155 names per 100 instead of 331.

Paired by paper, mean HR2 is not significantly different, but counting names 3.2 more true authors per 100 papers and invents 176 fewer names.

With a bigger budget there is a real trade-off: voting over five thinking answers gives the best HR2 and the fewest invented names, while a hundred ordinary samples name more true authors. Thinking makes each answer surer; sampling covers more.

## 06 Summary: In the Steering Unit: What Selection Adds on Known Papers

*About 1.25 min*

To close, the same unit as the steering results: true authors named per 100 known papers, here on 4,271 known test papers with our labels.

The default answer names 11.3. Thinking mode adds 5.2. Keeping names that appear in at least three of twenty samples adds 8, at the same cost as thinking, and invents about a third as many names as the default answer.

For reference, steering at layer 24 adds 0.7 to 3 over random controls, but on the team’s 566 known tune papers and labels, so this is not yet a direct comparison. The first next step is exactly that: rerun on split V3’s known tune papers.

Then I want more coverage per token, by asking for authors not yet named or for the first author alone, and a better selector that combines counting with the candidate choice. The point for the team: generate-then-select needs no model internals, and it sets the bar any white-box method should beat at equal compute. Thank you.
