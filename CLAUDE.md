# CLAUDE.md: Generate-then-Select track (Minyan Xu)

CS 7980 Research Capstone (Northeastern), team project "Enhancing Memory of LLMs".
Instructor and advisor: Yifan Hu. Teammates: Hana Feng (activation steering), Zhiqi Zhang (J-Lens).
The claude.ai history behind this project is in docs/chat/ (part 1 verbatim, part 2 summarised).

## How to work
- Reply to the user in Chinese. Write code, comments, slides and paper text in English.
- Never invent or estimate results. If a run is unfinished, say so.
- Check every new reference before adding it; the course penalises hallucinated citations.
- Shared LaTeX (docs/paper/sigconf.tex): never change teammates' text; append BibTeX at the end of references.bib.
- Never ask for or use the user's Google password. Colab runs go through the official Colab CLI after the user signs in.

## Decisions (2026-10-25)
- Work independently of Hana for now: no team labels, splits or Step 1 outputs. We rebuild the labels with the team's rules.
- Data and code come from the Dreaming Up Authors release (local copy: C:\Users\asc\Downloads\dreaming-up-authors-code-v0 1).
  Copied here: data/release/eval_sample_9108.csv.gz (9,108 papers) and qwen3_32b_greedy_scored.tsv.gz
  (the paper's own Qwen3-32B run: bf16, thinking off, greedy, max 96 new tokens).
- Model: Qwen3-32B through vLLM on a Colab A100. bf16 when the GPU has 80 GB and the disk has room for 61 GiB of
  weights; otherwise the official FP8 checkpoint (weight-only on Ampere). The run metadata records which.
- Open question: the paper's Section 7 Qwen prompt, verbatim (gts/prompts.py): zero-shot system + user message,
  semicolon-separated names, thinking off. So the shipped greedy run is a ready-made default-answer baseline.
- Sampling: 50 lists per paper, T=1.0, top_p=1.0, no top_k, max 96 new tokens, a fixed seed per paper.
  No truncation, so rare true authors are not cut from the distribution.
- Matching: the vendored matcher (gts/dua_matcher.py = the release's scholar_utils.py) with nameparser==1.1.3 and
  Unidecode==1.3.8. tests/test_reproduce_shipped.py reproduces all 9,108 shipped (M, H, U) rows exactly; newer
  nameparser changes 2 of them.
- Multiple choice: the paper's mc_recognition prompt; 4 cyclic option orders; every name rendered "F. Lastname" in ASCII;
  the answer is read from the first token's log-probabilities over A-D.
  - real_team: three teams of other papers, same field and author count (built: data/mc_items/real_team.jsonl.gz).
  - own_sample: wrong lists the model sampled for the paper (to build after the sampling run).
  - Controls on a stratified subset (up to 10 papers per field x bin, 1,023 papers): title hidden, title swapped.
- Labels (team rules, our data): recalled = greedy HR2 <= 0.5; otherwise known = right in >= 3 of 4 orders,
  notsure = 2, unknown = 0-1. Pure guessing reaches >= 3 of 4 for about 5% of papers.
- Unit for comparison with steering: true authors named per 100 known papers.

## Layout
- src/gts/: data, prompts, matching, mc (items), generate (vLLM runners), dua_matcher (vendored).
- tests/: pytest; run `.venv/Scripts/python -m pytest -q` (venv uses system site-packages plus the pins above).
- tools/: build_real_team_items.py, make_colab_bundle.py (writes colab/gts_bundle.zip and the notebook).
- colab/: gts_bundle.zip, 01_generate_and_mc.ipynb (manual route), remote_steps.py (CLI route).
- data/outputs/: results downloaded from Colab (greedy/, samples/, mc_real_team/, run_meta.json).
- docs/: chat history, paper drafts, plans, slides from the claude.ai phase.
- tools/slides/build_update2.js: Minyan's Update 2 slides and talk script (docs/slides/Generate_then_Select_Update2.*).
- Session 3 (2026-10-28):
  - tools/fetch_s2.py: Semantic Scholar authors, h-indexes and collaborators, written to data/external/.
  - tools/build_session3_items.py: the session's items.
  - tools/describe_session3_items.py: how the distractors came out.
  - tools/analyze_session3.py: the session's tables.
  - colab/session3.py: the steps run on Colab.

## Slides
- Font: Arial, the user's choice (as in their 2026-10-05 Generate_then_Select_Slides.pptx). The team's Google Slides
  deck uses Noto Sans KR, which this PC lacks: WPS then draws a handwriting font, and PDF export goes wrong too.
- The user opens decks in WPS. Check the real file, not a copy in another font: render it with WPS COM
  (KWPP.Application, Slides.Item(i).Export to PNG), save a PDF with SaveAs(path, 32), and list the PDF's fonts with
  PyMuPDF. Only ArialMT and Arial-BoldMT should appear.
- pptxgenjs gives chart text a Latin font only, so WPS draws ± and ’ in chart labels in DengXian; writeDeck() in the
  build adds the East Asian and complex-script fonts to the chart XML.

## Colab CLI
- Installed in .venv-colab (`.venv-colab/Scripts/colab.exe`); officially Linux/macOS only, but works on Windows here.
- Auth is oauth2 with a pasted code; the token is cached under ~/.config/colab-cli/. `colab skill` prints the agent guide.
- In Git Bash set `MSYS_NO_PATHCONV=1`, or remote paths like /content get rewritten to Windows paths.
- `colab exec` waits 30 s by default (`--timeout`); on timeout the kernel keeps running, BUT Colab deletes the runtime
  about 10 minutes after the last client disconnects, even mid-execution (lost two VMs this way on 2026-10-26; see
  github.com/googlecolab/colabtools/issues/5950). So run every long step with a client that stays connected:
  `echo 'step()' | colab exec -s gts --timeout 7000` as a background task (steps print an "alive" line every minute),
  and run tools/mirror_remote.sh <run_dir> alongside, which downloads each finished chunk, so a lost VM costs one chunk.
  To resume on a new VM, upload the mirrored chunks to /content/outputs/<dir>/ first; the runners skip existing chunks.
- GPUs on this account: A100 (40 GB standard; the 80 GB high-mem shape returned 503 on 2026-10-26), L4, T4, and
  G4 = RTX PRO 6000 Blackwell, 96 GB, 8.9 CU/h. H100 is not entitled. Use G4: bf16 fits, with HF_HOME on /dev/shm
  (87 GB) because the 113 GB disk is too small next to vLLM. vLLM 0.32.0, torch 2.13 (CUDA 13.0) installed fine.
- Route: `colab new -s gts --gpu G4`, `colab install -s gts vllm nameparser==1.1.3 Unidecode==1.3.8`, upload
  colab/gts_bundle.zip to /content, `colab exec -f colab/remote_steps.py`, then load(), greedy_step(),
  samples_then_real_team(); build own_sample and ptrue items locally from the downloaded samples, upload them, then
  second_half(); download /content/outputs.tar.gz; always `colab stop -s gts` at the end.
- Lessons from session 3:
  - A background Bash command is capped at 2 hours. Chain steps so that no single launch runs longer.
  - To end a running chain early, overwrite its next step file with a no-op, then relaunch the rest with a fresh
    budget. Restart tools/mirror_remote.sh the same way.
  - `colab upload` works while the kernel is busy. So a keep-alive exec can wait for a marker file while items
    are still being built locally, and the VM is not dropped.
  - Prompt log-probabilities (run_score) go at about 5,500 prompt tokens/s on G4; prefix caching does not apply
    to them.
- data/outputs/fp8_40gb/ holds an earlier greedy run (FP8 on a 40 GB A100): aggregate scores match the paper's bf16 run
  (1,224 vs 1,202 true authors named; 220 vs 222 recalled), but only 24% of answers are identical (52% for bf16 on G4).

## Deadlines
Update 2 on 11/9, Update 3 on 11/16, draft paper on 11/30, final talk on 12/14 (Update 1 was 10/26).

## Status (2026-10-26)
Session 1 is complete: data/outputs/g4_bf16/ (bf16 on G4; see its run_meta.json and analysis/summary.md,
analysis/selection_summary.md). Cost 26.2 compute units including two lost VMs; balance 90.91.
- Default answers match the paper's bf16 run (1,211 vs 1,202 true authors named; 222 vs 222 recalled).
- Reachable true authors per 100 papers, net of the topic-matched baseline: 48.4 overall, 64.5 on real-team known
  papers, 12.9 on unknown; rises with citations (11.5 at 0-2 citations, 139 at 4097+).
- Real-team MC: >= 3 of 4 orders for 70.3% of papers; controls 23.8% (title hidden) and 15.7% (title swapped).
  The swapped control, not 5%, is the false-positive floor: consistent option preferences inflate >= 3 of 4.
- Own-sample MC (the model's own wrong lists): 41.2%; controls about the same as real-team.
- P(True) separates true from false candidates worse than frequency (pairwise AUC 0.606 vs 0.740). Frequency
  thresholds beat the default answer (e.g. names in >= 5 of 50 samples: 20.8 true and 185 false names per 100
  papers, versus 13.3 and 514), at 50x the generation cost; thresholds were fixed, not tuned.

Session 2 is complete too (data/outputs/s2/; analysis in data/outputs/g4_bf16/analysis/). RESULTS.md at the
root is the consolidated write-up. Compute units left: 60.50.
- Split v1 (data/splits/split_v1.csv): 2,734 tune / 6,374 test. Test numbers in RESULTS.md were scored once.
- Year-matched real teams: title-hidden control falls from 0.238 to 0.176 (floor 0.158).
- Candidate MC beats most-frequent by +3.0 points [+0.2, +5.8]; it rarely rejects all-wrong pools.
- At about 650 tokens, frequency over 20 samples ties CoT on HR2 and names more true authors (+3.2 per 100
  [+1.9, +4.4]) with fewer false names. At about 3,200 tokens, CoT x5 self-consistency has the best HR2.
- Coverage is not saturated at 50 samples (0.241 at 50, 0.299 at 100, 0.358 at 200 on the subset).

Session 3 is complete (2026-10-28; data/outputs/s3/; RESULTS.md sections 6-7). Cost 33.9 compute units; 26.59 left.
- The ladder (L0-L5, plus L3t and L4r) and its controls are in RESULTS.md section 6.
  - With another paper's title, AUC is about 0.5 at every level except L4r.
  - Known is 0.703 at L1, 0.358 at L4 (one co-author swapped for a real collaborator, shared slot) and 0.412 at
    L5.
  - The release's collaborator swap (L4r, a slot per option) leaks: title-swapped control 0.406.
  - Hard-known (known at every level): 1,568 papers. Labels: g4_bf16/analysis/session3_labels.csv.
- Catch trials (L1 + "None of the above"): 48.9% of L1-known papers pass both the hit and the rejection.
- Yes/no is conservative (39% Yes to true lists) and weaker than joint choice (AUC 0.793 vs 0.896 at L1).
- Likelihood:
  - The PMI of each list nearly matches joint choice against L1 (0.877).
  - It fails against the model's own wrong lists, L5 (0.574, against 0.709 for choice).
- Selection: authors-to-title scoring adds to frequency (AUC +0.028 [+0.022, +0.035], top-1 +4.1 points).
  Forward PMI and P(True) do not help it.
- Not done: the GPT-4o replication (no OpenAI API key).

## Next
1. Generation strategies that raise coverage per token: exclusion prompting (ask for authors not yet named),
   decomposed generation (first author alone), enumerated first-letter cues.
2. Better selectors: count + authors-to-title as the default; joint MC with the samples in context.
3. Failure split as a function of k (reachable at 10 / 50 / 200 samples) by label and citation bin.
4. If Hana's labels and split become available, report in the team's unit on her 566 known tune papers.
