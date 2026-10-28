// Build Minyan's slides for the team results deck, in the team template:
// 10 x 5.625 in, Arial (as in the earlier Generate_then_Select_Slides deck; installed
// everywhere, so the deck looks the same on any computer and in PDF), blue accent 5B6CFF, presenter pill, key-message bar,
// page number = section number. Numbers come from data/outputs/g4_bf16/analysis.
//
//   node tools/slides/build_update2.js            -> docs/slides/Generate_then_Select_Update2.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");

const ROOT = path.resolve(__dirname, "..", "..");
const ANALYSIS = path.join(ROOT, "data", "outputs", "g4_bf16", "analysis");
const FONT = "Arial";
const OUT = process.env.OUT || path.join(ROOT, "docs", "slides", "Generate_then_Select_Update2.pptx");

const C = {
  blue: "5B6CFF", dblue: "3442C8", tint: "EEF0FF", gray: "F3F4F6", coral: "E2614A", coralTint: "FFF1ED",
  text: "191919", body: "333333", muted: "8A8A8A", line: "A1A1A1", border: "E5E7EB", white: "FFFFFF", slate: "4B5563",
};

function csv(name) {
  const [head, ...rows] = fs.readFileSync(path.join(ANALYSIS, name), "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => {
    const cells = r.match(/("[^"]*"|[^,]+)/g).map((c) => c.replace(/^"|"$/g, ""));
    return Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
  });
}

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Generate-then-Select: Results";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.defineSlideMaster({
  title: "TEAM_CONTENT",
  background: { color: C.white },
  objects: [{ text: { text: "CS 7980  ·  Enhancing Memory of LLMs", options: {
    x: 0.5, y: 5.25, w: 4.5, h: 0.22, fontFace: FONT, fontSize: 7, color: C.muted, margin: 0, valign: "middle" } } }],
});

const T = (o) => ({ fontFace: FONT, margin: 0, isTextBox: true, ...o });

const script = [];
let current = null;

function frame(slide, num, section, title, key) {
  current = { num, section, title };
  slide.addText([{ text: num, options: { bold: true, color: C.blue } }, { text: `  ${section}`, options: { color: C.muted } }],
    T({ x: 0.5, y: 0.26, w: 4, h: 0.25, fontSize: 9, valign: "top", objectName: "Section label" }));
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.72, y: 0.22, w: 1.83, h: 0.31, rectRadius: 0.08,
    fill: { color: C.tint }, line: { color: C.tint }, objectName: "Presenter pill" });
  slide.addText([{ text: "Presenter  ", options: { color: C.muted } }, { text: "Minyan Xu", options: { bold: true, color: C.dblue } }],
    T({ x: 7.72, y: 0.22, w: 1.83, h: 0.31, fontSize: 8, align: "center", valign: "middle", objectName: "Presenter" }));
  slide.addText(title, T({ x: 0.5, y: 0.56, w: 9, h: 0.45, fontSize: 22, bold: true, color: C.text, valign: "middle", objectName: "Title" }));
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.66, w: 9.05, h: 0.4, rectRadius: 0.06,
    fill: { color: C.tint }, line: { color: C.tint }, objectName: "Key message" });
  slide.addText(key, T({ x: 0.68, y: 4.66, w: 8.7, h: 0.4, fontSize: 9.5, bold: true, color: C.dblue, valign: "middle", objectName: "Key message text" }));
  slide.addText(num, T({ x: 9.05, y: 5.25, w: 0.5, h: 0.22, fontSize: 7, color: C.muted, align: "right", valign: "top", objectName: "Page number" }));
}

function addNotes(slide, paragraphs) {
  const text = Array.isArray(paragraphs) ? paragraphs.join("\n\n") : paragraphs;
  slide.addNotes(text);
  script.push({ ...current, text });
}

function card(slide, x, y, w, h, fill, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
}

function bullets(slide, items, o) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 10 }, breakLine: i < items.length - 1 } })),
    T({ fontSize: 9.5, color: C.body, valign: "top", paraSpaceAfter: 4, ...o }));
}

function circle(slide, x, y, label, fill) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.26, h: 0.26, fill: { color: fill }, line: { color: fill } });
  slide.addText(label, T({ x, y, w: 0.26, h: 0.26, fontSize: 9, bold: true, color: C.white, align: "center", valign: "middle" }));
}

const axis = { catAxisLabelColor: C.body, valAxisLabelColor: C.muted, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
  catAxisLabelFontSize: 8, valAxisLabelFontSize: 7, valGridLine: { color: C.border, size: 0.5 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false, dataLabelFontFace: FONT, dataLabelFontSize: 7.5, dataLabelColor: C.body,
  legendFontFace: FONT, legendFontSize: 8, legendColor: C.body };

// ------------------------------------------------------------------ 01 Setup
{
  const s = pres.addSlide({ masterName: "TEAM_CONTENT" });
  frame(s, "01", "Setup", "Sampling 50 Author Lists for Every Paper",
    "Same model, prompt and matcher as the paper’s Qwen3-32B run; every count is net of a chance baseline.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.14, w: 0.62, h: 0.24, rectRadius: 0.05, fill: { color: C.blue }, line: { color: C.blue } });
  s.addText("DONE", T({ x: 0.5, y: 1.14, w: 0.62, h: 0.24, fontSize: 8, bold: true, color: C.white, align: "center", valign: "middle" }));
  s.addText("All 9,108 papers  ·  455,400 author lists  ·  Qwen3-32B, bf16, thinking off",
    T({ x: 1.24, y: 1.14, w: 7, h: 0.24, fontSize: 9, color: C.muted, valign: "middle" }));
  const steps = [["Ask", "Qwen3-32B gets the title and year; no search, no retrieval"],
    ["Sample", "50 lists per paper at T = 1, with no top-p or top-k cut"],
    ["Match", "The paper’s own matcher: last name + first initial"],
    ["Pool", "All names from the 50 lists, with counts"]];
  steps.forEach(([head, text], i) => {
    const x = 0.5 + i * 2.33;
    card(s, x, 1.52, 2.06, 1.0, C.gray, `Step ${i + 1}`);
    circle(s, x + 0.14, 1.64, String(i + 1), C.blue);
    s.addText(head, T({ x: x + 0.5, y: 1.64, w: 1.45, h: 0.26, fontSize: 11, bold: true, color: C.text, valign: "middle" }));
    s.addText(text, T({ x: x + 0.14, y: 1.98, w: 1.8, h: 0.5, fontSize: 9, color: C.body, valign: "top" }));
    if (i < 3) s.addShape(pres.shapes.LINE, { x: x + 2.08, y: 2.02, w: 0.23, h: 0, line: { color: C.line, width: 1, endArrowType: "triangle" } });
  });
  card(s, 0.5, 2.72, 4.42, 1.76, C.tint, "Reproduction");
  s.addText("Our default answer reproduces the paper", T({ x: 0.68, y: 2.84, w: 4.1, h: 0.26, fontSize: 11, bold: true, color: C.dblue }));
  const hdr = { bold: true, color: C.muted, fontSize: 8 };
  const rows = [
    [{ text: "", options: hdr }, { text: "Ours", options: { ...hdr, align: "right" } }, { text: "Paper’s run", options: { ...hdr, align: "right" } }],
    ["True authors named (9,108 papers)", "1,211", "1,202"],
    ["Papers mostly named (HR2 ≤ 0.5)", "222", "222"],
    ["Mean HR2", "0.9691", "0.9691"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { align: j ? "right" : "left", bold: j === 1, color: j === 1 ? C.dblue : C.body } })));
  s.addTable(rows, { x: 0.68, y: 3.16, w: 4.06, colW: [2.46, 0.7, 0.9], fontFace: FONT, fontSize: 9, color: C.body,
    border: { type: "none" }, rowH: 0.28, margin: [0.02, 0.04, 0.02, 0.04] });
  card(s, 5.13, 2.72, 4.42, 1.76, C.coralTint, "Chance baseline");
  s.addText("Two chance baselines", T({ x: 5.31, y: 2.84, w: 4.1, h: 0.26, fontSize: 11, bold: true, color: C.coral }));
  bullets(s, ["Common names match by luck: 50 lists often include a “J. Wang”",
    "Random: authors of 3 other papers, same field and citation bin",
    "Topic (stricter): authors of the 3 most similar titles in the field",
    "Only matches above the baseline count"], { x: 5.31, y: 3.16, w: 4.1, h: 1.25 });
  addNotes(s, [
    "I’ll start with what I ran. For all 9,108 papers I gave Qwen3-32B the title and year, with no search and no retrieval, and asked for the author list fifty times, at temperature one with no truncation. That is 455,400 author lists in total. Each name is matched with the paper’s own matcher, last name plus first initial, and pooled with a count per paper.",
    "Before trusting any of it, I checked that my setup is the paper’s setup. My default answers reproduce the paper’s own Qwen3-32B run: 1,211 versus 1,202 true authors named, the same 222 papers mostly named, and the same mean HR2.",
    "One trap: with fifty lists, common surnames match by luck. So every count I show is net of a chance baseline, either the authors of random papers from the same field and citation bin, or, stricter, the authors of the three most similar titles. Only matches above the baseline count.",
  ].join("\n\n"));
}

// ------------------------------------------------------------------ 02 Recognition
{
  const s = pres.addSlide({ masterName: "TEAM_CONTENT" });
  frame(s, "02", "Recognition", "The Recognition Test Leaks; Own Wrong Lists Are Harder",
    "Read “known” against the 16% swapped-title floor; with the model’s own wrong lists, a third of known papers turn unknown.");
  s.addText("Papers whose true list is chosen in ≥ 3 of 4 option orders (%)",
    T({ x: 0.5, y: 1.14, w: 5.6, h: 0.24, fontSize: 9.5, bold: true, color: C.text, valign: "middle" }));
  s.addChart(pres.charts.BAR, [
    { name: "Standard", labels: ["Real teams", "Real teams, ±2 years", "Model’s own wrong lists"], values: [70.3, 66.8, 41.2] },
    { name: "Title hidden", labels: ["Real teams", "Real teams, ±2 years", "Model’s own wrong lists"], values: [23.8, 17.6, 25.2] },
    { name: "Title swapped", labels: ["Real teams", "Real teams, ±2 years", "Model’s own wrong lists"], values: [15.7, 15.8, 15.5] },
  ], { x: 0.4, y: 1.4, w: 5.75, h: 3.1, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [C.blue, C.coral, "B8BCC6"], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, showLegend: true, legendPos: "t", ...axis });
  const notes = [
    [C.coralTint, C.coral, "Floor ≈ 16%, not 5%", "With another paper’s title, 16% still pass: the model prefers some lists in every order."],
    [C.tint, C.dblue, "Hidden title: a year cue", "Teams from the same years cut it from 24% to 18%, close to the floor."],
    [C.gray, C.slate, "Own wrong lists: much harder", "Known falls from 70% to 41%: 2,009 of 6,179 known papers turn unknown; 82 move up."],
  ];
  notes.forEach(([fill, head, h, t], i) => {
    const y = 1.14 + i * 1.12;
    card(s, 6.3, y, 3.25, 1.0, fill, h);
    s.addText(h, T({ x: 6.46, y: y + 0.1, w: 2.95, h: 0.26, fontSize: 10.5, bold: true, color: head }));
    s.addText(t, T({ x: 6.46, y: y + 0.4, w: 2.95, h: 0.55, fontSize: 9, color: C.body, valign: "top" }));
  });
  s.addText("Labels rebuilt with the team’s rules on our runs: recalled 222 · known 6,179 · notsure 924 · unknown 1,783. Controls: 1,023 papers.",
    T({ x: 0.5, y: 4.48, w: 9.05, h: 0.16, fontSize: 7, color: C.muted, valign: "middle" }));
  addNotes(s, [
    "Multiple choice is our shared measurement: it decides which papers count as known. So before using it, I tested how much it can be passed without knowing the paper.",
    "The blue bars are the normal test, the orange bars hide the title, and the grey bars show another paper’s title. With the team’s design, real teams of other papers, 70% of papers pass. But with the title hidden, 24% still pass, and even with a wrong title 16% do. The model keeps picking the same list in every order, so the real floor is about 16%, not the 5% that pure guessing would give.",
    "Most of the hidden-title advantage is a year cue: when I draw the wrong teams from papers of the same years, it falls from 24% to 18%, close to the floor.",
    "Finally, my own harder test: the wrong options are the model’s own most frequent wrong answers. Known drops from 70% to 41%. Of the 6,179 known papers, 2,009 become unknown. The model often trusts its own invented lists more than the truth.",
  ].join("\n\n"));
}

// ------------------------------------------------------------------ 03 Failure split
{
  const s = pres.addSlide({ masterName: "TEAM_CONTENT" });
  frame(s, "03", "Failure split", "Known Papers: The Right Author Is Often in the Samples",
    "At 50 samples, 42% of known papers have a true author the default answer misses but some sample names.");
  const stats = [
    [C.tint, C.dblue, "Known  ·  6,179 papers", "11.4", "+64.5", "[61.8, 67.6]  ·  2,625 papers (42%)"],
    [C.gray, C.slate, "Unknown  ·  1,783 papers", "1.3", "+12.9", "[10.1, 15.9]  ·  254 papers (14%)"],
  ];
  stats.forEach(([fill, head, h, dflt, gain, sub], i) => {
    const y = 1.14 + i * 1.7;
    card(s, 0.5, y, 3.1, 1.58, fill, h);
    s.addText(h, T({ x: 0.66, y: y + 0.1, w: 2.8, h: 0.24, fontSize: 10.5, bold: true, color: head }));
    s.addText([{ text: "Named by default  ", options: { color: C.muted, fontSize: 8.5 } }, { text: dflt, options: { bold: true, color: C.body, fontSize: 11 } }],
      T({ x: 0.66, y: y + 0.38, w: 2.8, h: 0.26, valign: "middle" }));
    s.addText(gain, T({ x: 0.66, y: y + 0.66, w: 1.25, h: 0.5, fontSize: 26, bold: true, color: head, valign: "middle" }));
    s.addText("more true authors per 100 papers in the 50 samples, net of the topic baseline",
      T({ x: 1.92, y: y + 0.66, w: 1.6, h: 0.5, fontSize: 8, color: C.body, valign: "middle" }));
    s.addText(sub, T({ x: 0.66, y: y + 1.2, w: 2.8, h: 0.24, fontSize: 8, color: C.muted, valign: "middle" }));
  });
  const bins = csv("reach_by_bin.csv");
  const labels = bins.map((b) => b.citation_bin.replace("-", "–"));
  s.addText("True authors per 100 papers, by citation bin (all 9,108 papers)",
    T({ x: 3.85, y: 1.14, w: 5.7, h: 0.24, fontSize: 9.5, bold: true, color: C.text, valign: "middle" }));
  s.addChart(pres.charts.BAR, [
    { name: "Named by the default answer", labels, values: bins.map((b) => +b.named_by_default_per100) },
    { name: "Extra in 50 samples, net of topic baseline", labels, values: bins.map((b) => +b.net_of_topic_per100) },
  ], { x: 3.75, y: 1.36, w: 5.9, h: 2.5, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
    chartColors: ["B8BCC6", C.blue], showValue: false, showLegend: true, legendPos: "t", valAxisMinVal: 0, ...axis, catAxisLabelFontSize: 6.5 });
  card(s, 3.85, 3.96, 5.7, 0.52, C.gray, "More samples");
  s.addText([{ text: "More samples reach more  ", options: { bold: true, color: C.text } },
    { text: "(1,023-paper subset): 50 → 100 → 200 samples name 24% → 30% → 36% of true authors; topic baseline 2% → 3% → 4%.", options: { color: C.body } }],
    T({ x: 4.0, y: 3.96, w: 5.45, h: 0.52, fontSize: 8.5, valign: "middle" }));
  addNotes(s, [
    "This is the failure split: when the default answer misses an author, was it never generated, or generated but not chosen?",
    "On the known papers, the default answer names 11 true authors per 100 papers. But across the fifty samples, another 64 per 100 show up, after subtracting the topic baseline. 42% of known papers have at least one such author. For unknown papers it is only 13 per 100, five times less, which also tells us the recognition label does carry signal.",
    "On the right, by citation bin: the grey bars are the default answer, the blue bars what the samples add. It grows with citations, but even the least-cited papers are above the baseline.",
    "And this is not saturated. On a 1,023-paper subset, 50, 100 and 200 samples reach 24, 30 and 36% of true authors. So 'never generated' always has to say at how many samples.",
  ].join("\n\n"));
}

// ------------------------------------------------------------------ 04 Selection
{
  const s = pres.addSlide({ masterName: "TEAM_CONTENT" });
  frame(s, "04", "Selection", "Counting Beats Asking: Frequency Is the Selector to Beat",
    "Names that recur across samples are usually true: more true authors and far fewer invented ones than the default answer.");
  const rows = [
    [C.blue, "1", "Count: how many of the 50 samples name it", "AUC 0.74",
      "Ranks a true candidate above a false one of the same paper 74% of the time; top candidate true for 43%."],
    [C.coral, "2", "Ask: “Is X an author?” (P(True))", "AUC 0.61",
      "Worse than counting: the model’s own yes/no adds nothing beyond frequency; top candidate true for 31%."],
    [C.slate, "3", "Choose among the top 4 + “none of these”", "+3.0 pts",
      "Picks a true author 58.9% vs 55.9% for the most frequent [+0.2, +5.8]; says “none” only 30% of the time when none is true."],
  ];
  rows.forEach(([col, n, head, metric, text], i) => {
    const y = 1.14 + i * 1.12;
    card(s, 0.5, y, 5.3, 1.0, C.gray, `Method ${n}`);
    circle(s, 0.64, y + 0.12, n, col);
    s.addText(head, T({ x: 1.0, y: y + 0.1, w: 3.5, h: 0.3, fontSize: 10.5, bold: true, color: C.text, valign: "middle" }));
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.62, y: y + 0.13, w: 1.04, h: 0.25, rectRadius: 0.05, fill: { color: C.white }, line: { color: C.border } });
    s.addText(metric, T({ x: 4.62, y: y + 0.13, w: 1.04, h: 0.25, fontSize: 9, bold: true, color: col, align: "center", valign: "middle" }));
    s.addText(text, T({ x: 1.0, y: y + 0.44, w: 4.65, h: 0.5, fontSize: 9, color: C.body, valign: "top" }));
  });
  card(s, 6.0, 1.14, 3.55, 3.34, C.tint, "Test split");
  s.addText("Test split  ·  6,374 papers", T({ x: 6.18, y: 1.26, w: 3.2, h: 0.26, fontSize: 10.5, bold: true, color: C.dblue }));
  s.addText("Default answer vs names in ≥ 7 of 50 samples (rule chosen on the tune split)",
    T({ x: 6.18, y: 1.54, w: 3.2, h: 0.4, fontSize: 8, color: C.muted, valign: "top" }));
  const blocks = [["True authors / 100 papers", "13.4 → 17.2", "+29%"], ["Invented names / 100 papers", "516 → 101", "−80%"]];
  blocks.forEach(([label, value, delta], i) => {
    const y = 2.22 + i * 1.12;
    s.addText(label, T({ x: 6.18, y, w: 3.2, h: 0.22, fontSize: 9, color: C.body }));
    s.addText(value, T({ x: 6.18, y: y + 0.26, w: 2.2, h: 0.5, fontSize: 22, bold: true, color: C.text, valign: "middle" }));
    s.addText(delta, T({ x: 8.4, y: y + 0.26, w: 1.0, h: 0.5, fontSize: 16, bold: true, color: C.blue, align: "right", valign: "middle" }));
  });
  addNotes(s, [
    "If the right author is in the samples, can we pick it out? I compared three ways to select.",
    "First, simply counting: how many of the fifty samples name a candidate. True candidates beat false ones of the same paper 74% of the time.",
    "Second, asking the model directly, 'is X an author?', which is P(True). That is worse than counting, 61%: the model’s yes or no adds nothing beyond how often it writes the name.",
    "Third, a joint choice among the four most frequent candidates plus 'none of these'. It is a little better than taking the most frequent one, by three points, but it rarely says 'none' when none is true.",
    "On the right is the held-out result. I chose the rule on the tune split and scored the test split once. Keeping names that appear in at least seven of the fifty samples names 29% more true authors than the default answer and invents 80% fewer names.",
  ].join("\n\n"));
}

// ------------------------------------------------------------------ 05 Equal compute
{
  const s = pres.addSlide({ masterName: "TEAM_CONTENT" });
  frame(s, "05", "Equal compute", "At Equal Compute, Counting Matches Thinking",
    "At thinking mode’s cost, counting over 20 samples ties it on HR2, names more true authors and invents half as many.");
  const cats = ["Default", "CoT", "20 samples"];
  const charts = [["True authors / 100 papers", [13.4, 15.4, 18.6], 0.5], ["Invented names / 100 papers", [516, 331, 155], 3.25]];
  charts.forEach(([title, values, x]) => {
    s.addText(title, T({ x, y: 1.14, w: 2.6, h: 0.24, fontSize: 9.5, bold: true, color: C.text, valign: "middle" }));
    s.addText("tokens per paper: 33 · 646 · 645", T({ x, y: 1.38, w: 2.6, h: 0.2, fontSize: 8, color: C.muted, valign: "middle" }));
    s.addChart(pres.charts.BAR, [{ name: title, labels: cats, values }], { ...axis, x: x - 0.1, y: 1.6, w: 2.75, h: 2.45, barDir: "col",
      chartColors: ["B8BCC6", C.coral, C.blue], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: title.startsWith("True") ? "0.0" : "0",
      showLegend: false, valAxisMinVal: 0, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 8.5 });
  });
  s.addText("Test split, 6,374 papers. Per token, ordinary samples reach 2.5× more true authors than thinking answers.",
    T({ x: 0.5, y: 4.12, w: 5.4, h: 0.36, fontSize: 8, color: C.muted, valign: "middle" }));
  card(s, 6.1, 1.14, 3.45, 1.92, C.tint, "Paired");
  s.addText("Paired: 20 samples + count minus CoT", T({ x: 6.26, y: 1.24, w: 3.15, h: 0.26, fontSize: 10.5, bold: true, color: C.dblue }));
  const paired = [["Mean HR2", "+0.002 [−0.001, +0.005], n.s."], ["True authors", "+3.2 per 100 [+1.9, +4.4]"], ["Invented names", "−176 per 100 [−233, −131]"]];
  paired.forEach(([k, v], i) => {
    s.addText([{ text: `${k}  `, options: { bold: true, color: C.text } }, { text: v, options: { color: C.body } }],
      T({ x: 6.26, y: 1.6 + i * 0.44, w: 3.15, h: 0.36, fontSize: 9, valign: "middle" }));
  });
  card(s, 6.1, 3.18, 3.45, 1.3, C.gray, "Larger budget");
  s.addText("At ~3,200 tokens (1,023-paper subset)", T({ x: 6.26, y: 3.28, w: 3.15, h: 0.24, fontSize: 10, bold: true, color: C.slate }));
  s.addText("5 CoT answers + vote: best HR2 (−0.016) and fewest invented names. 100 samples + count: more true authors (+5.4 per 100).",
    T({ x: 6.26, y: 3.56, w: 3.15, h: 0.85, fontSize: 9, color: C.body, valign: "top" }));
  addNotes(s, [
    "The natural rival is letting the model think first. Thinking costs tokens, so I compare at equal compute. One thinking answer costs about 650 tokens, the same as twenty ordinary samples.",
    "On the left, test split: thinking mode names 15.4 true authors per 100 papers; counting over twenty samples names 18.6, and invents 155 names per 100 instead of 331.",
    "Paired by paper, mean HR2 is not significantly different, but counting names 3.2 more true authors per 100 papers and invents 176 fewer names.",
    "With a bigger budget there is a real trade-off: voting over five thinking answers gives the best HR2 and the fewest invented names, while a hundred ordinary samples name more true authors. Thinking makes each answer surer; sampling covers more.",
  ].join("\n\n"));
}

// ------------------------------------------------------------------ 06 Summary
{
  const s = pres.addSlide({ masterName: "TEAM_CONTENT" });
  frame(s, "06", "Summary", "In the Steering Unit: What Selection Adds on Known Papers",
    "Generate-then-select needs no model internals; it sets the bar white-box methods should beat at equal compute.");
  s.addText("Known test papers (4,271; our labels): true authors named per 100 papers",
    T({ x: 0.5, y: 1.14, w: 5.4, h: 0.24, fontSize: 9.5, bold: true, color: C.text, valign: "middle" }));
  const H = (t, a) => ({ text: t, options: { bold: true, color: C.muted, fontSize: 8, align: a } });
  const R = (cells, hi) => cells.map((c, j) => ({ text: c, options: { align: j ? "right" : "left", bold: hi && j > 0,
    color: hi && j > 0 ? C.dblue : C.body, fill: { color: hi ? C.tint : C.white } } }));
  s.addTable([
    [H("Method", "left"), H("True / 100", "right"), H("vs default [95% CI]", "right"), H("Invented / 100", "right")],
    R(["Default answer", "11.3", "—", "529"]),
    R(["Thinking (CoT), 1 answer", "16.5", "+5.2 [+3.7, +6.6]", "345"]),
    R(["Names in ≥ 3 of 20 samples", "19.3", "+8.0 [+6.6, +9.4]", "167"], true),
    R(["Names in ≥ 7 of 50 samples", "17.4", "+6.1 [+4.8, +7.4]", "111"]),
  ], { x: 0.5, y: 1.42, w: 5.4, colW: [2.0, 0.85, 1.55, 1.0], fontFace: FONT, fontSize: 9, color: C.body, rowH: 0.36,
    border: { type: "solid", pt: 0.5, color: C.border }, margin: [0.03, 0.06, 0.03, 0.06] });
  s.addText("Team steering at layer 24: +0.7 to +3.0 over random controls on the team’s 566 known tune papers. Different papers and labels: not a direct comparison yet.",
    T({ x: 0.5, y: 3.3, w: 5.4, h: 0.4, fontSize: 8, color: C.muted, valign: "top" }));
  card(s, 0.5, 3.8, 5.4, 0.68, C.tint, "Stricter label");
  s.addText([{ text: "Stricter label, same picture.  ", options: { bold: true, color: C.dblue } },
    { text: "Known under the model’s own wrong lists (2,481 test papers): default 16.1 per 100; names in ≥ 3 of 20 samples 27.9 (+11.7); CoT +8.5.", options: { color: C.body } }],
    T({ x: 0.66, y: 3.8, w: 5.1, h: 0.68, fontSize: 8.5, valign: "middle" }));
  card(s, 6.1, 1.14, 3.45, 3.34, C.gray, "Next");
  s.addText("Next", T({ x: 6.26, y: 1.24, w: 3.15, h: 0.26, fontSize: 10.5, bold: true, color: C.text }));
  const next = [["Same papers as steering", "Rerun on split V3’s 566 known tune papers"],
    ["More coverage per token", "Ask for authors not yet named; first author alone"],
    ["Better selection", "Counting plus candidate choice; reachable split at 10, 50 and 200 samples"]];
  next.forEach(([h, t], i) => {
    const y = 1.62 + i * 0.92;
    circle(s, 6.26, y, String(i + 1), C.blue);
    s.addText(h, T({ x: 6.62, y: y - 0.02, w: 2.8, h: 0.3, fontSize: 10, bold: true, color: C.text, valign: "middle" }));
    s.addText(t, T({ x: 6.62, y: y + 0.3, w: 2.8, h: 0.5, fontSize: 9, color: C.body, valign: "top" }));
  });
  addNotes(s, [
    "To close, the same unit as the steering results: true authors named per 100 known papers, here on 4,271 known test papers with our labels.",
    "The default answer names 11.3. Thinking mode adds 5.2. Keeping names that appear in at least three of twenty samples adds 8, at the same cost as thinking, and invents about a third as many names as the default answer.",
    "For reference, steering at layer 24 adds 0.7 to 3 over random controls, but on the team’s 566 known tune papers and labels, so this is not yet a direct comparison. The first next step is exactly that: rerun on split V3’s known tune papers.",
    "Then I want more coverage per token, by asking for authors not yet named or for the first author alone, and a better selector that combines counting with the candidate choice. The point for the team: generate-then-select needs no model internals, and it sets the bar any white-box method should beat at equal compute. Thank you.",
  ].join("\n\n"));
}

// pptxgenjs gives chart text a Latin font only; WPS then draws ± and ’ in chart labels with the
// theme's Chinese font (DengXian), so name the same font for East Asian and complex scripts too.
async function writeDeck() {
  const zip = await JSZip.loadAsync(await pres.write({ outputType: "nodebuffer" }));
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    zip.file(name, xml.replace(/<a:latin typeface="([^"]+)"\/>(?!<a:ea)/g, '<a:latin typeface="$1"/><a:ea typeface="$1"/><a:cs typeface="$1"/>'));
  }
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  return OUT;
}

writeDeck().then((f) => {
  console.log(`wrote ${f}`);
  const words = (t) => t.split(/\s+/).filter(Boolean).length;
  const total = script.reduce((n, sl) => n + words(sl.text), 0);
  const md = ["# Talk script: Generate-then-Select results (Minyan Xu)", "",
    `Six slides, ${total} words: about ${Math.round(total / 140)} minutes at 140 words a minute. The same text is in each slide’s speaker notes.`, "",
    ...script.flatMap((sl) => [`## ${sl.num} ${sl.section}: ${sl.title}`, "", `*About ${Math.max(1, Math.round(words(sl.text) / 140 * 4) / 4)} min*`, "", sl.text, ""])];
  const mdPath = OUT.replace(/\.pptx$/, "_script.md");
  fs.writeFileSync(mdPath, md.join("\n"), "utf8");
  console.log(`wrote ${mdPath}`);
});
