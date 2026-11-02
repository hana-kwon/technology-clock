# Coding guide: passage-level answers to the technology-clock questions (draft, 2026-11-01)

Coders read the same passage the language-model reader saw and answer the same questions, so that reader labels can be scored and later corrected with design-based supervised learning. Coders see the passage, call date, and company, and do not see any model answer.
Passage identifiers in parentheses (P..., H...) refer to the licensed pilot and holdout samples; their text is not included here.


## 1. Task

Read one transcript chunk (the same text the readers saw) and answer the reader question as an expert reader would. The question (prompt `prompts/timing_v4.txt`):

> When does the speaker expect quantum computing to be in regular commercial use, meaning that customers use quantum computers in their business to do useful work better than classical computers can?

Answer with one of seven choices: already in commercial use; within 2 years; in 3 to 5 years; in 6 to 10 years; more than 10 years from now; never or not foreseeable; no information is provided.

Coders see the chunk text, the call date, and the company. Coders do not see any reader answer, the other coder's answer, stock prices, or anything that happened after the call.

## 2. Fields

| Field | Values | Rule |
|---|---|---|
| `answer` | 7 choices | Main label under the rules in section 3. |
| `answer_if_analyst_counts` | 7 choices | Same chunk, but a time stated only by an analyst or relayed third party counts (rule 3.7). Equal to `answer` when no analyst time is involved. |
| `answer_if_firm_claim_counts` | 7 choices | Same chunk, but a company's claim that its product is useful today counts as "already" even without business results beyond classical computers (rule 3.4). Equal to `answer` otherwise. |
| `evidence_quote` | text | The exact words that decide the answer (copy from the chunk). Blank for "no information is provided" unless a near miss is worth recording. |
| `horizon_text` | text | The exact time words ("2 to 3 years", "end of this decade"). |
| `years_from_call` | number | The number used for the bin (section 3.6). Blank if no time. |
| `event` | commercial_use / advantage_practical / advantage_scientific / hardware_roadmap / crypto_break / market_size / other_technology / none | What the stated time refers to. Only `commercial_use` and `advantage_practical` can give a time bin (rules 3.2-3.3). |
| `speaker_role` | executive / company_other / analyst / third_party / unknown | Who states the time. `company_other` = company presenter or panelist who is not an officer. `third_party` = a cited report, consultant, or other firm. |
| `referent` | own / field / other_firm | Whose technology the time is about (as in codebook 62). |
| `stage` | research only / pilots or trials / paid early deployments / regular commercial use / no information is provided | The most advanced stage of quantum computing use that the speaker describes as already happening at the call date (rule 3.11). |
| `stage_quote` | text | Exact words that decide `stage`. |
| `hard_case` | yes / no | Yes if any rule in section 3 was needed to decide, or two rules point different ways. |
| `confidence` | 1-5 | Coder's confidence. |
| `note` | text | Which rule decided a hard case; any other time statements in the chunk. |

## 3. Decision rules (apply in this order)

**3.1 Read the whole chunk.** Do not read only the sentence with the keyword. In the pilot, a time ("5, 10 years from now") turned out to be about neuromorphic chips once the paragraph was read (P185), and a narrow reading window missed a "2 to 3 years" two paragraphs away (a holdout passage).

**3.2 Which event the time is about.** A time counts only if it is about customers using quantum computers for useful work better than classical computers. These do not count, and the answer is "no information is provided" unless another statement in the chunk gives a time for the target event:
- breaking encryption or "Q-Day", including post-quantum security deadlines (H050, H083, H004, H009, H024, H138);
- hardware or product milestones: logical-qubit counts, fault-tolerant machines, chip or system releases, data-center builds, manufacturing roadmaps (H057, H095, H129, H037, H121);
- market-size forecasts ("a $130B market by 2040", H115);
- other technologies (neuromorphic chips P185; quantum sensing H115).

**3.3 "Quantum advantage".** Count a time for quantum advantage only when the passage ties advantage to practical or commercial benefit (H027, H035, H069). Advantage on a scientific benchmark or as a claim to the research community does not count (H095; the supremacy result cited in H105).

**3.4 "Already in commercial use".** Choose this only if the speaker says or clearly implies that customers already get business results that classical computers cannot match (H052: a production pilot with a stated result against classical computing; H090, H123: the CEO says customers get real-world advantage today). These are not "already": selling, renting, or installing machines; pilots, tests, research partnerships; benchmark speedups; general claims that quantum is "valuable today", "real applications today", or "quantum is now" (H121, H105, H023, H097, H059, H085, H015). If such a company claim is the only basis, put "no information is provided" in `answer` and "already in commercial use" in `answer_if_firm_claim_counts`.

**3.5 A time must be stated.** Words that carry a count are times: "a couple of years" = 2, "a few years" or "the next few years" = 3 (H026), "a decade" = 10, "decades" = 20 (H143), "end of this decade" = 2030, "mid-2020s" = 2025. Words without a count are not times, so the answer is "no information is provided": "near term", "longer term", "soon", "coming fast", "much sooner than people expect", "a ways off", "may take a little time", "eventually" (H146, H011, H082, H072, H071, H125, H070). Do not infer a time from the firm's other statements, its roadmap, or what happened later.

**3.6 Converting to years.** `years_from_call` = target year − call year; a range ("2 to 5 years") uses its middle. Round to a whole number with halves going toward the call date (2.5 → 2, 4.5 → 4, 5.5 → 5, 10.5 → 10), then choose the bin: 0-2 within 2 years, 3-5, 6-10, 11 or more. A target year equal to the call year ("mid-2020s" in a 2025 call, H075) is "within 2 years", not "already", because the use has not started. Read obvious transcription errors as intended and say so in `note` ("4 fails" for "4 years", H128).

**3.7 Speaker.** The question asks what the speaker expects. In `answer`, count times stated by company speakers (`executive`, `company_other`). A time stated only by an analyst or a cited third party goes in `answer_if_analyst_counts`, and `answer` is "no information is provided" (H031, H103, H017). If a company speaker repeats or agrees with the analyst's time, it is the company's time. If a company speaker cites a report but gives a different view without a time ("not a decade away, but much sooner", H070), the answer is "no information is provided".

**3.8 Own product versus the field.** Both count. If the speaker gives one time for its own product and another for the field or competitors, use the statement about the target event that the speaker makes most directly about customers' use, and list the other in `note` (H123: own annealing systems already deliver advantage; "5-plus years" referred only to competitors' gate-model systems).

**3.9 Several times in one chunk.** Use the time stated for the target event by a company speaker. If two company statements give different times for the same event, use the more specific one and note the other. If they are equally specific, use the earlier one in the chunk and mark `hard_case`.

**3.10 "Never or not foreseeable".** Only when the speaker says regular commercial use may never come or cannot be foreseen. Doubt without a time ("we are skeptical") is "no information is provided".

**3.11 Stage (second question, prompt `prompts/stage.txt`).** Count only what is described as happening now; plans, targets, roadmaps, and forecasts do not count. research only = research, laboratory work, technical milestones (qubit counts, error rates, chips), or development spending. pilots or trials = the company, customers, or partners test quantum computers (pilots, proofs of concept, trials, experimental or cloud access) without paying for production use. paid early deployments = customers pay (sales, contracts, bookings, revenue, installed systems) but the speaker does not say they get regular business results better than classical computers. regular commercial use = the same definition as "already in commercial use" in rule 3.4. Quantum sensing, post-quantum encryption, and quantum-inspired software on classical computers are not quantum computing use. Suppliers of parts or equipment to quantum computer makers describe no use stage. If several stages are current, choose the most advanced.

## 4. Sample and procedure (pilot)

- Items: the 70 holdout chunks with independent LLM labels (to check those labels), plus all pilot chunks where any reader answer is informative and a random draw of the rest, with selection probabilities recorded for DSL (protocol 75 V6). Final sizes after `power_dsl()` on the pilot results.
- Two coders work independently. Disagreements are settled by discussion; the settled label is the reference label. Record both original labels.
- Training set: 20 chunks from the hard cases listed in the holdout `note` column; they are used to train coders and are not used to score readers.
- Coding sheets contain licensed transcript text, so they are kept with the licensed data and are not part of this repository.

## 5. Agreement and scoring

- Between coders: exact agreement and Cohen's kappa on the 7 choices; kappa on informative versus not; linearly weighted kappa on the five timed bins; share of `years_from_call` within one year.
- Readers against the reference label: the same measures, plus precision and recall of "informative", reported under the main rule and under both alternative rules.
- Report the reference-label share of informative chunks with its selection weights, because the item draw over-samples informative chunks.
