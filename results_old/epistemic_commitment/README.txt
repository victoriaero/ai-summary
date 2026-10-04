Epistemic commitment analysis
==============================

This directory contains two separate operationalizations. They are not merged
into a composite index.

Input coverage
--------------
Collection files discovered: 819
Non-empty AIO responses analyzed: 294

Hedging
-------
The lexicon is generated only from spans explicitly annotated with
type="speculation" in BioScope 1.0. No cue is added, removed, stemmed, or
lemmatized by hand. Unicode NFKC, whitespace collapse, and case-folding are used
only to create normalized_cue for matching. Phrase matching is token-based and
case-insensitive. When lexicon entries overlap, the longest non-overlapping span
is retained. Token counts exclude spaces and punctuation. Character offsets
start/end are zero-based half-open offsets in the complete AIO response;
sentence_start/sentence_end are relative to sentence_text.

BioScope is a biomedical/clinical resource and this is a direct lexical
transfer to AIO text, not a contextual hedge classifier. Broad cues and
formatting symbols remain when BioScope annotated them; no domain-adaptation or
manual pruning is applied. The English spaCy parser is used for every included
response, with the original language metadata retained for auditing.

Hedge occurrences found: 3652

Veridicality
------------
Scores come from the official normalized MegaVeridicality v2.1 file. A
predicate is considered only when its spaCy lemma occurs in the resource and it
has a dependency-parsed ccomp or xcomp complement. Matching uses predicate
lemma, syntactic frame, predicate polarity, and the non-conditional resource
configuration. Conditional-antecedent uses are unmatched because the official
normalized table contains only non-conditional tuples.

The resource distinguishes eventive and non-eventive infinitival frames. An
infinitive is matched when only one resource tuple is possible or when the
complement head is the resource's explicit do/have contrast. Otherwise it is
marked unmatched rather than assigned a score. n_veridical_predicates counts
only matched occurrences. Summary statistics are missing when that count is
zero.

Predicate candidates found: 1143
Matched and scored predicates: 459

Files
-----
- input_inventory.csv: every discovered collection file and inclusion status
- hedge_lexicon.csv: lexicon derived from BioScope annotations
- hedge_occurrences.csv: occurrence-level hedge matches
- hedging_response_metrics.csv: response-level hedging metrics
- predicate_occurrences.csv: matched and unmatched predicate candidates
- veridicality_response_metrics.csv: response-level veridicality summaries
- megaveridicality_resource_entries.csv: exact normalized tuples used
- resource_provenance.json: versions, URLs, checksums, and parser version

Reproduction
------------
Run from the repository root:

    python scripts/analyze_epistemic_commitment.py
