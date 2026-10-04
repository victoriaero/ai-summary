Epistemic commitment figures
=============================

All condition-comparison figures use the balanced Dallas collection only.
There are 21 outcomes for each explicit condition in each social dimension.
The generic people control has 21 outcomes and is displayed in every dimension
panel by reusing the same control response for that outcome.

Figures
-------
hedging_by_condition.png
    Response-level hedge occurrences per 100 tokens, shown by group type and
    social dimension. Boxes show the interquartile range and median; points are
    individual outcome responses.

veridicality_by_condition.png
    Response-level mean MegaVeridicality score among responses with at least one
    matched predicate. It contains 204 scored responses before the
    generic-control reuse. Panel headings give the scored response count for
    each group type. Scores are not imputed for responses without matched
    predicates.

veridicality_score_coverage.png
    Number and percent of responses with at least one matched predicate, out of
    21 outcomes per group-type cell.

most_frequent_hedge_cues.png
    Eighteen most frequent BioScope lexical matches in Dallas. All annotated
    cues are retained, including broad and formatting cues. This figure reports
    literal lexicon matches, not contextual uncertainty classification.

No combined hedging/veridicality index is computed. Los Angeles has only one
21-response group and New York has no completed responses, so neither is mixed
into the balanced Dallas condition figures.

Regenerate from the repository root:

    python scripts/plot_epistemic_commitment.py
