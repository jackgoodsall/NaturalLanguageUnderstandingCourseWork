# Research Schema

## Objective

Store research in a way that can directly drive experiments, decisions, and reversals.

This research layer exists to answer:

1. what should we try next
2. why that next step is justified
3. which sources support the decision
4. what would invalidate the decision

## Source Policy

Default source classes:

1. official library documentation
2. official model cards
3. official repositories
4. original papers

Avoid secondary blogs, summaries, and forum posts unless primary sources are insufficient.

If a non-primary source is ever used, label it explicitly.

## Record Structure

Each research question should produce one structured record following:

- `/Users/shivsaranshthakur/Projects/NaturalLanguageUnderstandingCourseWork/research/research_record.schema.json`

Each record must contain:

1. `research_id`
2. `question`
3. `created_utc`
4. `status`
5. `sources[]`
6. `local_evidence`
7. `recommendation`
8. `invalidation_condition`
9. `next_hypothesis_id`

## Required Source Fields

For each source, store:

1. `source_id`
2. `source_type`
3. `authority`
4. `title`
5. `url`
6. `accessed_utc`
7. `key_claim`
8. `action_implication`
9. `confidence`

## Decision Rules

Research is actionable only if it leads to one of:

1. a concrete experiment
2. a concrete rejection
3. a concrete instrumentation step

If it does not change action, it should not become a long report.

## Linkage Rules

Every experiment hypothesis after this point should link back to:

1. at least one research record
2. at least two primary sources

Every model selection decision should include:

1. the chosen option
2. at least one rejected alternative
3. why the alternative was rejected
4. what result would cause reversal

## Minimum Acceptance Test

A research record is valid only if:

1. all URLs resolve to primary sources
2. the recommendation is concrete
3. the next hypothesis is named
4. the invalidation condition is explicit
