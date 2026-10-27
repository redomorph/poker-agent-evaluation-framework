# Specification Authority and Engineering Principles

This document defines the authority of the scientific specification and the engineering principles used to implement the Poker Agent Evaluation Framework.

## Specification authority

- The active files listed in [the specification index](README.md) collectively define the binding scientific and technical specification.
- Unspecified or conflicting scientific behaviour must not be invented during implementation. Resolve ambiguities in the relevant versioned specification before using affected behaviour in a benchmark.
- Newer explicitly versioned definitions supersede older definitions of the same concept.
- Documents under `archive/` are retained for historical context only; they are not authoritative.

## Engineering principles

- **Test coverage:** Write automated tests for new functionality, including specified edge cases such as split pots, all-ins and invalid bets. A change is ready only when the relevant tests pass.
- **Type safety:** Use Python type annotations for functions and classes.
- **Immutable state:** Represent game state with immutable data structures (for example, frozen dataclasses) to avoid accidental side effects during evaluations.
- **Performance:** Avoid unnecessary deep copies in simulation hot paths. Benchmark workloads may involve millions of hands.
- **Traceability:** Link scientific implementation changes to the applicable specification section. Record changes to frozen rules and preserve the metadata needed to interpret historical results.

For scientific change control and precedence, see [the specification index](README.md).
