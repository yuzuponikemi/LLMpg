# Plan Complexity Evaluation

## Problem

When the LLM creates a plan for a complex query, it sometimes generates steps that try to do too many things at once. For example, when asked "what is the distance between Kyoto and Tokyo", it might create a step like:

```
Search for the geographic coordinates of Kyoto and Tokyo
```

However, this causes issues when the agent tries to execute this step because:

1. The LLM tends to generate multiple function calls (one for Kyoto, one for Tokyo) in a single response
2. The agent can only handle one function call at a time
3. After processing the first function call, the continuation flow doesn't properly capture both needed data points

## Solution

We've implemented a **Plan Complexity Evaluator** that automatically analyzes each step in a plan and breaks down complex steps into simpler, more executable steps.

### Key Features

1. **Automatic Detection**: The plan evaluator identifies steps that likely involve multiple discrete tasks by looking for patterns in the step description.

2. **Smart Breakdown**: Based on detected patterns, the evaluator breaks down complex steps into multiple simpler steps that can be executed sequentially.

3. **Seamless Integration**: This happens automatically when a new plan is created, so the agent's behavior is improved without changing the user experience.

### Example

For the query "what is the distance between Kyoto and Tokyo", the original plan might be:

```
1. Search for the geographic coordinates of Kyoto and Tokyo.
2. Calculate the distance using the Haversine formula.
3. Present the results.
```

But this would be automatically refined to:

```
1. Search for the geographic coordinates of Kyoto.
2. Search for the geographic coordinates of Tokyo.
3. Calculate the distance using the Haversine formula.
4. Present the results.
```

This ensures that each step contains exactly one clear task, making execution more reliable.

## Pattern Recognition

The evaluator recognizes several common patterns:

1. Gathering data about multiple locations
2. Multiple search operations in a single step
3. Comparison tasks that should first gather data separately

## Benefits

- Fewer errors during plan execution
- More reliable gathering of required information
- Cleaner, more focused function calls
- Better tracking of progress through the research process

## Testing

You can test this enhancement by running:

```bash
python examples/test_plan_complexity.py
```
