# Demo Notes

This project is evaluated around three locked demo prompts against `demo_repo/`:

1. `Explain how user authentication works in this repo`
2. `Find where rate limiting should be added for the login endpoint`
3. `Generate the minimal implementation plan for adding email verification`

## Strongest Demo Runs

### Best Demo Candidate 1

`Find where rate limiting should be added for the login endpoint`

Why it is strong:

- correctly identifies `src/api/auth_routes.py` as the primary insertion point
- keeps the file scope tight
- produces a clean implementation path
- reviewer marks the result as strong and grounded

### Best Demo Candidate 2

`Explain how user authentication works in this repo`

Why it is strong:

- traces the route -> auth service -> user lookup -> token issuance flow coherently
- uses grounded files from the demo backend
- gives a clean explanation-oriented final response rather than a speculative plan

## Best Verbose Walkthrough

`Find where rate limiting should be added for the login endpoint --verbose`

Why it is best for technical discussion:

- the stage progression is clear and concise
- it demonstrates the fixed Planner -> Analyst -> Implementation Planner -> Reviewer flow well
- the output is grounded enough that each stage is easy to explain during a walkthrough

## Notes On The Third Prompt

`Generate the minimal implementation plan for adding email verification`

What it demonstrates:

- the system can identify a plausible implementation path through auth logic, route logic, email-service extension points, and account-model state
- the system surfaces assumptions honestly when evidence is not perfect

Why it is slightly weaker than the top two:

- the Reviewer remains more cautious on this task
- the plan is still good for discussion, but it is not as cleanly “finished” as the rate-limiting run

## Recommended Demo Flow

1. Start with the auth-flow prompt to establish repo understanding.
2. Show the rate-limiting prompt in verbose mode as the main technical walkthrough.
3. Finish with the email-verification planning prompt to demonstrate forward-looking engineering planning with explicit assumptions.
