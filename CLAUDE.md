# Project Instructions

## Formatting

Always run `cargo fmt` before committing. Formatting is enforced in CI via GitHub Actions.

## Test-Driven Development

When implementing new features or fixing bugs:

1. Write tests first that check the desired behavior.
2. Verify the new tests fail (confirming they catch the issue / check the right thing).
3. Implement the fix or feature.
4. Verify all previously failing tests now pass.
