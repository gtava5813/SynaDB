# Contributing to Syna

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Syna-db.git
cd Syna-db

# Build
cargo build

# Run tests
cargo test

# Run clippy
cargo clippy -- -D warnings

# Format code
cargo fmt
```

## Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests: `cargo test`
5. Run lints: `cargo clippy -- -D warnings`
6. Format: `cargo fmt`
7. Commit with a clear message
8. Push and open a Pull Request

## Commit Messages

Use conventional commits:

```
feat(engine): add batch insert support
fix(ffi): handle null pointer in get_float
docs: update Python usage examples
test: add property test for compaction
```

## Code Style

- Follow Rust idioms
- Use `thiserror` for error types
- Add doc comments to public items
- Write property-based tests for new features

## Testing

We use property-based testing with `proptest`. When adding features:

1. Add unit tests for edge cases
2. Add property tests for invariants
3. Ensure all 16 existing properties still pass

```bash
# Run all tests
cargo test

# Run specific test file
cargo test --test atom_roundtrip

# Run with output
cargo test -- --nocapture
```

## Questions?

Open an issue or start a discussion. We're happy to help!
For any private inquiries email hello[at]synadb[dot]ai


