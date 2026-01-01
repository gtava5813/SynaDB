# Contributing to SynaDB

Thank you for your interest in contributing to SynaDB! This guide will help you get started.

## Ways to Contribute

- 🐛 **Bug Reports** - Found a bug? Open an issue!
- 💡 **Feature Requests** - Have an idea? We'd love to hear it!
- 📖 **Documentation** - Help improve our docs
- 🔧 **Code** - Fix bugs or implement features
- 🧪 **Testing** - Add tests or improve coverage

## Development Setup

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- Python 3.8+ (for Python wrapper)
- Git

### Clone and Build

```bash
git clone https://github.com/gtava5813/SynaDB.git
cd SynaDB

# Build
cargo build

# Run tests
cargo test

# Run clippy (linting)
cargo clippy -- -D warnings

# Check formatting
cargo fmt --check
```

### Project Structure

```
SynaDB/
├── src/                    # Rust core library
│   ├── lib.rs             # Public API
│   ├── engine.rs          # SynaDB implementation
│   ├── types.rs           # Atom, LogHeader
│   ├── compression.rs     # LZ4, delta encoding
│   ├── error.rs           # Error types
│   └── ffi.rs             # C-ABI interface
├── tests/                  # Integration tests
├── demos/python/           # Python wrapper
├── include/                # C headers
└── benchmarks/             # Performance benchmarks
```

## Code Style

### Rust

- Follow standard Rust conventions
- Use `cargo fmt` before committing
- No clippy warnings (`cargo clippy -- -D warnings`)
- Document public items with `///` comments

### Python

- Follow PEP 8
- Use type hints
- Document with docstrings

## Testing

### Property-Based Tests

We use `proptest` for property-based testing. Each property test validates correctness properties from the design document.

```rust
// **Feature: syna-db, Property 1: Atom Serialization Round-Trip**
// **Validates: Requirements 5.4, 10.1**
proptest! {
    #[test]
    fn prop_atom_roundtrip(atom in arb_atom()) {
        let encoded = bincode::serialize(&atom)?;
        let decoded: Atom = bincode::deserialize(&encoded)?;
        prop_assert_eq!(atom, decoded);
    }
}
```

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With output
cargo test -- --nocapture
```


## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feat/my-feature`)
3. **Make changes** and add tests
4. **Run tests** (`cargo test`)
5. **Run linting** (`cargo clippy -- -D warnings`)
6. **Format code** (`cargo fmt`)
7. **Commit** with a descriptive message
8. **Push** to your fork
9. **Open a PR** against `main`

### Commit Messages

Use conventional commits:

```
feat: add vector similarity search
fix: handle empty key in get()
docs: update Python examples
test: add property test for compression
refactor: simplify index rebuild logic
perf: optimize tensor extraction
chore: update dependencies
```

### PR Checklist

- [ ] Tests pass (`cargo test`)
- [ ] No clippy warnings
- [ ] Code formatted (`cargo fmt`)
- [ ] Documentation updated (if applicable)
- [ ] Changelog updated (for features/fixes)

## Architecture Guidelines

### The Feynman Principles

> "We didn't build a cathedral. We built a conveyor belt."

1. **Simplicity over cleverness** - Prefer readable code
2. **Append-only** - Never rewrite data
3. **Binary serialization** - Bincode, not JSON
4. **Zero-copy when possible** - Memory-mapped reads
5. **Panic safety** - `catch_unwind` at FFI boundary

### Adding New Features

1. Define requirements and design
2. Implement with property tests
3. Update documentation (wiki and READMEs)
4. Submit PR for review

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Chat**: Join our Discord (coming soon)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
