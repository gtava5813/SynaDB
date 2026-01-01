# Security Policy

## Supported Versions

The following versions of SynaDB are currently being supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| 0.1.x   | :x:                |

**Note:** We recommend always using the latest stable release for the best security and features.

## Reporting a Vulnerability

We take the security of SynaDB seriously. If you discover a security vulnerability, please follow these steps:

### Where to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues via one of these methods:

1. **Email:** Send details to security@synadb.ai
2. **GitHub Security Advisories:** Use the "Security" tab in the repository to privately report vulnerabilities

### What to Include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Potential impact
- Any suggested fixes (if available)

### Response Timeline

- **Initial Response:** Within 48 hours of report
- **Status Updates:** Every 5-7 days until resolved
- **Fix Timeline:** Critical issues within 7 days, others within 30 days

### What to Expect

**If Accepted:**
- We will confirm the vulnerability and work on a fix
- You will be credited in the security advisory (unless you prefer to remain anonymous)
- We will coordinate disclosure timing with you
- A CVE may be requested for significant vulnerabilities
- Security patch will be released as soon as possible

**If Declined:**
- We will explain why the issue is not considered a security vulnerability
- We may suggest alternative reporting channels if it's a bug rather than a security issue
- You are free to disclose the issue publicly if you disagree with our assessment

## Security Best Practices

When using SynaDB:

### FFI Safety
- Always validate input at FFI boundaries
- Never pass untrusted data directly to FFI functions without validation
- Use the provided Python/C wrappers rather than raw FFI when possible

### File Permissions
- Database files should have appropriate file system permissions
- Avoid storing databases in world-readable directories
- Use encryption at rest for sensitive data

### Memory Safety
- SynaDB is written in Rust for memory safety
- FFI boundary uses `catch_unwind` to prevent panics
- Always free allocated memory using provided free functions

### Data Validation
- Validate vector dimensions before insertion
- Check model checksums after loading
- Sanitize user input before using as keys

### Dependency Security
- Keep SynaDB updated to the latest version
- Monitor security advisories for dependencies
- Use `cargo audit` for Rust dependency scanning

## Known Security Considerations

### Append-Only Log
- Deleted data is marked with tombstones, not physically removed
- Use `compact()` to physically remove deleted data
- Consider encryption for sensitive data

### FFI Boundary
- C-ABI functions use integer error codes, not exceptions
- Panics are caught at FFI boundary to prevent undefined behavior
- Memory management is manual - always use provided free functions

### Concurrent Access
- Multiple processes can read simultaneously
- Only one writer at a time (enforced by mutex)
- No built-in authentication or authorization

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.5.1)
- Announced in release notes with `[SECURITY]` prefix
- Published to crates.io and PyPI immediately
- Documented in CHANGELOG.md

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged in:
- Security advisories
- Release notes
- This document (with permission)

Thank you for helping keep SynaDB and its users safe!
