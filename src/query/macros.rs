//! Query Macros — user-defined reusable query patterns.
//!
//! ```sql
//! DEFINE MACRO recent_avg(key_pattern, duration DEFAULT '1h') AS
//!   SELECT AVG(value) FROM $key_pattern
//!   WHERE timestamp > NOW() - INTERVAL $duration
//! ```

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// A stored macro definition.
#[derive(Debug, Clone)]
pub struct QueryMacro {
    /// Macro name.
    pub name: String,
    /// Parameters with optional defaults.
    pub params: Vec<MacroParam>,
    /// Body template with `$param` placeholders.
    pub body: String,
}

/// A macro parameter.
#[derive(Debug, Clone)]
pub struct MacroParam {
    /// Parameter name (without `$`).
    pub name: String,
    /// Optional default value.
    pub default: Option<String>,
}

/// Registry of defined macros.
#[derive(Debug, Default)]
pub struct MacroRegistry {
    macros: HashMap<String, QueryMacro>,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

impl MacroRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Define a new macro.
    pub fn define(&mut self, macro_def: QueryMacro) -> Result<(), String> {
        if macro_def.name.is_empty() {
            return Err("macro name cannot be empty".into());
        }
        self.macros.insert(macro_def.name.clone(), macro_def);
        Ok(())
    }

    /// Expand a macro invocation with the given arguments.
    ///
    /// Returns the expanded query string with parameters substituted.
    pub fn expand(&self, name: &str, args: &[&str]) -> Result<String, String> {
        let macro_def = self
            .macros
            .get(name)
            .ok_or_else(|| format!("undefined macro: {}", name))?;

        let mut body = macro_def.body.clone();

        for (i, param) in macro_def.params.iter().enumerate() {
            let value = if i < args.len() {
                args[i].to_string()
            } else if let Some(default) = &param.default {
                default.clone()
            } else {
                return Err(format!(
                    "missing argument '{}' for macro '{}'",
                    param.name, name
                ));
            };

            body = body.replace(&format!("${}", param.name), &value);
        }

        Ok(body)
    }

    /// List all defined macros.
    pub fn list(&self) -> Vec<&QueryMacro> {
        self.macros.values().collect()
    }

    /// Check if a macro exists.
    pub fn exists(&self, name: &str) -> bool {
        self.macros.contains_key(name)
    }

    /// Remove a macro.
    pub fn remove(&mut self, name: &str) -> bool {
        self.macros.remove(name).is_some()
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define_and_expand() {
        let mut registry = MacroRegistry::new();
        registry
            .define(QueryMacro {
                name: "recent_avg".into(),
                params: vec![
                    MacroParam {
                        name: "pattern".into(),
                        default: None,
                    },
                    MacroParam {
                        name: "duration".into(),
                        default: Some("1h".into()),
                    },
                ],
                body:
                    "SELECT AVG(value) FROM $pattern WHERE timestamp > NOW() - INTERVAL $duration"
                        .into(),
            })
            .unwrap();

        let expanded = registry
            .expand("recent_avg", &["\"sensor/*\"", "24h"])
            .unwrap();
        assert!(expanded.contains("\"sensor/*\""));
        assert!(expanded.contains("24h"));
    }

    #[test]
    fn test_default_parameter() {
        let mut registry = MacroRegistry::new();
        registry
            .define(QueryMacro {
                name: "check".into(),
                params: vec![
                    MacroParam {
                        name: "key".into(),
                        default: None,
                    },
                    MacroParam {
                        name: "limit".into(),
                        default: Some("10".into()),
                    },
                ],
                body: "SELECT * FROM $key LIMIT $limit".into(),
            })
            .unwrap();

        // Only provide first arg — second uses default
        let expanded = registry.expand("check", &["\"k\""]).unwrap();
        assert!(expanded.contains("LIMIT 10"));
    }

    #[test]
    fn test_missing_required_param() {
        let mut registry = MacroRegistry::new();
        registry
            .define(QueryMacro {
                name: "m".into(),
                params: vec![MacroParam {
                    name: "x".into(),
                    default: None,
                }],
                body: "$x".into(),
            })
            .unwrap();

        let err = registry.expand("m", &[]).unwrap_err();
        assert!(err.contains("missing argument"));
    }

    #[test]
    fn test_undefined_macro() {
        let registry = MacroRegistry::new();
        let err = registry.expand("nonexistent", &[]).unwrap_err();
        assert!(err.contains("undefined macro"));
    }

    #[test]
    fn test_list_and_remove() {
        let mut registry = MacroRegistry::new();
        registry
            .define(QueryMacro {
                name: "a".into(),
                params: vec![],
                body: "SELECT 1".into(),
            })
            .unwrap();

        assert_eq!(registry.list().len(), 1);
        assert!(registry.exists("a"));
        assert!(registry.remove("a"));
        assert!(!registry.exists("a"));
    }
}
