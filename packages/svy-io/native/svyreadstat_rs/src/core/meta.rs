use super::*;

#[derive(Serialize, Clone)]
pub(crate) struct UserMissing {
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub values: Vec<MissingValue>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub range: Option<Vec<f64>>,
}

/// A discrete user-missing code: a number, or a string variable's text.
#[derive(Serialize, Clone)]
#[serde(untagged)]
pub(crate) enum MissingValue {
    Num(f64),
    Str(String),
}

#[derive(Serialize, Clone)]
pub(crate) struct VarMeta {
    pub name: String,
    pub label: Option<String>,
    pub label_set: Option<String>,
    pub fmt: Option<String>,
    pub kind: String,
    /// `None` when undeclared (always in Stata); SPSS defaults numerics to "scale", so only nominal/ordinal are deliberate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measure: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_missing: Option<UserMissing>,
}

#[derive(Serialize)]
pub(crate) struct ValueLabels {
    pub set_name: String,
    pub mapping: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub(crate) struct MissingRule {
    pub var: String,
    pub discrete: Vec<String>,
    pub ranges: Vec<(String, String)>,
}

#[derive(Serialize)]
pub(crate) struct MetaOut {
    pub file_label: Option<String>,
    pub vars: Vec<VarMeta>,
    pub value_labels: Vec<ValueLabels>,
    pub user_missing: Vec<MissingRule>,
    pub n_rows: usize,
    pub tagged_missings: Vec<TaggedSpec>,
    pub notes: Vec<String>,
    /// Invalid UTF-8 was replaced by U+FFFD: usually a legacy code page needing an explicit `encoding=`.
    pub had_invalid_utf8: bool,
    /// Where the reader knows it (Stata: explicit, detected, or the format's fixed UTF-8).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// Why the SAS format catalog gave no value labels; the data still read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog_error: Option<String>,
}

#[derive(serde::Serialize)]
pub(crate) struct TaggedSpec {
    pub col: String,
    pub rows: Vec<usize>,
    pub tags: Vec<String>,
}
