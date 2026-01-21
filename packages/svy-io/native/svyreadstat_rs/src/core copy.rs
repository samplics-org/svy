// native/svyreadstat_rs/src/core.rs
use anyhow::Result;
use arrow::array::{ArrayRef, Float64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;

use readstat_sys::{
    readstat_double_value,
    readstat_get_file_label,
    readstat_metadata_t,
    readstat_string_value,
    readstat_type_class_e_READSTAT_TYPE_CLASS_STRING as TCLASS_STRING,
    readstat_type_e_READSTAT_TYPE_STRING as T_STRING,
    readstat_type_e_READSTAT_TYPE_STRING_REF as T_STRING_REF,
    readstat_value_is_system_missing,
    readstat_value_t,
    readstat_value_type as rs_value_type,
    readstat_variable_get_format,
    readstat_variable_get_label,
    readstat_variable_get_missing_range_hi,
    readstat_variable_get_missing_range_lo,
    // NEW: Add these for user-defined missing values
    readstat_variable_get_missing_ranges_count,
    readstat_variable_get_name,
    readstat_variable_get_type_class,
    readstat_variable_t,
};

pub(crate) const HANDLER_OK: c_int = 0;
pub(crate) const HANDLER_ABORT: c_int = 1;

/// ---------- Metadata we ship back to Python ----------

/// User-defined missing value specification
#[derive(Serialize, Clone)]
pub struct UserMissing {
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub values: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub range: Option<Vec<f64>>,
}

#[derive(Serialize, Clone)]
pub struct VarMeta {
    pub name: String,
    pub label: Option<String>,
    pub label_set: Option<String>,
    pub fmt: Option<String>,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_missing: Option<UserMissing>,
}

#[derive(Serialize)]
pub struct ValueLabels {
    pub set_name: String,
    pub mapping: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct MissingRule {
    pub var: String,
    pub discrete: Vec<String>,
    pub ranges: Vec<(String, String)>,
}

#[derive(Serialize)]
pub struct MetaOut {
    pub file_label: Option<String>,
    pub vars: Vec<VarMeta>,
    pub value_labels: Vec<ValueLabels>,
    pub user_missing: Vec<MissingRule>,
    pub n_rows: usize,
    pub tagged_missings: Vec<TaggedSpec>,
    pub notes: Vec<String>,
}

#[derive(serde::Serialize)]
pub struct TaggedSpec {
    pub col: String,
    pub rows: Vec<usize>,
    pub tags: Vec<String>,
}

/// ---------- Parse context ----------
pub(crate) struct ColBuilders {
    pub(crate) kind: ColKind,
    pub(crate) name: String,
    pub(crate) label: Option<String>,
    pub(crate) label_set: Option<String>,
    pub(crate) fmt: Option<String>,
    pub(crate) user_missing: Option<UserMissing>,
    pub(crate) sb: Option<StringBuilder>,
    pub(crate) fb: Option<Float64Builder>,
}

pub(crate) enum ColKind {
    Str,
    F64,
}

pub(crate) struct ParseCtx {
    pub(crate) cols: Vec<ColBuilders>,
    pub(crate) name_to_idx: HashMap<String, usize>,
    pub(crate) cols_skip: Option<HashMap<String, ()>>,
    pub(crate) rows_skip: usize,
    pub(crate) n_max: Option<usize>,
    pub(crate) n_rows_seen: usize,
    pub(crate) n_rows_emitted: usize,
    pub(crate) label_sets: HashMap<String, BTreeMap<String, String>>,
    pub(crate) file_label: Option<String>,
    pub(crate) last_err: Option<String>,
    pub(crate) tagged: HashMap<String, (Vec<usize>, Vec<String>)>,
    pub(crate) notes: Vec<String>,
    pub(crate) detect_tagged: bool,
}

/// ---------- Helpers on builders ----------
impl ColBuilders {
    pub(crate) fn push_missing(&mut self) {
        match self.kind {
            ColKind::Str => self.sb.as_mut().unwrap().append_null(),
            ColKind::F64 => self.fb.as_mut().unwrap().append_null(),
        }
    }
    pub(crate) fn push_str(&mut self, s: &str) {
        match self.kind {
            ColKind::Str => self.sb.as_mut().unwrap().append_value(s),
            ColKind::F64 => self.fb.as_mut().unwrap().append_null(),
        }
    }
    pub(crate) fn push_f64(&mut self, v: f64) {
        match self.kind {
            ColKind::F64 => self.fb.as_mut().unwrap().append_value(v),
            ColKind::Str => self.sb.as_mut().unwrap().append_value(&format!("{v}")),
        }
    }
}

// ---------- Common callbacks ----------

extern "C" {
    fn readstat_value_is_tagged_missing(
        value: readstat_sys::readstat_value_t,
    ) -> ::std::os::raw::c_int;
    fn readstat_value_tag(value: readstat_sys::readstat_value_t) -> ::std::os::raw::c_char;
}

pub(crate) unsafe extern "C" fn on_error_cb(message: *const c_char, ctx: *mut c_void) {
    if message.is_null() || ctx.is_null() {
        return;
    }
    let msg = CStr::from_ptr(message).to_string_lossy().into_owned();
    let rctx = &mut *(ctx as *mut ParseCtx);
    rctx.last_err = Some(msg);
}

pub(crate) unsafe extern "C" fn on_metadata_cb(
    metadata: *mut readstat_metadata_t,
    ctx: *mut c_void,
) -> c_int {
    if metadata.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    let rctx = &mut *(ctx as *mut ParseCtx);
    let label_ptr = readstat_get_file_label(metadata);
    if !label_ptr.is_null() {
        let label = CStr::from_ptr(label_ptr).to_string_lossy().into_owned();
        rctx.file_label = if label.trim().is_empty() {
            None
        } else {
            Some(label.trim().to_string())
        };
    }
    HANDLER_OK
}

pub(crate) unsafe extern "C" fn on_variable_cb(
    index: c_int,
    var: *mut readstat_variable_t,
    label_set_name: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if var.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    let rctx = &mut *(ctx as *mut ParseCtx);

    // --- Trim the incoming variable name (SPSS can pad)
    let name = {
        let p = readstat_variable_get_name(var);
        if p.is_null() {
            format!("V{index}")
        } else {
            CStr::from_ptr(p).to_string_lossy().trim().to_string()
        }
    };

    if let Some(skip) = &rctx.cols_skip {
        if skip.contains_key(&name) {
            let b = ColBuilders {
                kind: ColKind::Str,
                name: name.clone(),
                label: None,
                label_set: None,
                fmt: None,
                user_missing: None,
                sb: Some(StringBuilder::new()),
                fb: None,
            };
            rctx.name_to_idx.insert(name, rctx.cols.len());
            rctx.cols.push(b);
            return HANDLER_OK;
        }
    }

    // Trim label & format strings if present
    let label = {
        let p = readstat_variable_get_label(var);
        if p.is_null() {
            None
        } else {
            let s = CStr::from_ptr(p).to_string_lossy().to_string();
            let st = s.trim();
            if st.is_empty() {
                None
            } else {
                Some(st.to_string())
            }
        }
    };

    let fmt = {
        let p = readstat_variable_get_format(var);
        if p.is_null() {
            None
        } else {
            let s = CStr::from_ptr(p).to_string_lossy().to_string();
            let st = s.trim();
            if st.is_empty() {
                None
            } else {
                Some(st.to_string())
            }
        }
    };

    let kind = if readstat_variable_get_type_class(var) == TCLASS_STRING {
        ColKind::Str
    } else {
        ColKind::F64
    };

    let label_set = if label_set_name.is_null() {
        None
    } else {
        let s = CStr::from_ptr(label_set_name).to_string_lossy().to_string();
        let st = s.trim();
        if st.is_empty() {
            None
        } else {
            Some(st.to_string())
        }
    };

    // NEW: Extract user-defined missing values
    let user_missing = {
        let missing_count = readstat_variable_get_missing_ranges_count(var);
        if missing_count > 0 {
            let mut values = Vec::new();
            let mut range: Option<Vec<f64>> = None;

            let is_string = readstat_variable_get_type_class(var) == TCLASS_STRING;

            for i in 0..missing_count {
                let lo_val = readstat_variable_get_missing_range_lo(var, i);
                let hi_val = readstat_variable_get_missing_range_hi(var, i);

                if is_string {
                    // For string variables, we can't store string missing values in our current structure
                    // SPSS string missing values are not commonly used, so we skip them for now
                    // TODO: Add support for string missing values
                    continue;
                } else {
                    let lo = readstat_double_value(lo_val);
                    let hi = readstat_double_value(hi_val);

                    // Skip if system missing (NaN)
                    if lo.is_nan() || hi.is_nan() {
                        continue;
                    }

                    if (lo - hi).abs() < 1e-10 {
                        // Discrete value (lo == hi means single value)
                        values.push(lo);
                    } else {
                        // Range of values
                        range = Some(vec![lo, hi]);
                    }
                }
            }

            if !values.is_empty() || range.is_some() {
                Some(UserMissing { values, range })
            } else {
                None
            }
        } else {
            None
        }
    };

    let col = match kind {
        ColKind::Str => ColBuilders {
            kind,
            name: name.clone(),
            label,
            label_set,
            fmt,
            user_missing,
            sb: Some(StringBuilder::new()),
            fb: None,
        },
        ColKind::F64 => ColBuilders {
            kind,
            name: name.clone(),
            label,
            label_set,
            fmt,
            user_missing,
            sb: None,
            fb: Some(Float64Builder::new()),
        },
    };

    rctx.name_to_idx.insert(name, rctx.cols.len());
    rctx.cols.push(col);
    HANDLER_OK
}

pub(crate) unsafe extern "C" fn on_value_cb(
    row: c_int,
    var: *mut readstat_variable_t,
    value: readstat_value_t,
    ctx: *mut c_void,
) -> c_int {
    if var.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    let rctx = &mut *(ctx as *mut ParseCtx);

    let row_us = row as usize;
    rctx.n_rows_seen = rctx.n_rows_seen.max(row_us + 1);
    if row_us < rctx.rows_skip {
        return HANDLER_OK;
    }
    if let Some(nm) = rctx.n_max {
        let last_allowed = rctx.rows_skip.saturating_add(nm.saturating_sub(1));
        if row_us > last_allowed {
            return HANDLER_ABORT;
        }
    }

    // --- Trim here too so lookups match the map created in on_variable_cb
    let name = {
        let p = readstat_variable_get_name(var);
        if p.is_null() {
            return HANDLER_OK;
        }
        CStr::from_ptr(p).to_string_lossy().trim().to_string()
    };

    if let Some(skip) = &rctx.cols_skip {
        if skip.contains_key(&name) {
            return HANDLER_OK;
        }
    }

    let idx = match rctx.name_to_idx.get(&name) {
        Some(i) => *i,
        None => return HANDLER_OK,
    };
    let col = &mut rctx.cols[idx];

    if rctx.detect_tagged && unsafe { readstat_value_is_tagged_missing(value) } != 0 {
        let tag_ch = unsafe { readstat_value_tag(value) } as u8 as char;
        let (rows, tags) = rctx
            .tagged
            .entry(name.clone())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        rows.push(row as usize);
        tags.push(tag_ch.to_string());
        col.push_missing();
    } else if readstat_value_is_system_missing(value) != 0 {
        col.push_missing();
    } else {
        let vt = rs_value_type(value);
        if vt == T_STRING || vt == T_STRING_REF {
            let sp = readstat_string_value(value);
            if sp.is_null() {
                col.push_missing();
            } else {
                let s = CStr::from_ptr(sp).to_string_lossy().to_string();
                col.push_str(&s);
            }
        } else {
            let d = readstat_double_value(value);
            col.push_f64(d);
        }
    }

    if idx == 0 {
        rctx.n_rows_emitted += 1;
    }
    HANDLER_OK
}

pub(crate) unsafe extern "C" fn on_note_cb(
    _note_index: c_int,
    note: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if note.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    let rctx = &mut *(ctx as *mut ParseCtx);
    let s = CStr::from_ptr(note).to_string_lossy().into_owned();
    rctx.notes.push(s);
    HANDLER_OK
}

pub(crate) unsafe extern "C" fn on_value_label_cb(
    set_name: *const c_char,
    val: readstat_value_t,
    label: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if ctx.is_null() {
        return HANDLER_OK;
    }
    let rctx = &mut *(ctx as *mut ParseCtx);

    // Trim the set name; leave the human label as-is
    let set = if set_name.is_null() {
        "__default__".to_string()
    } else {
        CStr::from_ptr(set_name)
            .to_string_lossy()
            .trim()
            .to_string()
    };
    let lab = if label.is_null() {
        String::new()
    } else {
        CStr::from_ptr(label).to_string_lossy().into_owned()
    };

    let key = if readstat_value_is_system_missing(val) != 0 {
        "".to_string()
    } else if rs_value_type(val) == T_STRING || rs_value_type(val) == T_STRING_REF {
        let sp = readstat_string_value(val);
        if sp.is_null() {
            String::new()
        } else {
            CStr::from_ptr(sp).to_string_lossy().into_owned()
        }
    } else {
        format!("{}", readstat_double_value(val))
    };

    let map = rctx.label_sets.entry(set).or_default();
    map.insert(key, lab);
    HANDLER_OK
}

/// ---------- Finalize ----------
pub(crate) fn finalize_to_ipc(mut ctx: ParseCtx) -> Result<(Vec<u8>, MetaOut)> {
    use anyhow::anyhow;

    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let mut vars_meta = Vec::with_capacity(ctx.cols.len());

    for mut col in ctx.cols.drain(..) {
        if let Some(skip) = &ctx.cols_skip {
            if skip.contains_key(&col.name) {
                continue;
            }
        }

        let mut fmeta: HashMap<String, String> = HashMap::new();
        if let Some(lbl) = col.label.as_ref() {
            fmeta.insert("label".to_string(), lbl.clone());
        }
        if let Some(set) = col.label_set.as_ref() {
            fmeta.insert("label_set".to_string(), set.clone());
        }
        if let Some(fmt) = col.fmt.as_ref() {
            fmeta.insert("format".to_string(), fmt.clone());
        }

        match col.kind {
            ColKind::Str => {
                let arr = Arc::new(
                    col.sb
                        .take()
                        .ok_or_else(|| anyhow!("string builder missing"))?
                        .finish(),
                ) as ArrayRef;
                let mut field = Field::new(&col.name, DataType::Utf8, true);
                if !fmeta.is_empty() {
                    field = field.with_metadata(fmeta);
                }
                fields.push(field);
                arrays.push(arr);
                vars_meta.push(VarMeta {
                    name: col.name,
                    label: col.label,
                    label_set: col.label_set,
                    fmt: col.fmt,
                    kind: "string".into(),
                    user_missing: col.user_missing,
                });
            }
            ColKind::F64 => {
                let arr = Arc::new(
                    col.fb
                        .take()
                        .ok_or_else(|| anyhow!("float builder missing"))?
                        .finish(),
                ) as ArrayRef;
                let mut field = Field::new(&col.name, DataType::Float64, true);
                if !fmeta.is_empty() {
                    field = field.with_metadata(fmeta);
                }
                fields.push(field);
                arrays.push(arr);
                vars_meta.push(VarMeta {
                    name: col.name,
                    label: col.label,
                    label_set: col.label_set,
                    fmt: col.fmt,
                    kind: "double".into(),
                    user_missing: col.user_missing,
                });
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    let mut buf = Vec::new();
    {
        let mut w = FileWriter::try_new(&mut buf, &schema)?;
        w.write(&batch)?;
        w.finish()?;
    }

    let vlabels = ctx
        .label_sets
        .into_iter()
        .map(|(set_name, mapping)| ValueLabels { set_name, mapping })
        .collect::<Vec<_>>();

    let mut tagged_specs = Vec::<TaggedSpec>::new();
    for (col, (rows, tags)) in ctx.tagged.into_iter() {
        tagged_specs.push(TaggedSpec { col, rows, tags });
    }

    let meta = MetaOut {
        file_label: ctx.file_label,
        vars: vars_meta,
        value_labels: vlabels,
        user_missing: vec![],
        n_rows: ctx.n_rows_emitted,
        tagged_missings: tagged_specs,
        notes: ctx.notes,
    };

    Ok((buf, meta))
}
