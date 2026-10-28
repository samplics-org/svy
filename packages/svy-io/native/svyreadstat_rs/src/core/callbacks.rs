use super::*;

pub(super) fn describe_panic(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Turns a callback panic into `panic_err` and a parse abort; unwinding into C would abort the process.
/// A crafted file can trigger one: Arrow panics past 2 GiB of strings in one column.
///
/// # Safety
///
/// `ctx` must be a live `ParseCtx` nothing else touches during the call; `readstat_parse_*`
/// hands it to one callback at a time.
#[inline]
pub(super) unsafe fn guard_cb(ctx: *mut c_void, f: impl FnOnce(&mut ParseCtx) -> c_int) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the caller guarantees `ctx` is a live, unaliased `ParseCtx`.
        let rctx = unsafe { &mut *(ctx as *mut ParseCtx) };
        f(rctx)
    })) {
        Ok(rc) => rc,
        Err(p) => {
            // SAFETY: as above; the panicking closure and its borrow have been unwound.
            let rctx = unsafe { &mut *(ctx as *mut ParseCtx) };
            rctx.panic_err = Some(describe_panic(p.as_ref()));
            HANDLER_ABORT
        }
    }
}

/// # Safety
///
/// Called only by ReadStat: `message` is null or a NUL-terminated string, and `ctx` is
/// as `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_error_cb(message: *const c_char, ctx: *mut c_void) {
    if message.is_null() || ctx.is_null() {
        return;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `message` is non-null and NUL-terminated.
    unsafe {
        guard_cb(ctx, |rctx| {
            let msg = CStr::from_ptr(message).to_string_lossy().into_owned();
            rctx.last_err = Some(msg);
            HANDLER_OK
        })
    };
}

/// # Safety
///
/// Called only by ReadStat: `metadata` is null or live for the call, and `ctx` is as
/// `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_metadata_cb(
    metadata: *mut readstat_metadata_t,
    ctx: *mut c_void,
) -> c_int {
    if metadata.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `metadata` is non-null and live.
    unsafe { guard_cb(ctx, |rctx| on_metadata_impl(metadata, rctx)) }
}

/// # Safety
///
/// `metadata` must be non-null and live for the call.
unsafe fn on_metadata_impl(metadata: *mut readstat_metadata_t, rctx: &mut ParseCtx) -> c_int {
    // SAFETY: `metadata` is live (caller's guarantee).
    let label_ptr = unsafe { readstat_get_file_label(metadata) };
    if !label_ptr.is_null() {
        // SAFETY: ReadStat's metadata strings are NUL-terminated and live for the call.
        let label = rctx
            .decode(unsafe { CStr::from_ptr(label_ptr) })
            .into_owned();
        rctx.file_label = if label.trim().is_empty() {
            None
        } else {
            Some(label.trim().to_string())
        };
    }

    // SAFETY: `metadata` is live (caller's guarantee).
    let nrows = unsafe { readstat_get_row_count(metadata) };
    if nrows > 0 {
        rctx.row_capacity = Some(nrows as usize);
    }

    HANDLER_OK
}

/// # Safety
///
/// Called only by ReadStat: `var` is null or live for the call, `label_set_name` is null
/// or NUL-terminated, and `ctx` is as `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_variable_cb(
    index: c_int,
    var: *mut readstat_variable_t,
    label_set_name: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if var.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `var` is non-null and live.
    unsafe {
        guard_cb(ctx, |rctx| {
            on_variable_impl(index, var, label_set_name, rctx)
        })
    }
}

/// # Safety
///
/// `var` must be non-null and live for the call; `label_set_name` null or NUL-terminated.
unsafe fn on_variable_impl(
    index: c_int,
    var: *mut readstat_variable_t,
    label_set_name: *const c_char,
    rctx: &mut ParseCtx,
) -> c_int {
    let name = {
        // SAFETY: `var` is live (caller's guarantee).
        let p = unsafe { readstat_variable_get_name(var) };
        if p.is_null() {
            format!("V{index}")
        } else {
            // SAFETY: ReadStat's variable strings are NUL-terminated and live for the call.
            rctx.decode(unsafe { CStr::from_ptr(p) }).trim().to_string()
        }
    };

    // SAFETY: `var` is live (caller's guarantee).
    let Ok(var_index) = usize::try_from(unsafe { readstat_variable_get_index(var) }) else {
        return HANDLER_OK;
    };
    if rctx.var_cols.len() <= var_index {
        rctx.var_cols.resize(var_index + 1, None);
    }

    if let Some(skip) = &rctx.cols_skip
        && skip.contains_key(&name)
    {
        return HANDLER_OK;
    }

    let cap = safe_capacity(rctx.row_capacity);

    let label = {
        // SAFETY: `var` is live (caller's guarantee).
        let p = unsafe { readstat_variable_get_label(var) };
        if p.is_null() {
            None
        } else {
            // SAFETY: ReadStat's variable strings are NUL-terminated and live for the call.
            let s = rctx.decode(unsafe { CStr::from_ptr(p) }).into_owned();
            let st = s.trim();
            if st.is_empty() {
                None
            } else {
                Some(st.to_string())
            }
        }
    };

    let fmt = {
        // SAFETY: `var` is live (caller's guarantee).
        let p = unsafe { readstat_variable_get_format(var) };
        if p.is_null() {
            None
        } else {
            // SAFETY: ReadStat's variable strings are NUL-terminated and live for the call.
            let s = rctx.decode(unsafe { CStr::from_ptr(p) }).into_owned();
            let st = s.trim();
            if st.is_empty() {
                None
            } else {
                Some(st.to_string())
            }
        }
    };

    // Stata always reports UNKNOWN; None keeps "not declared" apart from "scale".
    // SAFETY: `var` is live (caller's guarantee).
    let measure = match unsafe { readstat_variable_get_measure(var) } {
        m if m == MEASURE_NOMINAL => Some("nominal".to_string()),
        m if m == MEASURE_ORDINAL => Some("ordinal".to_string()),
        m if m == MEASURE_SCALE => Some("scale".to_string()),
        _ => None,
    };

    // Only Stata stores integers; SAS and SPSS numerics are always doubles.
    // SAFETY: `var` is live (caller's guarantee).
    let kind = if unsafe { readstat_variable_get_type_class(var) } == TCLASS_STRING {
        ColKind::Str
    } else {
        // SAFETY: `var` is live (caller's guarantee).
        match unsafe { readstat_variable_get_type(var) } {
            t if t == T_INT8 => ColKind::I64 { storage: "int8" },
            t if t == T_INT16 => ColKind::I64 { storage: "int16" },
            t if t == T_INT32 => ColKind::I64 { storage: "int32" },
            t => ColKind::F64 {
                float32: t == T_FLOAT,
            },
        }
    };

    let label_set = if label_set_name.is_null() {
        None
    } else {
        // SAFETY: non-null, so NUL-terminated and live for the call (caller's guarantee).
        let s = rctx
            .decode(unsafe { CStr::from_ptr(label_set_name) })
            .into_owned();
        let st = s.trim();
        if st.is_empty() {
            None
        } else {
            Some(st.to_string())
        }
    };

    let user_missing = {
        // SAFETY: `var` is live (caller's guarantee).
        let missing_count = unsafe { readstat_variable_get_missing_ranges_count(var) };
        if missing_count > 0 {
            let mut values = Vec::new();
            let mut range: Option<Vec<f64>> = None;

            // SAFETY: `var` is live (caller's guarantee).
            let is_string = unsafe { readstat_variable_get_type_class(var) } == TCLASS_STRING;

            for i in 0..missing_count {
                // SAFETY: `var` is live and `i` is below its range count.
                let (lo_val, hi_val) = unsafe {
                    (
                        readstat_variable_get_missing_range_lo(var, i),
                        readstat_variable_get_missing_range_hi(var, i),
                    )
                };

                if is_string {
                    // SAV has no string ranges, so `lo_val` is the code.
                    // SAFETY: a string variable's missing value is a string whose text lives as long as `var`.
                    let sp = unsafe { readstat_string_value(lo_val) };
                    if !sp.is_null() {
                        // SAFETY: non-null ReadStat strings are NUL-terminated.
                        let code = rctx.decode(unsafe { CStr::from_ptr(sp) });
                        // SAV pads codes to 8 bytes; the data values carry no padding.
                        values.push(MissingValue::Str(code.trim_end().to_string()));
                    }
                } else {
                    // SAFETY: a numeric value is read by value; no pointer is followed.
                    let (lo, hi) =
                        unsafe { (readstat_double_value(lo_val), readstat_double_value(hi_val)) };
                    if lo.is_nan() || hi.is_nan() {
                        continue;
                    }

                    if (lo - hi).abs() < 1e-10 {
                        values.push(MissingValue::Num(lo));
                    } else {
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
            measure,
            user_missing,
            sb: Some(StringBuilder::with_capacity(
                cap,
                cap * STRING_CAPACITY_MULTIPLIER,
            )),
            fb: None,
            ib: None,
        },
        ColKind::F64 { .. } => ColBuilders {
            kind,
            name: name.clone(),
            label,
            label_set,
            fmt,
            measure,
            user_missing,
            sb: None,
            fb: Some(Float64Builder::with_capacity(cap)),
            ib: None,
        },
        ColKind::I64 { .. } => ColBuilders {
            kind,
            name: name.clone(),
            label,
            label_set,
            fmt,
            measure,
            user_missing,
            sb: None,
            fb: None,
            ib: Some(Int64Builder::with_capacity(cap)),
        },
    };

    rctx.var_cols[var_index] = Some(rctx.cols.len());
    rctx.cols.push(col);
    HANDLER_OK
}

/// # Safety
///
/// Called only by ReadStat: `var` is null or live for the call, `value` is live for the
/// call, and `ctx` is as `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_value_cb(
    row: c_int,
    var: *mut readstat_variable_t,
    value: readstat_value_t,
    ctx: *mut c_void,
) -> c_int {
    if var.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `var` is non-null and live.
    unsafe { guard_cb(ctx, |rctx| on_value_impl(row, var, value, rctx)) }
}

/// # Safety
///
/// `var` must be non-null and live for the call, and so must any string `value` holds.
unsafe fn on_value_impl(
    row: c_int,
    var: *mut readstat_variable_t,
    value: readstat_value_t,
    rctx: &mut ParseCtx,
) -> c_int {
    // Defensive: a negative row index would wrap to a huge usize below.
    if row < 0 {
        return HANDLER_OK;
    }
    let row_us = row as usize;
    rctx.n_rows_seen = rctx.n_rows_seen.max(row_us + 1);

    if row_us < rctx.rows_skip {
        return HANDLER_OK;
    }

    if let Some(nm) = rctx.n_max {
        // First disallowed row is rows_skip + n_max; n_max == 0 emits no rows.
        if row_us >= rctx.rows_skip.saturating_add(nm) {
            return HANDLER_ABORT;
        }
    }

    // Counted before the column lookup, so skipped columns cannot hide a row.
    if rctx.last_counted_row != Some(row_us) {
        rctx.last_counted_row = Some(row_us);
        rctx.n_rows_emitted += 1;
    }

    // SAFETY: `var` is live (caller's guarantee).
    let idx = match usize::try_from(unsafe { readstat_variable_get_index(var) })
        .ok()
        .and_then(|i| rctx.var_cols.get(i).copied().flatten())
    {
        Some(i) => i,
        None => return HANDLER_OK,
    };

    // SAFETY: these read `value`'s tag and type fields; no pointer is followed.
    let (is_tagged, is_missing, vt) = unsafe {
        (
            rctx.detect_tagged && readstat_value_is_tagged_missing(value) != 0,
            readstat_value_is_system_missing(value) != 0,
            rs_value_type(value),
        )
    };
    let is_str = vt == T_STRING || vt == T_STRING_REF;

    // Decoded before borrowing the column builder (decode needs &mut ctx).
    let str_val: Option<Cow<'_, str>> = if !is_tagged && !is_missing && is_str {
        // SAFETY: `value` is a string, so this reads its pointer field.
        let sp = unsafe { readstat_string_value(value) };
        if sp.is_null() {
            None
        } else {
            // SAFETY: ReadStat's string values are NUL-terminated and live for the call.
            Some(rctx.decode(unsafe { CStr::from_ptr(sp) }))
        }
    } else {
        None
    };

    let col = match rctx.cols.get_mut(idx) {
        Some(c) => c,
        None => return HANDLER_OK,
    };

    if is_tagged {
        // SAFETY: reads `value`'s tag field; no pointer is followed.
        let tag_ch = unsafe { readstat_value_tag(value) } as u8 as char;
        let (rows, tags) = rctx
            .tagged
            .entry(col.name.clone())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        rows.push(row as usize);
        tags.push(tag_ch.to_string());
        col.push_missing();
    } else if is_missing {
        col.push_missing();
    } else if is_str {
        match str_val {
            Some(s) => col.push_str(&s),
            None => col.push_missing(),
        }
    } else if vt == T_INT8 || vt == T_INT16 || vt == T_INT32 {
        // SAFETY: a numeric value is read by value; no pointer is followed.
        col.push_i64(unsafe { readstat_int32_value(value) } as i64);
    } else {
        // SAFETY: a numeric value is read by value; no pointer is followed.
        let d = unsafe { readstat_double_value(value) };
        col.push_f64(d);
    }

    HANDLER_OK
}

/// # Safety
///
/// Called only by ReadStat: `note` is null or NUL-terminated, and `ctx` is as `guard_cb`
/// requires.
pub(crate) unsafe extern "C" fn on_note_cb(
    _note_index: c_int,
    note: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if note.is_null() || ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; `note` is non-null and NUL-terminated.
    unsafe {
        guard_cb(ctx, |rctx| {
            let s = rctx.decode(CStr::from_ptr(note)).into_owned();
            rctx.notes.push(s);
            HANDLER_OK
        })
    }
}

/// # Safety
///
/// Called only by ReadStat: `set_name` and `label` are null or NUL-terminated, any string
/// `val` holds is live for the call, and `ctx` is as `guard_cb` requires.
pub(crate) unsafe extern "C" fn on_value_label_cb(
    set_name: *const c_char,
    val: readstat_value_t,
    label: *const c_char,
    ctx: *mut c_void,
) -> c_int {
    if ctx.is_null() {
        return HANDLER_OK;
    }
    // SAFETY: `ctx` is the reader's `ParseCtx`; the other arguments are ReadStat's.
    unsafe { guard_cb(ctx, |rctx| on_value_label_impl(set_name, val, label, rctx)) }
}

/// # Safety
///
/// `set_name` and `label` must be null or NUL-terminated, and any string `val` holds
/// must be live for the call.
unsafe fn on_value_label_impl(
    set_name: *const c_char,
    val: readstat_value_t,
    label: *const c_char,
    rctx: &mut ParseCtx,
) -> c_int {
    let set = if set_name.is_null() {
        "__default__".to_string()
    } else {
        // SAFETY: non-null, so NUL-terminated (caller's guarantee).
        rctx.decode(unsafe { CStr::from_ptr(set_name) })
            .trim()
            .to_string()
    };
    let lab = if label.is_null() {
        String::new()
    } else {
        // SAFETY: non-null, so NUL-terminated (caller's guarantee).
        rctx.decode(unsafe { CStr::from_ptr(label) }).into_owned()
    };

    // SAFETY: these read `val`'s missing flag and type; no pointer is followed.
    let (is_missing, vt) = unsafe {
        (
            readstat_value_is_system_missing(val) != 0,
            rs_value_type(val),
        )
    };
    let key = if is_missing {
        "".to_string()
    } else if vt == T_STRING || vt == T_STRING_REF {
        // SAFETY: `val` is a string, so this reads its pointer field.
        let sp = unsafe { readstat_string_value(val) };
        if sp.is_null() {
            String::new()
        } else {
            // SAFETY: non-null and NUL-terminated, live for the call (caller's guarantee).
            rctx.decode(unsafe { CStr::from_ptr(sp) }).into_owned()
        }
    } else {
        // SAFETY: a numeric value is read by value; no pointer is followed.
        format!("{}", unsafe { readstat_double_value(val) })
    };

    let map = rctx.label_sets.entry(set).or_default();
    map.insert(key, lab);
    HANDLER_OK
}

pub(crate) fn finalize_to_ipc(mut ctx: ParseCtx) -> Result<(Vec<u8>, MetaOut)> {
    let internal = |what: &str| Error::Internal(format!("{what} builder missing"));
    let arrow = |e: arrow::error::ArrowError| Error::Internal(format!("Arrow IPC: {e}"));

    {
        let mut seen = HashSet::with_capacity(ctx.cols.len());
        if let Some(dup) = ctx.cols.iter().find(|c| !seen.insert(c.name.as_str())) {
            return Err(Error::Parse(format!(
                "the file has more than one variable named {:?}",
                dup.name
            )));
        }
    }

    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let mut vars_meta = Vec::with_capacity(ctx.cols.len());

    for mut col in ctx.cols.drain(..) {
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
                let arr =
                    Arc::new(col.sb.take().ok_or_else(|| internal("string"))?.finish()) as ArrayRef;
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
                    measure: col.measure,
                    user_missing: col.user_missing,
                });
            }
            ColKind::F64 { float32 } => {
                let arr =
                    Arc::new(col.fb.take().ok_or_else(|| internal("float"))?.finish()) as ArrayRef;
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
                    kind: if float32 { "float" } else { "double" }.into(),
                    measure: col.measure,
                    user_missing: col.user_missing,
                });
            }
            ColKind::I64 { storage } => {
                let arr = Arc::new(col.ib.take().ok_or_else(|| internal("integer"))?.finish())
                    as ArrayRef;
                let mut field = Field::new(&col.name, DataType::Int64, true);
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
                    kind: storage.into(),
                    measure: col.measure,
                    user_missing: col.user_missing,
                });
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(arrow)?;

    // Uncompressed: faster to write.
    let write_options = IpcWriteOptions::default();
    let mut buf = Vec::new();
    {
        let mut w =
            FileWriter::try_new_with_options(&mut buf, &schema, write_options).map_err(arrow)?;
        w.write(&batch).map_err(arrow)?;
        w.finish().map_err(arrow)?;
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
        had_invalid_utf8: ctx.had_invalid_utf8,
        encoding: None,
        catalog_error: None,
    };

    Ok((buf, meta))
}
