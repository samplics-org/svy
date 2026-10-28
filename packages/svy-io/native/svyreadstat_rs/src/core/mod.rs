use arrow::array::{ArrayRef, Float64Builder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use arrow::record_batch::RecordBatch;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;

use readstat_sys::{
    readstat_double_value, readstat_get_file_label, readstat_get_row_count, readstat_int32_value,
    readstat_measure_e_READSTAT_MEASURE_NOMINAL as MEASURE_NOMINAL,
    readstat_measure_e_READSTAT_MEASURE_ORDINAL as MEASURE_ORDINAL,
    readstat_measure_e_READSTAT_MEASURE_SCALE as MEASURE_SCALE, readstat_metadata_t,
    readstat_string_value, readstat_type_class_e_READSTAT_TYPE_CLASS_STRING as TCLASS_STRING,
    readstat_type_e_READSTAT_TYPE_FLOAT as T_FLOAT, readstat_type_e_READSTAT_TYPE_INT8 as T_INT8,
    readstat_type_e_READSTAT_TYPE_INT16 as T_INT16, readstat_type_e_READSTAT_TYPE_INT32 as T_INT32,
    readstat_type_e_READSTAT_TYPE_STRING as T_STRING,
    readstat_type_e_READSTAT_TYPE_STRING_REF as T_STRING_REF, readstat_value_is_system_missing,
    readstat_value_is_tagged_missing, readstat_value_t, readstat_value_tag,
    readstat_value_type as rs_value_type, readstat_variable_get_format,
    readstat_variable_get_index, readstat_variable_get_label, readstat_variable_get_measure,
    readstat_variable_get_missing_range_hi, readstat_variable_get_missing_range_lo,
    readstat_variable_get_missing_ranges_count, readstat_variable_get_name,
    readstat_variable_get_type, readstat_variable_get_type_class, readstat_variable_t,
};

use crate::errors::{Error, Op, Result, readstat_message};

mod callbacks;
mod ctx;
mod guard;
mod meta;
mod setup;
#[cfg(test)]
mod tests;

pub(crate) use callbacks::*;
pub(crate) use ctx::*;
pub(crate) use guard::*;
pub(crate) use meta::*;
pub(crate) use setup::*;
