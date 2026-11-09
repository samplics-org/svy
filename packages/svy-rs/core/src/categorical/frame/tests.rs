use super::*;

#[test]
fn sort_labels_orders_numbers_by_value() {
    let mut ints = vec!["10".to_string(), "2".to_string(), "1".to_string()];
    sort_labels(&mut ints, &DataType::Int64);
    assert_eq!(ints, ["1", "2", "10"]);

    let mut floats = vec!["10.5".to_string(), "-1.0".to_string(), "2.0".to_string()];
    sort_labels(&mut floats, &DataType::Float64);
    assert_eq!(floats, ["-1.0", "2.0", "10.5"]);

    let mut text = vec!["b".to_string(), "10".to_string(), "2".to_string()];
    sort_labels(&mut text, &DataType::String);
    assert_eq!(text, ["10", "2", "b"]);
}
