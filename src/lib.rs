use std::collections::BTreeMap;
use std::io;

use arrow::array::PrimitiveArray;
use arrow::record_batch::RecordBatch;

use arrow::datatypes::ArrowPrimitiveType;

use arrow::datatypes::SchemaRef;

/// Computes the frequencies of each value in a [`PrimitiveArray`].
///
/// # Arguments
///
/// * `arr`: A [`PrimitiveArray`] of values.
///
/// # Returns
///
/// A [`Result`] containing a [`BTreeMap`] with the frequencies of each value,
/// or an [`io::Error`] if an error occurs.
///
/// # Example
///
/// ```
/// use arrow::array::Int32Array;
/// use rs_arrow_array2freq2rbat::frequencies;
///
/// let arr = Int32Array::from(vec![Some(2), Some(3), Some(3), Some(5), Some(5), Some(5)]);
/// let freqs = frequencies(arr).unwrap();
///
/// assert_eq!(freqs.get(&2), Some(&1));
/// assert_eq!(freqs.get(&3), Some(&2));
/// assert_eq!(freqs.get(&5), Some(&3));
/// ```
pub fn frequencies<T>(arr: PrimitiveArray<T>) -> Result<BTreeMap<T::Native, u64>, io::Error>
where
    T: ArrowPrimitiveType,
    T::Native: Ord,
{
    let mut freqs = BTreeMap::new();
    for item in arr.iter().flatten() {
        *freqs.entry(item).or_insert(0) += 1;
    }
    Ok(freqs)
}

/// Converts a `BTreeMap` of frequencies into an Arrow `RecordBatch`.
///
/// # Arguments
///
/// * `m`: A `BTreeMap` where keys are the values and values are their frequencies.
/// * `sch`: The `SchemaRef` for the output `RecordBatch`.
///
/// # Returns
///
/// A `Result` containing a `RecordBatch` with two columns, whose names are defined by the provided `SchemaRef`,
/// or an `io::Error` if an error occurs.
///
/// # Example
///
/// ```
/// use std::collections::BTreeMap;
/// use std::sync::Arc;
/// use arrow::datatypes::{DataType, Field, Schema};
/// use rs_arrow_array2freq2rbat::map2batch;
///
/// let mut freqs = BTreeMap::new();
/// freqs.insert(1, 1);
/// freqs.insert(2, 2);
/// freqs.insert(3, 3);
///
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("value", DataType::Int32, false),
///     Field::new("frequency", DataType::UInt64, false),
/// ]));
///
/// let batch = map2batch::<arrow::datatypes::Int32Type>(freqs, &schema).unwrap();
///
/// assert_eq!(batch.num_columns(), 2);
/// assert_eq!(batch.num_rows(), 3);
/// ```
pub fn map2batch<T>(m: BTreeMap<T::Native, u64>, sch: &SchemaRef) -> Result<RecordBatch, io::Error>
where
    T: ArrowPrimitiveType,
{
    let keys: Vec<T::Native> = m.keys().cloned().collect();
    let values: Vec<u64> = m.values().cloned().collect();

    let key_array = PrimitiveArray::<T>::from_iter_values(keys);
    let val_array = PrimitiveArray::<arrow::datatypes::UInt64Type>::from_iter(values);

    let batch = RecordBatch::try_new(
        sch.clone(),
        vec![
            std::sync::Arc::new(key_array),
            std::sync::Arc::new(val_array),
        ],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;

    Ok(batch)
}

/// Computes the frequencies of each value in a [`PrimitiveArray`] and converts the result into an Arrow [`RecordBatch`].
///
/// # Arguments
///
/// * `sch`: The `SchemaRef` for the output `RecordBatch`. This allows the caller to define custom column names and metadata for the resulting frequency and value columns. The schema is expected to have two fields, where the first field corresponds to the values from the `PrimitiveArray` (with `T::DATA_TYPE`) and the second field corresponds to the frequencies (with `DataType::UInt64`).
///
/// # Returns
///
/// A [`Result`] containing a [`RecordBatch`] with two columns, whose names are defined by the provided `SchemaRef`,
/// or an [`io::Error`] if an error occurs.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use arrow::array::Int32Array;
/// use arrow::datatypes::{DataType, Field, Schema};
/// use rs_arrow_array2freq2rbat::array2frequency2batch;
///
/// let arr = Int32Array::from(vec![Some(2), Some(3), Some(3), Some(5), Some(5), Some(5)]);
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("value", DataType::Int32, false),
///     Field::new("frequency", DataType::UInt64, false),
/// ]));
/// let batch = array2frequency2batch(arr, schema).unwrap();
///
/// assert_eq!(batch.num_columns(), 2);
/// assert_eq!(batch.num_rows(), 3);
/// ```
pub fn array2frequency2batch<T>(
    arr: PrimitiveArray<T>,
    sch: SchemaRef,
) -> Result<RecordBatch, io::Error>
where
    T: ArrowPrimitiveType,
    T::Native: Ord,
{
    let freqs = frequencies(arr)?;
    map2batch::<T>(freqs, &sch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;

    #[test]
    fn test_frequencies() {
        let arr = Int32Array::from(vec![1, 2, 2, 3, 3, 3]);
        let freqs = frequencies(arr).unwrap();
        assert_eq!(freqs.get(&1), Some(&1));
        assert_eq!(freqs.get(&2), Some(&2));
        assert_eq!(freqs.get(&3), Some(&3));
    }

    #[test]
    fn test_map2batch() {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let mut freqs = BTreeMap::new();
        freqs.insert(1, 1);
        freqs.insert(2, 2);
        freqs.insert(3, 3);

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new("frequency", DataType::UInt64, false),
        ]));

        let batch = map2batch::<arrow::datatypes::Int32Type>(freqs, &schema).unwrap();

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        let value_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let freq_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();

        assert_eq!(value_array.value(0), 1);
        assert_eq!(value_array.value(1), 2);
        assert_eq!(value_array.value(2), 3);

        assert_eq!(freq_array.value(0), 1);
        assert_eq!(freq_array.value(1), 2);
        assert_eq!(freq_array.value(2), 3);
    }

    #[test]
    fn test_array2frequency2batch() {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let arr = PrimitiveArray::<arrow::datatypes::Int32Type>::from(vec![1, 2, 2, 3, 3, 3]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new("frequency", DataType::UInt64, false),
        ]));
        let batch = array2frequency2batch(arr, schema).unwrap();

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        let value_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let freq_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();

        assert_eq!(value_array.value(0), 1);
        assert_eq!(value_array.value(1), 2);
        assert_eq!(value_array.value(2), 3);

        assert_eq!(freq_array.value(0), 1);
        assert_eq!(freq_array.value(1), 2);
        assert_eq!(freq_array.value(2), 3);
    }
}
