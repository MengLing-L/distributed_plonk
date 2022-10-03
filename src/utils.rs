use ark_ff::PrimeField;

#[derive(Clone, Debug)]
pub struct FftWorkload {
    pub row_start: usize,
    pub row_end: usize,
    pub col_start: usize,
    pub col_end: usize,
}

impl FftWorkload {
    pub const fn num_rows(&self) -> usize {
        self.row_end - self.row_start
    }

    pub const fn num_cols(&self) -> usize {
        self.col_end - self.col_start
    }
}

#[derive(Clone, Debug)]
pub struct MsmWorkload {
    pub start: usize,
    pub end: usize,
}

pub fn serialize<T>(v: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            v as *const _ as *const u8,
            v.len() * std::mem::size_of::<T>(),
        )
    }
}

pub fn deserialize<T>(r: &[u8]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            r as *const _ as *const T,
            r.len() / std::mem::size_of::<T>(),
        )
    }
}
