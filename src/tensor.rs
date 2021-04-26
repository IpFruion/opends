use std::{fmt, ops};
use std::fmt::{Debug, Formatter};
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

#[macro_export]
macro_rules! tensor_index {
    ($($x:expr),+) => {{
        let mut v: Vec<$crate::tensor::DimensionIndex> = Vec::new();
        $(
            v.push($x.into());
        )+
        v
    }}
}


#[derive(Debug, Clone)]
pub struct IndexError(String);

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

//TODO: Create TensorIndex structure to hold Vec[TensorIndexPiece...]
// pub struct TensorIndex {
//     index: Vec<DimensionIndex>
// }
//
// impl TensorIndex {
//     pub fn new() -> Self {
//         TensorIndex {
//             index: Vec::new()
//         }
//     }
// }

// impl<K> From<K> for TensorIndex where K: AsRef<[DimensionIndex]> {
//     fn from(k: K) -> Self {
//         TensorIndex {
//             index: Vec::from(k.as_ref())
//         }
//     }
// }

/// This is an index for the tensor for when t.get is called
#[derive(Clone)]
pub struct DimensionIndex {
    index: Vec<(Option<usize>, Option<usize>)>
}


impl From<usize> for DimensionIndex {
    fn from(u: usize) -> Self {
        let mut v = Vec::new();
        v.push((Some(u), Some(u)));
        DimensionIndex {
            index: v
        }
    }
}

impl From<Range<usize>> for DimensionIndex {
    fn from(r: Range<usize>) -> Self {
        let mut v = Vec::new();
        v.push((Some(r.start), Some(r.end)));
        DimensionIndex {
            index: v
        }
    }
}

impl From<RangeTo<usize>> for DimensionIndex {
    fn from(r: RangeTo<usize>) -> Self {
        let mut v = Vec::new();
        v.push((None, Some(r.end)));
        DimensionIndex {
            index: v
        }
    }
}

impl From<RangeFrom<usize>> for DimensionIndex {
    fn from(r: RangeFrom<usize>) -> Self {
        let mut v = Vec::new();
        v.push((Some(r.start), None));
        DimensionIndex {
            index: v
        }
    }
}

impl From<RangeFull> for DimensionIndex {
    fn from(_: RangeFull) -> Self {
        DimensionIndex {
            index: Vec::new()
        }
    }
}

impl From<Vec<DimensionIndex>> for DimensionIndex {
    fn from(f: Vec<DimensionIndex>) -> Self {
        if f.len() == 0 {
            panic!("Vector of Dimension Index can't be empty.")
        }
        let mut v = Vec::new();
        for x in f {
            if x.index.len() == 0 {
                return DimensionIndex {
                    index: Vec::new()
                };
            }
            v.push(x.index[0])
        }

        //TODO: Optimize v such that if there is overlap they are removed / addressed. i.e. vec![..4, ..2] is consumed by
        // ..4 so just return ..4 DimensionIndex
        DimensionIndex {
            index: v
        }
    }
}

impl DimensionIndex {
    //TODO: Fix with length of index i.e. test for more than one lengths
    fn retain<T>(&self, i: usize, shape: &Vec<usize>, factors: &Vec<usize>, data: &mut Vec<T>) -> Result<(), IndexError> {
        // Check valid bounds
        if self.index.len() == 0 {
            return Ok(());
        }
        let se_list: Vec<_> = self.index.iter()
            .map(|(start, end)| {
                if *start == None && end.unwrap() >= shape[i] {
                    return Err(IndexError(String::from("Index out of Dimension")));
                }

                let s = match start {
                    None => 0usize,
                    Some(x) => x * factors[i + 1]
                };
                let e = match end {
                    None => shape[i] * factors[i + 1],
                    Some(x) => (x + 1) * factors[i + 1]
                };

                if s >= e {
                    return Err(IndexError(String::from("Index start can't be more than index end")));
                }
                Ok((s, e))
            }).collect::<Result<Vec<_>, IndexError>>()?;
        let f = factors[i];
        let mut j = 0;
        data.retain(|_| {
            let k = j % f;
            for (s, e) in &se_list {
                if *s <= k && k < *e {
                    j += 1;
                    return true;
                }
            }
            j += 1;
            return false;
        });
        Ok(())
    }
}

impl Default for DimensionIndex {
    fn default() -> Self {
        (..).into()
    }
}


//TODO: implement situation where bound * bound?
impl<K: Copy> ops::Mul<K> for DimensionIndex where usize: ops::Mul<K, Output=usize> {
    type Output = DimensionIndex;

    fn mul(self, rhs: K) -> Self::Output {
        let mut v = Vec::new();
        for x in self.index {
            v.push((
                match x.0 {
                    None => None,
                    Some(x) => Some(x * rhs)
                },
                match x.1 {
                    None => None,
                    Some(x) => Some(x * rhs)
                }
            ))
        }
        DimensionIndex {
            index: v
        }
    }
}

impl Debug for DimensionIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for x in &self.index {
            write!(f, "{}, {}", match x.0 {
                None => String::from("None"),
                Some(x) => x.to_string()
            }, match x.1 {
                None => String::from("None"),
                Some(x) => x.to_string()
            })?
        }
        Ok(())
    }
}


// Need lifetime parameter because dimensions is stored inside this structure so who knows when it will be used up
// TODO: Replace Vec<Option<T>> with custom pointer at some point instead of it but for now it's fine

// Do we want all the variables stored in the matrix to be able to reference outside or be able to just store them inside
// and reference them there. i.e. mat[[2, 3]] = 43 instead of someplace = 43


/// A Tensor structure that can have any arbitrary rank > 0.
#[derive(Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

fn get_tensor_index(dimensions: &Vec<usize>, indices: &[usize]) -> usize {
    let mut index = 0;
    let mut factor = 1;
    for i in (0..indices.len() - 1).rev() {
        factor *= dimensions[i + 1];
        index += indices[i] * factor;
    }
    index += indices[indices.len() - 1];
    return index;
}

fn get_indices_from_index(dimensions: &Vec<usize>, index: usize) -> Vec<usize> {
    let mut indices = vec![0; dimensions.len()];
    let mut ind = index;
    for i in (0..dimensions.len()).rev() {
        indices[i] = ind % dimensions[i];
        ind /= dimensions[i];
    }
    return indices;
}

fn factors(shape: &Vec<usize>) -> Vec<usize> {
    let mut factor = 1;
    let mut factors = Vec::new();
    for i in (0..shape.len()).rev() {
        factors.push(factor);
        factor *= shape[i];
    }
    factors.push(factor);
    factors.reverse();
    factors
}

fn get_shape_and_size<K>(shape: K) -> (Vec<usize>, usize) where K: AsRef<[usize]> {
    let shape = shape.as_ref();
    if shape.len() == 0 {
        panic!("Tensor can't be of Zero Dimension since Tensor::new([1]) is the same thing.")
    }
    let mut s = Vec::new();
    let mut total = 1;
    for i in shape {
        if *i == 0 {
            panic!("Tensor can't be of Zero Size in a dimension i.e. Tensor::new([x]) x > 0.")
        }
        s.push(*i);
        total *= i;
    }
    return (s, total);
}

//TODO: Optimize. Can you instead of traversing Vec for the length of the shape
//Instead by once to calculate the ranges of each index?
fn get_internal<J, K>(shape: &Vec<usize>, data: &mut Vec<J>, indices: K) -> Result<(), IndexError> where K: AsRef<[DimensionIndex]> {
    let factors = factors(shape);
    let indices = indices.as_ref();
    for i in 0..shape.len() {
        if i < indices.len() {
            indices[i].retain(i, shape, &factors, data)?;
        } else {
            DimensionIndex::default().retain(i, shape, &factors, data)?;
        }
    }
    Ok(())
}

pub struct MultiplyError(String);

impl<T: Clone> Tensor<T> {

    pub fn fill<K>(shape: K, fill: T) -> Tensor<T> where K: AsRef<[usize]> {
        let (shape, total) = get_shape_and_size(shape);
        Tensor {
            data: vec![fill; total],
            shape,
        }
    }

    pub fn reshape_fill<K>(&mut self, new_shape: K, x: T) where K: AsRef<[usize]> {
        self.reshape_with(new_shape, || x.clone());
    }

    pub fn reshape_with_last<K>(&mut self, new_shape: K) where K: AsRef<[usize]> {
        let x = self.data.last().unwrap().clone();
        self.reshape_with(new_shape, move || x.clone());
    }

}

impl<T: Default> Tensor<T> {

    pub fn new<K>(shape: K) -> Tensor<T> where K: AsRef<[usize]> {
        let (shape, total) = get_shape_and_size(shape);
        Tensor {
            data: (0..total).into_iter().map(|_| Default::default()).collect(),
            shape,
        }
    }

    pub fn reshape<K>(&mut self, new_shape: K) where K: AsRef<[usize]> {
        self.reshape_with(new_shape, Default::default);
    }

}

impl<T> Tensor<T> {

    pub fn fill_with<K, F>(shape: K, f: F) -> Self where K: AsRef<[usize]>, F: Fn(usize) -> T {
        let (shape, total) = get_shape_and_size(shape);
        Tensor {
            shape,
            data: (0..total).map(f).collect()
        }
    }

    //TODO: Somehow translate these to use Index<K> and output the correct reference that isn't local
    pub fn get<K>(&self, indices: K) -> Result<Vec<&T>, IndexError> where K: AsRef<[DimensionIndex]> {
        let mut v: Vec<&T> = self.data.iter().collect();
        get_internal(&self.shape, &mut v, indices.as_ref()).and_then(|_| Ok(v))
    }

    pub fn get_mut<K>(&mut self, indices: K) -> Result<Vec<&mut T>, IndexError> where K: AsRef<[DimensionIndex]> {
        let mut v: Vec<&mut T> = self.data.iter_mut().collect();
        get_internal(&self.shape, &mut v, indices.as_ref()).and_then(|_| Ok(v))
    }

    pub fn iter(&self) -> impl Iterator<Item=(Vec<usize>, &T)> {
        let shape = &self.shape;
        self.data.iter()
            .enumerate()
            .map(move |(i, x)| {
                (get_indices_from_index(shape, i), x)
            })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=(Vec<usize>, &mut T)> {
        let shape = &self.shape;
        self.data.iter_mut()
            .enumerate()
            .map(move |(i, x)| {
                (get_indices_from_index(shape, i), x)
            })
    }

    pub fn reshape_with<K, F>(&mut self, new_shape: K, f: F) where K: AsRef<[usize]>, F: FnMut() -> T {
        self.shape.clear();
        for x in new_shape.as_ref() {
            if *x == 0 {
                //TODO: Handle this better maybe not panic?
                panic!("Shape can't have zero dimension")
            }
            self.shape.push(*x);
        }
        self.data.resize_with(self.shape.iter().product(), f)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Asynchronous tensor multiplication around two axes in each tensor
    /// Currently this only works with tensors of size less than 2
    /// Working on async use of futures
    // pub async fn multiply_async<K>(self, rhs: Tensor<K>) -> Result<Tensor<T>, MultiplyError>
    //     where T: Copy + ops::Mul<K, Output=T> + ops::Add<T, Output=T>, K: Copy {
    //     let s_len = self.shape.len();
    //     let r_len = rhs.shape.len();
    //     if s_len > 2 || r_len > 2 {
    //         return Err(MultiplyError("Can't do higher dimensions yet. Sorry".to_string()));
    //     }
    //     if s_len != 2 {
    //         return Err(MultiplyError("Self dimension needs to be 2 for MxN * NxP".to_string()));
    //     }
    //     if self.shape[1] != rhs.shape[0] {
    //         return Err(MultiplyError("Shapes need same column and row MxN * NxP".to_string()));
    //     }
    //
    //     let new_shape;
    //     if r_len == 1 {
    //         new_shape = vec![self.shape[0]];
    //     } else {
    //         new_shape = vec![self.shape[0], rhs.shape[1]]
    //     }
    //
    //     async fn muliply_row<T, K>(i: usize, j: usize, p: usize, s_rows: usize, r_rows: usize, s_data: &Vec<T>, r_data: &Vec<K>) -> Option<T>
    //         where T: Copy + ops::Mul<K, Output=T> + ops::Add<T, Output=T>, K: Copy {
    //         let mut sum = None;
    //         for k in 0..p {
    //             let v = s_data[s_rows * i + k] * r_data[k * r_rows + j];
    //             if k == 0 {
    //                 sum = Some(v);
    //             } else {
    //                 sum = Some(sum.unwrap() + v);
    //             }
    //         }
    //         sum
    //     }
    //     // let mut new_data = Vec::new();
    //     let mut futures = Vec::new();
    //
    //     let p = self.shape[1];
    //     for i in 0..self.shape[0] {
    //         for j in 0..rhs.shape[1] {
    //             futures.push(muliply_row(i, j, p, self.shape[0], rhs.shape[0], &self.data, &rhs.data))
    //         }
    //     }
    //
    //
    //     Err(MultiplyError("Not implemented".to_string()))
    // }

    pub fn multiply<K>(self, rhs: Tensor<K>) -> Result<Tensor<T>, MultiplyError>
        where T: Copy + ops::Mul<K, Output=T> + ops::Add<T, Output=T>, K: Copy {
        let s_len = self.shape.len();
        let r_len = rhs.shape.len();
        if s_len > 2 || r_len > 2 {
            return Err(MultiplyError("Can't do higher dimensions yet. Sorry".to_string()));
        }
        if s_len != 2 {
            return Err(MultiplyError("Self dimension needs to be 2 for MxN * NxP".to_string()));
        }
        if self.shape[1] != rhs.shape[0] {
            return Err(MultiplyError("Shapes need same column and row MxN * NxP".to_string()));
        }

        let new_shape = vec![self.shape[0], rhs.shape[1]];
        let mut new_data = Vec::with_capacity(new_shape.iter().product());
        for i in 0..self.shape[0] {
            for j in 0..rhs.shape[1] {
                let mut sum = None;
                let m;
                if s_len == 1 {
                    m = 1;
                } else {
                    m = self.shape[1];
                }
                for k in 0..m {
                    let v = self.data[i * self.shape[0] + k] * rhs.data[k * rhs.shape[0] + j];
                    if k == 0 {
                        sum = Some(v);
                    } else {
                        sum = Some(sum.unwrap() + v);
                    }
                }
                new_data.push(sum.unwrap());
            }
        }
        Ok(Tensor {
            shape: new_shape,
            data: new_data,
        })
    }
}

impl<K, T: Clone> From<(K, Vec<T>)> for Tensor<T> where K: AsRef<[usize]> {
    fn from((shape, mut data): (K, Vec<T>)) -> Self {
        let (shape, total) = get_shape_and_size(shape);
        if data.len() != total {
            let x = data.last().unwrap().clone();
            data.resize_with(total, || x.clone());
        }
        Tensor {
            shape: Vec::from(shape.as_ref()),
            data,
        }
    }
}

impl<T, K> ops::Index<K> for Tensor<T> where K: AsRef<[usize]> {
    type Output = T;

    fn index(&self, index: K) -> &Self::Output {
        &self.data[get_tensor_index(&self.shape, index.as_ref())]
    }
}

impl<T, K> ops::IndexMut<K> for Tensor<T> where K: AsRef<[usize]> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        &mut self.data[get_tensor_index(&self.shape, index.as_ref())]
    }
}

impl<T, K> ops::Add<Tensor<K>> for Tensor<T>
    where T: Copy + ops::Add<K, Output=T>, K: Copy {
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<K>) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("Tensor dimensions need to match to do element wise addition.")
        }
        let mut s_iter = self.data.iter();
        let mut r_iter = rhs.data.iter();
        let mut v = Vec::new();
        loop {
            let s = s_iter.next();
            match s {
                None => {
                    break;
                }
                Some(x) => {
                    v.push(*x + *r_iter.next().unwrap());
                }
            }
        }
        Tensor {
            shape: self.shape,
            data: v,
        }
    }
}

impl<T, K> ops::Add<K> for Tensor<T>
    where T: Copy + ops::Add<K, Output=T>, K: Copy {
    type Output = Tensor<T>;
    fn add(self, rhs: K) -> Tensor<T> {
        Tensor {
            data: self.data.iter().map(|x| *x + rhs).collect(),
            shape: self.shape,
        }
    }
}

impl<T, K> ops::Mul<K> for Tensor<T>
    where T: Copy + ops::Mul<K, Output=T>, K: Copy {
    type Output = Tensor<T>;

    fn mul(self, rhs: K) -> Self::Output {
        Self {
            data: self.data.iter().map(|x| *x * rhs).collect(),
            shape: self.shape,
        }
    }
}

impl<T, K> ops::Div<K> for Tensor<T>
    where T: Copy + ops::Div<K, Output=T>, K: Copy {
    type Output = Tensor<T>;

    fn div(self, rhs: K) -> Self::Output {
        Self {
            data: self.data.iter().map(|x| *x / rhs).collect(),
            shape: self.shape,
        }
    }
}

impl<T, K> ops::Sub<K> for Tensor<T>
    where T: Copy + ops::Sub<K, Output=T>, K: Copy {
    type Output = Tensor<T>;

    fn sub(self, rhs: K) -> Self::Output {
        Self {
            data: self.data.iter().map(|x| *x - rhs).collect(),
            shape: self.shape,
        }
    }
}

impl<T: ToString> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut out_str = String::new();
        for i in 0..self.data.len() {
            let mut factor = 1;
            for j in (0..self.shape.len()).rev() {
                if i % (factor * self.shape[j]) == 0 {
                    out_str.push_str("{");
                }
                factor *= self.shape[j];
            }
            out_str.push_str(&self.data[i].to_string());
            out_str.push_str(", ");
            factor = 1;
            for j in (0..self.shape.len()).rev() {
                if (i > 0) && ((i + 1) % (factor * self.shape[j]) == 0) {
                    out_str = out_str.chars().take(out_str.len() - 2).collect();
                    out_str.push_str("}, ");
                }
                factor *= self.shape[j];
            }
        }
        out_str = out_str.chars().take(out_str.len() - 2).collect();
        write!(f, "{}", out_str)
    }
}


pub mod matrix {
    use std::fmt::{Display, Formatter, Debug};
    use std::fmt;
    use std::ops::{Add, Mul, Index, IndexMut, Div, Sub};

    use crate::tensor::{MultiplyError, Tensor, DimensionIndex, IndexError};

    pub struct Matrix<T> {
        tensor: Tensor<T>,
    }

    impl<T: Default> Matrix<T> {
        pub fn new(height: usize, width: usize) -> Self {
            Matrix {
                tensor: Tensor::new([height, width])
            }
        }

        pub fn reshape(&mut self, rows: usize, columns: usize) {
            self.tensor.reshape([rows, columns]);
        }

    }

    impl<T: Clone> Matrix<T> {
        pub fn fill(rows: usize, columns: usize, x: T) -> Self {
            Matrix {
                tensor: Tensor::fill([rows, columns], x)
            }
        }

        pub fn reshape_fill(&mut self, rows: usize, columns: usize, x: T) {
            self.tensor.reshape_fill([rows, columns], x);
        }

        pub fn reshape_with_last(&mut self, rows: usize, columns: usize) {
            self.tensor.reshape_with_last([rows, columns]);
        }

    }

    impl<T> Matrix<T> {

        pub fn fill_with<F>(rows: usize, columns: usize, f: F) -> Self where F: Fn(usize) -> T {
            Matrix {
                tensor: Tensor::fill_with([rows, columns], f)
            }
        }

        pub fn reshape_with<F>(&mut self, rows: usize, columns: usize, f: F) where F: FnMut() -> T {
            self.tensor.reshape_with([rows, columns], f);
        }

        pub fn rows(&self) -> usize {
            self.tensor.shape[0]
        }

        pub fn columns(&self) -> usize {
            self.tensor.shape[1]
        }

        pub fn get(&self, index: [DimensionIndex; 2]) -> Result<Vec<&T>, IndexError> {
            self.tensor.get(index)
        }

        pub fn get_mut(&mut self, index: [DimensionIndex; 2]) -> Result<Vec<&mut T>, IndexError> {
            self.tensor.get_mut(index)
        }

        // pub async fn multiply<K>(self, rhs: Matrix<K>) -> Matrix<T> {
        //
        // }

    }

    impl<T> From<Tensor<T>> for Matrix<T> {
        fn from(t: Tensor<T>) -> Self {
            if t.shape.len() != 2 {
                panic!("Tensor rank has to be 2")
            }
            Matrix {
                tensor: t
            }
        }
    }

    impl<T, K> Mul<K> for Matrix<T>
        where T: Copy + Mul<K, Output=T>, K: Copy {
        type Output = Matrix<T>;

        fn mul(self, rhs: K) -> Self::Output {
            (self.tensor * rhs).into()
        }
    }

    impl<T, K> Div<K> for Matrix<T>
        where T: Copy + Div<K, Output=T>, K: Copy {
        type Output = Matrix<T>;

        fn div(self, rhs: K) -> Self::Output {
            (self.tensor / rhs).into()
        }
    }

    impl<T, K> Add<K> for Matrix<T>
        where T: Copy + Add<K, Output=T>, K: Copy {
        type Output = Matrix<T>;

        fn add(self, rhs: K) -> Self::Output {
            (self.tensor + rhs).into()
        }
    }

    impl<T, K> Sub<K> for Matrix<T>
        where T: Copy + Sub<K, Output=T>, K: Copy {
        type Output = Matrix<T>;

        fn sub(self, rhs: K) -> Self::Output {
            (self.tensor - rhs).into()
        }
    }

    impl<T, K> Mul<Matrix<K>> for Matrix<T>
        where T: Copy + Mul<K, Output=T> + Add<T, Output=T>, K: Copy {
        type Output = Matrix<T>;

        fn mul(self, rhs: Matrix<K>) -> Self::Output {
            if self.tensor.shape[1] != rhs.tensor.shape[0] {
                panic!("Shapes need same column and row MxN * NxP");
            }
            let new_shape = vec![self.tensor.shape[0], rhs.tensor.shape[1]];
            let mut new_data = Vec::with_capacity(new_shape.iter().product());
            for i in 0..self.tensor.shape[0] {
                for j in 0..rhs.tensor.shape[1] {
                    let mut sum = None;
                    for k in 0..self.tensor.shape[1] {
                        let v = self.tensor.data[i * self.tensor.shape[0] + k] * rhs.tensor.data[k * rhs.tensor.shape[0] + j];
                        if k == 0 {
                            sum = Some(v);
                        } else {
                            sum = Some(sum.unwrap() + v);
                        }
                    }
                    new_data.push(sum.unwrap());
                }
            }
            Matrix {
                tensor: Tensor::from((new_shape, new_data))
            }
        }
    }

    //TODO: Make functionality for doing Vec * Matrix and Matrix * Vec

    impl<T, K> Add<Matrix<K>> for Matrix<T>
        where T: Copy + Add<K, Output=T>, K: Copy {
        type Output = Matrix<T>;

        fn add(self, rhs: Matrix<K>) -> Self::Output {
            (self.tensor + rhs.tensor).into()
        }
    }

    impl<T: ToString> Display for Matrix<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            for (index, c) in self.tensor.iter() {
                write!(f, "{}", c.to_string())?;
                if index[1] == self.tensor.shape[1] - 1 {
                    write!(f, "\n")?;
                }
            }
            Ok(())
        }
    }

    impl<T: ToString> Debug for Matrix<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            self.tensor.fmt(f)
        }
    }

    impl<T> Index<[usize; 2]> for Matrix<T> {
        type Output = T;

        fn index(&self, index: [usize; 2]) -> &Self::Output {
            &self.tensor[index]
        }
    }

    impl<T> IndexMut<[usize; 2]> for Matrix<T> {
        fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
            &mut self.tensor[index]
        }
    }

    impl<T: Clone> Clone for Matrix<T> {
        fn clone(&self) -> Self {
            Matrix {
                tensor: self.tensor.clone()
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use crate::tensor::{DimensionIndex, Tensor};

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    #[should_panic]
    fn zero_dim_tensor() {
        Tensor::<bool>::new([]);
    }

    #[test]
    #[should_panic]
    fn zero_size_tensor() {
        Tensor::<bool>::new([0]);
    }

    #[test]
    fn vector() {
        let t = Tensor::<bool>::new([10]);
        assert_eq!(t.shape().len(), 1);
        assert_eq!(t.shape()[0], 10);
        assert_eq!(t[[0]], false);
    }

    #[test]
    fn vector_debug() {
        let t = Tensor::<bool>::new([3]);
        assert_eq!(format!("{:?}", t), "{false, false, false}");
    }

    #[test]
    fn tensor_matrix() {
        let m = Tensor::<bool>::new([2, 2]);
        assert_eq!(m.shape().len(), 2);
        assert_eq!(m[[0, 0]], false);
    }

    #[test]
    fn tensor_fill() {
        let t = Tensor::fill([2, 2, 2], true);
        assert_eq!(t.shape().len(), 3);
        assert_eq!(t[[0, 0, 0]], true);
    }

    #[test]
    fn tensor_mut_index() {
        let mut t = Tensor::new([2, 2]);
        t[[0, 0]] = 1;
        assert_eq!(format!("{:?}", t), "{{1, 0}, {0, 0}}")
    }

    #[test]
    fn tensor_from_vec() {
        let v = vec![3; 8];
        let t: Tensor<_> = ([4, 4], v).into();
        assert_eq!(t.shape, vec![4, 4]);
        assert_eq!(t.data.len(), 16);
    }

    #[test]
    fn tensor_index_full() {
        let t = DimensionIndex::from(..);
        assert_eq!(t.index.len(), 0);
    }

    #[test]
    fn tensor_index_from() {
        let t = DimensionIndex::from(0..);
        assert_eq!(t.index[0].0, Some(0));
        assert_eq!(t.index[0].1, None);
    }

    #[test]
    fn tensor_index_to() {
        let t = DimensionIndex::from(..1);
        assert_eq!(t.index[0].0, None);
        assert_eq!(t.index[0].1, Some(1));
    }

    #[test]
    fn tensor_index_usize() {
        let t = DimensionIndex::from(0);
        assert_eq!(t.index[0].0, Some(0));
        assert_eq!(t.index[0].1, Some(0));
    }

    #[test]
    fn tensor_index_vec() {
        let t = DimensionIndex::from(vec![DimensionIndex::from(0), DimensionIndex::from(2)]);
        assert_eq!(t.index.len(), 2);
        assert_eq!(t.index[0].0, Some(0));
        assert_eq!(t.index[0].1, Some(0));
        assert_eq!(t.index[1].0, Some(2));
        assert_eq!(t.index[1].1, Some(2));
    }

    #[test]
    fn tensor_slice() {
        let t = Tensor::<i32>::new([4, 4]);
        let k = t.get(tensor_index![0, ..]).unwrap();
        assert_eq!(k.len(), 4)
    }

    #[test]
    fn tensor_mut_slice() {
        let mut t = Tensor::<i32>::new([4, 4]);
        let mut k = t.get_mut(tensor_index![0, ..]).unwrap();
        *k[3] = 23;
        assert_eq!(k.len(), 4);
        assert_eq!(t[[0, 3]], 23);
    }

    #[test]
    fn tensor_mult() {
        let k = Tensor::<i32>::from(([1, 3], vec![1, 2, 3]));
        let j = Tensor::<i32>::fill([3, 3], 1);
        let i = k.multiply(j).ok().unwrap();
        assert_eq!(i.shape, vec![1, 3]);
        assert_eq!(format!("{:?}", i), "{{6, 6, 6}}");
    }
}