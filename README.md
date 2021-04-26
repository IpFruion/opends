# opends

opends stands for Open Data Structures.

I wanted to create a very simple library (using as little external crate dependencies as possible) that I believe should be added to the standardard library.

This project is really meant for personal learning experience of Rust. Though I do believe in collaborative work and so I am releasing it here for others to add things and help make changes to mistakes (I am sure I have) that have arose.

My current status of this repository is that I have a `Tensor` structure that uses behind the scenes a `Vec` to create the mathmatical structure known as a Tensor with N Dimensions. Along with this is a structure called `Matrix` that uses some of the `Tensors` features and expands on it like multiplication.
**One thing to note is that `Matrix` I wish would delegate to `Tensor` so I wouldn't have to repeat so many functions.**

This a very new library and there are a few tests (and barely any documentation). I plan on expanding on those in the future.
