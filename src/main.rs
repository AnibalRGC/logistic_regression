extern crate image;

use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::OwnedRepr;
use ndarray::{arr2, s, Array, ArrayBase, Axis, Dim, Ix1, Ix2, Ix4};
use plotters::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/*Save image
fn save_image() {
let image_index: usize = 25;
let mut img: RgbImage = ImageBuffer::from_fn(
    64,
    64,
    |i, j| {
        //println!("[{},{}] : {:#?}", &i, &j, &test_set_x_orig.slice(s![0, i, j, ..]));
        let test_arr = train_set_x_orig.slice(s![image_index , j as usize, i as usize,
        ..]).to_vec();
        let r = test_arr[0];
        let g = test_arr[1];
        let b = test_arr[2];

        Rgb([r, g, b])
    });

img.save("picture.png").unwrap();
}
---------------------------- */

fn propagate(
    w: &Array<f64, Ix2>,
    b: f64,
    x: &Array<f64, Ix2>,
    y: &Array<f64, Ix2>,
) -> (HashMap<&'static str, Box<dyn Debug + 'static>>, f64) {
    let m = x.shape()[1] as f64;
    let A = (w.t().dot(x) + b).map(|&x| sigmoid(x));

    let cost = (-1.0 / m)
        * (y * &A.map(|&x| f64::ln(x)) + (y.map(|&x| 1.0 - x)) * &A.map(|&a| f64::ln(1.0 - a)))
            .sum();

    let dw = (1.0 / m) * x.dot(&(&A - y).t());
    let db = (1.0 / m) * &(&A - y).sum();

    let mut grads: HashMap<&'static str, Box<dyn Debug + 'static>> = HashMap::new();
    grads.insert("dw", Box::new(dw));
    grads.insert("db", Box::new(db));

    (grads, cost)
}

type CostHash = (
    HashMap<&'static str, Box<dyn Debug + 'static>>,
    HashMap<&'static str, Box<dyn Debug + 'static>>,
    Vec<f64>,
);

fn optimize(
    w: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    b: f64,
    x: &Array<f64, Ix2>,
    y: &Array<f64, Ix2>,
    _learning_rate: f64,
    _print_cost: bool,
) -> CostHash {
    let costs: Vec<f64> = Vec::<f64>::new();
    let (grads, cost) = propagate(&w, b, &x, &y);

    println!("\ndw : {:?}", grads["dw"]);
    println!("\ndb : {:?}", grads["db"]);
    println!("\nCost: {:#?}", cost);

    let mut params: HashMap<&'static str, Box<dyn Debug + 'static>> = HashMap::new();
    params.insert("w", Box::new(w));
    params.insert("b", Box::new(b));

    (params, grads, costs)
}

fn main() -> hdf5::Result<()> {
    let _e = hdf5::silence_errors();

    {
        let file_train = hdf5::File::open(
            "/home/rce/dev/learning/deeplearning\
    .ai/logistic_regression/data/datasets/train_catvnoncat.h5",
        )?;
        let file_test = hdf5::File::open(
            "/home/rce/dev/learning/deeplearning\
    .ai/logistic_regression/data/datasets/test_catvnoncat.h5",
        )?;

        let train_set_x_dataset = file_train.dataset("train_set_x")?;
        let train_set_y_dataset = file_train.dataset("train_set_y")?;
        let test_set_x_dataset = file_test.dataset("test_set_x")?;
        let test_set_y_dataset = file_test.dataset("test_set_y")?;

        println!("Train set x shape : {:?}", train_set_x_dataset.shape());
        println!("Train set y shape : {:?}", train_set_y_dataset.shape());
        println!("Test set x shape : {:?}", test_set_x_dataset.shape());
        println!("Test set y shape : {:?}", test_set_y_dataset.shape());

        let train_set_x_orig = train_set_x_dataset.read::<f64, Ix4>()?;
        let train_set_y_orig = train_set_y_dataset.read::<u8, Ix1>()?;
        let test_set_x_orig = test_set_x_dataset.read::<f64, Ix4>()?;
        println!("{:?}", train_set_x_orig.shape());
        let _test_set_y_orig = test_set_y_dataset.read::<u8, Ix1>()?;
        //let test_set_x_orig_first_image = test_set_x_orig.into_shape((1, 614400)).unwrap();
        //let test_set_x_orig_first_image = test_set_x_orig.into_shape((1, 614400)).unwrap();
        //println!("{:#?}", &test_set_x_orig.slice(s![0, .., .., ..]));

        // Reshape the training and test examples
        let train_set_x_flatten = train_set_x_orig.into_shape((209, 12288)).unwrap();
        let test_set_x_flatten = test_set_x_orig.into_shape((50, 12288)).unwrap();
        println!(
            "Train set flatten x shape : {:?}",
            train_set_x_flatten.shape()
        );
        println!(
            "Test set flatten x shape : {:?}",
            test_set_x_flatten.shape()
        );

        //println!("{:?}", train_set_x_flatten.slice(s![0, ..]));

        // Standardize the dataset
        let _train_x = train_set_x_flatten / 255.0;
        let _train_y = train_set_y_orig;
        let _test_x = test_set_x_flatten / 255.0;

        // Initialize with zeros
        //let w: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = Array::zeros((209, 12288));
        //let b = 0.0;

        // Propagate
        // test purpose
        let w = arr2(&[[1.0], [2.0]]);
        let b = 2.0;
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let y = arr2(&[[1.0, 0.0]]);

        let (params, grads, costs) = optimize(w.clone(), b, &x, &y, 0.5, false);
    }

    Ok(())
}

