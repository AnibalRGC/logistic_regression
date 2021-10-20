#![allow(unused_variables)]
#![allow(clippy::many_single_char_names)]
extern crate image;

use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::OwnedRepr;
use ndarray::{s, Array, ArrayBase, Axis, Ix1, Ix2, Ix4};

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn save_image(train_set_x_orig: &Array<f64, Ix4>, index: usize) {
    let img: RgbImage = ImageBuffer::from_fn(64, 64, |i, j| {
        //println!("[{},{}] : {:#?}", &i, &j, &test_set_x_orig.slice(s![0, i, j, ..]));
        let test_arr = train_set_x_orig
            .slice(s![index, j as usize, i as usize, ..])
            .to_vec();
        let r = test_arr[0] as u8;
        let g = test_arr[1] as u8;
        let b = test_arr[2] as u8;

        Rgb([r, g, b])
    });

    img.save("picture.png").unwrap();
}

fn propagate(
    w: &Array<f64, Ix2>,
    b: f64,
    x: &Array<f64, Ix2>,
    y: &Array<f64, Ix2>,
) -> (Array<f64, Ix2>, f64, f64) {
    let m = x.shape()[1] as f64;
    let A = (w.t().dot(x) + b).map(|&x| sigmoid(x));

    let cost = (-1.0 / m)
        * (y * &A.map(|&x| f64::ln(x)) + (y.map(|&x| 1.0 - x)) * &A.map(|&a| f64::ln(1.0 - a)))
            .sum();

    let dw = (1.0 / m) * x.dot(&(&A - y).t());
    let db = (1.0 / m) * &(&A - y).sum();

    (dw, db, cost)
}

type CostHash = (Array<f64, Ix2>, f64, Array<f64, Ix2>, f64, Vec<f64>);

fn optimize(
    mut w: Array<f64, Ix2>,
    mut b: f64,
    x: &Array<f64, Ix2>,
    y: &Array<f64, Ix2>,
    num_iterations: usize,
    learning_rate: f64,
    print_cost: bool,
) -> CostHash {
    let mut costs: Vec<f64> = Vec::<f64>::new();
    //let mut dw: Array<f64, Ix2> = Array::<f64, Ix2>::zeros((2, 1));
    //let mut db: f64 = 0.0;

    let (mut dw, mut db, cost) = propagate(&w, b, x, y);

    for i in 0..num_iterations {
        // Cost and gradient calculation
        let (dw_, db_, cost) = propagate(&w, b, x, y);
        dw = dw_.clone();
        db = db_.clone();

        // update rule
        w = w - learning_rate * &dw;
        b -= learning_rate * db;

        if i % 100 == 0 {
            costs.push(cost);
        }

        if print_cost && (i % 100 == 0) {
            println!("Cost after iteration {}: {}", i, cost);
        }
    }

    (w, b, dw, db, costs)
}

fn predict(
    w: ArrayBase<OwnedRepr<f64>, Ix2>,
    b: f64,
    x: &Array<f64, Ix2>,
) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    let m = x.shape()[1];
    let mut y_prediction: ArrayBase<OwnedRepr<f64>, Ix2> = Array::zeros((1, 209));
    let w = w.into_shape((x.shape()[0], 1)).unwrap();

    // Compute vector "A" predicting the probabilities of a cat being present in the picture
    let a = (w.t().dot(x) + b).map(|&x| sigmoid(x));

    for i in 0..a.shape()[1] {
        // Convert probabilities A[0,i] to actual predictions p[0,i]
        if a[[0, i]] <= 0.5 {
            y_prediction[[0, i]] = 0.0;
        } else {
            y_prediction[[0, i]] = 1.0;
        }
    }

    y_prediction
}

fn model(
    x_train: &Array<f64, Ix2>,
    y_train: &Array<f64, Ix2>,
    x_test: &Array<f64, Ix2>,
    y_test: &Array<f64, Ix2>,
    num_iterations: usize,
    learning_rate: f64,
    print_cost: bool,
) {
    // Initialize with zeros
    let w: ArrayBase<OwnedRepr<f64>, Ix2> = Array::zeros((12288, 1));
    let b = 0.0;

    let (w, b, dw, db, costs) = optimize(
        w,
        b,
        x_train,
        y_train,
        num_iterations,
        learning_rate,
        print_cost,
    );

    // Predict test/train set examples (≈ 2 lines of code)
    let y_prediction_train = predict(w.clone(), b, x_train);
    let y_prediction_test_extended = predict(w.clone(), b, x_test);

    let mut y_prediction_test: ArrayBase<OwnedRepr<f64>, Ix2> = Array::zeros((1, 50));
    for i in 0..50 {
        y_prediction_test[(0, i)] = y_prediction_test_extended[(0, i)];
    }

    let diff_train = (y_prediction_train - y_train).map(|diff| diff.abs());
    let diff_train_sum: f64 = diff_train.iter().sum();
    let diff_train_mean: f64 = diff_train_sum as f64 / diff_train.len() as f64;

    let diff_test = (y_prediction_test - y_test).map(|diff| diff.abs());
    let diff_test_sum: f64 = diff_test.iter().sum();
    let diff_test_mean: f64 = diff_test_sum as f64 / diff_test.len() as f64;

    // Print train/test Errors
    println!("train accuracy: {} %", 100.0 - (diff_train_mean * 100.0));
    println!("test accuracy: {} %", 100.0 - (diff_test_mean * 100.0));
}

fn main() -> hdf5::Result<()> {
    let _e = hdf5::silence_errors();

    {
        // --------------- load_dataset() -------------------
        let file_train = hdf5::File::open("data/datasets/train_catvnoncat.h5")?;
        let file_test = hdf5::File::open("data/datasets/test_catvnoncat.h5")?;

        let train_set_x_dataset = file_train.dataset("train_set_x")?;
        let train_set_y_dataset = file_train.dataset("train_set_y")?;
        let test_set_x_dataset = file_test.dataset("test_set_x")?;
        let test_set_y_dataset = file_test.dataset("test_set_y")?;

        println!(
            "Number of training examples: m_train = {}",
            train_set_x_dataset.shape()[0]
        );
        println!(
            "Number of testing examples: m_test = {}",
            test_set_x_dataset.shape()[0]
        );
        println!(
            "Height/Width of each image: num_px = {}",
            train_set_x_dataset.shape()[1]
        );
        println!("Train set x shape : {:?}", train_set_x_dataset.shape());
        println!("Train set y shape : {:?}", train_set_y_dataset.shape());
        println!("Test set x shape : {:?}", test_set_x_dataset.shape());
        println!("Test set y shape : {:?}", test_set_y_dataset.shape());
        println!("------------------------------------------------------");

        // Example of a picture
        let train_set_x_orig = train_set_x_dataset.read::<f64, Ix4>()?;
        let train_set_y_orig = train_set_y_dataset.read::<f64, Ix1>()?;
        let test_set_x_orig = test_set_x_dataset.read::<f64, Ix4>()?;
        let test_set_y_orig = test_set_y_dataset.read::<f64, Ix1>()?;
        save_image(&train_set_x_orig, 26);

        //println!("{:#?}", &test_set_x_orig.slice(s![0, .., .., ..]));

        // Reshape the training and test examples
        let train_set_x_flatten = train_set_x_orig
            .into_shape((209, 12288))
            .unwrap()
            .reversed_axes();
        let test_set_x_flatten = test_set_x_orig
            .into_shape((50, 12288))
            .unwrap()
            .reversed_axes();
        println!(
            "Train set x flatten shape: {:?}",
            train_set_x_flatten.shape()
        );
        println!("Train set y shape: {:?}", train_set_y_orig.shape());
        println!("Test set x flatten shape: {:?}", test_set_x_flatten.shape());
        println!("Test set y shape: {:?}", test_set_y_orig.shape());
        println!(
            "sanity check after reshaping: {:?}",
            train_set_x_flatten
                .lanes(Axis(0))
                .into_iter()
                .next()
                .unwrap()
                .to_vec()[0..5]
                .to_vec()
        );
        //for col in train_set_x_flatten.lanes(Axis(0)).into_iter() { println!("-> {}", col); }
        //println!("{:?}", train_set_x_flatten.slice(s![0, ..]));
        assert_eq!(
            train_set_x_flatten
                .lanes(Axis(0))
                .into_iter()
                .next()
                .unwrap()
                .to_vec()[0..5]
                .to_vec(),
            vec![17.0, 31.0, 56.0, 22.0, 33.0]
        );
        println!("------------------------------------------------------");

        // Standardize the dataset
        let train_set_x = train_set_x_flatten / 255.0;
        let test_set_x = test_set_x_flatten / 255.0;
        let train_set_y_orig_array =
            Array::from_shape_vec((1, 209), train_set_y_orig.to_vec()).unwrap();
        let test_set_y_orig_array =
            Array::from_shape_vec((1, 50), test_set_y_orig.to_vec()).unwrap();

        model(
            &train_set_x,
            &train_set_y_orig_array,
            &test_set_x,
            &test_set_y_orig_array,
            2000,
            0.005,
            true,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, ArrayView};

    #[test]
    fn test_propagate() {
        let mut w = arr2(&[[1.0], [2.0]]);
        let mut b = 2.0;
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let y = arr2(&[[1.0, 0.0]]);

        let (mut dw, mut db, cost) = propagate(&w, b, &x, &y);

        assert!(
            dw.index_axis(Axis(1), 0) == ArrayView::from(&[0.9999321585374046, 1.9998026197868162])
        );
        assert_eq!(db, 0.49993523062470574);
        assert_eq!(cost, 6.000064773192205);
    }

    #[test]
    fn test_propagate2() {
        let mut w = arr2(&[[1.0], [2.0], [1.0]]);
        let mut b = 2.0;
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let y = arr2(&[[1.0, 0.0]]);

        let (mut dw, mut db, cost) = propagate(&w, b, &x, &y);

        println!("{:?}", dw);
        assert!(
            dw.index_axis(Axis(1), 0)
                == ArrayView::from(&[0.9999995690060066, 1.9999987222479993, 2.999997875489992])
        );
        assert_eq!(db, 0.49999957662099637);
        assert_eq!(cost, 9.000000423805828);
    }

    #[test]
    fn test_optimize() {
        let mut w = arr2(&[[1.0], [2.0]]);
        let mut b = 2.0;
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let y = arr2(&[[1.0, 0.0]]);

        let (w, b, dw, db, costs) = optimize(w.clone(), b, &x, &y, 100, 0.009, true);

        assert!(
            w.index_axis(Axis(1), 0) == ArrayView::from(&[0.1124578970863514, 0.2310677467761294])
        );
        assert_eq!(b, 1.5593049248448891);
        assert!(
            dw.index_axis(Axis(1), 0) == ArrayView::from(&[0.9015842801285687, 1.7625084234859343])
        );
        assert_eq!(db, 0.4304620716786828);
    }

    #[test]
    fn test_optimize2() {
        let mut w = arr2(&[[1.0], [2.0], [1.0]]);
        let mut b = 2.0;
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let y = arr2(&[[1.0, 0.0]]);

        let (w, b, dw, db, costs) = optimize(w.clone(), b, &x, &y, 100, 0.009, true);

        println!("{:?}", db);
        assert!(
            w.index_axis(Axis(1), 0)
                == ArrayView::from(&[0.3170627070069936, 0.7152088891154749, -0.8866449287760431])
        );
        assert_eq!(b, 1.6990730910542418);
        assert!(
            dw.index_axis(Axis(1), 0)
                == ArrayView::from(&[
                    0.18672293477143087,
                    0.090135993506539,
                    -0.006450947758352754
                ])
        );
        assert_eq!(db, -0.048293470632445934);
    }

    #[test]
    fn test_predict() {
        let w = arr2(&[[1.0], [2.0]]);
        let mut b = 2.0;
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let y_prediction = predict(w.clone(), b, &x);

        assert!(
            y_prediction
                .lanes(Axis(1))
                .into_iter()
                .next()
                .unwrap()
                .to_vec()[0..2]
                .to_vec()
                == vec![1.0, 1.0]
        );
    }
}
