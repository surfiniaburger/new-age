// use traits::Into;
use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};

use multiple_linear_regresion::datasets::aave_data::aave_x_features::aave_x_features;
use multiple_linear_regresion::datasets::aave_data::aave_y_labels::aave_y_labels; 
use multiple_linear_regresion::datasets::user_inputs_data::aave_weth_revenue_data_input::{aave_weth_revenue_data_input };  

use multiple_linear_regresion::model::multiple_linear_regression_model::{
     MultipleLinearRegressionModel, MultipleLinearRegression, MultipleLinearRegressionModelTrait
};
use multiple_linear_regresion::data_preprocessing::{Dataset, DatasetTrait};
use multiple_linear_regresion::helper_functions::{get_tensor_data_by_row, transpose_tensor, calculate_mean , 
calculate_r_score, normalize_user_x_inputs, rescale_predictions};

use orion::numbers::{FP16x16,  FixedTrait};


use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, U32Tensor, U32TensorAdd, 
    FP16x16TensorSub, FP16x16TensorAdd, FP16x16TensorDiv, FP16x16TensorMul};

#[test]
#[available_gas(99999999999999999)]
fn multiple_linear_regression_test() {

//Constructing our model
let mut main_x_vals = aave_x_features();
let mut main_y_vals = aave_y_labels();
let mut dataset = Dataset{x_values: main_x_vals,y_values:main_y_vals};
let mut normalized_dataset = dataset.normalize_dataset();
let mut model = MultipleLinearRegression(normalized_dataset);
let mut model_coefficients = model.coefficients;
let mut reconstructed_ys = model.predict (normalized_dataset.x_values);
let mut r_squared_score = calculate_r_score(normalized_dataset.y_values,reconstructed_ys);
// r_squared_score.print(); // model accuracy around 0.9969482421875 

// checking if data has been normalized correctly
assert(normalized_dataset.x_values.max_in_tensor() <= FixedTrait::new(65536, false), 'normalized x not between 0-1');
assert(normalized_dataset.x_values.min_in_tensor() >= FixedTrait::new(0, false), 'normalized x not between 0-1');
assert(normalized_dataset.y_values.max_in_tensor() <= FixedTrait::new(65536, false), 'normalized y not between 0-1');
assert(normalized_dataset.x_values.min_in_tensor() >= FixedTrait::new(0, false), 'normalized y not between 0-1');
// performing checks on the shape of normalized data
assert(normalized_dataset.x_values.data.len()== main_x_vals.data.len() && 
normalized_dataset.y_values.data.len()== main_y_vals.data.len() , 'normalized data shape mismatch');
// performing checks on the shape of coefficient values (gradient vals + bias)
assert(model.coefficients.data.len() == *main_x_vals.shape.at(1)+1, 'coefficient data shape mismatch');
// model accuracy deviance checks
assert(r_squared_score >= FixedTrait::new(62259, false), 'AAVE model acc. less than 95%');

// using the model to forecast aave's 7-day WETH net lifetime repayments forecast  
let last_7_days_aave_data = aave_weth_revenue_data_input();
let last_7_days_aave_data_normalized = normalize_user_x_inputs(last_7_days_aave_data, main_x_vals );
let mut forecast_results  = model.predict (last_7_days_aave_data_normalized); 
let mut rescale_forecasts = rescale_predictions(forecast_results, main_y_vals);  // PS. ** the rescaled forecasted outputs are denominated in thousands of ETH
// (*rescale_forecasts.data.at(0)).print();  // day1 forecast: 95.66773986816406 
// (*rescale_forecasts.data.at(1)).print();  // day2: 96.64869689941406
// (*rescale_forecasts.data.at(5)).print();  // day6: 99.44300842285156
// (*rescale_forecasts.data.at(6)).print();  // day7: 100.57145690917969
}