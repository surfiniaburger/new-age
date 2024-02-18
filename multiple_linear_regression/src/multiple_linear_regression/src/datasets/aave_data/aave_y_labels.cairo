use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq };
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn aave_y_labels() ->  Tensor<FP16x16>  {
    let tensor = TensorTrait::<FP16x16>::new( 
    shape: array![5].span(),
    data: array![ 
    FixedTrait::new(5072486, false ),
    FixedTrait::new(5072486, false ),
    FixedTrait::new(5079040, false ),
    FixedTrait::new(5085593, false ),
    FixedTrait::new(5111808, false ),
].span() 
 
);

return tensor; 
}