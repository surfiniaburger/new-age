use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq };
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn aave_x_features() ->  Tensor<FP16x16>  {
    let tensor = TensorTrait::<FP16x16>::new( 
    shape: array![5,10].span(),
    data: array![ 
    FixedTrait::new(1966080, false ),
    FixedTrait::new(61, false ),
    FixedTrait::new(484966, false ),
    FixedTrait::new(812646, false ),
    FixedTrait::new(13369344, false ),
    FixedTrait::new(3604, false ),
    FixedTrait::new(7798784, false ),
    FixedTrait::new(1880883, false ),
    FixedTrait::new(5006950, false ),
    FixedTrait::new(220856320, false ),
    FixedTrait::new(1900544, false ),
    FixedTrait::new(87, false ),
    FixedTrait::new(488243, false ),
    FixedTrait::new(812646, false ),
    FixedTrait::new(13434880, false ),
    FixedTrait::new(3604, false ),
    FixedTrait::new(7798784, false ),
    FixedTrait::new(1880883, false ),
    FixedTrait::new(5006950, false ),
    FixedTrait::new(220856320, false ),
    FixedTrait::new(1835008, false ),
    FixedTrait::new(114, false ),
    FixedTrait::new(525598, false ),
    FixedTrait::new(812646, false ),
    FixedTrait::new(13565952, false ),
    FixedTrait::new(3604, false ),
    FixedTrait::new(7798784, false ),
    FixedTrait::new(1887436, false ),
    FixedTrait::new(5013504, false ),
    FixedTrait::new(217579520, false ),
    FixedTrait::new(1769472, false ),
    FixedTrait::new(138, false ),
    FixedTrait::new(628490, false ),
    FixedTrait::new(838860, false ),
    FixedTrait::new(13893632, false ),
    FixedTrait::new(3604, false ),
    FixedTrait::new(8126464, false ),
    FixedTrait::new(1874329, false ),
    FixedTrait::new(5046272, false ),
    FixedTrait::new(208404480, false ),
    FixedTrait::new(1703936, false ),
    FixedTrait::new(1, false ),
    FixedTrait::new(655360, false ),
    FixedTrait::new(924057, false ),
    FixedTrait::new(14090240, false ),
    FixedTrait::new(3768, false ),
    FixedTrait::new(8388608, false ),
    FixedTrait::new(1880883, false ),
    FixedTrait::new(5065932, false ),
    FixedTrait::new(206438400, false ),
].span() 
 
);

return tensor; 
}