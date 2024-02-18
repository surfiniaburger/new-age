use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_l2_weight() -> Tensor<FP16x16> {
    let mut shape = array![10, 10];

    let mut data = array![FP16x16 { mag: 55691, sign: false }, FP16x16 { mag: 60981, sign: true }, FP16x16 { mag: 2167, sign: false }, FP16x16 { mag: 35522, sign: false }, FP16x16 { mag: 50580, sign: true }, FP16x16 { mag: 36901, sign: true }, FP16x16 { mag: 27832, sign: false }, FP16x16 { mag: 70238, sign: true }, FP16x16 { mag: 7378, sign: false }, FP16x16 { mag: 1045, sign: false }, FP16x16 { mag: 64911, sign: true }, FP16x16 { mag: 11577, sign: true }, FP16x16 { mag: 52988, sign: true }, FP16x16 { mag: 59834, sign: true }, FP16x16 { mag: 47475, sign: false }, FP16x16 { mag: 65942, sign: true }, FP16x16 { mag: 77267, sign: true }, FP16x16 { mag: 16957, sign: false }, FP16x16 { mag: 8135, sign: true }, FP16x16 { mag: 55504, sign: false }, FP16x16 { mag: 39940, sign: true }, FP16x16 { mag: 51320, sign: false }, FP16x16 { mag: 7006, sign: true }, FP16x16 { mag: 3112, sign: false }, FP16x16 { mag: 6244, sign: false }, FP16x16 { mag: 40488, sign: true }, FP16x16 { mag: 44324, sign: false }, FP16x16 { mag: 31386, sign: true }, FP16x16 { mag: 52834, sign: true }, FP16x16 { mag: 29150, sign: false }, FP16x16 { mag: 40079, sign: true }, FP16x16 { mag: 38529, sign: true }, FP16x16 { mag: 29347, sign: true }, FP16x16 { mag: 62096, sign: false }, FP16x16 { mag: 36843, sign: true }, FP16x16 { mag: 17589, sign: true }, FP16x16 { mag: 57873, sign: true }, FP16x16 { mag: 25434, sign: false }, FP16x16 { mag: 53882, sign: true }, FP16x16 { mag: 28448, sign: false }, FP16x16 { mag: 21436, sign: true }, FP16x16 { mag: 29774, sign: false }, FP16x16 { mag: 69024, sign: false }, FP16x16 { mag: 1613, sign: false }, FP16x16 { mag: 41542, sign: false }, FP16x16 { mag: 34974, sign: false }, FP16x16 { mag: 40918, sign: true }, FP16x16 { mag: 8148, sign: true }, FP16x16 { mag: 11929, sign: true }, FP16x16 { mag: 61370, sign: true }, FP16x16 { mag: 42157, sign: false }, FP16x16 { mag: 72066, sign: true }, FP16x16 { mag: 19606, sign: true }, FP16x16 { mag: 35384, sign: false }, FP16x16 { mag: 43307, sign: false }, FP16x16 { mag: 49945, sign: true }, FP16x16 { mag: 606, sign: true }, FP16x16 { mag: 40111, sign: false }, FP16x16 { mag: 19353, sign: true }, FP16x16 { mag: 56818, sign: true }, FP16x16 { mag: 37247, sign: true }, FP16x16 { mag: 32370, sign: true }, FP16x16 { mag: 32904, sign: true }, FP16x16 { mag: 8793, sign: true }, FP16x16 { mag: 43652, sign: false }, FP16x16 { mag: 53144, sign: false }, FP16x16 { mag: 30691, sign: false }, FP16x16 { mag: 56392, sign: true }, FP16x16 { mag: 52287, sign: false }, FP16x16 { mag: 17727, sign: true }, FP16x16 { mag: 48823, sign: false }, FP16x16 { mag: 11143, sign: true }, FP16x16 { mag: 24018, sign: false }, FP16x16 { mag: 45997, sign: true }, FP16x16 { mag: 70922, sign: true }, FP16x16 { mag: 39306, sign: false }, FP16x16 { mag: 46105, sign: true }, FP16x16 { mag: 22233, sign: false }, FP16x16 { mag: 82898, sign: true }, FP16x16 { mag: 21462, sign: false }, FP16x16 { mag: 30100, sign: false }, FP16x16 { mag: 60264, sign: false }, FP16x16 { mag: 51128, sign: true }, FP16x16 { mag: 21939, sign: false }, FP16x16 { mag: 11800, sign: true }, FP16x16 { mag: 42498, sign: true }, FP16x16 { mag: 57284, sign: true }, FP16x16 { mag: 2405, sign: true }, FP16x16 { mag: 57232, sign: false }, FP16x16 { mag: 21120, sign: true }, FP16x16 { mag: 35463, sign: true }, FP16x16 { mag: 8617, sign: true }, FP16x16 { mag: 49446, sign: false }, FP16x16 { mag: 23066, sign: true }, FP16x16 { mag: 95310, sign: true }, FP16x16 { mag: 828, sign: false }, FP16x16 { mag: 6067, sign: true }, FP16x16 { mag: 39178, sign: false }, FP16x16 { mag: 49958, sign: false }, FP16x16 { mag: 9718, sign: true }];

    TensorTrait::new(shape.span(), data.span())
}