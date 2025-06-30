module {
  func.func @main(%arg0: !torch.vtensor<[?,4],f32>, %arg1: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[],f32> {
    %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[?,4],f32>, !torch.vtensor<[?,4],f32> -> !torch.vtensor<[?,4],f32>
    %none = torch.constant.none
    %1 = torch.aten.sum %0, %none : !torch.vtensor<[?,4],f32>, !torch.none -> !torch.vtensor<[],f32>
    return %1 : !torch.vtensor<[],f32>
  }
}
