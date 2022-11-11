// RUN: circt-opt %s -hls-unroll-loops -canonicalize -split-input-file | FileCheck %s

module {
  func.func @main(%A: memref<1024xi32>, %B : memref<1024xi32>) {
    %c1_i32 = arith.constant 1 : i32
    affine.for %arg2 = 0 to 1024 step 1 {
      %7 = affine.load %A[%arg2] : memref<1024xi32>
      %8 = arith.addi %7, %c1_i32 : i32
      affine.store %8, %B[%arg2] : memref<1024xi32>
    } {hls.unroll = 2 : i64}
    return
  }
}
