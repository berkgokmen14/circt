//===- PassDetails.h - HLS pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the stuff shared between the different HLS passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the
// header guard on some systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_HLS_TRANSFORMS_PASSDETAILS_H
#define DIALECT_HLS_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace circt {
namespace hls {

#define GEN_PASS_CLASSES
#include "circt/Dialect/HLS/HLSPasses.h.inc"

} // namespace hls
} // namespace circt

#endif // DIALECT_HLS_TRANSFORMS_PASSDETAILS_H
