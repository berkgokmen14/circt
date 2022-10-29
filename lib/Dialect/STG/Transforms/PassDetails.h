//===- PassDetails.h - Pipeline pass class details --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the stuff shared between the different Pipeline passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the
// header guard on some systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_STG_TRANSFORMS_PASSDETAILS_H
#define DIALECT_STG_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/STG/STG.h"
#include "circt/Dialect/STG/STGPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace stg {

#define GEN_PASS_CLASSES
#include "circt/Dialect/STG/STGPasses.h.inc"

} // namespace stg
} // namespace circt

#endif // DIALECT_STG_TRANSFORMS_PASSDETAILS_H
