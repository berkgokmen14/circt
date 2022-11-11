//===- HLSDialect.cpp - Implement the HLS dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HLS dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HLS/HLSDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace circt::hls;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

namespace {

// We implement the OpAsmDialectInterface so that HLS dialect operations
// automatically interpret the name attribute on operations as their SSA name.
struct HLSOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op, OpAsmSetValueNameFn setNameFn) const {}
};

} // end anonymous namespace

void HLSDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HLS/HLS.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<HLSOpAsmDialectInterface>();
}

// Provide implementations for the enums and attributes we use.
#include "circt/Dialect/HLS/HLSDialect.cpp.inc"
