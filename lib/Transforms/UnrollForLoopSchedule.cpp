//===- UnrollForLoopSchedule.cpp - Unroll nested loops for parallelism ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct UnrollForLoopSchedule
    : public circt::UnrollForLoopScheduleBase<UnrollForLoopSchedule> {
  using UnrollForLoopScheduleBase<
      UnrollForLoopSchedule>::UnrollForLoopScheduleBase;
  void runOnOperation() override;

private:
};
} // namespace

void UnrollForLoopSchedule::runOnOperation() {
  llvm::errs() << "UnrollForLoopSchedule\n";
  auto func = getOperation();
}

namespace circt {
std::unique_ptr<mlir::Pass> createUnrollForLoopSchedulePass() {
  return std::make_unique<UnrollForLoopSchedule>();
}
} // namespace circt
