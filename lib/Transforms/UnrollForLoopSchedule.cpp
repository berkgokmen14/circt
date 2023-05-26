//===- UnrollForLoopSchedule.cpp - Unroll nested loops for parallelism ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::affine;

namespace {
struct UnrollForLoopSchedule
    : public circt::UnrollForLoopScheduleBase<UnrollForLoopSchedule> {
  using UnrollForLoopScheduleBase<
      UnrollForLoopSchedule>::UnrollForLoopScheduleBase;
  void runOnOperation() override;

private:
  DenseSet<AffineMapAccessInterface> getMemoryOps(AffineForOp forOp);
};
} // namespace

DenseSet<AffineMapAccessInterface>
UnrollForLoopSchedule::getMemoryOps(AffineForOp forOp) {
  DenseSet<AffineMapAccessInterface> memOps;
  auto &bodyRegion = forOp.getLoopBody();
  // Only affine memory dependencies are supported
  auto affineLdOps = bodyRegion.getOps<AffineLoadOp>();
  auto affineStOps = bodyRegion.getOps<AffineStoreOp>();
  memOps.insert(affineLdOps.begin(), affineLdOps.end());
  memOps.insert(affineStOps.begin(), affineStOps.end());
  return memOps;
}

void UnrollForLoopSchedule::runOnOperation() {
  SmallVector<SmallVector<AffineForOp>> loopNests;
  for (auto rootForLoop : getOperation().getOps<AffineForOp>()) {
    SmallVector<AffineForOp> loopNest;
    loopNest.push_back(rootForLoop);
    getPerfectlyNestedLoops(loopNest, rootForLoop);
    loopNests.push_back(loopNest);
  }

  auto memDepAnalysis =
      getAnalysis<circt::analysis::MemoryDependenceAnalysis>();

  for (auto &loopNest : loopNests) {
    auto innerLoop = loopNest.back();

    if (!innerLoop.getLoopBody().getOps<memref::LoadOp>().empty() ||
        !innerLoop.getLoopBody().getOps<memref::StoreOp>().empty())
      return signalPassFailure();

    auto memOps = getMemoryOps(innerLoop);
    for (auto dest : memOps) {
      auto inboudDeps = memDepAnalysis.getDependences(dest);
      for (const auto &inboudDep : inboudDeps) {
        if (llvm::isa<AffineMapAccessInterface>(*inboudDep.source) &&
            memOps.contains(
                llvm::cast<AffineMapAccessInterface>(*inboudDep.source)) &&
            inboudDep.dependenceType != DependenceResult::ResultEnum::Failure) {
          if (inboudDep.dependenceType ==
              DependenceResult::ResultEnum::HasDependence)
            llvm::errs() << "there is a ";
          else
            llvm::errs() << "there is NOT a ";
          llvm::errs() << "dep from " << *inboudDep.source << " to " << dest
                       << " with distance " << inboudDep.dependenceType << "\n";
        }
      }
    }
  }
}

namespace circt {
std::unique_ptr<mlir::Pass> createUnrollForLoopSchedulePass() {
  return std::make_unique<UnrollForLoopSchedule>();
}
} // namespace circt
