//===- UnrollForLoopSchedule.cpp - Unroll nested loops for parallelism ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace circt::analysis;

namespace {
struct UnrollForLoopSchedule
    : public circt::UnrollForLoopScheduleBase<UnrollForLoopSchedule> {
  using UnrollForLoopScheduleBase<
      UnrollForLoopSchedule>::UnrollForLoopScheduleBase;
  void runOnOperation() override;

private:
  std::tuple<unsigned, AffineForOp>
  getDeepestNestedLoop(std::tuple<unsigned, AffineForOp> root);
  DenseSet<AffineMapAccessInterface> updateMemoryOps(AffineForOp innermostLoop);
  unsigned getMinDepDistance(AffineForOp innermostLoop);
  SmallVector<AffineForOp> nestedLoops;
  DenseMap<AffineForOp, DenseSet<AffineMapAccessInterface>> loopToMemOps;
  MemoryDependenceAnalysis *memDepAnalysis;
};
} // namespace

std::tuple<unsigned, AffineForOp> UnrollForLoopSchedule::getDeepestNestedLoop(
    std::tuple<unsigned, AffineForOp> root) {
  auto forOp = std::get<AffineForOp>(root);
  auto depth = std::get<unsigned>(root);
  if (forOp.getLoopBody().getOps<AffineForOp>().empty())
    return root;

  unsigned deepestDepth = depth;
  AffineForOp deepestLoop = forOp;
  for (auto loop : forOp.getLoopBody().getOps<AffineForOp>()) {
    auto result = getDeepestNestedLoop(std::tuple(depth + 1, loop));
    auto newDepth = std::get<unsigned>(result);
    if (newDepth > deepestDepth) {
      deepestDepth = newDepth;
      deepestLoop = std::get<AffineForOp>(result);
    }
  }
  return std::tuple(deepestDepth, deepestLoop);
}

DenseSet<AffineMapAccessInterface>
UnrollForLoopSchedule::updateMemoryOps(AffineForOp innermostLoop) {
  DenseSet<AffineMapAccessInterface> memOps;
  auto &bodyRegion = innermostLoop.getLoopBody();
  // Only affine memory dependencies are supported
  auto affineLdOps = bodyRegion.getOps<AffineLoadOp>();
  auto affineStOps = bodyRegion.getOps<AffineStoreOp>();
  memOps.insert(affineLdOps.begin(), affineLdOps.end());
  memOps.insert(affineStOps.begin(), affineStOps.end());
  loopToMemOps[innermostLoop] = memOps;
  return memOps;
}

unsigned UnrollForLoopSchedule::getMinDepDistance(AffineForOp innermostLoop) {
  auto memOps = loopToMemOps[innermostLoop];
  unsigned minDistance = std::numeric_limits<unsigned>::max();
  for (auto memOp : memOps) {

    ArrayRef<MemoryDependence> dependences =
        memDepAnalysis->getDependences(memOp);
    if (dependences.empty())
      continue;

    for (MemoryDependence memoryDep : dependences) {
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      unsigned distance = *memoryDep.dependenceComponents.back().lb;
      minDistance = std::min(minDistance, distance);
    }
  }

  return minDistance;
}

// LogicalResult stToLdBypass(AffineStoreOp storeOp, AffineLoadOp loadOp) {
//   loadOp->replaceAllUsesWith(storeOp.getValueToStore());
//   if (storeOp->getUses().empty())
//     storeOp->erase();
//   loadOp->erase();
//   return success();
// }

void UnrollForLoopSchedule::runOnOperation() {
  // Get loopNests in Function
  for (auto rootForLoop :
       getOperation().getFunctionBody().getOps<AffineForOp>())
    nestedLoops.push_back(std::get<AffineForOp>(getDeepestNestedLoop(
        std::tuple(std::numeric_limits<unsigned>::min(), rootForLoop))));

  // Gert scheduling analyses
  memDepAnalysis = &getAnalysis<MemoryDependenceAnalysis>();

  for (auto innerLoop : nestedLoops) {
    updateMemoryOps(innerLoop);
    unsigned minDepDistance = getMinDepDistance(innerLoop);
    if (minDepDistance == std::numeric_limits<unsigned>::max()) {
      llvm::errs() << "unrolling by factor "
                   << "full\n";
      auto result = loopUnrollFull(innerLoop);
      if (result.failed())
        return signalPassFailure();
    } else if (minDepDistance - 1 > 1) {
      llvm::errs() << "unrolling by factor " << minDepDistance - 1 << "\n";
      auto result = loopUnrollByFactor(innerLoop, minDepDistance - 1);
      if (result.failed())
        return signalPassFailure();
    }
  }
}

namespace circt {
std::unique_ptr<mlir::Pass> createUnrollForLoopSchedulePass() {
  return std::make_unique<UnrollForLoopSchedule>();
}
} // namespace circt
